#include "visual_graph_slam/loop_closure/loop_detector.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include <DBoW2/DBoW2.h>
#include <DBoW2/FORB.h>

namespace {

namespace fs = std::filesystem;

std::vector<DBoW2::FORB::TDescriptor> toBowDescriptors(const cv::Mat& descriptors) {
    std::vector<DBoW2::FORB::TDescriptor> out;
    if (descriptors.empty()) {
        return out;
    }

    out.reserve(descriptors.rows);
    for (int i = 0; i < descriptors.rows; ++i) {
        DBoW2::FORB::TDescriptor d;
        descriptors.row(i).copyTo(d);
        out.push_back(d);
    }
    return out;
}

class BinaryOrbVocabulary
    : public DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> {
public:
    using Base = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;
    using Base::Base;

    static constexpr uint32_t kNoWord = 0xFFFFFFFFu;

    bool loadFromFastBinary(const std::string& filename) {
        std::ifstream f(filename, std::ios::binary);
        if (!f) return false;

        char magic[4];
        f.read(magic, 4);
        if (!f || magic[0]!='D' || magic[1]!='B' || magic[2]!='V' || magic[3]!='C')
            return false;

        uint32_t version;
        f.read(reinterpret_cast<char*>(&version), 4);
        if (!f || version != 1u) return false;

        int32_t k, L, sc, wt;
        f.read(reinterpret_cast<char*>(&k),  4);
        f.read(reinterpret_cast<char*>(&L),  4);
        f.read(reinterpret_cast<char*>(&sc), 4);
        f.read(reinterpret_cast<char*>(&wt), 4);

        uint32_t nn, nw;
        f.read(reinterpret_cast<char*>(&nn), 4);
        f.read(reinterpret_cast<char*>(&nw), 4);
        if (!f) return false;

        m_k         = static_cast<int>(k);
        m_L         = static_cast<int>(L);
        m_scoring   = static_cast<DBoW2::ScoringType>(sc);
        m_weighting = static_cast<DBoW2::WeightingType>(wt);
        createScoringObject();

        m_nodes.clear();
        m_nodes.resize(nn);
        m_words.assign(nw, nullptr);

        for (uint32_t i = 0; i < nn; ++i) {
            uint32_t id, par, wid_raw;
            f.read(reinterpret_cast<char*>(&id),      4);
            f.read(reinterpret_cast<char*>(&par),     4);
            f.read(reinterpret_cast<char*>(&wid_raw), 4);
            double weight;
            f.read(reinterpret_cast<char*>(&weight),  8);

            auto& nd   = m_nodes[i];
            nd.id      = static_cast<DBoW2::NodeId>(id);
            nd.parent  = static_cast<DBoW2::NodeId>(par);
            nd.weight  = weight;

            nd.descriptor = cv::Mat(1, 32, CV_8U);
            f.read(reinterpret_cast<char*>(nd.descriptor.data), 32);

            uint32_t nc;
            f.read(reinterpret_cast<char*>(&nc), 4);
            nd.children.resize(nc);
            for (uint32_t j = 0; j < nc; ++j) {
                uint32_t c;
                f.read(reinterpret_cast<char*>(&c), 4);
                nd.children[j] = static_cast<DBoW2::NodeId>(c);
            }

            if (wid_raw != kNoWord && wid_raw < nw) {
                nd.word_id       = static_cast<DBoW2::WordId>(wid_raw);
                m_words[wid_raw] = &nd;
            }
        }
        return f.good() && (nw > 0);
    }
};

}  // namespace

namespace slam::loop_closure {

struct DBoW2Detector::Impl {
    BinaryOrbVocabulary voc{10, 6, DBoW2::TF_IDF, DBoW2::L1_NORM};
    DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB> db{voc, false, 0};
    std::unordered_map<int, int> db_entry_to_keyframe_id;
};

DBoW2Detector::DBoW2Detector()
    : impl_(std::make_unique<Impl>()) {}

DBoW2Detector::~DBoW2Detector() = default;

std::string DBoW2Detector::name() const {
    return "dbow2";
}

bool DBoW2Detector::loadVocabulary(const std::string& path) {
    if (!fs::exists(path)) {
        throw std::runtime_error("Binary vocabulary file not found: " + path);
    }

    if (!impl_->voc.loadFromFastBinary(path) || impl_->voc.size() == 0) {
        throw std::runtime_error("Failed to load binary vocabulary: " + path);
    }

    impl_->db.setVocabulary(impl_->voc, false, 0);

    return true;
}

bool DBoW2Detector::createAndSaveVocabulary(
    const std::string& path,
    const std::vector<cv::Mat>& training_descriptors) {
    if (training_descriptors.empty()) {
        return false;
    }

    std::vector<std::vector<DBoW2::FORB::TDescriptor>> bow_training;
    bow_training.reserve(training_descriptors.size());
    for (const auto& desc : training_descriptors) {
        auto converted = toBowDescriptors(desc);
        if (!converted.empty()) {
            bow_training.push_back(std::move(converted));
        }
    }

    if (bow_training.empty()) {
        return false;
    }

    impl_->voc.create(bow_training);
    if (impl_->voc.size() == 0) {
        return false;
    }

    impl_->voc.save(path);
    return true;
}

std::size_t DBoW2Detector::vocabularySize() const {
    return static_cast<std::size_t>(impl_->voc.size());
}

bool DBoW2Detector::addKeyframe(int keyframe_id,
                                 const cv::Mat& descriptors) {
    const auto bow_descriptors = toBowDescriptors(descriptors);
    if (bow_descriptors.empty() || impl_->voc.size() == 0) {
        return false;
    }

    const int entry_id = impl_->db.add(bow_descriptors);
    impl_->db_entry_to_keyframe_id[entry_id] = keyframe_id;
    return true;
}

std::optional<VprMatchCandidate> DBoW2Detector::queryBestCandidate(
    int current_keyframe_id,
    const cv::Mat& current_descriptors,
    int temporal_loop_gap,
    double min_score,
    std::size_t max_results) {
    const auto current_bow_descriptors = toBowDescriptors(current_descriptors);
    if (current_bow_descriptors.empty() || impl_->db.size() < 2) {
        return std::nullopt;
    }

    DBoW2::QueryResults results;
    impl_->db.query(current_bow_descriptors, results, static_cast<int>(max_results));

    // 1. Find the Baseline Score (similarity with the previous frame)
    double baseline_score = 1e-6; // prevent division by zero
    for (const auto& r : results) {
        auto map_it = impl_->db_entry_to_keyframe_id.find(r.Id);
        if (map_it != impl_->db_entry_to_keyframe_id.end() && map_it->second == current_keyframe_id - 1) {
            baseline_score = std::max(baseline_score, r.Score);
            break;
        }
    }

    // 2. Find the best normalized loop candidate
    VprMatchCandidate best;
    for (const auto& r : results) {
        auto map_it = impl_->db_entry_to_keyframe_id.find(r.Id);
        if (map_it == impl_->db_entry_to_keyframe_id.end()) {
            continue;
        }

        const int candidate_id = map_it->second;
        if (candidate_id == current_keyframe_id) {
            continue;
        }
        if (std::abs(candidate_id - current_keyframe_id) < temporal_loop_gap) {
            continue;
        }

        double normalized_score = r.Score / baseline_score;

        if (normalized_score > best.score) {
            best.score = normalized_score; // store normalized score
            best.matched_keyframe_id = candidate_id;
        }
    }

    if (best.matched_keyframe_id < 0 || best.score < min_score) {
        return std::nullopt;
    }

    return best;
}

}  // namespace slam::loop_closure
