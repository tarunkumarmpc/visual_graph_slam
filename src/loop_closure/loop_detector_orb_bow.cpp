#include "visual_graph_slam/loop_closure/loop_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <unordered_map>
#include <utility>

#include <opencv2/core.hpp>

namespace slam::loop_closure {

struct OrbBowDetector::Impl {
    cv::Mat vocabulary;  // rows = words, cols = descriptor dim (CV_32F)
    std::unordered_map<int, cv::Mat> keyframe_histograms;  // keyframe_id -> 1xK CV_32F (L2 normalized)
    std::mutex mutex;

    int num_words{256};
    int descriptor_dim{32};
    int max_training_descriptors{20000};
};

namespace {

constexpr const char* kVocabularyNode = "vocabulary";
constexpr const char* kNumWordsNode = "num_words";
constexpr const char* kDescriptorDimNode = "descriptor_dim";
constexpr const char* kMaxTrainingDescriptorsNode = "max_training_descriptors";

cv::Mat toDescriptorFloat(const cv::Mat& descriptors) {
    if (descriptors.empty()) {
        return cv::Mat();
    }

    cv::Mat d;
    if (descriptors.type() != CV_8U && descriptors.type() != CV_32F) {
        return cv::Mat();
    }

    descriptors.convertTo(d, CV_32F);
    return d;
}

cv::Mat sampleRows(const cv::Mat& src, int max_rows) {
    if (src.empty() || src.rows <= max_rows) {
        return src.clone();
    }

    std::vector<int> indices(src.rows);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(max_rows);

    cv::Mat sampled(max_rows, src.cols, src.type());
    for (int i = 0; i < max_rows; ++i) {
        src.row(indices[i]).copyTo(sampled.row(i));
    }
    return sampled;
}

int nearestWord(const cv::Mat& vocabulary, const float* descriptor_ptr, int dim) {
    int best_idx = -1;
    float best_dist = std::numeric_limits<float>::max();

    for (int r = 0; r < vocabulary.rows; ++r) {
        const float* word_ptr = vocabulary.ptr<float>(r);
        float d = 0.0f;
        for (int c = 0; c < dim; ++c) {
            const float diff = descriptor_ptr[c] - word_ptr[c];
            d += diff * diff;
        }

        if (d < best_dist) {
            best_dist = d;
            best_idx = r;
        }
    }

    return best_idx;
}

cv::Mat computeHistogram(const cv::Mat& vocabulary, const cv::Mat& descriptors_float) {
    if (vocabulary.empty() || descriptors_float.empty() || descriptors_float.cols != vocabulary.cols) {
        return cv::Mat();
    }

    cv::Mat hist = cv::Mat::zeros(1, vocabulary.rows, CV_32F);
    for (int i = 0; i < descriptors_float.rows; ++i) {
        const float* d = descriptors_float.ptr<float>(i);
        const int idx = nearestWord(vocabulary, d, descriptors_float.cols);
        if (idx >= 0) {
            hist.at<float>(0, idx) += 1.0f;
        }
    }

    const double norm_l2 = cv::norm(hist, cv::NORM_L2);
    if (norm_l2 > 1e-9) {
        hist /= static_cast<float>(norm_l2);
    }
    return hist;
}

double cosineScore(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty() || a.cols != b.cols) {
        return 0.0;
    }
    return static_cast<double>(a.dot(b));
}

}  // namespace

OrbBowDetector::OrbBowDetector()
    : impl_(std::make_unique<Impl>()) {}

OrbBowDetector::~OrbBowDetector() = default;

std::string OrbBowDetector::name() const {
    return "orb_bow";
}

bool OrbBowDetector::loadVocabulary(const std::string& path) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    cv::Mat vocabulary;
    fs[kVocabularyNode] >> vocabulary;
    if (vocabulary.empty()) {
        return false;
    }

    int num_words = impl_->num_words;
    int descriptor_dim = impl_->descriptor_dim;
    int max_training_descriptors = impl_->max_training_descriptors;
    fs[kNumWordsNode] >> num_words;
    fs[kDescriptorDimNode] >> descriptor_dim;
    fs[kMaxTrainingDescriptorsNode] >> max_training_descriptors;

    vocabulary.convertTo(impl_->vocabulary, CV_32F);
    impl_->num_words = std::max(16, num_words);
    impl_->descriptor_dim = std::max(16, descriptor_dim);
    impl_->max_training_descriptors = std::max(1000, max_training_descriptors);
    impl_->keyframe_histograms.clear();
    return true;
}

bool OrbBowDetector::createAndSaveVocabulary(
    const std::string& path,
    const std::vector<cv::Mat>& training_descriptors) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    std::vector<cv::Mat> float_blocks;
    int total_rows = 0;

    for (const auto& desc : training_descriptors) {
        cv::Mat df = toDescriptorFloat(desc);
        if (df.empty()) {
            continue;
        }

        if (df.cols != impl_->descriptor_dim) {
            continue;
        }

        float_blocks.push_back(df);
        total_rows += df.rows;
    }

    if (float_blocks.empty() || total_rows < impl_->num_words) {
        return false;
    }

    cv::Mat all;
    cv::vconcat(float_blocks, all);
    all = sampleRows(all, impl_->max_training_descriptors);

    if (all.rows < impl_->num_words) {
        return false;
    }

    cv::Mat labels;
    cv::Mat centers;
    cv::kmeans(all,
               impl_->num_words,
               labels,
               cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 80, 1e-3),
               3,
               cv::KMEANS_PP_CENTERS,
               centers);

    if (centers.empty()) {
        return false;
    }

    centers.convertTo(impl_->vocabulary, CV_32F);
    impl_->keyframe_histograms.clear();

    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        return false;
    }
    fs << kVocabularyNode << impl_->vocabulary;
    fs << kNumWordsNode << impl_->num_words;
    fs << kDescriptorDimNode << impl_->descriptor_dim;
    fs << kMaxTrainingDescriptorsNode << impl_->max_training_descriptors;
    fs.release();

    return true;
}

std::size_t OrbBowDetector::vocabularySize() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return static_cast<std::size_t>(impl_->vocabulary.rows);
}

bool OrbBowDetector::addKeyframe(int keyframe_id, const cv::Mat& descriptors) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (impl_->vocabulary.empty()) {
        return false;
    }

    cv::Mat df = toDescriptorFloat(descriptors);
    if (df.empty() || df.cols != impl_->vocabulary.cols) {
        return false;
    }

    cv::Mat hist = computeHistogram(impl_->vocabulary, df);
    if (hist.empty()) {
        return false;
    }

    impl_->keyframe_histograms[keyframe_id] = hist;
    return true;
}

std::optional<VprMatchCandidate> OrbBowDetector::queryBestCandidate(
    int current_keyframe_id,
    const cv::Mat& current_descriptors,
    int temporal_loop_gap,
    double min_score,
    std::size_t max_results) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (impl_->vocabulary.empty() || impl_->keyframe_histograms.size() < 2) {
        return std::nullopt;
    }

    cv::Mat df = toDescriptorFloat(current_descriptors);
    if (df.empty() || df.cols != impl_->vocabulary.cols) {
        return std::nullopt;
    }

    const cv::Mat current_hist = computeHistogram(impl_->vocabulary, df);
    if (current_hist.empty()) {
        return std::nullopt;
    }

    std::vector<VprMatchCandidate> candidates;
    candidates.reserve(impl_->keyframe_histograms.size());

    for (const auto& [candidate_id, candidate_hist] : impl_->keyframe_histograms) {
        if (candidate_id == current_keyframe_id) {
            continue;
        }
        if (std::abs(candidate_id - current_keyframe_id) < temporal_loop_gap) {
            continue;
        }

        const double score = cosineScore(current_hist, candidate_hist);
        if (score > 0.0) {
            candidates.push_back(VprMatchCandidate{candidate_id, score});
        }
    }

    if (candidates.empty()) {
        return std::nullopt;
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.score > b.score;
    });

    const std::size_t limit = std::max<std::size_t>(1, max_results);
    const std::size_t take = std::min(limit, candidates.size());

    const auto best_it = std::max_element(candidates.begin(), candidates.begin() + static_cast<long>(take),
                                          [](const auto& a, const auto& b) { return a.score < b.score; });
    if (best_it == candidates.end() || best_it->score < min_score) {
        return std::nullopt;
    }

    return *best_it;
}

}  // namespace slam::loop_closure
