#include "visual_graph_slam/loop_closure/loop_detector.hpp"

namespace slam::loop_closure {

struct DBoW2Detector::Impl {};

DBoW2Detector::DBoW2Detector() = default;
DBoW2Detector::~DBoW2Detector() = default;

std::string DBoW2Detector::name() const {
    return "dbow2-unavailable";
}

bool DBoW2Detector::loadVocabulary(const std::string&) {
    return false;
}

bool DBoW2Detector::createAndSaveVocabulary(const std::string&,
                                             const std::vector<cv::Mat>&) {
    return false;
}

std::size_t DBoW2Detector::vocabularySize() const {
    return 0;
}

bool DBoW2Detector::addKeyframe(int,
                                 const cv::Mat&) {
    return false;
}

std::optional<VprMatchCandidate> DBoW2Detector::queryBestCandidate(
    int,
    const cv::Mat&,
    int,
    double,
    std::size_t) {
    return std::nullopt;
}

}  // namespace slam::loop_closure
