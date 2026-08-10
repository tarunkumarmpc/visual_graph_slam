#include "visual_graph_slam/frontend/feature_pipeline.hpp"

#include <opencv2/imgproc.hpp>

namespace {

cv::Mat toGray(const cv::Mat& image_bgr_or_gray) {
    if (image_bgr_or_gray.channels() == 1) {
        return image_bgr_or_gray.clone();
    }

    cv::Mat gray;
    cv::cvtColor(image_bgr_or_gray, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

}  // namespace

namespace slam::frontend {

OrbFeaturePipeline::OrbFeaturePipeline(int max_features)
    : orb_(cv::ORB::create(max_features)) {}

std::string OrbFeaturePipeline::name() const {
    return "orb";
}

// FIX #4: Implement descriptor type query
int OrbFeaturePipeline::descriptorType() const {
    return cv::NORM_HAMMING;  // ORB uses binary descriptors
}

int OrbFeaturePipeline::descriptorSize() const {
    return 32;  // 256 bits / 8 bytes
}

void OrbFeaturePipeline::extract(const cv::Mat& image_bgr_or_gray,
                                 std::vector<cv::KeyPoint>& keypoints,
                                 cv::Mat& descriptors) const {
    cv::Mat gray = toGray(image_bgr_or_gray);
    orb_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
}

}  // namespace slam::frontend
