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

SuperPointFeaturePipeline::SuperPointFeaturePipeline(int fallback_orb_features)
    : fallback_orb_(cv::ORB::create(fallback_orb_features)) {}

std::string SuperPointFeaturePipeline::name() const {
    return "superpoint-fallback-orb";
}

// FIX #4: Implement descriptor type query
int SuperPointFeaturePipeline::descriptorType() const {
    // SuperPoint would use NORM_L2 when real implementation available
    // For now, fallback to ORB (binary)
    return cv::NORM_HAMMING;
}

int SuperPointFeaturePipeline::descriptorSize() const {
    // When SuperPoint is properly implemented: 256 * sizeof(float) = 1024
    // For now, fallback to ORB: 32 bytes
    return 32;
}

void SuperPointFeaturePipeline::extract(const cv::Mat& image_bgr_or_gray,
                                        std::vector<cv::KeyPoint>& keypoints,
                                        cv::Mat& descriptors) const {
    cv::Mat gray = toGray(image_bgr_or_gray);
    fallback_orb_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
}

}  // namespace slam::frontend
