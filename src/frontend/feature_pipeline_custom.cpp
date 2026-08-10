#include "visual_graph_slam/frontend/feature_pipeline.hpp"

#include <algorithm>

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

CustomFeaturePipeline::CustomFeaturePipeline(int max_features,
                                             double clahe_clip_limit,
                                             int clahe_tile_grid)
    : orb_(cv::ORB::create(std::max(200, max_features))),
      clahe_(cv::createCLAHE(std::max(0.5, clahe_clip_limit),
                             cv::Size(std::max(2, clahe_tile_grid), std::max(2, clahe_tile_grid)))) {}

std::string CustomFeaturePipeline::name() const {
    return "custom-clahe-orb";
}

// FIX #4: Implement descriptor type query
int CustomFeaturePipeline::descriptorType() const {
    return cv::NORM_HAMMING;  // Uses ORB base, binary descriptors
}

int CustomFeaturePipeline::descriptorSize() const {
    return 32;  // 256 bits / 8 bytes (ORB-based)
}

void CustomFeaturePipeline::extract(const cv::Mat& image_bgr_or_gray,
                                    std::vector<cv::KeyPoint>& keypoints,
                                    cv::Mat& descriptors) const {
    cv::Mat gray = toGray(image_bgr_or_gray);
    cv::Mat enhanced;
    clahe_->apply(gray, enhanced);
    orb_->detectAndCompute(enhanced, cv::noArray(), keypoints, descriptors);
}

}  // namespace slam::frontend
