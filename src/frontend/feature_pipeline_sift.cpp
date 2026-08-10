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

SiftFeaturePipeline::SiftFeaturePipeline(int max_features)
    : sift_(cv::SIFT::create(std::max(0, max_features))) {}

std::string SiftFeaturePipeline::name() const {
    return "sift";
}

// FIX #4: Implement descriptor type query for SIFT (uses float descriptors)
int SiftFeaturePipeline::descriptorType() const {
    return cv::NORM_L2;  // SIFT uses 128-D float descriptors
}

int SiftFeaturePipeline::descriptorSize() const {
    return 128 * sizeof(float);  // 512 bytes per descriptor
}

void SiftFeaturePipeline::extract(const cv::Mat& image_bgr_or_gray,
                                  std::vector<cv::KeyPoint>& keypoints,
                                  cv::Mat& descriptors) const {
    cv::Mat gray = toGray(image_bgr_or_gray);
    sift_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
}

}  // namespace slam::frontend
