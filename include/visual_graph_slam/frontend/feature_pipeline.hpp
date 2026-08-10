#ifndef VISUAL_GRAPH_SLAM_FRONTEND_FEATURE_PIPELINE_HPP
#define VISUAL_GRAPH_SLAM_FRONTEND_FEATURE_PIPELINE_HPP

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace slam::frontend {

class FeaturePipeline {
public:
    virtual ~FeaturePipeline() = default;

    virtual std::string name() const = 0;
    virtual int descriptorType() const = 0;  // Returns cv::NORM_HAMMING or cv::NORM_L2
    virtual int descriptorSize() const = 0;  // Returns byte size per descriptor
    virtual void extract(const cv::Mat& image_bgr_or_gray,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::Mat& descriptors) const = 0;
};

class OrbFeaturePipeline final : public FeaturePipeline {
public:
    explicit OrbFeaturePipeline(int max_features = 1200);

    std::string name() const override;
    int descriptorType() const override;  // FIX #4
    int descriptorSize() const override;   // FIX #4
    void extract(const cv::Mat& image_bgr_or_gray,
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors) const override;

private:
    cv::Ptr<cv::ORB> orb_;
};

class SuperPointFeaturePipeline final : public FeaturePipeline {
public:
    // Placeholder implementation to keep architecture swappable.
    // It falls back to ORB until a true SuperPoint runtime is integrated.
    explicit SuperPointFeaturePipeline(int fallback_orb_features = 1500);

    std::string name() const override;
    int descriptorType() const override;  // FIX #4
    int descriptorSize() const override;   // FIX #4
    void extract(const cv::Mat& image_bgr_or_gray,
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors) const override;

private:
    cv::Ptr<cv::ORB> fallback_orb_;
};

class SiftFeaturePipeline final : public FeaturePipeline {
public:
    explicit SiftFeaturePipeline(int max_features = 0);

    std::string name() const override;
    int descriptorType() const override;  // FIX #4
    int descriptorSize() const override;   // FIX #4
    void extract(const cv::Mat& image_bgr_or_gray,
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors) const override;

private:
    cv::Ptr<cv::SIFT> sift_;
};

class CustomFeaturePipeline final : public FeaturePipeline {
public:
    explicit CustomFeaturePipeline(int max_features = 1800,
                                   double clahe_clip_limit = 2.0,
                                   int clahe_tile_grid = 8);

    std::string name() const override;
    int descriptorType() const override;  // FIX #4
    int descriptorSize() const override;   // FIX #4
    void extract(const cv::Mat& image_bgr_or_gray,
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors) const override;

private:
    cv::Ptr<cv::ORB> orb_;
    cv::Ptr<cv::CLAHE> clahe_;
};

std::unique_ptr<FeaturePipeline> createFeaturePipeline(const std::string& frontend_name);

}  // namespace slam::frontend

#endif  // VISUAL_GRAPH_SLAM_FRONTEND_FEATURE_PIPELINE_HPP
