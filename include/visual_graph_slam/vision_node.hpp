#ifndef VISION_NODE_HPP
#define VISION_NODE_HPP

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <optional>
#include <future>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/rclcpp.hpp>

#include "visual_graph_slam/frontend/feature_pipeline.hpp"
#include "visual_graph_slam/frontend/visual_odometry_frontend.hpp"
#include "visual_graph_slam/loop_closure/loop_detector.hpp"
#include "visual_graph_slam/sensor/camera_model.hpp"

struct VisionNodeConfig {
    int temporal_loop_gap{20};
    int vpr_query_max_results{100};
    double loop_score_threshold{0.08};
    double ratio_test_threshold{0.75};
    std::string vpr_backend_name{"orb_bow"};
    std::string detector_name{"orb"};
    std::string vocabulary_path{""};
    int vo_min_tracked_pts{80};
    int vo_reinit_threshold{40};
    int vo_max_corners{500};
    double vo_min_flow_px{1.5};
    cv::Mat camera_matrix;
    std::string camera_model_name{"pinhole"};
    Eigen::Vector4d camera_distortion{Eigen::Vector4d::Zero()};
};

class VisionNode {
public:
    struct FrontendResult {
        /// Estimation method used — determines information-matrix weight in graph.
        enum class Method { NONE, LK_ESSENTIAL, FEATURE_ESSENTIAL, LOCAL_PNP_RECOVERY };

        bool   pose_estimated{false};
        Method method{Method::NONE};
        geometry_msgs::msg::Transform relative_transform{};
        int    inliers{0};
        int    tracked_points{0};  ///< correspondences fed to the estimator
        double inlier_ratio{0.0};
        double confidence{0.0};    ///< [0,1] — drives edge information weight
        std::string failure_reason{"none"};
        double mean_flow_px{0.0};
        int fallback_match_count{0};
        int reference_age{1};
    };

    struct LoopClosureResult {
        bool detected{false};
        int current_keyframe_id{-1};
        int matched_keyframe_id{-1};
        double score{0.0};
        int inliers{0};
        int matches{0};
        double inlier_ratio{0.0};
        geometry_msgs::msg::Transform relative_transform{};
    };

    explicit VisionNode(const VisionNodeConfig& config, rclcpp::Logger logger, rclcpp::Clock::SharedPtr clock);

    FrontendResult processFrame(const cv::Mat& image_bgr_or_gray,
                                std::optional<double> odom_displacement = std::nullopt);
    void updateCameraCalibration(const sensor_msgs::msg::CameraInfo& camera_info);
    void addKeyframe(int keyframe_id,
                     const cv::Mat& image_bgr_or_gray,
                     const geometry_msgs::msg::Pose& pose,
                     const rclcpp::Time& timestamp);
    std::optional<LoopClosureResult> detectLoopClosureWithVpr(int current_keyframe_id);
    void extractFeatures(const cv::Mat& image_bgr_or_gray,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::Mat& descriptors) const;
    void resetFrontend();

    /// Keypoints extracted from the most recent processFrame() call.
    /// Use these when creating a keyframe to avoid re-extracting.
    const std::vector<cv::KeyPoint>& currentKeypoints()   const;
    const cv::Mat&                   currentDescriptors() const;

private:
    struct KeyframeRecord {
        int id{-1};
        geometry_msgs::msg::Pose pose{};
        rclcpp::Time timestamp{0, 0, RCL_ROS_TIME};
        cv::Mat image_gray;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        int db_entry_id{-1};
    };

    bool load_or_create_vocabulary();
    bool ensure_vocabulary_ready();
    bool estimateRelativeTransformFromMatches(const KeyframeRecord& from,
                                              const KeyframeRecord& to,
                                              geometry_msgs::msg::Transform& transform_out,
                                              int& inliers_out,
                                              int& matches_out) const;

    static cv::Mat toGray(const cv::Mat& image_bgr_or_gray);
    static geometry_msgs::msg::Transform rtToTransform(const cv::Mat& rotation, const cv::Mat& translation);

    std::unique_ptr<slam::loop_closure::LoopDetector>        loop_detector_;
    std::unique_ptr<slam::frontend::FeaturePipeline>          feature_pipeline_;
    std::unique_ptr<slam::frontend::VisualOdometryFrontend>   vo_frontend_;
    cv::BFMatcher matcher_;

    std::unordered_map<int, KeyframeRecord> keyframes_;
    std::optional<KeyframeRecord> previous_frame_;

    int temporal_loop_gap_;
    int vpr_query_max_results_;
    double loop_score_threshold_;
    double ratio_test_threshold_;
    std::string vpr_backend_name_;
    std::string detector_name_;
    std::string vocabulary_path_;
    bool vocabulary_load_started_{false};
    std::future<bool> vocabulary_load_future_;
    bool vocabulary_initialized_{false};
    bool vocabulary_available_{false};
    std::mutex vpr_mutex_;
    rclcpp::Logger logger_;
    rclcpp::Clock::SharedPtr clock_;
    int    vo_min_tracked_pts_;
    int    vo_reinit_threshold_;
    int    vo_max_corners_;
    double vo_min_flow_px_;
    cv::Mat camera_matrix_;
    std::string camera_model_name_;
    Eigen::Vector4d camera_distortion_;
    std::unique_ptr<slam::sensor::ICameraModel> camera_model_;
};

#endif // VISION_NODE_HPP
