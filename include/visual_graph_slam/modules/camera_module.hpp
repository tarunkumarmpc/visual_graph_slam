#pragma once

#include <deque>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <string>
#include <optional>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <pluginlib/class_loader.hpp>

#include "visual_graph_slam/core/sensor_module.hpp"
#include "visual_graph_slam/plugins/frontend.hpp"
#include "visual_graph_slam/plugins/loop_closure.hpp"
#include "visual_graph_slam/telemetry/slam_telemetry.hpp"

namespace slam::modules {

enum class TrackingState {
    OK,
    WEAK,
    LOST
};

class CameraModule : public slam::core::SensorModule {
public:
    CameraModule();
    ~CameraModule() override = default;

    void initialize(rclcpp::Node::SharedPtr node, const std::string& name, slam::core::GraphInterface* graph) override;
    
    void preProcess(const slam::core::SensorFrame& frame) override;
    
    void setCameraIntrinsics(const slam::sensor::CameraIntrinsics& intrinsics) override;

    bool wantsKeyframe() const override;
    void onKeyframeCreated(std::shared_ptr<slam::GraphNode> node, slam::core::GraphInterface* graph) override;
    void onGlobalAlignment(double scale, const tf2::Quaternion& q_align) override;

    slam::telemetry::TelemetrySnapshot getLatestTelemetry() const;

private:
    mutable std::mutex mutex_;
    void setTrackingState(TrackingState new_state, const char* reason);
    const char* trackingStateToString(TrackingState state) const;
    void registerTrackingReject(const char* reason, bool reset_frontend_if_lost = true);
    
    std::optional<slam::core::FrontendResult> tryLocalKeyframePnpRecovery(const cv::Mat& current_descriptors, const std::vector<cv::KeyPoint>& current_keypoints);
    
    bool shouldAddVisionKeyframe(const geometry_msgs::msg::Pose& candidate_pose, const rclcpp::Time& stamp) const;

    rclcpp::Node::SharedPtr node_;
    rclcpp::Logger logger_{rclcpp::get_logger("CameraModule")};
    slam::core::GraphInterface* graph_{nullptr};

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    pluginlib::ClassLoader<slam::core::Frontend> frontend_loader_;
    pluginlib::ClassLoader<slam::core::LoopClosure> loop_closure_loader_;

    std::shared_ptr<slam::core::Frontend> frontend_;
    std::shared_ptr<slam::core::LoopClosure> loop_closure_;

    TrackingState tracking_state_{TrackingState::OK};
    int consecutive_vo_rejects_{0};
    int consecutive_good_vo_frames_{0};
    
    // Parameters
    int weak_tracking_reject_threshold_{4};
    int lost_tracking_reject_threshold_{12};
    int weak_to_ok_min_good_frames_{3};
    double keyframe_distance_threshold_{0.25};
    double keyframe_rotation_threshold_deg_{8.0};
    double keyframe_time_threshold_sec_{0.35};
    int pnp_keyframe_window_{8};
    int pnp_min_inliers_{15};
    double pnp_min_inlier_ratio_{0.20};
    double pnp_max_rotation_deg_{10.0};
    bool planar_motion_constraint_{true};
    std::string mode_{"mono"};
    
    std::string base_frame_id_{"base_link"};
    std::string camera_frame_id_{"camera_link"};
    cv::Mat camera_matrix_;

    std::deque<int> recent_keyframe_ids_;
    rclcpp::Time last_keyframe_time_{0, 0, RCL_ROS_TIME};
    geometry_msgs::msg::Pose last_keyframe_vision_pose_;
    geometry_msgs::msg::Pose vision_integrated_pose_;
    // track the actual previous keyframe ID (mirrors ARCH-2 fix in ImuModule).
    // Using kf_id-1 breaks after any tracking failure where a keyframe is skipped.
    int last_keyframe_id_{-1};

    std::optional<slam::core::FrontendResult> latest_vo_result_;
    bool wants_keyframe_cache_{false};
    slam::core::SensorFrame latest_frame_;
    slam::telemetry::TelemetrySnapshot latest_telemetry_;

    static bool isFiniteTransform(const geometry_msgs::msg::Transform& tf);
    static double transformTranslationNorm(const geometry_msgs::msg::Transform& tf);
    static double quaternionAngularDistanceRad(const geometry_msgs::msg::Quaternion& a, const geometry_msgs::msg::Quaternion& b);
    static geometry_msgs::msg::Transform calculateRelativeTransform(const geometry_msgs::msg::Pose& from, const geometry_msgs::msg::Pose& to);
};

} // namespace slam::modules
