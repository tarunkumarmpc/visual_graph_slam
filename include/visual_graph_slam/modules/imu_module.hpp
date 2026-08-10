#pragma once

#include <memory>
#include <deque>
#include <rclcpp/rclcpp.hpp>

#include "visual_graph_slam/core/sensor_module.hpp"
#include "visual_graph_slam/vins/imu_preintegrator.hpp"
#include "visual_graph_slam/vins/vins_initializer.hpp"
#include "visual_graph_slam/vins/gtsam_imu_builder.hpp"

namespace slam::modules {

class ImuModule : public slam::core::SensorModule {
public:
    ImuModule();
    ~ImuModule() override = default;

    void initialize(rclcpp::Node::SharedPtr node, const std::string& name, slam::core::GraphInterface* graph) override;
    
    void preProcess(const slam::core::SensorFrame& frame) override;
    bool wantsKeyframe() const override;
    void onKeyframeCreated(std::shared_ptr<slam::GraphNode> node, slam::core::GraphInterface* graph) override;
    std::optional<slam::core::SensorModule::GlobalAlignmentRequest> wantsGlobalAlignment() const override;
    void onGlobalAlignment(double scale, const tf2::Quaternion& q_align) override;

    bool isVinsInitialized() const { return vins_initialized_; }
    double getEstimatedScale() const { return estimated_scale_; }
    const Eigen::Vector3d& getEstimatedGravity() const { return estimated_gravity_; }
    Eigen::Vector3d getLatestGyroBias() const {
        if (!initialization_window_.empty() && initialization_window_.back()) {
            return initialization_window_.back()->getGyroBias();
        }
        return Eigen::Vector3d::Zero();
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Logger logger_{rclcpp::get_logger("ImuModule")};
    slam::core::GraphInterface* graph_{nullptr};

    std::unique_ptr<slam::vins::ImuPreintegrator> imu_preintegrator_;
    std::unique_ptr<slam::vins::VinsInitializer> vins_initializer_;

    std::vector<std::shared_ptr<slam::GraphNode>> initialization_window_;
    std::vector<slam::vins::ImuPreintegrator::PreintegratedData> imu_preintegrated_keyframes_;
    std::vector<std::vector<sensor_msgs::msg::Imu>> buffered_raw_imu_samples_;
    
    bool vins_initialized_{false};
    double estimated_scale_{1.0};
    Eigen::Vector3d estimated_gravity_{0, 0, -9.81};

    // DESIGN-FLAW-FIX: VINS init window size from YAML (was hardcoded 15 in .cpp)
    int vins_min_keyframes_for_init_{15};

    bool wants_keyframe_cache_{false};
    rclcpp::Time last_keyframe_time_{0, 0, RCL_ROS_TIME};
    int last_keyframe_id_{-1};  // ARCH-2 fix: track actual previous keyframe ID

    // DESIGN-FLAW-FIX: mutable so wantsGlobalAlignment() can consume-and-clear
    // the request in a const method, preventing double-application of scale/rotation.
    mutable std::optional<slam::core::SensorModule::GlobalAlignmentRequest> pending_alignment_request_;

    // Current uncommitted preintegrated data
    std::optional<slam::vins::ImuPreintegrator::PreintegratedData> pending_preintegration_;
    std::vector<sensor_msgs::msg::Imu> raw_imu_msgs_;
};

} // namespace slam::modules
