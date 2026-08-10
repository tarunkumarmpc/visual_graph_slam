#pragma once

#include <vector>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "visual_graph_slam/plugins/system.hpp"
#include "visual_graph_slam/core/sensor_module.hpp"

namespace slam::plugins {

class SystemCoordinator : public slam::core::System {
public:
    SystemCoordinator() = default;
    ~SystemCoordinator() override = default;

    void initialize(rclcpp::Node::SharedPtr node, 
                    const std::string& plugin_name,
                    std::shared_ptr<slam::core::GraphInterface> graph) override;

    void processSensorFrame(const slam::core::SensorFrame& frame) override;
    
    void setCameraIntrinsics(const slam::sensor::CameraIntrinsics& intrinsics) override;

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Logger logger_{rclcpp::get_logger("SystemCoordinator")};
    std::shared_ptr<slam::core::GraphInterface> graph_;

    std::vector<std::shared_ptr<slam::core::SensorModule>> modules_;

    int current_keyframe_id_{0};
    int local_mapping_window_size_{20};
    
    void processGlobalAlignment(double scale, const tf2::Quaternion& q_align);
};

} // namespace slam::plugins
