#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <opencv2/core.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

#include "visual_graph_slam/core/measurement_edge.hpp"
#include "visual_graph_slam/map/map_manager.hpp"
#include "visual_graph_slam/core/graph.hpp"
#include "visual_graph_slam/sensor/camera_model.hpp"

namespace slam::core {

struct SensorFrame {
    cv::Mat image;
    rclcpp::Time stamp;
    std::vector<sensor_msgs::msg::Imu> imu_measurements; // Accumulated since last frame
    std::optional<nav_msgs::msg::Odometry> wheel_odom;
};

class GraphInterface {
public:
    virtual ~GraphInterface() = default;

    /**
     * @brief Submit a measurement edge configuration to the optimizer backend
     */
    virtual void submitMeasurement(const slam::MeasurementEdgeConfig& config) = 0;

    /**
     * @brief Access the global map manager for tracking recovery
     */
    virtual std::shared_ptr<slam::map::MapManager> getMapManager() = 0;

    /**
     * @brief Access the pose graph for tracking recovery
     */
    virtual std::shared_ptr<slam::Graph> getGraph() = 0;

    /**
     * @brief Signal the optimizer to run
     */
    virtual void signalOptimization(bool global, const char* reason = nullptr, int from_keyframe_id = -1, int to_keyframe_id = -1) = 0;
};

class System {
public:
    using SharedPtr = std::shared_ptr<System>;

    virtual ~System() = default;

    /**
     * @brief Initialize the plugin with parameters from the node
     */
    virtual void initialize(rclcpp::Node::SharedPtr node, 
                            const std::string& plugin_name,
                            std::shared_ptr<GraphInterface> graph) = 0;
                            
    /**
     * @brief Process a synchronized bundle of sensor data
     */
    virtual void processSensorFrame(const SensorFrame& frame) = 0;
    
    /**
     * @brief Dynamically update camera calibration
     */
    virtual void setCameraIntrinsics(const slam::sensor::CameraIntrinsics& intrinsics) {}
};

} // namespace slam::core
