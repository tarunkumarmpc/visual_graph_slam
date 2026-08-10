#pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include "visual_graph_slam/plugins/system.hpp"
#include "visual_graph_slam/sensor/camera_model.hpp"
#include "visual_graph_slam/core/graph_node.hpp"

namespace slam::core {

class SensorModule {
public:
    virtual ~SensorModule() = default;

    /**
     * @brief Initialize the module with ROS node, parameters, and graph.
     */
    virtual void initialize(rclcpp::Node::SharedPtr node, const std::string& name, GraphInterface* graph) = 0;

    /**
     * @brief Pre-process the sensor data (e.g. IMU preintegration).
     */
    virtual void preProcess(const SensorFrame& frame) = 0;
    
    /**
     * @brief Receive updated camera calibration dynamically
     */
    virtual void setCameraIntrinsics(const slam::sensor::CameraIntrinsics& intrinsics) {}

    /**
     * @brief The module votes on whether a keyframe should be created.
     */
    virtual bool wantsKeyframe() const = 0;

    /**
     * @brief Called when the Coordinator creates a new keyframe node. 
     * The module should attach its edges to the graph here.
     */
    virtual void onKeyframeCreated(std::shared_ptr<GraphNode> node, GraphInterface* graph) = 0;

    struct GlobalAlignmentRequest {
        double scale;
        tf2::Quaternion q_align;
    };

    /**
     * @brief The module can request a global alignment (e.g. after VINS init).
     */
    virtual std::optional<GlobalAlignmentRequest> wantsGlobalAlignment() const { return std::nullopt; }

    /**
     * @brief Called when the graph undergoes a global alignment (e.g. VINS Init).
     */
    virtual void onGlobalAlignment(double scale, const tf2::Quaternion& q_align) = 0;
};

} // namespace slam::core
