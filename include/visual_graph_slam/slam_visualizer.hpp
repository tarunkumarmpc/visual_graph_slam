#ifndef SLAM_VISUALIZER_HPP
#define SLAM_VISUALIZER_HPP


#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include "visual_graph_slam/core/graph.hpp"

namespace slam {

class Visualizer {
public:
    Visualizer(rclcpp::Node::SharedPtr node, std::shared_ptr<Graph> graph);
    void triggerUpdate();

private:
    void timerCallback();
    void onGraphChanged(const Graph& graph);
    visualization_msgs::msg::MarkerArray computeMarkerArray(const Graph& graph);
    nav_msgs::msg::Path computePathMessage(const Graph& graph) const;
    geometry_msgs::msg::PoseStamped computeCurrentPoseMessage(const Graph& graph) const;
    nav_msgs::msg::Odometry computeLocalizationMessage(const Graph& graph) const;

    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<Graph> graph_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr localization_publisher_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    visualization_msgs::msg::MarkerArray last_marker_array_;
    nav_msgs::msg::Path last_path_;
    geometry_msgs::msg::PoseStamped last_pose_;
    nav_msgs::msg::Odometry last_localization_;
    std::string map_frame_id_;
    std::string base_frame_id_;
    bool publish_map_to_base_tf_;
    bool visualizer_initialized_;
};

} // namespace slam

#endif // SLAM_VISUALIZER_HPP
