#include "visual_graph_slam/slam_visualizer.hpp"
#include <geometry_msgs/msg/point.hpp>

#include <algorithm>

namespace slam {

Visualizer::Visualizer(rclcpp::Node::SharedPtr node, std::shared_ptr<Graph> graph)
    : node_(node), graph_(graph) {
    node_->get_parameter_or<std::string>("map_frame_id", map_frame_id_, "map");
    node_->get_parameter_or<std::string>("base_frame_id", base_frame_id_, "base_link");
    node_->get_parameter_or<bool>("publish_map_to_base_tf", publish_map_to_base_tf_, true);

    marker_publisher_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("slam_graph", 10);
    path_publisher_ = node_->create_publisher<nav_msgs::msg::Path>("slam/path", 10);
    pose_publisher_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>("slam/pose", 10);
    localization_publisher_ = node_->create_publisher<nav_msgs::msg::Odometry>("slam/localization", 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);

    
    graph_->addObserver([this](const Graph& g) { this->onGraphChanged(g); });
}

void Visualizer::triggerUpdate() {
    timerCallback();
}

void Visualizer::timerCallback() {
    if (graph_->isGraphChanged()) {
        last_marker_array_ = computeMarkerArray(*graph_);
        last_path_ = computePathMessage(*graph_);
        last_pose_ = computeCurrentPoseMessage(*graph_);
        last_localization_ = computeLocalizationMessage(*graph_);
        graph_->setGraphChanged(false);
    }

    if (publish_map_to_base_tf_ && tf_broadcaster_) {
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = last_pose_.header.stamp;
        tf_msg.header.frame_id = map_frame_id_;
        tf_msg.child_frame_id = base_frame_id_;
        tf_msg.transform.translation.x = last_pose_.pose.position.x;
        tf_msg.transform.translation.y = last_pose_.pose.position.y;
        tf_msg.transform.translation.z = last_pose_.pose.position.z;
        tf_msg.transform.rotation = last_pose_.pose.orientation;
        tf_broadcaster_->sendTransform(tf_msg);
    }

    marker_publisher_->publish(last_marker_array_);
    path_publisher_->publish(last_path_);
    pose_publisher_->publish(last_pose_);
    localization_publisher_->publish(last_localization_);
}

void Visualizer::onGraphChanged(const Graph& /*graph*/) {
    // Intentionally empty: triggerUpdate() / timerCallback() drives all publishing.
    // The graph_ member (stored in ctor) is used directly there.
}

visualization_msgs::msg::MarkerArray Visualizer::computeMarkerArray(const Graph& graph) {
    visualization_msgs::msg::MarkerArray marker_array;

    visualization_msgs::msg::Marker node_marker;
    node_marker.header.frame_id = map_frame_id_;
    node_marker.header.stamp = node_->now();
    node_marker.ns = "nodes";
    node_marker.type = visualization_msgs::msg::Marker::SPHERE;
    node_marker.action = visualization_msgs::msg::Marker::ADD;
    node_marker.scale.x = 0.1;
    node_marker.scale.y = 0.1;
    node_marker.scale.z = 0.1;
    node_marker.color.r = 1.0;
    node_marker.color.g = 0.0;
    node_marker.color.b = 0.0;
    node_marker.color.a = 1.0;

    for (const auto& [id, node] : graph.getNodes()) {
        node_marker.id = node->getId();
        node_marker.pose = node->getPose();
        marker_array.markers.push_back(node_marker);
    }

    visualization_msgs::msg::Marker edge_marker;
    edge_marker.header.frame_id = map_frame_id_;
    edge_marker.header.stamp = node_->now();
    edge_marker.ns = "edges";
    edge_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    edge_marker.action = visualization_msgs::msg::Marker::ADD;
    edge_marker.scale.x = 0.02;
    edge_marker.color.r = 0.0;
    edge_marker.color.g = 1.0;
    edge_marker.color.b = 0.0;
    edge_marker.color.a = 1.0;

    for (const auto& edge : graph.getEdges()) {
        const auto& from_node = graph.getNode(edge->getFromId());
        const auto& to_node = graph.getNode(edge->getToId());

        if (from_node && to_node) {
            geometry_msgs::msg::Point from_point;
            from_point.x = from_node->getPose().position.x;
            from_point.y = from_node->getPose().position.y;
            from_point.z = from_node->getPose().position.z;

            geometry_msgs::msg::Point to_point;
            to_point.x = to_node->getPose().position.x;
            to_point.y = to_node->getPose().position.y;
            to_point.z = to_node->getPose().position.z;

            edge_marker.points.push_back(from_point);
            edge_marker.points.push_back(to_point);
        }
    }

    marker_array.markers.push_back(edge_marker);

    return marker_array;
}

nav_msgs::msg::Path Visualizer::computePathMessage(const Graph& graph) const {
    nav_msgs::msg::Path path;
    path.header.frame_id = map_frame_id_;
    path.header.stamp = node_->now();

    auto nodes = graph.getNodes();
    std::vector<int> ordered_ids;
    ordered_ids.reserve(nodes.size());
    for (const auto& [id, _] : nodes) {
        ordered_ids.push_back(id);
    }
    std::sort(ordered_ids.begin(), ordered_ids.end());

    path.poses.reserve(ordered_ids.size());
    for (int id : ordered_ids) {
        auto it = nodes.find(id);
        if (it == nodes.end() || !it->second) {
            continue;
        }

        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = map_frame_id_;
        pose.header.stamp = it->second->getTimestamp();
        pose.pose = it->second->getPose();
        path.poses.push_back(pose);
    }

    return path;
}

geometry_msgs::msg::PoseStamped Visualizer::computeCurrentPoseMessage(const Graph& graph) const {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = map_frame_id_;
    pose.header.stamp = node_->now();

    const int last_id = graph.getLastNodeId();
    auto node = graph.getNode(last_id);
    if (node) {
        pose.header.stamp = node->getTimestamp();
        pose.pose = node->getPose();
    }

    return pose;
}

nav_msgs::msg::Odometry Visualizer::computeLocalizationMessage(const Graph& graph) const {
    nav_msgs::msg::Odometry odom;
    odom.header.frame_id = map_frame_id_;
    odom.child_frame_id = base_frame_id_;
    odom.header.stamp = node_->now();

    const int last_id = graph.getLastNodeId();
    auto node = graph.getNode(last_id);
    if (node) {
        odom.header.stamp = node->getTimestamp();
        odom.pose.pose = node->getPose();
    }

    return odom;
}

} 