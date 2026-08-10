#include "visual_graph_slam/core/graph_node.hpp"

namespace slam {

GraphNode::GraphNode(int id, const geometry_msgs::msg::Pose& pose, const rclcpp::Time& timestamp)
    : id_(id), pose_(pose), timestamp_(timestamp) {}

int GraphNode::getId() const {
    return id_;
}

geometry_msgs::msg::Pose GraphNode::getPose() const {
    return pose_;
}

rclcpp::Time GraphNode::getTimestamp() const {
    return timestamp_;
}

const std::vector<uint8_t>& GraphNode::getImageFeatures() const {
    return image_features_;
}

const std::vector<cv::KeyPoint>& GraphNode::getKeypoints() const {
    return keypoints_;
}

const cv::Mat& GraphNode::getDescriptors() const {
    return descriptors_;
}

void GraphNode::updatePose(const geometry_msgs::msg::Pose& new_pose) {
    pose_ = new_pose;
}

void GraphNode::setImageFeatures(const std::vector<uint8_t>& features) {
    image_features_ = features;
}

void GraphNode::setVisualData(const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors) {
    keypoints_ = keypoints;
    descriptors_ = descriptors.clone();
}

} // namespace slam
