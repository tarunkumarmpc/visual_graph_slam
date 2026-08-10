#ifndef GRAPH_NODE_HPP
#define GRAPH_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <Eigen/Dense>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace slam {

class GraphNode {
public:
    GraphNode(int id, const geometry_msgs::msg::Pose& pose, const rclcpp::Time& timestamp);

    int getId() const;
    geometry_msgs::msg::Pose getPose() const;
    rclcpp::Time getTimestamp() const;
    const std::vector<uint8_t>& getImageFeatures() const;
    const std::vector<cv::KeyPoint>& getKeypoints() const;
    const cv::Mat& getDescriptors() const;

    Eigen::Vector3d getVelocity() const { return velocity_; }
    Eigen::Vector3d getAccelBias() const { return accel_bias_; }
    Eigen::Vector3d getGyroBias() const { return gyro_bias_; }

    void updatePose(const geometry_msgs::msg::Pose& new_pose);
    void updateVelocity(const Eigen::Vector3d& new_vel) { velocity_ = new_vel; }
    void updateBiases(const Eigen::Vector3d& accel_bias, const Eigen::Vector3d& gyro_bias) {
        accel_bias_ = accel_bias;
        gyro_bias_ = gyro_bias;
    }
    void setImageFeatures(const std::vector<uint8_t>& features);
    void setVisualData(const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors);

private:
    int id_;
    geometry_msgs::msg::Pose pose_;
    Eigen::Vector3d velocity_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d accel_bias_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d gyro_bias_{Eigen::Vector3d::Zero()};
    rclcpp::Time timestamp_;
    std::vector<uint8_t> image_features_;
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
};

} // namespace slam

#endif // GRAPH_NODE_HPP
