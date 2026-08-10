#pragma once

#include <string>
#include <optional>
#include <memory>
#include <opencv2/core.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>

namespace slam::core {

struct FrontendResult {
    bool valid{false};
    geometry_msgs::msg::Transform relative_transform;
    double confidence{0.0};
    int inliers{0};
    int tracked_points{0};
    double inlier_ratio{0.0};
    int reference_age{1};
    double unit_sphere_parallax{0.0};
};

class Frontend {
public:
    using SharedPtr = std::shared_ptr<Frontend>;

    virtual ~Frontend() = default;

    /**
     * @brief Initialize the plugin with parameters from the lifecycle node
     */
    virtual void initialize(rclcpp::Node::SharedPtr node, const std::string& plugin_name) = 0;

    /**
     * @brief Extract features, track against previous frame, and return displacement
     * @param image Current image frame
     * @param external_displacement_prior Optional metric scale prior from external odometry
     * @param imu_rotation Optional relative rotation predicted by IMU preintegration
     */
    virtual FrontendResult processFrame(const cv::Mat& image,
                                        std::optional<double> external_displacement_prior = std::nullopt,
                                        const std::optional<Eigen::Matrix3d>& imu_rotation = std::nullopt) = 0;
    
    /**
     * @brief Add current frame to internal map/keyframe database
     */
    virtual void addKeyframe() = 0;

    /**
     * @brief Reset tracking state
     */
    virtual void reset() = 0;
    
    /**
     * @brief Dynamically update camera matrix (e.g. after camera_info)
     */
    virtual void setCameraMatrix(const cv::Mat& camera_matrix) {}

    /**
     * @brief Access to current keypoints for PnP recovery
     */
    virtual std::vector<cv::KeyPoint> getCurrentKeypoints() const = 0;

    /**
     * @brief Access to current descriptors for PnP recovery
     */
    virtual cv::Mat getCurrentDescriptors() const = 0;
};

} // namespace slam::core
