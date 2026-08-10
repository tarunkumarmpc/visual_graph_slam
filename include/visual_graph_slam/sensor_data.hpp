#ifndef SENSOR_DATA_HPP
#define SENSOR_DATA_HPP

#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <rclcpp/rclcpp.hpp>
#include <optional>

namespace slam {

struct SensorData {
    sensor_msgs::msg::Image::ConstSharedPtr image;
    nav_msgs::msg::Odometry::ConstSharedPtr odometry;
    std::optional<sensor_msgs::msg::Imu> imu;
    std::vector<sensor_msgs::msg::Imu> imu_buffer;
    rclcpp::Time timestamp;

    // Default constructor
    SensorData() = default;

    // Parameterized constructor
    SensorData(sensor_msgs::msg::Image::ConstSharedPtr img,
               nav_msgs::msg::Odometry::ConstSharedPtr odom,
               const std::optional<sensor_msgs::msg::Imu>& imu_msg = std::nullopt,
               const std::vector<sensor_msgs::msg::Imu>& imus = {})
        : image(img), odometry(odom), imu(imu_msg), imu_buffer(imus) {
        if (img) {
            timestamp = img->header.stamp; // Set timestamp from image
        }
    }
};

} // namespace slam

#endif // SENSOR_DATA_HPP
