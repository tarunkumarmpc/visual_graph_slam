#ifndef SENSOR_DATA_MANAGER_HPP
#define SENSOR_DATA_MANAGER_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <optional>
#include <deque>
#include <memory>
#include <string>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include "visual_graph_slam/sensor_data.hpp" // Include the SensorData header
#include "visual_graph_slam/core/graph_slam.hpp"
#include "visual_graph_slam/slam_visualizer.hpp"


namespace slam {

class SensorDataManager : public rclcpp::Node {

public:
    explicit SensorDataManager(std::shared_ptr<GraphSlam> graph_slam_);

private:
    enum class Mode {
        VISION_ONLY,
        VISION_IMU,
        VISION_WHEEL,
        VISION_EXTERNAL,
        VISION_FULL
    };

    static Mode parseMode(const std::string& mode);
    static const char* modeToString(Mode mode);

    void image_callback(const sensor_msgs::msg::Image::SharedPtr image_msg);
    void synced_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                        const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg);
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu_msg);
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr camera_info_msg);

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, nav_msgs::msg::Odometry>;

    // Synchronized subscribers
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> image_sub_;
    std::shared_ptr<message_filters::Subscriber<nav_msgs::msg::Odometry>> odom_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    
    std::mutex imu_buffer_mutex_;
    std::vector<sensor_msgs::msg::Imu> imu_buffer_;
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_raw_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    
    std::optional<sensor_msgs::msg::Imu> latest_imu_;
    std::optional<SensorData> synchronized_data_;

    std::string image_topic_;
    std::string odom_topic_;
    std::string imu_topic_;
    std::string camera_info_topic_;
    std::string mode_name_;
    int sync_queue_size_;
    bool camera_info_received_;
    Mode mode_;
    rclcpp::Time last_frame_timestamp_;

    std::shared_ptr<GraphSlam> graph_slam_;
    std::shared_ptr<Visualizer>visualizer_;
};

} // namespace slam

#endif // SENSOR_DATA_MANAGER_HPP
