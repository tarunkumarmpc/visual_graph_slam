#include "visual_graph_slam/sensor_data_manager.hpp"
#include "visual_graph_slam/sensor_data.hpp"
#include <chrono>
#include <functional>
#include <algorithm>
#include <cctype>

namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

rmw_qos_reliability_policy_t parseReliability(const std::string& value) {
    const std::string normalized = toLower(value);
    if (normalized == "reliable") {
        return RMW_QOS_POLICY_RELIABILITY_RELIABLE;
    }
    if (normalized == "best_effort" || normalized == "besteffort") {
        return RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
    }
    return RMW_QOS_POLICY_RELIABILITY_RELIABLE;
}

}  // namespace

using namespace slam;

SensorDataManager::Mode SensorDataManager::parseMode(const std::string& mode) {
    const std::string normalized = toLower(mode);
    // Canonical mode names are mono-based. Keep legacy vision_* aliases for compatibility.
    if (normalized == "mono" || normalized == "mono_only" || normalized == "vision_only") {
        return Mode::VISION_ONLY;
    }
    if (normalized == "mono_imu" || normalized == "vision_imu") {
        return Mode::VISION_IMU;
    }
    if (normalized == "mono_wheel" || normalized == "vision_wheel") {
        return Mode::VISION_WHEEL;
    }
    if (normalized == "mono_external" || normalized == "vision_external") {
        return Mode::VISION_EXTERNAL;
    }
    if (normalized == "mono_imu_wheel" || normalized == "vision_full") {
        return Mode::VISION_FULL;
    }
    return Mode::VISION_FULL;
}

const char* SensorDataManager::modeToString(Mode mode) {
    switch (mode) {
        case Mode::VISION_ONLY: return "mono";
        case Mode::VISION_IMU: return "mono_imu";
        case Mode::VISION_WHEEL: return "mono_wheel";
        case Mode::VISION_EXTERNAL: return "mono_external";
        case Mode::VISION_FULL: return "mono_imu_wheel";
        default: return "mono_imu_wheel";
    }
}

SensorDataManager::SensorDataManager(std::shared_ptr<GraphSlam> graph_slam)
    : Node("sensor_data_manager"), graph_slam_(graph_slam), camera_info_received_(false), last_frame_timestamp_(0, 0, RCL_ROS_TIME) {
    RCLCPP_INFO(this->get_logger(), "Initializing SensorDataManager...");

    mode_name_ = this->declare_parameter<std::string>("mode", "mono_imu_wheel");
    mode_ = parseMode(mode_name_);

    image_topic_ = this->declare_parameter<std::string>("topics.image", "/camera/rgb/image_color");
    odom_topic_ = this->declare_parameter<std::string>("topics.odom", "/odom");
    imu_topic_ = this->declare_parameter<std::string>("topics.imu", "/imu/data");
    camera_info_topic_ = this->declare_parameter<std::string>("topics.camera_info", "/camera/rgb/camera_info");
    sync_queue_size_ = this->declare_parameter<int>("sync.queue_size", 20);
    const auto image_qos_reliability = parseReliability(
        this->declare_parameter<std::string>("topics.image_qos.reliability", "reliable"));
    const int image_qos_depth = this->declare_parameter<int>("topics.image_qos.depth", 10);

    rclcpp::QoS image_qos(static_cast<size_t>(std::max(1, image_qos_depth)));
    if (image_qos_reliability == RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT) {
        image_qos.best_effort();
    } else {
        image_qos.reliable();
    }
    image_qos.durability_volatile();

    rmw_qos_profile_t image_qos_profile = rmw_qos_profile_default;
    image_qos_profile.reliability = image_qos_reliability;
    image_qos_profile.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
    image_qos_profile.depth = static_cast<size_t>(std::max(1, image_qos_depth));
    image_qos_profile.durability = RMW_QOS_POLICY_DURABILITY_VOLATILE;

    const bool image_only_mode = (mode_ == Mode::VISION_ONLY || mode_ == Mode::VISION_IMU);
    if (image_only_mode) {
        image_raw_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            image_topic_, image_qos,
            std::bind(&SensorDataManager::image_callback, this, std::placeholders::_1));
    } else {
        image_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, image_topic_, image_qos_profile);
        odom_sub_ = std::make_shared<message_filters::Subscriber<nav_msgs::msg::Odometry>>(this, odom_topic_, image_qos_profile);

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(sync_queue_size_), *image_sub_, *odom_sub_);
        sync_->registerCallback(std::bind(&SensorDataManager::synced_callback, this, std::placeholders::_1, std::placeholders::_2));
    }

    if (mode_ == Mode::VISION_IMU || mode_ == Mode::VISION_FULL) {
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, 100,
            std::bind(&SensorDataManager::imu_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "IMU subscriber enabled: %s", imu_topic_.c_str());
    } else {
        RCLCPP_INFO(this->get_logger(), "IMU subscriber disabled for mode: %s", modeToString(mode_));
    }

    // camera_info subscription is optional — skip if topic is empty
    if (!camera_info_topic_.empty()) {
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic_, rclcpp::QoS(10),
            std::bind(&SensorDataManager::camera_info_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "camera_info subscriber: %s", camera_info_topic_.c_str());
    } else {
        RCLCPP_INFO(this->get_logger(),
                    "No camera_info topic configured — using YAML camera parameters only.");
        camera_info_received_ = true;  // treat as "already set" so we never wait for it
    }

    RCLCPP_INFO(this->get_logger(),
                "SensorDataManager initialized (mode=%s, image=%s, odom=%s, imu=%s, queue=%d, image_qos=%s depth=%d).",
                modeToString(mode_),
                image_topic_.c_str(),
                odom_topic_.c_str(),
                imu_topic_.c_str(),
                sync_queue_size_,
                image_qos_reliability == RMW_QOS_POLICY_RELIABILITY_RELIABLE ? "reliable" : "best_effort",
                image_qos_depth);
}

void SensorDataManager::image_callback(const sensor_msgs::msg::Image::SharedPtr image_msg) {
    if (!image_msg) {
        return;
    }

    RCLCPP_INFO_ONCE(this->get_logger(), "First image received! Topic: %s", image_topic_.c_str());
    
    const rclcpp::Time current_stamp = image_msg->header.stamp;
    std::vector<sensor_msgs::msg::Imu> imu_to_process;
    {
        std::lock_guard<std::mutex> lock(imu_buffer_mutex_);
        auto it = imu_buffer_.begin();
        while (it != imu_buffer_.end()) {
            const rclcpp::Time imu_stamp = it->header.stamp;
            if (imu_stamp <= last_frame_timestamp_) {
                it = imu_buffer_.erase(it);
            } else if (imu_stamp <= current_stamp) {
                imu_to_process.push_back(*it);
                ++it;
            } else {
                break;
            }
        }
    }
    
    last_frame_timestamp_ = current_stamp;
    graph_slam_->process_image_frame(image_msg, imu_to_process);
}

void SensorDataManager::synced_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                                        const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg) {
    RCLCPP_INFO_ONCE(this->get_logger(), "First synchronized image+odom pair received!");
    
    const rclcpp::Time current_stamp = image_msg->header.stamp;
    std::vector<sensor_msgs::msg::Imu> imu_to_process;
    {
        std::lock_guard<std::mutex> lock(imu_buffer_mutex_);
        auto it = imu_buffer_.begin();
        while (it != imu_buffer_.end()) {
            const rclcpp::Time imu_stamp = it->header.stamp;
            if (imu_stamp <= last_frame_timestamp_) {
                it = imu_buffer_.erase(it);
            } else if (imu_stamp <= current_stamp) {
                imu_to_process.push_back(*it);
                ++it;
            } else {
                break;
            }
        }
    }

    last_frame_timestamp_ = current_stamp;
    SensorData data(image_msg, odom_msg, latest_imu_, imu_to_process);
    graph_slam_->process_data(data);
}

void SensorDataManager::imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu_msg) {
    static int imu_count = 0;
    if (imu_count % 100 == 0) {
        RCLCPP_INFO_ONCE(this->get_logger(), "IMU data stream detected!");
    }
    imu_count++;

    // Keep IMU measurements in body frame (base_link / oxts_link, Z-up).
    // DO NOT manually rotate to camera frame, as our backend graph, VO poses, and GTSAM planar priors operate in body frame.
    sensor_msgs::msg::Imu transformed_imu = *imu_msg;

    latest_imu_ = transformed_imu;
    {
        std::lock_guard<std::mutex> lock(imu_buffer_mutex_);
        imu_buffer_.push_back(transformed_imu);
        if (imu_buffer_.size() > 1000) { // Safety limit
            imu_buffer_.erase(imu_buffer_.begin());
        }
    }
}

void SensorDataManager::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr camera_info_msg) {
    if (!camera_info_msg || camera_info_received_) {
        return;
    }

    graph_slam_->updateCameraCalibration(*camera_info_msg);
    camera_info_received_ = true;
    RCLCPP_INFO(this->get_logger(), "Camera calibration received from %s", camera_info_topic_.c_str());
}
