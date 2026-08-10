#ifndef VISUAL_GRAPH_SLAM_TELEMETRY_SLAM_TELEMETRY_HPP
#define VISUAL_GRAPH_SLAM_TELEMETRY_SLAM_TELEMETRY_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>
#include <optional>

namespace slam::telemetry {

struct TelemetrySnapshot {
    uint64_t frame_id = 0;

    // Stage 1: Raw Sensor Streams & Input State (A to Z)
    double t_camera_s = 0.0;
    double t_imu_s = 0.0;
    double latency_ms = 0.0;
    int imu_packet_count = 0;
    Eigen::Vector3d raw_gyro = Eigen::Vector3d::Zero();
    Eigen::Vector3d raw_accel = Eigen::Vector3d::Zero();

    // Stage 2: Frontend State — VO & IMU Preintegration (A to Z)
    int vo_tracked_points = 0;
    double vo_inlier_ratio = 0.0;
    Eigen::Vector3d vo_euler_deg = Eigen::Vector3d::Zero(); // [Roll, Pitch, Yaw]
    Eigen::Vector4d vo_quat = Eigen::Vector4d(0, 0, 0, 1);  // [x, y, z, w]
    Eigen::Vector3d vo_trans = Eigen::Vector3d::Zero();     // [tx, ty, tz]

    Eigen::Vector3d imu_euler_deg = Eigen::Vector3d::Zero(); // [Roll, Pitch, Yaw]
    Eigen::Vector4d imu_quat = Eigen::Vector4d(0, 0, 0, 1);  // [x, y, z, w]
    Eigen::Vector3d imu_trans_accel = Eigen::Vector3d::Zero(); // Accel transformed to cam0_link

    double yaw_diff_deg = 0.0;
    bool sign_flip_alert = false;

    // Stage 3: Local Optimization State — VINS & Local Backend (A to Z)
    geometry_msgs::msg::Pose local_fused_pose;
    bool vins_initialized = false;
    Eigen::Vector3d vins_gyro_bias = Eigen::Vector3d::Zero();
    double vins_scale = 1.0;
    Eigen::Vector3d vins_gravity = Eigen::Vector3d(0, 0, -9.81);

    // Stage 4: Global Optimization State — Backend Graph & Loop Closure (A to Z)
    geometry_msgs::msg::Pose global_optimized_pose;
    int active_factor_count = 0;
    bool global_alignment_applied = false;

    // Stage 5: Root-Cause Diagnostics & Drift Analysis (A to Z)
    double trans_drift_m = 0.0;
    double rot_drift_deg = 0.0;
    bool gt_available = false;
    geometry_msgs::msg::Pose gt_pose;
};

class TelemetryLogger {
public:
    /**
     * @brief Render and print the complete multi-stage diagnostic dashboard to the terminal.
     */
    static void renderReport(const TelemetrySnapshot& snap, rclcpp::Logger logger);

    /**
     * @brief Compute Euler angles [Roll, Pitch, Yaw] in degrees from a Quaternion [x, y, z, w].
     */
    static Eigen::Vector3d computeEulerFromQuat(const Eigen::Quaterniond& q);

    /**
     * @brief Compute Euler angles [Roll, Pitch, Yaw] in degrees from a Rotation Matrix.
     */
    static Eigen::Vector3d computeEulerFromRot(const Eigen::Matrix3d& R);
};

} // namespace slam::telemetry

#endif // VISUAL_GRAPH_SLAM_TELEMETRY_SLAM_TELEMETRY_HPP
