#include "visual_graph_slam/telemetry/slam_telemetry.hpp"
#include <iomanip>
#include <sstream>
#include <cmath>

namespace slam::telemetry {

Eigen::Vector3d TelemetryLogger::computeEulerFromRot(const Eigen::Matrix3d& R) {
    // Roll (x), Pitch (y), Yaw (z) in degrees
    // Using standard aerospace sequence Z-Y-X (Yaw-Pitch-Roll)
    Eigen::Vector3d euler_rad = R.eulerAngles(2, 1, 0); // [Yaw, Pitch, Roll]
    // Reorder to [Roll, Pitch, Yaw] and convert to degrees
    Eigen::Vector3d euler_deg;
    euler_deg(0) = euler_rad(2) * 180.0 / M_PI;
    euler_deg(1) = euler_rad(1) * 180.0 / M_PI;
    euler_deg(2) = euler_rad(0) * 180.0 / M_PI;
    return euler_deg;
}

Eigen::Vector3d TelemetryLogger::computeEulerFromQuat(const Eigen::Quaterniond& q) {
    return computeEulerFromRot(q.toRotationMatrix());
}

void TelemetryLogger::renderReport(const TelemetrySnapshot& snap, rclcpp::Logger logger) {
    std::stringstream ss;
    ss << "\n"
       << "================================================================================\n"
       << ">>> [TELEMETRY DASHBOARD | FRAME ID: " << std::setw(6) << snap.frame_id << "] <<<\n"
       << "================================================================================\n";

    if (snap.sign_flip_alert) {
        ss << "********************************************************************************\n"
           << "*** [ALERT: POSSIBLE AXIS SIGN FLIP DETECTED]                                ***\n"
           << "*** VO Yaw: " << std::setw(8) << std::fixed << std::setprecision(2) << snap.vo_euler_deg(2) 
           << " deg | IMU Yaw: " << std::setw(8) << snap.imu_euler_deg(2) << " deg (Opposite Signs!) ***\n"
           << "********************************************************************************\n";
    }

    // Stage 1: Input State
    ss << "--- [Stage 1: Input State & Timestamps (A to Z)] ---\n"
       << "  t_cam: " << std::fixed << std::setprecision(4) << snap.t_camera_s << " s | t_imu: " << snap.t_imu_s << " s\n"
       << "  Lag (t_cam - t_imu): " << std::setprecision(2) << snap.latency_ms << " ms | IMU Packets: " << snap.imu_packet_count << "\n"
       << "  Raw Gyro [base_link]:   [" << std::setprecision(3) << snap.raw_gyro(0) << ", " << snap.raw_gyro(1) << ", " << snap.raw_gyro(2) << "] rad/s\n"
       << "  Raw Accel [base_link]:  [" << snap.raw_accel(0) << ", " << snap.raw_accel(1) << ", " << snap.raw_accel(2) << "] m/s^2\n\n";

    // Stage 2: Frontend State
    ss << "--- [Stage 2: Frontend State — VO & IMU Preintegration (A to Z)] ---\n"
       << "  VO Tracked Pts: " << snap.vo_tracked_points << " | Inlier Ratio: " << std::setprecision(1) << (snap.vo_inlier_ratio * 100.0) << " %\n"
       << "  VO Rel Euler (RPY):   [" << std::setprecision(2) << snap.vo_euler_deg(0) << ", " << snap.vo_euler_deg(1) << ", " << snap.vo_euler_deg(2) << "] deg\n"
       << "  VO Rel Quat (xyzw):   [" << std::setprecision(4) << snap.vo_quat(0) << ", " << snap.vo_quat(1) << ", " << snap.vo_quat(2) << ", " << snap.vo_quat(3) << "]\n"
       << "  VO Rel Trans (xyz):   [" << std::setprecision(3) << snap.vo_trans(0) << ", " << snap.vo_trans(1) << ", " << snap.vo_trans(2) << "] m\n"
       << "  IMU Rel Euler (RPY):  [" << std::setprecision(2) << snap.imu_euler_deg(0) << ", " << snap.imu_euler_deg(1) << ", " << snap.imu_euler_deg(2) << "] deg\n"
       << "  IMU Rel Quat (xyzw):  [" << std::setprecision(4) << snap.imu_quat(0) << ", " << snap.imu_quat(1) << ", " << snap.imu_quat(2) << ", " << snap.imu_quat(3) << "]\n"
       << "  IMU Accel [cam0_link]:[" << std::setprecision(3) << snap.imu_trans_accel(0) << ", " << snap.imu_trans_accel(1) << ", " << snap.imu_trans_accel(2) << "] m/s^2\n"
       << "  Yaw Diff (VO - IMU):   " << std::setprecision(2) << snap.yaw_diff_deg << " deg\n\n";

    // Stage 3: Local Optimization State
    Eigen::Quaterniond q_loc(snap.local_fused_pose.orientation.w, snap.local_fused_pose.orientation.x, snap.local_fused_pose.orientation.y, snap.local_fused_pose.orientation.z);
    Eigen::Vector3d loc_euler = computeEulerFromQuat(q_loc);
    ss << "--- [Stage 3: Local Optimization State — VINS & Local Backend (A to Z)] ---\n"
       << "  Local Fused Trans (xyz): [" << std::setprecision(2) << snap.local_fused_pose.position.x << ", " 
       << snap.local_fused_pose.position.y << ", " << snap.local_fused_pose.position.z << "] m\n"
       << "  Local Fused Euler (RPY): [" << std::setprecision(2) << loc_euler(0) << ", " << loc_euler(1) << ", " << loc_euler(2) << "] deg\n"
       << "  Local Fused Quat (xyzw): [" << std::setprecision(4) << q_loc.x() << ", " << q_loc.y() << ", " << q_loc.z() << ", " << q_loc.w() << "]\n"
       << "  VINS Initialized:        " << (snap.vins_initialized ? "YES" : "NO") << " | Scale (s): " << std::setprecision(4) << snap.vins_scale << "\n"
       << "  VINS Gyro Bias:          [" << std::setprecision(4) << snap.vins_gyro_bias(0) << ", " << snap.vins_gyro_bias(1) << ", " << snap.vins_gyro_bias(2) << "] rad/s\n"
       << "  VINS Gravity:            [" << std::setprecision(2) << snap.vins_gravity(0) << ", " << snap.vins_gravity(1) << ", " << snap.vins_gravity(2) << "] m/s^2\n\n";

    // Stage 4: Global Optimization State
    Eigen::Quaterniond q_glob(snap.global_optimized_pose.orientation.w, snap.global_optimized_pose.orientation.x, snap.global_optimized_pose.orientation.y, snap.global_optimized_pose.orientation.z);
    Eigen::Vector3d glob_euler = computeEulerFromQuat(q_glob);
    ss << "--- [Stage 4: Global Optimization State — Backend Graph (A to Z)] ---\n"
       << "  Global Graph Trans (xyz): [" << std::setprecision(2) << snap.global_optimized_pose.position.x << ", " 
       << snap.global_optimized_pose.position.y << ", " << snap.global_optimized_pose.position.z << "] m\n"
       << "  Global Graph Euler (RPY): [" << std::setprecision(2) << glob_euler(0) << ", " << glob_euler(1) << ", " << glob_euler(2) << "] deg\n"
       << "  Global Graph Quat (xyzw): [" << std::setprecision(4) << q_glob.x() << ", " << q_glob.y() << ", " << q_glob.z() << ", " << q_glob.w() << "]\n"
       << "  Active Factors:           " << snap.active_factor_count << " | Global Alignment Applied: " << (snap.global_alignment_applied ? "YES" : "NO") << "\n\n";

    // Stage 5: Diagnostics & Drift Analysis
    ss << "--- [Stage 5: Root-Cause Diagnostics & Drift Analysis (A to Z)] ---\n"
       << "  Trans Drift (Loc vs Glob): " << std::setprecision(3) << snap.trans_drift_m << " m | Rot Drift: " << std::setprecision(2) << snap.rot_drift_deg << " deg\n";
    if (snap.gt_available) {
        Eigen::Quaterniond q_gt(snap.gt_pose.orientation.w, snap.gt_pose.orientation.x, snap.gt_pose.orientation.y, snap.gt_pose.orientation.z);
        Eigen::Vector3d gt_euler = computeEulerFromQuat(q_gt);
        ss << "  Ground Truth Trans (xyz): [" << std::setprecision(2) << snap.gt_pose.position.x << ", " 
           << snap.gt_pose.position.y << ", " << snap.gt_pose.position.z << "] m\n"
           << "  Ground Truth Euler (RPY): [" << std::setprecision(2) << gt_euler(0) << ", " << gt_euler(1) << ", " << gt_euler(2) << "] deg\n"
           << "  Ground Truth Quat (xyzw): [" << std::setprecision(4) << q_gt.x() << ", " << q_gt.y() << ", " << q_gt.z() << ", " << q_gt.w() << "]\n";
    }
    ss << "================================================================================\n";

    RCLCPP_INFO_STREAM(logger, ss.str());
}

} // namespace slam::telemetry
