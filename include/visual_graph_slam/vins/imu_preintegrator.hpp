#ifndef VISUAL_GRAPH_SLAM_VINS_IMU_PREINTEGRATOR_HPP
#define VISUAL_GRAPH_SLAM_VINS_IMU_PREINTEGRATOR_HPP

#include <Eigen/Dense>
#include <vector>
#include <sensor_msgs/msg/imu.hpp>

namespace slam::vins {

struct ImuMeasurement {
    double timestamp{0.0};
    Eigen::Vector3d accel{Eigen::Vector3d::Zero()};
    Eigen::Vector3d gyro{Eigen::Vector3d::Zero()};
};

class ImuPreintegrator {
public:
    // ─────────────────────────────────────────────────────────────────────────
    // Preintegrated IMU state between two keyframes.
    //
    // Notation follows Forster et al. (TRO 2017), "On-Manifold Preintegration":
    //   delta_q  — accumulated rotation  (body frame)
    //   delta_v  — accumulated velocity  (world frame)
    //   delta_p  — accumulated position  (world frame)
    //   dt       — total integration time [s]
    //
    // Jacobians w.r.t. linearisation-point biases allow first-order correction
    // when bias estimates change without re-integrating the raw measurements:
    //   dr_dbg   — d(delta_R) / d(gyro_bias)
    //   dv_dba   — d(delta_v) / d(accel_bias)
    //   dv_dbg   — d(delta_v) / d(gyro_bias)
    //   dp_dba   — d(delta_p) / d(accel_bias)
    //   dp_dbg   — d(delta_p) / d(gyro_bias)
    //
    // covariance — full 15×15 noise covariance matrix propagated
    //              using the continuous-discrete noise model.
    //              Order: [delta_phi (3), delta_v (3), delta_p (3),
    //                      accel_bias (3), gyro_bias (3)]
    // ─────────────────────────────────────────────────────────────────────────
    struct PreintegratedData {
        Eigen::Quaterniond delta_q{Eigen::Quaterniond::Identity()};
        Eigen::Vector3d    delta_v{Eigen::Vector3d::Zero()};
        Eigen::Vector3d    delta_p{Eigen::Vector3d::Zero()};
        double             dt{0.0};

        // Bias Jacobians (all 3×3)
        Eigen::Matrix3d dr_dbg{Eigen::Matrix3d::Zero()};
        Eigen::Matrix3d dv_dbg{Eigen::Matrix3d::Zero()};
        Eigen::Matrix3d dv_dba{Eigen::Matrix3d::Zero()};
        Eigen::Matrix3d dp_dbg{Eigen::Matrix3d::Zero()};
        Eigen::Matrix3d dp_dba{Eigen::Matrix3d::Zero()};

        // Full 15×15 covariance (phi, v, p, ba, bg)
        Eigen::Matrix<double, 15, 15> covariance{
            Eigen::Matrix<double, 15, 15>::Identity() * 1e-4};
            
        // Continuous-time noise densities (needed by GTSAM factor)
        double accel_noise{0.02};
        double gyro_noise{0.0017};
    };

    // ─────────────────────────────────────────────────────────────────────────
    // Construction / reset
    // ─────────────────────────────────────────────────────────────────────────
    explicit ImuPreintegrator(
        const Eigen::Vector3d& accel_bias = Eigen::Vector3d::Zero(),
        const Eigen::Vector3d& gyro_bias  = Eigen::Vector3d::Zero());

    void reset();

    // ─────────────────────────────────────────────────────────────────────────
    // Bias / noise configuration (call before reset() takes effect)
    // ─────────────────────────────────────────────────────────────────────────
    void setBiases(const Eigen::Vector3d& accel_bias,
                   const Eigen::Vector3d& gyro_bias);

    /// Set continuous-time IMU noise standard deviations.
    /// accel_noise     — accelerometer white noise [m/s²/√Hz]
    /// gyro_noise      — gyroscope white noise     [rad/s/√Hz]
    /// accel_bias_walk — accelerometer bias random walk [m/s³/√Hz]
    /// gyro_bias_walk  — gyroscope bias random walk    [rad/s²/√Hz]
    void setNoiseParams(double accel_noise,
                        double gyro_noise,
                        double accel_bias_walk,
                        double gyro_bias_walk);

    // ─────────────────────────────────────────────────────────────────────────
    // Measurement ingestion
    // ─────────────────────────────────────────────────────────────────────────
    void update(const sensor_msgs::msg::Imu& msg);

    // ─────────────────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────────────────
    const PreintegratedData&         getPreintegratedData() const { return data_; }
    const std::vector<ImuMeasurement>& getMeasurements()    const { return measurements_; }

private:
    // Midpoint integration step — updates data_ given two consecutive samples.
    void propagate(const ImuMeasurement& m1, const ImuMeasurement& m2);

    Eigen::Vector3d accel_bias_;
    Eigen::Vector3d gyro_bias_;

    PreintegratedData          data_;
    std::vector<ImuMeasurement> measurements_;

    // Continuous-time noise standard deviations (configurable)
    double accel_noise_     = 0.1;      // [m/s²/√Hz]
    double gyro_noise_      = 0.01;     // [rad/s/√Hz]
    double accel_bias_walk_ = 0.001;    // [m/s³/√Hz]
    double gyro_bias_walk_  = 0.0001;   // [rad/s²/√Hz]
};

}  // namespace slam::vins

#endif  // VISUAL_GRAPH_SLAM_VINS_IMU_PREINTEGRATOR_HPP
