// imu_preintegrator.cpp
//
// Full IMU preintegration on manifold following:
//   Forster et al., "On-Manifold Preintegration for Real-Time
//   Visual-Inertial Odometry", TRO 2017.
//
// Key fixes versus the original stub:
//   1. Full 5-term Jacobians (dr_dbg, dv_dba, dv_dbg, dp_dba, dp_dbg)
//      propagated via the state-transition matrix at each midpoint step.
//   2. Full 15×15 continuous-discrete noise covariance propagation.
//   3. Configurable noise params via setNoiseParams().

#include "visual_graph_slam/vins/imu_preintegrator.hpp"
#include <cmath>

namespace slam::vins {

namespace {

/// Skew-symmetric matrix of a 3-vector.
inline Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d S;
    S <<  0.0, -v.z(),  v.y(),
         v.z(),   0.0, -v.x(),
        -v.y(),  v.x(),   0.0;
    return S;
}

/// Rodrigues' rotation for a rotation vector omega*dt.
/// Falls back to identity when the angle is negligible.
inline Eigen::Matrix3d Exp(const Eigen::Vector3d& phi)
{
    const double angle = phi.norm();
    if (angle < 1e-10) {
        // I + skew(phi) is NOT a rotation matrix (det ≈ 1 only to first order).
        // For negligible angle, identity is the correct SO(3) element.
        return Eigen::Matrix3d::Identity();
    }
    const Eigen::Vector3d axis = phi / angle;
    return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
ImuPreintegrator::ImuPreintegrator(const Eigen::Vector3d& accel_bias,
                                   const Eigen::Vector3d& gyro_bias)
    : accel_bias_(accel_bias), gyro_bias_(gyro_bias)
{
    reset();
}

// ─────────────────────────────────────────────────────────────────────────────
void ImuPreintegrator::reset()
{
    data_ = PreintegratedData{};
    data_.accel_noise = accel_noise_;
    data_.gyro_noise = gyro_noise_;
    measurements_.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
void ImuPreintegrator::setBiases(const Eigen::Vector3d& accel_bias,
                                 const Eigen::Vector3d& gyro_bias)
{
    accel_bias_ = accel_bias;
    gyro_bias_  = gyro_bias;
}

// ─────────────────────────────────────────────────────────────────────────────
void ImuPreintegrator::setNoiseParams(double accel_noise,
                                      double gyro_noise,
                                      double accel_bias_walk,
                                      double gyro_bias_walk)
{
    accel_noise_     = accel_noise;
    gyro_noise_      = gyro_noise;
    accel_bias_walk_ = accel_bias_walk;
    gyro_bias_walk_  = gyro_bias_walk;
}

// ─────────────────────────────────────────────────────────────────────────────
void ImuPreintegrator::update(const sensor_msgs::msg::Imu& msg)
{
    ImuMeasurement m;
    m.timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9;
    m.accel  = Eigen::Vector3d(msg.linear_acceleration.x,
                               msg.linear_acceleration.y,
                               msg.linear_acceleration.z);
    m.gyro   = Eigen::Vector3d(msg.angular_velocity.x,
                               msg.angular_velocity.y,
                               msg.angular_velocity.z);

    if (!measurements_.empty()) {
        propagate(measurements_.back(), m);
    }
    measurements_.push_back(m);
}

// ─────────────────────────────────────────────────────────────────────────────
void ImuPreintegrator::propagate(const ImuMeasurement& m1,
                                 const ImuMeasurement& m2)
{
    const double dt = m2.timestamp - m1.timestamp;
    if (dt <= 0.0 || dt > 1.0) {   // sanity: skip stale or too-large steps
        return;
    }

    // ── Bias-corrected measurements ──────────────────────────────────────────
    const Eigen::Vector3d a1  = m1.accel - accel_bias_;
    const Eigen::Vector3d g1  = m1.gyro  - gyro_bias_;
    const Eigen::Vector3d a2  = m2.accel - accel_bias_;
    const Eigen::Vector3d g2  = m2.gyro  - gyro_bias_;

    // ── Midpoint gyro ─────────────────────────────────────────────────────────
    const Eigen::Vector3d g_mid = 0.5 * (g1 + g2);
    // NOTE: body-frame a_mid is NOT used because the correct midpoint is computed
    // in world frame below (a_world_mid), which accounts for rotation between steps.

    // ── Rotation update (midpoint angle-axis) ────────────────────────────────
    const Eigen::Matrix3d dR  = Exp(g_mid * dt);                  // incremental rotation
    const Eigen::Matrix3d R_i = data_.delta_q.toRotationMatrix(); // rotation at step i
    const Eigen::Matrix3d R_j = R_i * dR;                        // rotation at step j

    // ── Velocity & position update (midpoint) ────────────────────────────────
    const Eigen::Vector3d a_world_i = R_i * a1;
    const Eigen::Vector3d a_world_j = R_j * a2;
    const Eigen::Vector3d a_world_mid = 0.5 * (a_world_i + a_world_j);

    data_.delta_p  = data_.delta_p + data_.delta_v * dt + 0.5 * a_world_mid * dt * dt;
    data_.delta_v  = data_.delta_v + a_world_mid * dt;
    data_.delta_q  = Eigen::Quaterniond(R_j).normalized();
    data_.dt      += dt;

    // ── Full Jacobian propagation (state-transition matrix) ──────────────────
    //
    // State: [phi (3), v (3), p (3), ba (3), bg (3)]
    // Using the midpoint Jacobians from Forster TRO 2017, Appendix.
    //
    // dR_j / dbg:
    //   d(R_i * dR) / dbg = R_i * d(dR)/dbg
    //   d(Exp(g_mid*dt)) / dbg ≈ -dR * dt  (first-order right Jacobian approximation)
    const Eigen::Matrix3d dR_dbg_step = -dR * dt;

    // dr_dbg  (rotation Jacobian w.r.t. gyro bias)
    //   J_j = dR^T * J_i + dR_dbg_step
    const Eigen::Matrix3d dr_dbg_new = dR.transpose() * data_.dr_dbg + dR_dbg_step;

    // dv_dba (velocity Jacobian w.r.t. accel bias)
    //   J_j = J_i + (R_i + R_j) * 0.5 * dt  =  J_i + R_mid * dt
    const Eigen::Matrix3d dv_dba_new = data_.dv_dba + 0.5 * (R_i + R_j) * dt;

    // dv_dbg (velocity Jacobian w.r.t. gyro bias)
    //   The gyro bias affects dR which in turn rotates acceleration.
    //   dv_j = dv_i + (-R_i*skew(a1)*dr_dbg_i - R_j*skew(a2)*dr_dbg_new) * 0.5 * dt
    const Eigen::Matrix3d dv_dbg_new =
        data_.dv_dbg
        + 0.5 * (-R_i * skew(a1) * data_.dr_dbg
                 - R_j * skew(a2) * dr_dbg_new) * dt;

    // dp_dba (position Jacobian w.r.t. accel bias)
    //   dp_j = dp_i + dv_i*dt + 0.5*(R_i + R_j)*0.5*dt²
    const Eigen::Matrix3d dp_dba_new =
        data_.dp_dba + data_.dv_dba * dt + 0.5 * dv_dba_new * dt * dt;

    // dp_dbg (position Jacobian w.r.t. gyro bias)
    //   dp_j = dp_i + dv_dbg_i*dt + 0.5*dv_dbg_j*dt²
    const Eigen::Matrix3d dp_dbg_new =
        data_.dp_dbg + data_.dv_dbg * dt + 0.5 * dv_dbg_new * dt * dt;

    data_.dr_dbg = dr_dbg_new;
    data_.dv_dba = dv_dba_new;
    data_.dv_dbg = dv_dbg_new;
    data_.dp_dba = dp_dba_new;
    data_.dp_dbg = dp_dbg_new;

    // ── 15×15 Noise Covariance Propagation ───────────────────────────────────
    // Discrete noise model: continuous PSD → discrete variances.
    // The 15×15 state transition matrix F (first-order) and noise matrix G
    // map to the covariance update:  P_j = F * P_i * F^T + G * Q * G^T
    //
    // Block structure: [phi(3), v(3), p(3), ba(3), bg(3)]
    Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Identity();

    // F[v, phi]:  d(delta_v) / d(delta_phi) — a_world rotated
    F.block<3, 3>(3, 0)  = -0.5 * (R_i * skew(a1) + R_j * skew(a2)) * dt;
    // F[p, phi]:  d(delta_p) / d(delta_phi)
    F.block<3, 3>(6, 0)  = -0.25 * (R_i * skew(a1) + R_j * skew(a2)) * dt * dt;
    // F[v, ba]:   d(delta_v) / d(ba)
    F.block<3, 3>(3, 9)  = -0.5 * (R_i + R_j) * dt;
    // F[p, ba]:   d(delta_p) / d(ba)
    F.block<3, 3>(6, 9)  = -0.25 * (R_i + R_j) * dt * dt;
    // F[p, v]:    d(delta_p) / d(delta_v) — integration time
    F.block<3, 3>(6, 3)  = Eigen::Matrix3d::Identity() * dt;
    // F[phi, bg]: d(delta_phi) / d(bg)
    F.block<3, 3>(0, 12) = -Eigen::Matrix3d::Identity() * dt;
    // F[v, bg]:   d(delta_v) / d(bg)
    F.block<3, 3>(3, 12) = 0.5 * (R_i * skew(a1) * data_.dr_dbg
                                 + R_j * skew(a2) * dr_dbg_new) * dt;
    // F[p, bg]:   d(delta_p) / d(bg)
    F.block<3, 3>(6, 12) = 0.25 * (R_i * skew(a1) * data_.dr_dbg
                                  + R_j * skew(a2) * dr_dbg_new) * dt * dt;

    // Discrete noise variances (σ² = σ_cont² / dt)
    const double sigma_a  = accel_noise_;
    const double sigma_g  = gyro_noise_;
    const double sigma_ba = accel_bias_walk_;
    const double sigma_bg = gyro_bias_walk_;

    Eigen::Matrix<double, 15, 15> Q = Eigen::Matrix<double, 15, 15>::Zero();
    Q.block<3, 3>(0, 0).diagonal().setConstant(sigma_g  * sigma_g  * dt);
    Q.block<3, 3>(3, 3).diagonal().setConstant(sigma_a  * sigma_a  * dt);
    Q.block<3, 3>(6, 6).diagonal().setConstant(0.25 * sigma_a * sigma_a * dt * dt * dt);
    Q.block<3, 3>(9, 9).diagonal().setConstant(sigma_ba * sigma_ba * dt);
    Q.block<3, 3>(12, 12).diagonal().setConstant(sigma_bg * sigma_bg * dt);

    data_.covariance = F * data_.covariance * F.transpose() + Q;
}

}  // namespace slam::vins
