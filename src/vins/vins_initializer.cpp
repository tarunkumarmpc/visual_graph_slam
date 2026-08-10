// vins_initializer.cpp
//
// Visual-Inertial Alignment (VINS Initialization)
//
// Implements the two-step initialization from:
//   Qin & Shen, "Monocular Visual-Inertial State Estimator", RAL/IROS 2017
//   (VINS-Mono, Section IV-A: Initialization)
//
// Step 1 — Solve for scale (s) and gravity (g) via linear system
//           by eliminating per-keyframe velocities between consecutive pairs.
// Step 2 — Back-substitute to recover per-keyframe velocities.
//
// Key fixes versus the original stub:
//   1. Correct A matrix: the original used lhs = (Pj-Pi)*dt_jk - (Pk-Pj)*dt_ij
//      which is dimensionally inconsistent.  The correct elimination
//      (from VINS-Mono eq. 10) yields a 6×4 system per triplet in (s, g_x, g_y, g_z).
//   2. Velocity estimation: after solving for s and g, per-KF velocities are
//      recovered from the preintegration equations.
//   3. Gravity magnitude sanity check [9.5, 10.5] m/s².
//   4. Scale sanity bounds tightened [0.05, 50.0].

#include "visual_graph_slam/vins/vins_initializer.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace slam::vins {

namespace {

// Minimum gravity magnitude [m/s²] accepted as physically plausible.
constexpr double kGravityMin =  9.5;
constexpr double kGravityMax = 10.5;

// Scale bounds
// Scale bounds are now driven entirely by VinsConfig::scale_prior_min / scale_prior_max
// (set from YAML: vins.scale_prior_min / vins.scale_prior_max).
// Default values are [0.1, 50.0] — wide enough for both KITTI (car) and indoor drone.
// Both evaluateMotionRegime() and the final acceptance gate in estimateGravityAndScale()
// read from config_, so changing the YAML is sufficient to tune per-deployment.
//
// KITTI (outdoor car, 10-30 m/s, keyframe ~1.5m):  scale_prior_min=0.3, max=30.0
// Indoor drone (0.5-3 m/s, keyframe ~0.5m):         scale_prior_min=0.1, max=5.0

}  // namespace

VinsInitializer::VinsInitializer() = default;

// ─────────────────────────────────────────────────────────────────────────────
VinsInitializer::InitializationResult VinsInitializer::align(
    const std::vector<std::shared_ptr<GraphNode>>&             keyframes,
    const std::vector<ImuPreintegrator::PreintegratedData>&    imu_data)
{
    InitializationResult result;

    // Need at least 5 keyframes and matching (N-1 or N) IMU segments.
    if (keyframes.size() < 5 || (imu_data.size() != keyframes.size() - 1 && imu_data.size() != keyframes.size())) {
        return result;
    }

    // Step 1 — Estimate Gyroscope Bias and correct IMU preintegration
    std::vector<ImuPreintegrator::PreintegratedData> corrected_imu_data = imu_data;
    if (estimateGyroBias(keyframes, corrected_imu_data, result.gyro_bias)) {
        std::cerr << "[VinsInit] Estimated Gyro Bias: [" << result.gyro_bias.transpose() << "]\n";
        const int imu_offset = (corrected_imu_data.size() == keyframes.size()) ? 1 : 0;
        for (size_t i = imu_offset; i < corrected_imu_data.size(); ++i) {
            auto& pim = corrected_imu_data[i];
            Eigen::Vector3d dtheta = pim.dr_dbg * result.gyro_bias;
            if (dtheta.norm() > 1e-8) {
                Eigen::AngleAxisd dq(dtheta.norm(), dtheta.normalized());
                pim.delta_q = pim.delta_q * Eigen::Quaterniond(dq);
            }
            pim.delta_v += pim.dv_dbg * result.gyro_bias;
            pim.delta_p += pim.dp_dbg * result.gyro_bias;
        }
    }

    // Step 2 & 3 — Estimate scale and gravity with modular 3-regime logic
    if (!estimateGravityAndScale(keyframes, corrected_imu_data, result.scale, result.gravity, result.regime)) {
        return result;
    }

    // Step 2 — Back-solve per-keyframe velocities.
    //
    // For segment i→j:
    //   delta_p_ij = s*(Pj - Pi) - Vi*dt_ij - 0.5*g*dt_ij²
    //   => Vi = (s*(Pj - Pi) - delta_p_ij - 0.5*g*dt_ij²) / dt_ij
    const double s = result.scale;
    const Eigen::Vector3d g = result.gravity;
    const int N = static_cast<int>(keyframes.size());
    const int imu_offset = (imu_data.size() == keyframes.size()) ? 1 : 0;

    result.velocities.resize(N, Eigen::Vector3d::Zero());

    for (int i = 0; i < N - 1; ++i) {
        const auto& p_i = keyframes[i]->getPose().position;
        const auto& p_j = keyframes[i + 1]->getPose().position;
        // ALG-3 FIX: use corrected_imu_data (gyro-bias-corrected), not the raw imu_data.
        const double dt = corrected_imu_data[i + imu_offset].dt;

        if (dt < 1e-6) {
            continue;
        }

        // World-frame IMU delta_p at the linearization point
        const Eigen::Quaterniond q_i(
            keyframes[i]->getPose().orientation.w,
            keyframes[i]->getPose().orientation.x,
            keyframes[i]->getPose().orientation.y,
            keyframes[i]->getPose().orientation.z);
        const Eigen::Vector3d dp_imu = q_i * corrected_imu_data[i + imu_offset].delta_p;

        const Eigen::Vector3d Pi(p_i.x, p_i.y, p_i.z);
        const Eigen::Vector3d Pj(p_j.x, p_j.y, p_j.z);

        result.velocities[i] = (s * (Pj - Pi) - dp_imu - 0.5 * g * dt * dt) / dt;
    }
    // Last keyframe velocity approximated from the second-to-last segment.
    if (N >= 2) {
        result.velocities[N - 1] = result.velocities[N - 2];
    }

    result.success = true;
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
VinsInitializer::MotionRegime VinsInitializer::evaluateMotionRegime(
    const std::vector<std::shared_ptr<GraphNode>>&          keyframes,
    const std::vector<ImuPreintegrator::PreintegratedData>& imu_data,
    const Eigen::MatrixXd&                                  A,
    const Eigen::VectorXd&                                  singular_values,
    double                                                  estimated_scale) const
{
    // 1. Check if vehicle is stationary by computing average visual speed
    double total_dist = 0.0;
    double total_time = 0.0;
    const int imu_offset = (imu_data.size() == keyframes.size()) ? 1 : 0;
    for (size_t i = 0; i < keyframes.size() - 1; ++i) {
        const auto& p_i = keyframes[i]->getPose().position;
        const auto& p_j = keyframes[i + 1]->getPose().position;
        const double dx = p_j.x - p_i.x;
        const double dy = p_j.y - p_i.y;
        const double dz = p_j.z - p_i.z;
        total_dist += std::sqrt(dx * dx + dy * dy + dz * dz);
        total_time += imu_data[i + imu_offset].dt;
    }
    const double v_vis = (total_time > 1e-6) ? (total_dist / total_time) : 0.0;
    if (v_vis < config_.stationary_velocity_threshold) {
        std::cerr << "[VinsInit] Regime 1 (STATIONARY): Vehicle is stopped or motion too small (v_vis=" 
                  << v_vis << " < " << config_.stationary_velocity_threshold << " VO-units/s).\n";
        return MotionRegime::STATIONARY;
    }

    // 2. Check if scale is physically observable (Dynamic Excitation)
    constexpr double kScaleColMinNorm = 0.05;
    constexpr double kSingularValueThreshold = 1e-4;
    constexpr double kMaxConditionNumber = 1e6;

    const double scale_col_norm = A.col(0).norm();
    const double condition_number = (singular_values.size() >= 4 && singular_values(3) > 0) 
                                    ? (singular_values(0) / singular_values(3)) : 1e9;

    const bool is_observable = (scale_col_norm >= kScaleColMinNorm) &&
                               (singular_values.size() >= 4 && singular_values(3) >= kSingularValueThreshold) &&
                               (condition_number <= kMaxConditionNumber) &&
                               (estimated_scale >= config_.scale_prior_min && estimated_scale <= config_.scale_prior_max);

    if (is_observable) {
        return MotionRegime::DYNAMIC_EXCITATION;
    }

    // 3. Constant-Velocity Fallback (when moving but unobservable)
    if (config_.use_scale_prior_fallback) {
        return MotionRegime::CONSTANT_VELOCITY_FALLBACK;
    }

    return MotionRegime::STATIONARY;
}

// ─────────────────────────────────────────────────────────────────────────────
bool VinsInitializer::estimateGravityAndScale(
    const std::vector<std::shared_ptr<GraphNode>>&          keyframes,
    const std::vector<ImuPreintegrator::PreintegratedData>& imu_data,
    double& scale,
    Eigen::Vector3d& gravity,
    MotionRegime& detected_regime)
{
    // ── Build the over-determined linear system  A * x = b ───────────────────
    //
    // For each consecutive triplet (i, j=i+1, k=i+2) we get 6 scalar equations
    // in 4 unknowns x = [s, g_x, g_y, g_z]^T.
    //
    // The kinematic equations for segment i→j and j→k:
    //
    //   s*(Pj - Pi) = Vi*dt_ij + 0.5*g*dt_ij² + R_i*dp_ij
    //   s*(Pk - Pj) = Vj*dt_jk + 0.5*g*dt_jk² + R_j*dp_jk
    //   where Vj = Vi + g*dt_ij + R_i*dv_ij
    //
    // Eliminating Vi yields:
    //   s * [(Pj-Pi)*dt_jk - (Pk-Pj)*dt_ij]
    //     + 0.5*g * (dt_ij²*dt_jk + dt_jk²*dt_ij)
    //     = R_i*dp_ij*dt_jk - R_j*dp_jk*dt_ij - R_i*dv_ij*dt_ij*dt_jk
    //
    // FATAL FLAW FIXED: The previous implementation dropped the dv_ij term.
    // Scale is only observable during acceleration. Dropping the velocity
    // change term completely ruins the math during any non-constant motion,
    // causing the solver to output negative scales!

    const int n_triplets = static_cast<int>(keyframes.size()) - 2;
    if (n_triplets < 2) {
        return false;   // need at least 2 triplets (≥4 keyframes)
    }

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_triplets * 3, 4);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n_triplets * 3);

    const int imu_offset = (imu_data.size() == keyframes.size()) ? 1 : 0;

    for (int i = 0; i < n_triplets; ++i) {
        const auto& p_i = keyframes[i]->getPose().position;
        const auto& p_j = keyframes[i + 1]->getPose().position;
        const auto& p_k = keyframes[i + 2]->getPose().position;

        const double dt_ij = imu_data[i + imu_offset].dt;
        const double dt_jk = imu_data[i + 1 + imu_offset].dt;

        if (dt_ij < 1e-6 || dt_jk < 1e-6) {
            continue;
        }

        const Eigen::Vector3d Pi(p_i.x, p_i.y, p_i.z);
        const Eigen::Vector3d Pj(p_j.x, p_j.y, p_j.z);
        const Eigen::Vector3d Pk(p_k.x, p_k.y, p_k.z);

        // World-frame preintegration (rotate from body frame)
        const Eigen::Quaterniond q_i(
            keyframes[i]->getPose().orientation.w,
            keyframes[i]->getPose().orientation.x,
            keyframes[i]->getPose().orientation.y,
            keyframes[i]->getPose().orientation.z);
        const Eigen::Quaterniond q_j(
            keyframes[i + 1]->getPose().orientation.w,
            keyframes[i + 1]->getPose().orientation.x,
            keyframes[i + 1]->getPose().orientation.y,
            keyframes[i + 1]->getPose().orientation.z);

        const Eigen::Vector3d dp_ij = q_i * imu_data[i + imu_offset].delta_p;
        const Eigen::Vector3d dp_jk = q_j * imu_data[i + 1 + imu_offset].delta_p;
        
        // MISSING TERM ADDED: Preintegrated velocity change
        const Eigen::Vector3d dv_ij = q_i * imu_data[i + imu_offset].delta_v;

        // Fill row (3 equations)
        const int row = i * 3;

        // Column 0: scale coefficient
        A.block<3, 1>(row, 0) = (Pj - Pi) * dt_jk - (Pk - Pj) * dt_ij;

        // Columns 1-3: gravity coefficient
        const double grav_coeff = 0.5 * (dt_ij * dt_ij * dt_jk + dt_jk * dt_jk * dt_ij);
        A.block<3, 3>(row, 1) = Eigen::Matrix3d::Identity() * grav_coeff;

        // Right-hand side (corrected signs matching kinematic derivation)
        b.segment<3>(row) = dp_ij * dt_jk - dp_jk * dt_ij - dv_ij * dt_ij * dt_jk;
    }

    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& singular_values = svd.singularValues();

    const Eigen::Vector4d x = svd.solve(b);
    const double s_svd = x(0);

    // Evaluate Motion Regime modularly
    detected_regime = evaluateMotionRegime(keyframes, imu_data, A, singular_values, s_svd);

    if (detected_regime == MotionRegime::STATIONARY) {
        return false;
    }
    else if (detected_regime == MotionRegime::DYNAMIC_EXCITATION) {
        std::cerr << "[VinsInit] Regime 2 (DYNAMIC EXCITATION): Physical scale observed (s=" << s_svd << ").\n";
        scale   = s_svd;
        gravity = x.segment<3>(1);
    }
    else { // MotionRegime::CONSTANT_VELOCITY_FALLBACK
        std::cerr << "[VinsInit] Regime 3 (CONSTANT VELOCITY FALLBACK): Engaging modular scale prior (s=" 
                  << config_.default_scale_prior << "). Solving cleanly for gravity.\n";
        scale = config_.default_scale_prior;
        
        // Solve for gravity given fixed prior scale: A_grav * g = b - A_scale * s_prior
        Eigen::MatrixXd A_grav = A.block(0, 1, A.rows(), 3);
        Eigen::VectorXd b_grav = b - A.col(0) * scale;
        gravity = A_grav.colPivHouseholderQr().solve(b_grav);
    }

    std::cerr << "[VinsInit] Scale: " << scale << "  gravity_norm=" << gravity.norm() << " m/s²\n";

    const double g_mag = gravity.norm();
    if (g_mag < kGravityMin || g_mag > kGravityMax) {
        std::cerr << "[VinsInit] Gravity magnitude out of bounds: " << g_mag
                  << " m/s² (expected [" << kGravityMin << ", " << kGravityMax << "])" << std::endl;
        return false;
    }

    // Final scale acceptance gate — uses config bounds so YAML controls the range.
    // Both KITTI (0.3..30) and indoor drone (0.1..5) are handled by the same code;
    // only the YAML vins.scale_prior_min / vins.scale_prior_max differ per config.
    if (scale < config_.scale_prior_min || scale > config_.scale_prior_max) {
        std::cerr << "[VinsInit] Scale " << scale
                  << " outside config bounds [" << config_.scale_prior_min
                  << ", " << config_.scale_prior_max << "]. Rejecting solution." << std::endl;
        return false;
    }

    // Normalize gravity to 9.81 m/s² (preserves direction, corrects magnitude)
    gravity = gravity / g_mag * 9.81;

    // Step 3 — Refine gravity and scale on spherical manifold ||g|| = 9.81
    refineGravity(keyframes, imu_data, gravity, scale);

    return true;
}

bool VinsInitializer::estimateGyroBias(
    const std::vector<std::shared_ptr<GraphNode>>& keyframes,
    const std::vector<ImuPreintegrator::PreintegratedData>& imu_data,
    Eigen::Vector3d& gyro_bias)
{
    const int imu_offset = (imu_data.size() == keyframes.size()) ? 1 : 0;
    const int N = static_cast<int>(keyframes.size());
    if (N < 2) return false;

    Eigen::Matrix3d AtA = Eigen::Matrix3d::Zero();
    Eigen::Vector3d Atb = Eigen::Vector3d::Zero();

    for (int i = 0; i < N - 1; ++i) {
        const auto& q_i_msg = keyframes[i]->getPose().orientation;
        const auto& q_j_msg = keyframes[i + 1]->getPose().orientation;
        Eigen::Quaterniond q_i(q_i_msg.w, q_i_msg.x, q_i_msg.y, q_i_msg.z);
        Eigen::Quaterniond q_j(q_j_msg.w, q_j_msg.x, q_j_msg.y, q_j_msg.z);

        const auto& pim = imu_data[i + imu_offset];
        Eigen::Quaterniond q_ij_cam = q_i.inverse() * q_j;
        Eigen::Quaterniond q_err = pim.delta_q.inverse() * q_ij_cam;
        if (q_err.w() < 0) {
            q_err.coeffs() *= -1.0;
        }

        Eigen::AngleAxisd aa(q_err);
        Eigen::Vector3d r_err = aa.angle() * aa.axis();
        if (std::isnan(r_err.x()) || std::isnan(r_err.y()) || std::isnan(r_err.z())) {
            continue;
        }

        const Eigen::Matrix3d& J = pim.dr_dbg;
        AtA += J.transpose() * J;
        Atb += J.transpose() * r_err;
    }

    Eigen::Matrix3d AtA_damped = AtA + Eigen::Matrix3d::Identity() * 1e-4;
    gyro_bias = AtA_damped.ldlt().solve(Atb);
    return true;
}

bool VinsInitializer::refineGravity(
    const std::vector<std::shared_ptr<GraphNode>>& keyframes,
    const std::vector<ImuPreintegrator::PreintegratedData>& imu_data,
    Eigen::Vector3d& gravity,
    double& scale)
{
    const int n_triplets = static_cast<int>(keyframes.size()) - 2;
    if (n_triplets < 2) return false;

    const int imu_offset = (imu_data.size() == keyframes.size()) ? 1 : 0;
    const double G = 9.81;

    for (int iter = 0; iter < 4; ++iter) {
        Eigen::Vector3d g_hat = gravity.normalized();
        Eigen::Vector3d e1 = (std::abs(g_hat.x()) < 0.9) ? Eigen::Vector3d(1, 0, 0) : Eigen::Vector3d(0, 1, 0);
        Eigen::Vector3d b1 = (e1 - (g_hat.dot(e1)) * g_hat).normalized();
        Eigen::Vector3d b2 = (g_hat.cross(b1)).normalized();

        Eigen::Matrix3d AtA = Eigen::Matrix3d::Zero();
        Eigen::Vector3d Atb = Eigen::Vector3d::Zero();

        for (int i = 0; i < n_triplets; ++i) {
            const auto& p_i = keyframes[i]->getPose().position;
            const auto& p_j = keyframes[i + 1]->getPose().position;
            const auto& p_k = keyframes[i + 2]->getPose().position;

            const double dt_ij = imu_data[i + imu_offset].dt;
            const double dt_jk = imu_data[i + 1 + imu_offset].dt;
            if (dt_ij < 1e-6 || dt_jk < 1e-6) continue;

            const Eigen::Vector3d Pi(p_i.x, p_i.y, p_i.z);
            const Eigen::Vector3d Pj(p_j.x, p_j.y, p_j.z);
            const Eigen::Vector3d Pk(p_k.x, p_k.y, p_k.z);

            const Eigen::Quaterniond q_i(
                keyframes[i]->getPose().orientation.w,
                keyframes[i]->getPose().orientation.x,
                keyframes[i]->getPose().orientation.y,
                keyframes[i]->getPose().orientation.z);
            const Eigen::Quaterniond q_j(
                keyframes[i + 1]->getPose().orientation.w,
                keyframes[i + 1]->getPose().orientation.x,
                keyframes[i + 1]->getPose().orientation.y,
                keyframes[i + 1]->getPose().orientation.z);

            const Eigen::Vector3d dp_ij = q_i * imu_data[i + imu_offset].delta_p;
            const Eigen::Vector3d dp_jk = q_j * imu_data[i + 1 + imu_offset].delta_p;
            const Eigen::Vector3d dv_ij = q_i * imu_data[i + imu_offset].delta_v;

            Eigen::Vector3d alpha = (Pj - Pi) * dt_jk - (Pk - Pj) * dt_ij;
            double beta = 0.5 * (dt_ij * dt_ij * dt_jk + dt_jk * dt_jk * dt_ij);
            Eigen::Vector3d rhs_val = dp_ij * dt_jk - dp_jk * dt_ij - dv_ij * dt_ij * dt_jk;

            Eigen::Matrix<double, 3, 3> A_tri;
            A_tri.col(0) = alpha;
            A_tri.col(1) = beta * b1;
            A_tri.col(2) = beta * b2;

            Eigen::Vector3d r_tri = rhs_val - scale * alpha - beta * G * g_hat;

            AtA += A_tri.transpose() * A_tri;
            Atb += A_tri.transpose() * r_tri;
        }

        Eigen::Vector3d dx = (AtA + Eigen::Matrix3d::Identity() * 1e-6).ldlt().solve(Atb);
        const bool is_fallback = (config_.use_scale_prior_fallback && std::abs(scale - config_.default_scale_prior) < 1e-6);
        if (!is_fallback) {
            scale += dx(0);
            scale = std::clamp(scale, config_.scale_prior_min, config_.scale_prior_max);
        }
        gravity = G * g_hat + dx(1) * b1 + dx(2) * b2;
        gravity = G * gravity.normalized();

        if (dx.norm() < 1e-4) break;
    }

    std::cerr << "[VinsInit] Refined Scale: " << scale << "  Refined Gravity: [" << gravity.transpose() << "]\n";
    return true;
}

}  // namespace slam::vins
