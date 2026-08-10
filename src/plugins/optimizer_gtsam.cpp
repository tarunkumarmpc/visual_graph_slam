// optimizer_gtsam.cpp
//
// GTSAM Pose-Graph Optimizer with full IMU Factor support.
//
// Key fixes:
//   1. IMU edges now use gtsam::ImuFactor (PreintegratedImuMeasurements)
//      instead of the placeholder BetweenFactor<Pose3>.
//   2. Velocity (v) and bias (b) states are properly constrained:
//      - Prior on v_0 and b_0 at the anchor node
//      - BetweenFactor<imuBias::ConstantBias> for bias random-walk continuity
//   3. Noise models derived from the 15×15 preintegrated covariance.
//   4. Falls back cleanly to BetweenFactor<Pose3> for non-IMU edges.

#if VISUAL_GRAPH_SLAM_WITH_GTSAM

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/navigation/ManifoldPreintegration.h>

#pragma GCC diagnostic pop

#endif  // VISUAL_GRAPH_SLAM_WITH_GTSAM

#include "visual_graph_slam/backend/optimizer_backend.hpp"
#include "visual_graph_slam/vins/gtsam_imu_builder.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <algorithm>
#include <iostream>

namespace slam::core {

std::string GtsamOptimizerBackend::name() const
{
#if VISUAL_GRAPH_SLAM_WITH_GTSAM
    return "gtsam";
#else
    return "gtsam-unavailable";
#endif
}

bool GtsamOptimizerBackend::optimizePoseGraph(Graph& graph, int iterations)
{
#if !VISUAL_GRAPH_SLAM_WITH_GTSAM
    (void)graph;
    (void)iterations;
    return false;

#else
    if (graph.getNodes().size() < 3 || graph.getEdges().empty()) {
        return false;
    }

    // ── Symbols ───────────────────────────────────────────────────────────────
    // x(i) = pose, v(i) = velocity, b(i) = IMU bias
    using gtsam::Symbol;
    auto X = [](uint64_t i) { return Symbol('x', i); };
    auto V = [](uint64_t i) { return Symbol('v', i); };
    auto B = [](uint64_t i) { return Symbol('b', i); };

    gtsam::NonlinearFactorGraph factor_graph;
    gtsam::Values initial;

    // ── Noise models ──────────────────────────────────────────────────────────
    // Tight anchor prior on pose, velocity, bias at node 0.
    const auto prior_pose_noise  = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());
    const auto prior_vel_noise   = gtsam::noiseModel::Isotropic::Sigma(3, 0.3);
    const auto prior_bias_noise  = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.01, 0.01, 0.01, 5e-4, 5e-4, 5e-4).finished());

    // Bias random-walk between consecutive keyframes.
    const auto bias_between_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 3e-3, 3e-3, 3e-3, 3e-4, 3e-4, 3e-4).finished());

    // Default VO-edge noise (used when covariance unavailable).
    const auto default_between_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());

    // ── Ground-vehicle planar prior ───────────────────────────────────────────
    // Cars don't fly or tilt side-to-side. We softly penalise z-deviation with σ=0.15m
    // and roll/pitch deviation with σ=0.05 rad (~2.8 deg), while leaving yaw, x, and y
    // completely unconstrained (kFreeSigma). Huber kernel handles bumps and slopes gracefully.
    constexpr double kFreeSigma = 1e6;   // effectively unconstrained
    constexpr double kZSigma    = 0.15;  // 15 cm 1-sigma for ground vehicle Z
    constexpr double kRotSigma  = 0.05;  // ~2.8 deg 1-sigma for Roll and Pitch
    const auto z_prior_base = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << kRotSigma, kRotSigma, kFreeSigma,  // rotation r,p,y
                             kFreeSigma, kFreeSigma, kZSigma       // translation x,y,z
        ).finished());
    const auto z_prior_noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(0.5), z_prior_base);

    // ── Get consistent snapshots to prevent data races ─────────────────────────
    const auto nodes = graph.getNodes();
    const auto edges = graph.getEdges();

    // ── Insert initial values for every node ──────────────────────────────────
    bool prior_added = false;
    int oldest_id = std::numeric_limits<int>::max();
    std::unordered_set<uint64_t> imu_nodes;
    
    for (const auto& [id, node] : nodes) {
        if (id < oldest_id) {
            oldest_id = id;
        }
    }
    
    for (const auto& edge : edges) {
        if (edge && edge->getType() == GraphEdge::Type::IMU_PREINTEGRATION) {
            imu_nodes.insert(static_cast<uint64_t>(edge->getFromId()));
            imu_nodes.insert(static_cast<uint64_t>(edge->getToId()));
        }
    }

    for (const auto& [id, node] : nodes) {
        if (!node) {
            continue;
        }
        const uint64_t uid = static_cast<uint64_t>(id);

        const auto& pose = node->getPose();
        const gtsam::Rot3 rot = gtsam::Rot3::Quaternion(
            pose.orientation.w, pose.orientation.x,
            pose.orientation.y, pose.orientation.z);
        const gtsam::Pose3 pose3(rot, gtsam::Point3(
            pose.position.x, pose.position.y, pose.position.z));

        initial.insert(X(uid), pose3);
        if (imu_nodes.count(uid)) {
            initial.insert(V(uid), gtsam::Velocity3(node->getVelocity()));
            initial.insert(B(uid), gtsam::imuBias::ConstantBias(
                node->getAccelBias(), node->getGyroBias()));
        }

        // Ground-vehicle planar prior: target z=0, roll=0, pitch=0 (ground plane).
        if (graph.usePlanarConstraint()) {
            gtsam::Rot3 yaw_only_rot = gtsam::Rot3::Ypr(rot.yaw(), 0.0, 0.0);
            const gtsam::Pose3 planar_ref(yaw_only_rot, gtsam::Point3(
                pose.position.x, pose.position.y, 0.0));
            factor_graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                X(uid), planar_ref, z_prior_noise);
        }

        // Prior at the anchor node (the oldest node in the window)
        if (!prior_added && id == oldest_id) {
            factor_graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                X(uid), pose3, prior_pose_noise);
            if (imu_nodes.count(uid)) {
                factor_graph.emplace_shared<gtsam::PriorFactor<gtsam::Velocity3>>(
                    V(uid), gtsam::Velocity3(node->getVelocity()), prior_vel_noise);
                factor_graph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
                    B(uid), gtsam::imuBias::ConstantBias(
                        node->getAccelBias(), node->getGyroBias()), prior_bias_noise);
            }
            prior_added = true;
        }
    }

    if (!prior_added) {
        return false;
    }

    // ── Insert factors for every edge ─────────────────────────────────────────
    for (const auto& edge : edges) {
        if (!edge) {
            continue;
        }

        const uint64_t from_uid = static_cast<uint64_t>(edge->getFromId());
        const uint64_t to_uid   = static_cast<uint64_t>(edge->getToId());

        // Check that both nodes exist in the initial set (guard against stale edges)
        if (!initial.exists(X(from_uid)) || !initial.exists(X(to_uid))) {
            continue;
        }

        if (edge->getType() == GraphEdge::Type::IMU_PREINTEGRATION) {
            // ── Real IMU Factor ───────────────────────────────────────────────
            
            // Retrieve bias at the 'from' node
            gtsam::imuBias::ConstantBias from_bias;
            if (initial.exists(B(from_uid))) {
                from_bias = initial.at<gtsam::imuBias::ConstantBias>(B(from_uid));
            }

            const double acc_mult = graph.isMetricScaleInitialized() ? 1.0 : 1000.0;
            gtsam::PreintegratedImuMeasurements pim = vins::GtsamImuBuilder::buildPIM(
                *edge, from_bias, 9.81, acc_mult);

            factor_graph.emplace_shared<gtsam::ImuFactor>(
                X(from_uid), V(from_uid),
                X(to_uid),   V(to_uid),
                B(from_uid),
                pim);

            RCLCPP_WARN(rclcpp::get_logger("optimizer_gtsam"), "Added ImuFactor between node %lu and %lu", from_uid, to_uid);

            // Bias random-walk constraint (keeps bias smoothly varying)
            factor_graph.emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
                B(from_uid), B(to_uid),
                gtsam::imuBias::ConstantBias(),
                bias_between_noise);

        } else {
            // ── Pose-only BetweenFactor (VO, loop closure, wheel odom) ────────
            const auto& tf = edge->getRelativeTransform();
            const gtsam::Rot3 rel_rot = gtsam::Rot3::Quaternion(
                tf.rotation.w, tf.rotation.x, tf.rotation.y, tf.rotation.z);
            const gtsam::Pose3 rel_pose(rel_rot, gtsam::Point3(
                tf.translation.x, tf.translation.y, tf.translation.z));

            // Convert 6×6 information matrix (pose-graph style) to noise model.
            // GTSAM wants sqrt-information (= Cholesky factor of information).
            const Eigen::Matrix<double, 6, 6> info = edge->getInformationMatrix();
            // Guard against ill-conditioned matrices.
            Eigen::Matrix<double, 6, 6> info_safe =
                info + Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
            if (edge->getType() == GraphEdge::Type::VISUAL_ODOMETRY) {
                // Anisotropic VO weighting: monocular VO has scale ambiguity along the forward
                // (depth) axis, so we already halve the forward-translation weight in the
                // MeasurementWeightCalculator. No additional manipulation needed here.
                // the previous code zeroed the rotation block (info→1e-6) whenever
                // ANY IMU node existed. This killed rotation constraints before VINS init
                // (when ImuFactor starts from zero-velocity and is itself degenerate), leaving
                // the system with NO valid rotation source for the first 15+ keyframes.
                // Both VO and IMU should agree on rotation; if they disagree the Huber kernel
                // handles the outlier gracefully.
            }
            // Wrap with Huber robust kernel: this makes the factor ignore
            // catastrophic outlier VO measurements (bad matches, degenerate configs)
            // while still weighting good measurements by their information.
            // Huber threshold k=4.0 — preserves exact 90-degree turns without clipping rotation constraints.
            const auto base_noise = gtsam::noiseModel::Gaussian::Information(info_safe);
            const auto noise = gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(4.0), base_noise);

            factor_graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(from_uid), X(to_uid), rel_pose, noise);
        }
    }

    // ── Optimize ──────────────────────────────────────────────────────────────
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations           = static_cast<unsigned int>(std::max(1, iterations));
    params.relativeErrorTol        = 1e-5;
    params.absoluteErrorTol        = 1e-5;
    params.verbosity               = gtsam::NonlinearOptimizerParams::SILENT;
    params.verbosityLM             = gtsam::LevenbergMarquardtParams::SILENT;

    gtsam::Values optimized;
    try {
        gtsam::LevenbergMarquardtOptimizer optimizer(factor_graph, initial, params);

        // Debug: check for missing keys
        for (const auto& factor : factor_graph) {
            if (factor) {
                for (const gtsam::Key& key : factor->keys()) {
                    if (!initial.exists(key)) {
                        RCLCPP_ERROR(rclcpp::get_logger("optimizer_gtsam"), "Factor requires key %lu (chr %c) which is NOT in initial values!", key, gtsam::Symbol(key).chr());
                    }
                }
            }
        }

        optimized = optimizer.optimize();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("optimizer_gtsam"), "Optimization failed: %s", e.what());
        return false;
    }

    // ── Write back optimized poses and IMU states ─────────────────────────────
    for (const auto& [id, node] : nodes) {
        if (!node) {
            continue;
        }
        const uint64_t uid = static_cast<uint64_t>(id);

        // Pose
        if (!optimized.exists(X(uid))) {
            continue;
        }
        const gtsam::Pose3 pose3 = optimized.at<gtsam::Pose3>(X(uid));
        const auto q = pose3.rotation().toQuaternion();
        const auto t = pose3.translation();

        geometry_msgs::msg::Pose optimized_pose;
        optimized_pose.position.x    = t.x();
        optimized_pose.position.y    = t.y();
        optimized_pose.position.z    = t.z();
        optimized_pose.orientation.x = q.x();
        optimized_pose.orientation.y = q.y();
        optimized_pose.orientation.z = q.z();
        optimized_pose.orientation.w = q.w();
        graph.updateNodePose(id, optimized_pose);

        // Velocity and bias (write back so next solve starts warm)
        if (optimized.exists(V(uid))) {
            const gtsam::Velocity3 vel = optimized.at<gtsam::Velocity3>(V(uid));
            node->updateVelocity(vel);
        }
        if (optimized.exists(B(uid))) {
            const auto bias = optimized.at<gtsam::imuBias::ConstantBias>(B(uid));
            node->updateBiases(bias.accelerometer(), bias.gyroscope());
        }
    }

    return true;
#endif  // VISUAL_GRAPH_SLAM_WITH_GTSAM
}

}  // namespace slam::core

PLUGINLIB_EXPORT_CLASS(slam::core::GtsamOptimizerBackend, slam::core::OptimizerBackend)
