#include "visual_graph_slam/vins/tightly_coupled_local_ba.hpp"
#include "visual_graph_slam/vins/gtsam_imu_builder.hpp"
#include <rclcpp/rclcpp.hpp>

// GTSAM
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/ProjectionFactor.h>

#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

using gtsam::symbol_shorthand::X;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::L;

namespace slam::vins {

bool TightlyCoupledLocalBA::optimizeWindow(
    Graph& graph,
    const std::vector<int>& window_ids,
    const std::unordered_set<int>& fixed_ids,
    std::vector<LocalBaLandmark>& landmarks,
    const cv::Mat& camera_matrix,
    const Eigen::Isometry3d& base_T_camera,
    double reprojection_info_scale,
    double huber_delta,
    bool enable_height_prior,
    double height_prior_value,
    double height_prior_stddev)
{
    if (window_ids.size() < 2) {
        return false;
    }

    gtsam::NonlinearFactorGraph factor_graph;
    gtsam::Values initial_values;

    // Extract Camera Calibration
    const double fx = camera_matrix.at<double>(0, 0);
    const double fy = camera_matrix.at<double>(1, 1);
    const double cx = camera_matrix.at<double>(0, 2);
    const double cy = camera_matrix.at<double>(1, 2);
    auto K = gtsam::Cal3_S2::shared_ptr(new gtsam::Cal3_S2(fx, fy, 0.0, cx, cy));

    // Extrinsics
    gtsam::Pose3 body_P_sensor(
        gtsam::Rot3(base_T_camera.linear()),
        gtsam::Point3(base_T_camera.translation()));

    auto proj_noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(huber_delta),
        gtsam::noiseModel::Isotropic::Sigma(2, 1.0 / reprojection_info_scale));

    // 1. Add Poses, Velocities, Biases to Values and Priors for Fixed Nodes
    auto pose_prior_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-6);
    auto vel_prior_noise = gtsam::noiseModel::Isotropic::Sigma(3, 1e-6);
    auto bias_prior_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-6);
    
    // Height Prior Noise Model
    gtsam::noiseModel::Diagonal::shared_ptr height_noise = nullptr;
    if (enable_height_prior) {
        Eigen::VectorXd sigmas(6);
        sigmas << 1e6, 1e6, 1e6, 1e6, 1e6, height_prior_stddev; // Only constrain Z translation
        height_noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
    }

    // 1. Add Prior Factors (for fixed keys) and Insert Initial Values
    for (int id : window_ids) {
        auto node = graph.getNode(id);
        if (!node) continue;

        const auto& pose_msg = node->getPose();
        gtsam::Pose3 pose(
            gtsam::Rot3::Quaternion(pose_msg.orientation.w, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z),
            gtsam::Point3(pose_msg.position.x, pose_msg.position.y, pose_msg.position.z));
        
        gtsam::Velocity3 vel = node->getVelocity();
        gtsam::imuBias::ConstantBias bias(node->getAccelBias(), node->getGyroBias());

        initial_values.insert(X(id), pose);
        initial_values.insert(V(id), vel);
        initial_values.insert(B(id), bias);

        if (enable_height_prior) {
            // Constrain only the Z height of each node relative to its current
            // position — NOT against the world-frame origin. The prior pins
            // z=height_prior_value while leaving X, Y, and rotation free.
            gtsam::Pose3 height_constrained_pose(
                pose.rotation(),
                gtsam::Point3(pose.translation().x(), pose.translation().y(), height_prior_value));
            factor_graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                X(id), height_constrained_pose, height_noise);
        }

        if (fixed_ids.find(id) != fixed_ids.end()) {
            factor_graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(id), pose, pose_prior_noise);
            factor_graph.emplace_shared<gtsam::PriorFactor<gtsam::Velocity3>>(V(id), vel, vel_prior_noise);
            factor_graph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(B(id), bias, bias_prior_noise);
        }
    }

    // 2. Add IMU Preintegration Edges
    for (size_t i = 0; i < window_ids.size() - 1; ++i) {
        int from_id = window_ids[i];
        int to_id = window_ids[i+1];

        if (!initial_values.exists(X(from_id)) || !initial_values.exists(X(to_id))) {
            continue;
        }

        // Find IMU edge and/or Visual Odometry relative pose edge between from_id and to_id
        bool found_edge = false;
        bool found_vo_edge = false;
        for (const auto& edge : graph.getEdges()) {
            if (edge->getFromId() == from_id && edge->getToId() == to_id) {
                if (edge->getType() == GraphEdge::Type::IMU_PREINTEGRATION && !found_edge) {
                    found_edge = true;
                    const auto& pim_data = edge->getImuPreintegration();
                    gtsam::imuBias::ConstantBias from_bias;
                    if (initial_values.exists(B(from_id))) {
                        from_bias = initial_values.at<gtsam::imuBias::ConstantBias>(B(from_id));
                    }

                    const double acc_mult = graph.isMetricScaleInitialized() ? 1.0 : 1000.0;
                    gtsam::PreintegratedImuMeasurements pim = GtsamImuBuilder::buildPIM(
                        *edge, from_bias, 9.81, acc_mult);

                    factor_graph.emplace_shared<gtsam::ImuFactor>(X(from_id), V(from_id), X(to_id), V(to_id), B(from_id), pim);

                    // Bias random walk factor to constrain B(to_id)
                    auto bias_rw_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-4);
                    factor_graph.emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
                        B(from_id), B(to_id), gtsam::imuBias::ConstantBias(), bias_rw_noise);
                }
                else if (edge->getType() == GraphEdge::Type::VISUAL_ODOMETRY && !found_vo_edge) {
                    found_vo_edge = true;
                    const auto& tf = edge->getRelativeTransform();
                    const gtsam::Rot3 rel_rot = gtsam::Rot3::Quaternion(
                        tf.rotation.w, tf.rotation.x, tf.rotation.y, tf.rotation.z);
                    const gtsam::Pose3 rel_pose(rel_rot, gtsam::Point3(
                        tf.translation.x, tf.translation.y, tf.translation.z));

                    const Eigen::Matrix<double, 6, 6> info = edge->getInformationMatrix();
                    Eigen::Matrix<double, 6, 6> info_safe =
                        info + Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
                    if (found_edge) {
                        // When ImuFactor is active between from_id and to_id, the gyroscope provides
                        // superior short-term orientation accuracy without optical flow motion blur during 90-degree turns.
                        // We downweight the visual rotation block by 10x relative to translation so the IMU
                        // orientation dominates local rotation while VO anchors translation and baseline geometry.
                        info_safe.block<3, 3>(0, 0) *= 0.1;
                    }
                    const auto base_noise = gtsam::noiseModel::Gaussian::Information(info_safe);
                    const auto noise = gtsam::noiseModel::Robust::Create(
                        gtsam::noiseModel::mEstimator::Huber::Create(huber_delta), base_noise);

                    factor_graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                        X(from_id), X(to_id), rel_pose, noise);
                }
            }
        }
        
        // If IMU edge is missing for this segment, V(to_id) and B(to_id) have no ImuFactor connecting them.
        // We MUST pin them with a loose prior so GTSAM does not throw variable ordering / disconnected graph errors.
        if (!found_edge) {
            const double kLooseSigma = 1.0;
            auto loose_vel_noise  = gtsam::noiseModel::Isotropic::Sigma(3, kLooseSigma);
            auto loose_bias_noise = gtsam::noiseModel::Isotropic::Sigma(6, kLooseSigma);

            if (initial_values.exists(V(to_id))) {
                factor_graph.emplace_shared<gtsam::PriorFactor<gtsam::Velocity3>>(
                    V(to_id), initial_values.at<gtsam::Velocity3>(V(to_id)), loose_vel_noise);
            }
            if (initial_values.exists(B(to_id))) {
                factor_graph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
                    B(to_id), initial_values.at<gtsam::imuBias::ConstantBias>(B(to_id)), loose_bias_noise);
            }
        }

        // If both IMU and VO pose edges are missing for this segment, X(to_id) is unconstrained by odometry.
        // Pin X(to_id) with a loose prior so projection factors alone don't cause degenerate gauge drift.
        if (!found_edge && !found_vo_edge) {
            const double kLooseSigma = 1.0;  // 1.0 m / 1.0 rad — very loose
            auto loose_pose_noise  = gtsam::noiseModel::Isotropic::Sigma(6, kLooseSigma);

            if (initial_values.exists(X(to_id))) {
                factor_graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                    X(to_id), initial_values.at<gtsam::Pose3>(X(to_id)), loose_pose_noise);
            }
        }
    }

    // 3. Add Visual Landmarks and Projection Factors
    for (auto& lm : landmarks) {
        bool added_factor = false;
        for (const auto& obs : lm.observations) {
            if (!initial_values.exists(X(obs.keyframe_id))) continue;
            
            factor_graph.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>>(
                obs.pixel, proj_noise, X(obs.keyframe_id), L(lm.id), K, body_P_sensor);
            added_factor = true;
        }

        if (added_factor) {
            initial_values.insert(L(lm.id), gtsam::Point3(lm.position));
        }
    }

    // 4. Optimize
    try {
        gtsam::LevenbergMarquardtParams params;
        params.setMaxIterations(10);
        gtsam::LevenbergMarquardtOptimizer optimizer(factor_graph, initial_values, params);
        gtsam::Values result = optimizer.optimize();

        // 5. Write back results
        for (int id : window_ids) {
            if (!result.exists(X(id))) continue;

            auto node = graph.getNode(id);
            if (!node) continue;

            auto pose = result.at<gtsam::Pose3>(X(id));
            auto vel = result.at<gtsam::Velocity3>(V(id));
            auto bias = result.at<gtsam::imuBias::ConstantBias>(B(id));

            geometry_msgs::msg::Pose pose_msg;
            pose_msg.position.x = pose.translation().x();
            pose_msg.position.y = pose.translation().y();
            pose_msg.position.z = pose.translation().z();
            auto q = pose.rotation().toQuaternion();
            pose_msg.orientation.w = q.w();
            pose_msg.orientation.x = q.x();
            pose_msg.orientation.y = q.y();
            pose_msg.orientation.z = q.z();

            node->updatePose(pose_msg);
            node->updateVelocity(vel);
            node->updateBiases(bias.accelerometer(), bias.gyroscope());
        }

        for (auto& lm : landmarks) {
            if (result.exists(L(lm.id))) {
                lm.position = result.at<gtsam::Point3>(L(lm.id));
            }
        }

        return true;

    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("TightlyCoupledLocalBA"), "GTSAM Optimization failed: %s", e.what());
        return false;
    }
}

} // namespace slam::vins
