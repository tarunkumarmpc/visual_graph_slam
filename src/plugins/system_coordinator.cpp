#include "visual_graph_slam/plugins/system_coordinator.hpp"
#include "visual_graph_slam/modules/camera_module.hpp"
#include "visual_graph_slam/modules/imu_module.hpp"
#include "visual_graph_slam/telemetry/slam_telemetry.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace slam::plugins {

void SystemCoordinator::initialize(rclcpp::Node::SharedPtr node, 
                                   const std::string& plugin_name,
                                   std::shared_ptr<slam::core::GraphInterface> graph)
{
    node_ = node;
    logger_ = node_->get_logger();
    graph_ = graph;

    // We can read a param to decide which modules to load.
    // For now, we will read the global "mode" parameter.
    std::string mode = "mono"; // Default
    node_->get_parameter_or("mode", mode, mode);

    RCLCPP_INFO(logger_, "Initializing SystemCoordinator in mode: %s", mode.c_str());

    // 1. Camera Module is always required in Visual Graph SLAM
    auto camera_module = std::make_shared<slam::modules::CameraModule>();
    camera_module->initialize(node_, "camera", graph_.get());
    modules_.push_back(camera_module);

    // 2. Load IMU Module if requested
    if (mode == "mono_imu" || mode == "stereo_imu") {
        auto imu_module = std::make_shared<slam::modules::ImuModule>();
        imu_module->initialize(node_, "imu", graph_.get());
        modules_.push_back(imu_module);
    }

    if (!node_->get_parameter("local_mapping_window_size", local_mapping_window_size_)) {
        local_mapping_window_size_ = 20; // Default fallback
    }
}

void SystemCoordinator::processSensorFrame(const slam::core::SensorFrame& frame)
{
    if (!graph_) return;

    // 1. PreProcess all modules
    for (auto& module : modules_) {
        module->preProcess(frame);
    }

    // 2. Gather Votes
    bool make_keyframe = false;
    for (const auto& module : modules_) {
        if (module->wantsKeyframe()) {
            make_keyframe = true;
            break;
        }
    }

    // 3. Execute
    if (make_keyframe) {
        // Create the new node with an empty pose (will be filled by CameraModule if present)
        geometry_msgs::msg::Pose empty_pose;
        empty_pose.orientation.w = 1.0;
        auto new_node = std::make_shared<slam::GraphNode>(current_keyframe_id_, empty_pose, frame.stamp);
        
        // Modules will populate the visual pose, descriptors, etc during onKeyframeCreated!
        
        graph_->getGraph()->addNode(new_node);

        // Tell all modules to attach their edges to the graph
        for (auto& module : modules_) {
            module->onKeyframeCreated(new_node, graph_.get());
        }

        // 4. Check for Global Alignment Requests (e.g. from IMU VINS Init)
        bool global_alignment_ran = false;
        for (auto& module : modules_) {
            auto req = module->wantsGlobalAlignment();
            if (req) {
                processGlobalAlignment(req->scale, req->q_align);
                global_alignment_ran = true;
            }
        }

        // 5. Trigger Optimization (if global alignment already triggered global optimization, skip local)
        if (!global_alignment_ran) {
            int from_id = std::max(0, current_keyframe_id_ - local_mapping_window_size_);
            graph_->signalOptimization(false, "New keyframe added", from_id, current_keyframe_id_);
        }

        // 6. Gather Telemetry across all stages and render dashboard
        std::shared_ptr<slam::modules::CameraModule> cam_mod;
        std::shared_ptr<slam::modules::ImuModule> imu_mod;
        for (auto& mod : modules_) {
            if (auto c = std::dynamic_pointer_cast<slam::modules::CameraModule>(mod)) cam_mod = c;
            if (auto i = std::dynamic_pointer_cast<slam::modules::ImuModule>(mod)) imu_mod = i;
        }

        if (cam_mod) {
            auto snap = cam_mod->getLatestTelemetry();
            if (imu_mod) {
                snap.vins_initialized = imu_mod->isVinsInitialized();
                snap.vins_scale = imu_mod->getEstimatedScale();
                snap.vins_gravity = imu_mod->getEstimatedGravity();
                snap.vins_gyro_bias = imu_mod->getLatestGyroBias();
            }
            
            // Stage 4: Global Optimization State
            auto opt_node = graph_->getGraph()->getNode(new_node->getId());
            if (opt_node) {
                snap.global_optimized_pose = opt_node->getPose();
            } else {
                snap.global_optimized_pose = snap.local_fused_pose;
            }
            snap.active_factor_count = static_cast<int>(graph_->getGraph()->getEdges().size());
            snap.global_alignment_applied = (snap.vins_initialized && snap.vins_scale != 1.0);

            // Stage 5: Diagnostics & Drift
            double dx = snap.local_fused_pose.position.x - snap.global_optimized_pose.position.x;
            double dy = snap.local_fused_pose.position.y - snap.global_optimized_pose.position.y;
            double dz = snap.local_fused_pose.position.z - snap.global_optimized_pose.position.z;
            snap.trans_drift_m = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            tf2::Quaternion q_loc, q_glob;
            tf2::fromMsg(snap.local_fused_pose.orientation, q_loc);
            tf2::fromMsg(snap.global_optimized_pose.orientation, q_glob);
            double angle_rad = q_loc.angleShortestPath(q_glob);
            snap.rot_drift_deg = angle_rad * 180.0 / M_PI;

            slam::telemetry::TelemetryLogger::renderReport(snap, logger_);
        }

        current_keyframe_id_++;
    }
}

void SystemCoordinator::processGlobalAlignment(double scale, const tf2::Quaternion& q_align)
{
    RCLCPP_INFO(logger_, "Coordinator processing Global Alignment Request (Scale: %.2f)", scale);

    // Scale and align all existing nodes in the graph memory
    auto nodes = graph_->getGraph()->getNodes();
    for (auto& [id, node] : nodes) {
        auto pose = node->getPose();
        
        // Scale translation
        pose.position.x *= scale;
        pose.position.y *= scale;
        pose.position.z *= scale;
        
        // Align pose to gravity
        tf2::Transform t_pose;
        tf2::fromMsg(pose, t_pose);
        tf2::Transform t_aligned = tf2::Transform(q_align) * t_pose;
        tf2::toMsg(t_aligned, pose);
        
        node->updatePose(pose);

        // Align velocity to gravity
        Eigen::Vector3d v_old = node->getVelocity();
        tf2::Vector3 v_tf(v_old.x(), v_old.y(), v_old.z());
        tf2::Vector3 v_rot = tf2::quatRotate(q_align, v_tf);
        node->updateVelocity(Eigen::Vector3d(v_rot.x(), v_rot.y(), v_rot.z()));
    }

    // Scale all existing VISUAL_ODOMETRY edges
    for (auto& edge : graph_->getGraph()->getEdges()) {
        if (edge->getType() == slam::GraphEdge::Type::VISUAL_ODOMETRY) {
            auto tf = edge->getRelativeTransform();
            tf.translation.x *= scale;
            tf.translation.y *= scale;
            tf.translation.z *= scale;
            edge->setRelativeTransform(tf);
        }
    }

    // Broadcast the alignment to all modules (so Camera can scale its internal vision pose)
    for (auto& module : modules_) {
        module->onGlobalAlignment(scale, q_align);
    }

    graph_->getGraph()->setMetricScaleInitialized(true);
    graph_->signalOptimization(true, "Global Alignment Triggered");
}

void SystemCoordinator::setCameraIntrinsics(const slam::sensor::CameraIntrinsics& intrinsics)
{
    for (auto& module : modules_) {
        module->setCameraIntrinsics(intrinsics);
    }
}

} // namespace slam::plugins

PLUGINLIB_EXPORT_CLASS(slam::plugins::SystemCoordinator, slam::core::System)
