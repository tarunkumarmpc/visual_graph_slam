#include "visual_graph_slam/modules/imu_module.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace slam::modules {

ImuModule::ImuModule()
{
}

void ImuModule::initialize(rclcpp::Node::SharedPtr node, const std::string& name, slam::core::GraphInterface* graph)
{
    node_ = node;
    graph_ = graph;
    
    imu_preintegrator_ = std::make_unique<slam::vins::ImuPreintegrator>();
    vins_initializer_ = std::make_unique<slam::vins::VinsInitializer>();

    // Parameter names must match the YAML keys under graph_slam.ros__parameters.
                // All params are declared on the shared graph_slam node, so use get_parameter_or.
    // These params are loaded from YAML by the ROS2 parameter server automatically.
    auto get_double = [&](const std::string& key, double def) -> double {
        double v = def;
        node_->get_parameter_or(key, v, def);
        return v;
    };
    auto get_bool = [&](const std::string& key, bool def) -> bool {
        bool v = def;
        node_->get_parameter_or(key, v, def);
        return v;
    };

    const double acc_n = get_double("imu.accel_noise",     0.02);
    const double gyr_n = get_double("imu.gyro_noise",      0.0017);
    const double acc_w = get_double("imu.accel_bias_walk", 0.0002);
    const double gyr_w = get_double("imu.gyro_bias_walk",  0.00002);

    imu_preintegrator_->setNoiseParams(acc_n, gyr_n, acc_w, gyr_w);

    // Load Modular VINS parameters — also fixed namespace (was "imu.vins.*")
    slam::vins::VinsInitializer::VinsConfig vins_config;
    vins_config.stationary_velocity_threshold = get_double("vins.stationary_velocity_threshold", 0.10);
    vins_config.use_scale_prior_fallback      = get_bool("vins.use_scale_prior_fallback", true);
    vins_config.default_scale_prior           = get_double("vins.default_scale_prior", 1.0);
    // Ensure observable scale gate matches our tightened bounds or reads from YAML config
    vins_config.scale_prior_min = get_double("vins.scale_prior_min", 0.3);
    vins_config.scale_prior_max = get_double("vins.scale_prior_max", 30.0);
    vins_initializer_->setConfig(vins_config);

    // DESIGN-FLAW-FIX: read vins_min_keyframes from YAML instead of hardcoding 15 later.
    // Stored in member so onKeyframeCreated() can use it.
    // NOTE: must use get_parameter_or<int> — YAML stores this as an integer literal,
    // and ROS2 throws InvalidParameterTypeException if you request double for an int param.
    {
        int v = 15;
        node_->get_parameter_or("vins.min_keyframes_for_init", v, v);
        vins_min_keyframes_for_init_ = v;
    }

    RCLCPP_INFO(logger_,
        "IMU Module Initialized. acc_n=%.5f gyr_n=%.5f acc_w=%.6f gyr_w=%.7f "
        "vins_fallback=%s scale_prior=%.3f vins_init_kfs=%d",
        acc_n, gyr_n, acc_w, gyr_w,
        vins_config.use_scale_prior_fallback ? "YES" : "NO",
        vins_config.default_scale_prior,
        vins_min_keyframes_for_init_);
}

void ImuModule::preProcess(const slam::core::SensorFrame& frame)
{
    if (frame.imu_measurements.empty()) {
        wants_keyframe_cache_ = false;
        return;
    }

    for (const auto& msg : frame.imu_measurements) {
        imu_preintegrator_->update(msg);
        raw_imu_msgs_.push_back(msg);
    }
    pending_preintegration_ = imu_preintegrator_->getPreintegratedData();
    
    if (last_keyframe_time_.nanoseconds() == 0) {
        last_keyframe_time_ = frame.stamp;
    }

    // Voting Logic for Keyframe
    wants_keyframe_cache_ = false;
    
    // 1. Time heuristic (bound drift)
    double dt = (frame.stamp - last_keyframe_time_).seconds();
    if (dt > 0.5) {
        wants_keyframe_cache_ = true;
    }

    // 2. Rotation heuristic (catch fast spins)
    if (pending_preintegration_) {
        Eigen::AngleAxisd aa(pending_preintegration_->delta_q);
        double angle_deg = aa.angle() * 180.0 / M_PI;
        // TURN-FIX: Was hardcoded to 15.0, ignoring the camera's 7.0 setting.
        // During a fast spin, if camera drops frames, IMU must trigger KFs.
        if (angle_deg > 7.0) {
            wants_keyframe_cache_ = true;
        }
    }
}

bool ImuModule::wantsKeyframe() const
{
    return wants_keyframe_cache_;
}

void ImuModule::onKeyframeCreated(std::shared_ptr<slam::GraphNode> node, slam::core::GraphInterface* graph)
{
    last_keyframe_time_ = node->getTimestamp(); // Reset time anchor

    if (!pending_preintegration_) {
        return; // No IMU data to add
    }

    const int kf_id = node->getId();

    // VINS Initialization logic vs Active VIO mode
    if (!vins_initialized_) {
        initialization_window_.push_back(node);
        if (initialization_window_.size() > 1) {
            imu_preintegrated_keyframes_.push_back(*pending_preintegration_);
            buffered_raw_imu_samples_.push_back(raw_imu_msgs_);
        }

        // DESIGN-FLAW-FIX: use vins_min_keyframes_for_init_ loaded from YAML (was hardcoded 15,
        // silently ignoring the vins.min_keyframes_for_init YAML param set to 20).
        if (static_cast<int>(initialization_window_.size()) >= vins_min_keyframes_for_init_ &&
            imu_preintegrated_keyframes_.size() == initialization_window_.size() - 1) {
            auto init_result = vins_initializer_->align(initialization_window_, imu_preintegrated_keyframes_);
            
            // Require realistic metric scale [0.3, 30.0] before locking in initialization.
            // If the vehicle is in steady-state straight driving, scale is unobservable and can solve
            // to a degenerate value (~0.0014). We slide the window until genuine acceleration is observed.
            const bool scale_realistic = (init_result.scale >= 0.3 && init_result.scale <= 30.0);
            if (init_result.success && scale_realistic) {
                vins_initialized_ = true;
                estimated_scale_ = init_result.scale;
                estimated_gravity_ = init_result.gravity;
                
                RCLCPP_INFO(logger_, "VINS Initialization SUCCESS! Scale: %.3f, Gravity: [%.2f, %.2f, %.2f]",
                            estimated_scale_, estimated_gravity_.x(), estimated_gravity_.y(), estimated_gravity_.z());

                // Compute alignment rotation
                Eigen::Vector3d g_est = init_result.gravity;
                Eigen::Vector3d g_true(0, 0, -9.81);
                Eigen::Quaterniond q_align = Eigen::Quaterniond::FromTwoVectors(g_est, g_true);
                tf2::Quaternion tf_q_align(q_align.x(), q_align.y(), q_align.z(), q_align.w());

                // Update Biases and Velocities for window
                for (size_t i = 0; i < initialization_window_.size(); ++i) {
                    if (i < init_result.velocities.size()) {
                        Eigen::Vector3d v_aligned = q_align * init_result.velocities[i];
                        initialization_window_[i]->updateVelocity(v_aligned);
                    }
                    initialization_window_[i]->updateBiases(init_result.accel_bias, init_result.gyro_bias);
                }

                // Retroactively submit all buffered IMU factors for the initialized window.
                // We withheld them during bootstrap so uncalibrated/unscaled IMU factors wouldn't
                // distort unscaled visual odometry edges. Now that scale & biases are solved, we add them.
                for (size_t i = 0; i + 1 < initialization_window_.size(); ++i) {
                    slam::MeasurementEdgeConfig imu_edge;
                    imu_edge.source = slam::MeasurementSource::IMU_PREINTEGRATION;
                    imu_edge.from_keyframe_id = initialization_window_[i]->getId();
                    imu_edge.to_keyframe_id = initialization_window_[i + 1]->getId();
                    imu_edge.imu_data = imu_preintegrated_keyframes_[i];
                    imu_edge.raw_imu_samples = buffered_raw_imu_samples_[i];
                    graph->submitMeasurement(imu_edge);
                }

                // Clear buffers now that they have been submitted to the graph
                initialization_window_.clear();
                imu_preintegrated_keyframes_.clear();
                buffered_raw_imu_samples_.clear();

                // Queue Global Alignment request for the Coordinator
                slam::core::SensorModule::GlobalAlignmentRequest req;
                req.scale = estimated_scale_;
                req.q_align = tf_q_align;
                pending_alignment_request_ = req;
            } else {
                RCLCPP_INFO_THROTTLE(logger_, *node_->get_clock(), 1000,
                    "VINS init pending/unobservable (success=%s, scale=%.4f). Sliding window...",
                    init_result.success ? "true" : "false", init_result.scale);
                // If failed or scale unobservable, pop the oldest to keep sliding window active
                initialization_window_.erase(initialization_window_.begin());
                imu_preintegrated_keyframes_.erase(imu_preintegrated_keyframes_.begin());
                buffered_raw_imu_samples_.erase(buffered_raw_imu_samples_.begin());
            }
        }
    } else if (last_keyframe_id_ >= 0) {
        // Active VIO mode (post-initialization): submit IMU factors live as keyframes are created
        slam::MeasurementEdgeConfig imu_edge;
        imu_edge.source = slam::MeasurementSource::IMU_PREINTEGRATION;
        imu_edge.from_keyframe_id = last_keyframe_id_;
        imu_edge.to_keyframe_id = kf_id;
        imu_edge.imu_data = pending_preintegration_;
        imu_edge.raw_imu_samples = raw_imu_msgs_;
        graph->submitMeasurement(imu_edge);
    }

    // Reset preintegrator for the next segment
    imu_preintegrator_->reset();
    pending_preintegration_ = std::nullopt;
    raw_imu_msgs_.clear();

    // ARCH-2 FIX: advance the from-ID anchor to this keyframe.
    last_keyframe_id_ = kf_id;
}

std::optional<slam::core::SensorModule::GlobalAlignmentRequest> ImuModule::wantsGlobalAlignment() const
{
    // DESIGN-FLAW-FIX: consume-on-read so that if the Coordinator queries this on two
    // consecutive keyframes (e.g. if onGlobalAlignment was never called due to an error
    // in processGlobalAlignment), the scale is not applied twice.
    auto req = pending_alignment_request_;
    pending_alignment_request_ = std::nullopt;   // clear regardless of whether coordinator consumes it
    return req;
}

void ImuModule::onGlobalAlignment(double scale, const tf2::Quaternion& q_align)
{
    // IMU doesn't hold spatial state like Camera does, so nothing to align locally.
    // The graph alignment is handled by the Coordinator.
    
    // Clear the pending request if it was us who requested it
    pending_alignment_request_ = std::nullopt;
}

} // namespace slam::modules
