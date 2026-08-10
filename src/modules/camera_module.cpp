#include "visual_graph_slam/modules/camera_module.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <opencv2/calib3d.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>

namespace slam::modules {

CameraModule::CameraModule()
    : frontend_loader_("visual_graph_slam", "slam::core::Frontend"),
      loop_closure_loader_("visual_graph_slam", "slam::core::LoopClosure") 
{
    camera_matrix_ = (cv::Mat_<double>(3, 3) << 525.0, 0.0, 320.0, 0.0, 525.0, 240.0, 0.0, 0.0, 1.0);
}

void CameraModule::initialize(rclcpp::Node::SharedPtr node, const std::string& name, slam::core::GraphInterface* graph)
{
    node_ = node;
    graph_ = graph;
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    std::string frontend_plugin = node_->declare_parameter<std::string>("frontend_plugin", "slam::plugins::FrontendOrb");
    std::string loop_plugin = node_->declare_parameter<std::string>("loop_closure_plugin", "slam::plugins::LoopClosureDBoW2");
    node_->get_parameter_or("mode", mode_, mode_);

    weak_tracking_reject_threshold_ = node_->declare_parameter<int>("vision.weak_tracking_reject_threshold", weak_tracking_reject_threshold_);
    lost_tracking_reject_threshold_ = node_->declare_parameter<int>("vision.lost_tracking_reject_threshold", lost_tracking_reject_threshold_);
    weak_to_ok_min_good_frames_ = node_->declare_parameter<int>("vision.weak_to_ok_min_good_frames", weak_to_ok_min_good_frames_);
    node_->get_parameter_or("keyframe_distance_threshold", keyframe_distance_threshold_, keyframe_distance_threshold_);
    keyframe_rotation_threshold_deg_ = node_->declare_parameter<double>("vision.keyframe_rotation_threshold_deg", keyframe_rotation_threshold_deg_);
    keyframe_time_threshold_sec_ = node_->declare_parameter<double>("vision.keyframe_time_threshold_sec", keyframe_time_threshold_sec_);
    
    pnp_keyframe_window_ = node_->declare_parameter<int>("pnp.keyframe_window", pnp_keyframe_window_);
    pnp_min_inliers_ = node_->declare_parameter<int>("pnp.min_inliers", pnp_min_inliers_);
    pnp_min_inlier_ratio_ = node_->declare_parameter<double>("pnp.min_inlier_ratio", pnp_min_inlier_ratio_);
    pnp_max_rotation_deg_ = node_->declare_parameter<double>("pnp.max_rotation_deg", pnp_max_rotation_deg_);
    node_->get_parameter_or("mono.planar_motion_constraint", planar_motion_constraint_, planar_motion_constraint_);

    node_->get_parameter_or("base_frame_id", base_frame_id_, base_frame_id_);
    node_->get_parameter_or("camera_frame_id", camera_frame_id_, camera_frame_id_);

    double fx = camera_matrix_.at<double>(0, 0);
    double fy = camera_matrix_.at<double>(1, 1);
    double cx = camera_matrix_.at<double>(0, 2);
    double cy = camera_matrix_.at<double>(1, 2);
    
    node_->get_parameter_or("camera.fx", fx, fx);
    node_->get_parameter_or("camera.fy", fy, fy);
    node_->get_parameter_or("camera.cx", cx, cx);
    node_->get_parameter_or("camera.cy", cy, cy);
    camera_matrix_.at<double>(0, 0) = fx;
    camera_matrix_.at<double>(1, 1) = fy;
    camera_matrix_.at<double>(0, 2) = cx;
    camera_matrix_.at<double>(1, 2) = cy;

    vision_integrated_pose_.orientation.x = 0.0;
    vision_integrated_pose_.orientation.y = 0.0;
    vision_integrated_pose_.orientation.z = 0.0;
    vision_integrated_pose_.orientation.w = 1.0;

    try {
        frontend_ = frontend_loader_.createSharedInstance(frontend_plugin);
        frontend_->initialize(node_, frontend_plugin);
        RCLCPP_INFO(logger_, "Loaded Frontend: %s", frontend_plugin.c_str());
    } catch (const pluginlib::PluginlibException& ex) {
        RCLCPP_ERROR(logger_, "Frontend plugin failed to load: %s", ex.what());
    }

    try {
        loop_closure_ = loop_closure_loader_.createSharedInstance(loop_plugin);
        loop_closure_->initialize(node_, loop_plugin);
        RCLCPP_INFO(logger_, "Loaded LoopClosure: %s", loop_plugin.c_str());
    } catch (const pluginlib::PluginlibException& ex) {
        RCLCPP_ERROR(logger_, "LoopClosure plugin failed to load: %s", ex.what());
    }
}

void CameraModule::setCameraIntrinsics(const slam::sensor::CameraIntrinsics& intrinsics)
{
    camera_matrix_.at<double>(0, 0) = intrinsics.fx;
    camera_matrix_.at<double>(1, 1) = intrinsics.fy;
    camera_matrix_.at<double>(0, 2) = intrinsics.cx;
    camera_matrix_.at<double>(1, 2) = intrinsics.cy;

    if (frontend_) {
        frontend_->setCameraMatrix(camera_matrix_);
        RCLCPP_INFO(logger_, "Camera calibration dynamically updated. Frontend matrix updated.");
    }
}

void CameraModule::preProcess(const slam::core::SensorFrame& frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frame.image.empty() || !frontend_) {
        latest_vo_result_ = std::nullopt;
        wants_keyframe_cache_ = false;
        return;
    }
    
    latest_frame_ = frame;
    
    // Stage 1: Input State (Raw Sensors & Timestamps A to Z)
    latest_telemetry_.t_camera_s = frame.stamp.seconds();
    latest_telemetry_.imu_packet_count = static_cast<int>(frame.imu_measurements.size());
    if (!frame.imu_measurements.empty()) {
        const auto& last_imu = frame.imu_measurements.back();
        latest_telemetry_.t_imu_s = last_imu.header.stamp.sec + last_imu.header.stamp.nanosec * 1e-9;
        latest_telemetry_.latency_ms = (latest_telemetry_.t_camera_s - latest_telemetry_.t_imu_s) * 1000.0;
        latest_telemetry_.raw_gyro = Eigen::Vector3d(last_imu.angular_velocity.x, last_imu.angular_velocity.y, last_imu.angular_velocity.z);
        latest_telemetry_.raw_accel = Eigen::Vector3d(last_imu.linear_acceleration.x, last_imu.linear_acceleration.y, last_imu.linear_acceleration.z);
    }
    
    std::optional<Eigen::Matrix3d> imu_rotation = std::nullopt;
    if (!frame.imu_measurements.empty()) {
        Eigen::Matrix3d R_base_cam = Eigen::Matrix3d::Identity();
        try {
            if (tf_buffer_ && !base_frame_id_.empty() && !camera_frame_id_.empty() && base_frame_id_ != camera_frame_id_) {
                const geometry_msgs::msg::TransformStamped t_base_cam_msg =
                    tf_buffer_->lookupTransform(base_frame_id_, camera_frame_id_, tf2::TimePointZero);
                tf2::Quaternion q_bc;
                tf2::fromMsg(t_base_cam_msg.transform.rotation, q_bc);
                Eigen::Quaterniond eigen_q(q_bc.w(), q_bc.x(), q_bc.y(), q_bc.z());
                R_base_cam = eigen_q.toRotationMatrix();
            }
        } catch (const tf2::TransformException& ex) {
            // Default to identity
        }

        // Read the best known gyro bias from the graph (populated by VINS after init).
        // Falls back to zero before initialization — better than ignoring bias entirely.
        Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
        if (graph_) {
            auto graph_ptr = graph_->getGraph();
            if (graph_ptr) {
                auto last_node = graph_ptr->getNode(graph_ptr->getLastNodeId());
                if (last_node) {
                    gyro_bias = last_node->getGyroBias();
                }
            }
        }

        Eigen::Matrix3d R_acc = Eigen::Matrix3d::Identity();
        double last_t = frame.imu_measurements[0].header.stamp.sec + frame.imu_measurements[0].header.stamp.nanosec * 1e-9;
        for (std::size_t i = 0; i < frame.imu_measurements.size(); ++i) {
            const auto& imu = frame.imu_measurements[i];
            double curr_t = imu.header.stamp.sec + imu.header.stamp.nanosec * 1e-9;
            double dt = (i == 0) ? 0.005 : (curr_t - last_t);
            last_t = curr_t;
            if (dt < 0.0 || dt > 0.1) dt = 0.005;

            // Subtract gyro bias before integrating ().
            Eigen::Vector3d omega_base(imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z);
            Eigen::Vector3d omega = R_base_cam.transpose() * (omega_base - gyro_bias);
            double angle = omega.norm() * dt;
            if (angle > 1e-6) {
                Eigen::Vector3d axis = omega.normalized();
                Eigen::AngleAxisd aa(angle, axis);
                R_acc = R_acc * aa.toRotationMatrix();
            }
        }
        imu_rotation = R_acc;
        
        if (imu_rotation) {
            latest_telemetry_.imu_euler_deg = slam::telemetry::TelemetryLogger::computeEulerFromRot(*imu_rotation);
            Eigen::Quaterniond q_imu(*imu_rotation);
            latest_telemetry_.imu_quat = Eigen::Vector4d(q_imu.x(), q_imu.y(), q_imu.z(), q_imu.w());
            latest_telemetry_.imu_trans_accel = R_base_cam.transpose() * latest_telemetry_.raw_accel;
        }
    }

    // Process image through frontend with IMU rotation prediction
    latest_vo_result_ = frontend_->processFrame(frame.image, std::nullopt, imu_rotation);
    if (latest_vo_result_ && latest_vo_result_->valid) {
        try {
            if (tf_buffer_ && !base_frame_id_.empty() && !camera_frame_id_.empty() && base_frame_id_ != camera_frame_id_) {
                const geometry_msgs::msg::TransformStamped t_base_cam_msg =
                    tf_buffer_->lookupTransform(base_frame_id_, camera_frame_id_, tf2::TimePointZero);
                tf2::Transform base_T_cam;
                tf2::fromMsg(t_base_cam_msg.transform, base_T_cam);
                tf2::Transform t_rel_cam;
                tf2::fromMsg(latest_vo_result_->relative_transform, t_rel_cam);
                tf2::Transform t_rel_base = base_T_cam * t_rel_cam * base_T_cam.inverse();
                latest_vo_result_->relative_transform = tf2::toMsg(t_rel_base);
            }
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(logger_, *node_->get_clock(), 5000, "Missing TF %s->%s for VO frame conversion (%s)", base_frame_id_.c_str(), camera_frame_id_.c_str(), ex.what());
        }
        
        // Stage 2: VO Telemetry & Directional Check
        latest_telemetry_.vo_tracked_points = latest_vo_result_->tracked_points;
        latest_telemetry_.vo_inlier_ratio = (latest_telemetry_.vo_tracked_points > 0) ? 
            static_cast<double>(latest_vo_result_->inliers) / latest_telemetry_.vo_tracked_points : 0.0;
        tf2::Quaternion q_vo;
        tf2::fromMsg(latest_vo_result_->relative_transform.rotation, q_vo);
        Eigen::Quaterniond eq_vo(q_vo.w(), q_vo.x(), q_vo.y(), q_vo.z());
        latest_telemetry_.vo_euler_deg = slam::telemetry::TelemetryLogger::computeEulerFromQuat(eq_vo);
        latest_telemetry_.vo_quat = Eigen::Vector4d(eq_vo.x(), eq_vo.y(), eq_vo.z(), eq_vo.w());
        latest_telemetry_.vo_trans = Eigen::Vector3d(
            latest_vo_result_->relative_transform.translation.x,
            latest_vo_result_->relative_transform.translation.y,
            latest_vo_result_->relative_transform.translation.z);
            
        latest_telemetry_.yaw_diff_deg = latest_telemetry_.vo_euler_deg(2) - latest_telemetry_.imu_euler_deg(2);
        if (std::abs(latest_telemetry_.vo_euler_deg(2)) > 0.5 && std::abs(latest_telemetry_.imu_euler_deg(2)) > 0.5 &&
            (latest_telemetry_.vo_euler_deg(2) * latest_telemetry_.imu_euler_deg(2) < 0.0)) {
            latest_telemetry_.sign_flip_alert = true;
        } else {
            latest_telemetry_.sign_flip_alert = false;
        }
    }
    
    // Try PnP if VO failed and tracking is LOST
    if ((!latest_vo_result_ || !latest_vo_result_->valid) && tracking_state_ == TrackingState::LOST) {
        auto pnp_result = tryLocalKeyframePnpRecovery(
            frontend_->getCurrentDescriptors(),
            frontend_->getCurrentKeypoints());
        if (pnp_result) {
            latest_vo_result_ = pnp_result;
            RCLCPP_INFO(logger_, "PnP recovery succeeded");
        }
    }
    
    if (!latest_vo_result_ || !latest_vo_result_->valid) {
        registerTrackingReject("VO failed, PnP failed");
        wants_keyframe_cache_ = false;
        return;
    }
    
    if (!isFiniteTransform(latest_vo_result_->relative_transform)) {
        registerTrackingReject("Non-finite VO transform");
        wants_keyframe_cache_ = false;
        return;
    }
    
    setTrackingState(TrackingState::OK, "Tracking OK");
    
    // Accumulate visual pose.
    // IMPORTANT: vision_integrated_pose_ is only updated on successful frames (we
    // return early above on failure). So when the frontend falls back to a reference
    // frame N frames ago (reference_age > 1), vision_integrated_pose_ is still at
    // that older reference frame pose — applying t_rel on top of it is correct.
    tf2::Transform t_prev, t_rel;
    tf2::fromMsg(vision_integrated_pose_, t_prev);
    tf2::fromMsg(latest_vo_result_->relative_transform, t_rel);
    

    tf2::Transform t_new = t_prev * t_rel;
    tf2::toMsg(t_new, vision_integrated_pose_);
    
    wants_keyframe_cache_ = shouldAddVisionKeyframe(vision_integrated_pose_, frame.stamp);
}

bool CameraModule::wantsKeyframe() const {
    return wants_keyframe_cache_;
}

void CameraModule::onKeyframeCreated(std::shared_ptr<slam::GraphNode> node, slam::core::GraphInterface* graph) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Always update node pose with our best guess (even if frozen due to lost tracking)
    node->updatePose(vision_integrated_pose_);
    latest_telemetry_.local_fused_pose = vision_integrated_pose_;
    latest_telemetry_.frame_id = node->getId();
    
    if (latest_vo_result_ && latest_vo_result_->valid) {
        node->setVisualData(
            frontend_->getCurrentKeypoints(),
            frontend_->getCurrentDescriptors());
    }
        
    // Update camera internal state
    const int kf_id = node->getId();
    recent_keyframe_ids_.push_back(kf_id);
    if (recent_keyframe_ids_.size() > static_cast<size_t>(pnp_keyframe_window_ * 2)) {
        recent_keyframe_ids_.pop_front();
    }
    
    // ARCH-8 FIX: Actively triangulate 2D keypoint matches into 3D world landmarks
    if (graph) {
        auto map_manager = graph->getMapManager();
        auto graph_ptr = graph->getGraph();
        if (map_manager && graph_ptr && !camera_matrix_.empty()) {
            map_manager->updateLandmarkMapForLatestKeyframe(
                graph_ptr,
                kf_id,
                recent_keyframe_ids_,
                camera_matrix_,
                0.75, // match_ratio_test
                8,    // min_triangulation_parallax_px
                0.5,  // min_depth
                50.0  // max_depth
            );
        }
    }
    
    // Submit VO edge if not origin AND tracking is valid
    // use last_keyframe_id_ (the actual previous keyframe) instead of kf_id-1.
    // If any keyframe was skipped due to tracking failure, kf_id-1 references a
    // non-existent node, producing a dangling edge that corrupts optimization.
    if (kf_id > 0 && last_keyframe_id_ >= 0 && latest_vo_result_ && latest_vo_result_->valid) {
        slam::MeasurementEdgeConfig vo_edge;
        vo_edge.source = slam::MeasurementSource::VISUAL_ODOMETRY;
        vo_edge.from_keyframe_id = last_keyframe_id_;
        vo_edge.to_keyframe_id = kf_id;
        vo_edge.relative_transform = calculateRelativeTransform(last_keyframe_vision_pose_, vision_integrated_pose_);
        vo_edge.confidence = latest_vo_result_->confidence;
        graph->submitMeasurement(vo_edge);
    }
    
    // Always update the anchor so we don't hallucinate skipped keyframes
    last_keyframe_time_ = latest_frame_.stamp;
    last_keyframe_vision_pose_ = vision_integrated_pose_;
    // advance the from-ID anchor to this keyframe (regardless of VO validity),
    // so that if VO fails on the next frame, we don't skip back to an old node.
    last_keyframe_id_ = kf_id;

    if (latest_vo_result_ && latest_vo_result_->valid) {
        frontend_->addKeyframe();
        if (loop_closure_) {
            loop_closure_->addKeyframe(kf_id, frontend_->getCurrentDescriptors());
        }
    }
}

void CameraModule::onGlobalAlignment(double scale, const tf2::Quaternion& q_align) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Scale and rotate the internal vision poses to match the new global gravity/scale
    tf2::Transform t_vision;
    tf2::fromMsg(vision_integrated_pose_, t_vision);
    t_vision.getOrigin() *= scale;
    tf2::Transform t_vision_aligned = tf2::Transform(q_align) * t_vision;
    tf2::toMsg(t_vision_aligned, vision_integrated_pose_);
    
    tf2::Transform t_last;
    tf2::fromMsg(last_keyframe_vision_pose_, t_last);
    t_last.getOrigin() *= scale;
    tf2::Transform t_last_aligned = tf2::Transform(q_align) * t_last;
    tf2::toMsg(t_last_aligned, last_keyframe_vision_pose_);
}

slam::telemetry::TelemetrySnapshot CameraModule::getLatestTelemetry() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_telemetry_;
}

const char* CameraModule::trackingStateToString(TrackingState state) const {
    switch (state) {
        case TrackingState::OK: return "OK";
        case TrackingState::WEAK: return "WEAK";
        case TrackingState::LOST: return "LOST";
        default: return "UNKNOWN";
    }
}

void CameraModule::setTrackingState(TrackingState new_state, const char* reason) {
    if (tracking_state_ == new_state) return;

    RCLCPP_WARN(logger_, "Tracking state %s -> %s (%s)",
                trackingStateToString(tracking_state_),
                trackingStateToString(new_state), reason);
    tracking_state_ = new_state;
}

void CameraModule::registerTrackingReject(const char* reason, bool reset_frontend_if_lost) {
    consecutive_vo_rejects_++;
    consecutive_good_vo_frames_ = 0;

    if (consecutive_vo_rejects_ >= lost_tracking_reject_threshold_) {
        const bool entering_lost = tracking_state_ != TrackingState::LOST;
        setTrackingState(TrackingState::LOST, reason);
        if (entering_lost && reset_frontend_if_lost && frontend_) {
            frontend_->reset();
        }
        return;
    }

    if (consecutive_vo_rejects_ >= weak_tracking_reject_threshold_) {
        setTrackingState(TrackingState::WEAK, reason);
    }
}

bool CameraModule::isFiniteTransform(const geometry_msgs::msg::Transform& tf) {
    return std::isfinite(tf.translation.x) && std::isfinite(tf.translation.y) && std::isfinite(tf.translation.z) &&
           std::isfinite(tf.rotation.x) && std::isfinite(tf.rotation.y) && std::isfinite(tf.rotation.z) && std::isfinite(tf.rotation.w) &&
           (tf.rotation.x * tf.rotation.x + tf.rotation.y * tf.rotation.y + tf.rotation.z * tf.rotation.z + tf.rotation.w * tf.rotation.w > 0.99);
}

double CameraModule::transformTranslationNorm(const geometry_msgs::msg::Transform& tf) {
    return std::sqrt(tf.translation.x * tf.translation.x + tf.translation.y * tf.translation.y + tf.translation.z * tf.translation.z);
}

double CameraModule::quaternionAngularDistanceRad(const geometry_msgs::msg::Quaternion& a, const geometry_msgs::msg::Quaternion& b) {
    double dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    if (dot < 0) dot = -dot;
    if (dot > 1.0) dot = 1.0;
    return 2.0 * std::acos(dot);
}

geometry_msgs::msg::Transform CameraModule::calculateRelativeTransform(const geometry_msgs::msg::Pose& from, const geometry_msgs::msg::Pose& to) {
    tf2::Transform tf_from, tf_to;
    tf2::fromMsg(from, tf_from);
    tf2::fromMsg(to, tf_to);
    tf2::Transform tf_rel = tf_from.inverse() * tf_to;
    return tf2::toMsg(tf_rel);
}

bool CameraModule::shouldAddVisionKeyframe(const geometry_msgs::msg::Pose& candidate_pose, const rclcpp::Time& stamp) const {
    const double dx = candidate_pose.position.x - last_keyframe_vision_pose_.position.x;
    const double dy = candidate_pose.position.y - last_keyframe_vision_pose_.position.y;
    const double dz = candidate_pose.position.z - last_keyframe_vision_pose_.position.z;
    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

    const bool by_distance = (distance + 1e-6) >= keyframe_distance_threshold_;
    const double rot_rad = quaternionAngularDistanceRad(last_keyframe_vision_pose_.orientation, candidate_pose.orientation);
    
    // Allow visual rotation keyframe triggering across all modes (including mono_imu/stereo_imu)
    // so that rapid 90-degree turns get crisp, short-baseline keyframe spacing rather than
    // suffering optical flow feature dropout across large angular intervals.
    const bool is_imu_mode = (mode_ == "mono_imu" || mode_ == "stereo_imu");
    const bool by_rotation = (rot_rad >= (keyframe_rotation_threshold_deg_ * M_PI / 180.0));
    
    constexpr double kMinMotionForTimeTrigger = 0.05;
    const bool by_time = (distance > kMinMotionForTimeTrigger) &&
                         ((last_keyframe_time_.nanoseconds() == 0) ||
                          ((stamp - last_keyframe_time_).seconds() >= keyframe_time_threshold_sec_));

    const bool by_parallax = !is_imu_mode && latest_vo_result_ && (latest_vo_result_->unit_sphere_parallax >= 0.02);

    return by_distance || by_time || by_rotation || by_parallax;
}

std::optional<slam::core::FrontendResult> CameraModule::tryLocalKeyframePnpRecovery(const cv::Mat& current_descriptors, const std::vector<cv::KeyPoint>& current_keypoints) {
    if (recent_keyframe_ids_.empty() || !graph_) return std::nullopt;
    if (current_keypoints.empty() || current_descriptors.empty()) return std::nullopt;

    auto map_manager = graph_->getMapManager();
    auto graph_ptr = graph_->getGraph();
    if (!map_manager || !graph_ptr) return std::nullopt;

    struct CandidateMatch {
        int landmark_id{-1};
        int current_keypoint_index{-1};
        float distance{0.0f};
    };

    std::unordered_map<int, CandidateMatch> best_by_landmark;
    constexpr float kRatioTest = 0.75f;

    std::size_t used_keyframes = 0;
    for (auto it = recent_keyframe_ids_.rbegin();
         it != recent_keyframe_ids_.rend() && used_keyframes < static_cast<std::size_t>(pnp_keyframe_window_);
         ++it, ++used_keyframes)
    {
        auto node = graph_ptr->getNode(*it);
        if (!node) continue;

        const auto& node_descriptors = node->getDescriptors();
        if (node_descriptors.empty() || node_descriptors.type() != current_descriptors.type()) continue;

        const int norm = (node_descriptors.type() == CV_8U) ? cv::NORM_HAMMING : cv::NORM_L2;
        cv::BFMatcher matcher(norm, false);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(node_descriptors, current_descriptors, knn_matches, 2);

        for (const auto& pair : knn_matches) {
            if (pair.size() < 2 || !(pair[0].distance < kRatioTest * pair[1].distance)) continue;

            const int query_index = pair[0].queryIdx;
            const int train_index = pair[0].trainIdx;
            
            int landmark_id = map_manager->getLandmarkIdFromObservation(*it, query_index);
            if (landmark_id < 0) continue;

            const auto* landmark = map_manager->getLandmark(landmark_id);
            if (!landmark || landmark->bad || !landmark->position.allFinite()) continue;

            auto best_it = best_by_landmark.find(landmark_id);
            if (best_it == best_by_landmark.end() || pair[0].distance < best_it->second.distance) {
                best_by_landmark[landmark_id] = CandidateMatch{landmark_id, train_index, pair[0].distance};
            }
        }
    }

    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    object_points.reserve(best_by_landmark.size());
    image_points.reserve(best_by_landmark.size());

    for (const auto& [landmark_id, match] : best_by_landmark) {
        const auto* landmark = map_manager->getLandmark(landmark_id);
        if (!landmark || match.current_keypoint_index < 0 || match.current_keypoint_index >= static_cast<int>(current_keypoints.size())) continue;

        const auto& p = landmark->position;
        object_points.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z()));
        image_points.push_back(current_keypoints[match.current_keypoint_index].pt);
    }

    if (object_points.size() < 8) return std::nullopt;

    cv::Mat rvec, tvec, inliers;
    const cv::Mat distortion = cv::Mat::zeros(4, 1, CV_64F);
    const bool solved = cv::solvePnPRansac(object_points, image_points, camera_matrix_, distortion,
                                           rvec, tvec, false, 100, 4.0, 0.99, inliers, cv::SOLVEPNP_EPNP);
    if (!solved || inliers.rows < 12) return std::nullopt;

    const double inlier_ratio = static_cast<double>(inliers.rows) / static_cast<double>(object_points.size());
    if (inliers.rows < pnp_min_inliers_ || inlier_ratio < pnp_min_inlier_ratio_) return std::nullopt;

    cv::Mat rotation_cv;
    cv::Rodrigues(rvec, rotation_cv);
    Eigen::Matrix3d Rcw;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            Rcw(row, col) = rotation_cv.at<double>(row, col);
        }
    }
    const Eigen::Vector3d tcw(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    const Eigen::Matrix3d Rwc = Rcw.transpose();
    const Eigen::Vector3d twc = -Rwc * tcw;

    tf2::Transform world_T_camera;
    world_T_camera.setOrigin(tf2::Vector3(twc.x(), twc.y(), twc.z()));
    world_T_camera.setRotation(tf2::Quaternion(Eigen::Quaterniond(Rwc).x(), Eigen::Quaterniond(Rwc).y(), Eigen::Quaterniond(Rwc).z(), Eigen::Quaterniond(Rwc).w()));

    geometry_msgs::msg::Pose recovered_pose;
    try {
        const geometry_msgs::msg::TransformStamped t_base_cam_msg = tf_buffer_->lookupTransform(base_frame_id_, camera_frame_id_, tf2::TimePointZero);
        tf2::Transform base_T_camera;
        tf2::fromMsg(t_base_cam_msg.transform, base_T_camera);
        const tf2::Transform world_T_base = world_T_camera * base_T_camera.inverse();

        const tf2::Vector3 origin = world_T_base.getOrigin();
        const tf2::Quaternion q = world_T_base.getRotation().normalized();
        recovered_pose.position.x = origin.x();
        recovered_pose.position.y = origin.y();
        recovered_pose.position.z = origin.z();
        recovered_pose.orientation = tf2::toMsg(q);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(logger_, *node_->get_clock(), 2000, "Local PnP recovery skipped: missing base<->camera TF (%s)", ex.what());
        return std::nullopt;
    }

    slam::core::FrontendResult result;
    result.valid = true;
    result.relative_transform = calculateRelativeTransform(last_keyframe_vision_pose_, recovered_pose);
    result.inliers = inliers.rows;
    result.inlier_ratio = inlier_ratio;
    result.confidence = std::clamp(0.35 + (0.65 * result.inlier_ratio), 0.0, 1.0);

    if (!isFiniteTransform(result.relative_transform)) return std::nullopt;

    geometry_msgs::msg::Quaternion identity_q;
    identity_q.x = 0.0; identity_q.y = 0.0; identity_q.z = 0.0; identity_q.w = 1.0;
    const double pnp_rot_err_deg = quaternionAngularDistanceRad(identity_q, result.relative_transform.rotation) * 180.0 / M_PI;
    if (pnp_rot_err_deg > pnp_max_rotation_deg_) return std::nullopt;

    const double pnp_translation = transformTranslationNorm(result.relative_transform);
    const double max_pnp_translation = std::max(3.0, 4.0 * keyframe_distance_threshold_);
    if (pnp_translation > max_pnp_translation) return std::nullopt;

    RCLCPP_WARN(logger_, "Local PnP recovery succeeded: inliers=%d ratio=%.2f", result.inliers, result.inlier_ratio);
    return result;
}

} // namespace slam::modules
