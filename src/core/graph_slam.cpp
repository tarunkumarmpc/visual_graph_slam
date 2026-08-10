#include "visual_graph_slam/core/graph_slam.hpp"
#include "visual_graph_slam/vins/tightly_coupled_local_ba.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

namespace slam
{

namespace {

constexpr double kDegToRad = M_PI / 180.0;

Eigen::Matrix<double, 6, 6> makeSe3ExpmapInformation(double rotation_weight,
                                                     double translation_weight)
{
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
    information.diagonal() << rotation_weight, rotation_weight, rotation_weight,
                               translation_weight, translation_weight, translation_weight;
    return information;
}

int64_t makeDirectedEdgeKey(int from_id, int to_id)
{
    return static_cast<int64_t>((static_cast<uint64_t>(static_cast<uint32_t>(from_id)) << 32) |
                                static_cast<uint32_t>(to_id));
}

double clamp01(double value)
{
    return std::clamp(value, 0.0, 1.0);
}

double transformTranslationNorm(const geometry_msgs::msg::Transform& tf)
{
    return std::sqrt(tf.translation.x * tf.translation.x +
                     tf.translation.y * tf.translation.y +
                     tf.translation.z * tf.translation.z);
}

double poseDistance(const geometry_msgs::msg::Pose& a, const geometry_msgs::msg::Pose& b)
{
    const double dx = a.position.x - b.position.x;
    const double dy = a.position.y - b.position.y;
    const double dz = a.position.z - b.position.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool isMarginalVoResult(const VisionNode::FrontendResult& result,
                        int min_inliers,
                        double min_ratio,
                        double min_confidence)
{
    const int near_inliers = std::max(10, static_cast<int>(std::ceil(static_cast<double>(min_inliers) * 0.75)));
    return result.inliers >= near_inliers &&
           result.inlier_ratio >= (min_ratio * 0.85) &&
           result.confidence >= (min_confidence * 0.85);
}

class EdgeProjectXYZ2UVBasePose final
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjectXYZ2UVBasePose(double fx,
                              double fy,
                              double cx,
                              double cy,
                              const Eigen::Isometry3d& base_T_camera)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), base_T_camera_(base_T_camera)
    {
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    void computeError() override
    {
        const auto* point = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
        const auto* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const Eigen::Vector2d projected = project(pose->estimate(), point->estimate());
        _error = projected - _measurement;
    }

    // WEAKNESS3-FIX (corrected): Analytic Jacobians for projection edge.
    //
    // Optimization variable : world_T_base  (g2o::VertexSE3Expmap stores the
    //                          direct camera-to-world pose, i.e. position +
    //                          orientation of base_link IN world frame).
    //
    // Full camera pose      : world_T_cam = world_T_base * base_T_cam
    //
    // Projection            : Pc = camera_T_world * Pw
    //                        π(Pc) = [fx*Pc.x/Pc.z + cx, fy*Pc.y/Pc.z + cy]
    //
    // ── Point Jacobian (J_xi, 2×3) ──────────────────────────────────────────
    //   dPc/dPw = R_cw   →   J_xi = dπ/dPc * R_cw
    //
    // ── Pose Jacobian (J_xj, 2×6) ───────────────────────────────────────────
    //   Under LEFT perturbation:  T_wb_new = exp(δ) * T_wb
    //   Pc_new = T_bc⁻¹ * (exp(δ)*T_wb)⁻¹ * Pw
    //          ≈ Pc + R_cw * ([Pw]× δω - δv)   (first order)
    //
    //   So: dPc/dδ = R_cw * [[Pw]×, -I₃]
    //
    //   NOTE: The common textbook form [-[Pc]×, I₃] is only correct when the
    //   optimization variable IS world_T_camera (identity extrinsic). With a
    //   non-identity base_T_camera the above world-frame form is required.
    void linearizeOplus() override
    {
        const auto* v_point = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
        const auto* v_pose  = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

        const Eigen::Vector3d Pw = v_point->estimate();

        const g2o::SE3Quat& world_T_base = v_pose->estimate();
        Eigen::Isometry3d world_T_base_iso = Eigen::Isometry3d::Identity();
        world_T_base_iso.linear()      = world_T_base.rotation().toRotationMatrix();
        world_T_base_iso.translation() = world_T_base.translation();

        const Eigen::Isometry3d world_T_cam  = world_T_base_iso * base_T_camera_;
        const Eigen::Isometry3d camera_T_world = world_T_cam.inverse();

        // R_cw: rotation from world frame to camera frame
        const Eigen::Matrix3d R_cw = camera_T_world.linear();

        // Landmark in camera frame
        const Eigen::Vector3d Pc = camera_T_world * Pw;
        const double z    = std::max(1e-9, Pc.z());
        const double iz   = 1.0 / z;
        const double iz2  = iz * iz;

        // dπ/dPc  (2×3)
        Eigen::Matrix<double, 2, 3> dpi;
        dpi << fx_ * iz,      0.0,  -fx_ * Pc.x() * iz2,
                   0.0,  fy_ * iz,  -fy_ * Pc.y() * iz2;

        // ── J_xi : dπ/dPw = dπ/dPc * dPc/dPw = dπ/dPc * R_cw ──────────────
        _jacobianOplusXi = dpi * R_cw;

        // ── J_xj : dπ/dδ = dπ/dPc * R_cw * [[Pw]×, -I₃] ───────────────────
        // Build skew-symmetric [Pw]×
        Eigen::Matrix<double, 3, 3> Pw_skew;
        Pw_skew <<      0.0,  -Pw.z(),   Pw.y(),
                    Pw.z(),      0.0,  -Pw.x(),
                   -Pw.y(),   Pw.x(),      0.0;

        // dPc/dδ = R_cw * [[Pw]×, -I₃]   (3×6)
        Eigen::Matrix<double, 3, 6> dPc_dxi;
        dPc_dxi.block<3, 3>(0, 0) =  R_cw * Pw_skew;
        dPc_dxi.block<3, 3>(0, 3) = -R_cw;

        _jacobianOplusXj = dpi * dPc_dxi;
    }

private:
    Eigen::Vector2d project(const g2o::SE3Quat& world_T_base,
                            const Eigen::Vector3d& point_world) const
    {
        Eigen::Isometry3d world_T_base_iso = Eigen::Isometry3d::Identity();
        world_T_base_iso.linear()      = world_T_base.rotation().toRotationMatrix();
        world_T_base_iso.translation() = world_T_base.translation();

        const Eigen::Isometry3d world_T_camera = world_T_base_iso * base_T_camera_;
        const Eigen::Vector3d point_camera = world_T_camera.inverse() * point_world;

        const double z = std::max(1e-9, point_camera.z());
        return Eigen::Vector2d(
            fx_ * (point_camera.x() / z) + cx_,
            fy_ * (point_camera.y() / z) + cy_);
    }

    double fx_;
    double fy_;
    double cx_;
    double cy_;
    Eigen::Isometry3d base_T_camera_;
};

}  // namespace

GraphSlam::GraphSlam()
    : Node("graph_slam"),
      graph_(std::make_shared<Graph>()),
      camera_matrix_((cv::Mat_<double>(3, 3) << 525.0, 0.0, 320.0, 0.0, 525.0, 240.0, 0.0, 0.0, 1.0)),
      last_keyframe_time_(0, 0, RCL_ROS_TIME),
      system_loader_("visual_graph_slam", "slam::core::System")
{
    map_manager_ = std::make_shared<slam::map::MapManager>(this->get_logger());
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    (void)this->declare_parameter<std::string>("map_frame_id", "map");
    base_frame_id_ = this->declare_parameter<std::string>("base_frame_id", "base_link");
    camera_frame_id_ = this->declare_parameter<std::string>("camera_frame_id", "camera_link");
    imu_frame_id_ = this->declare_parameter<std::string>("imu.frame_id", "imu_link");
    (void)this->declare_parameter<bool>("publish_map_to_base_tf", true);
    mode_ = this->declare_parameter<std::string>("mode", mode_);

    keyframe_distance_threshold_ = this->declare_parameter<double>("keyframe_distance_threshold", keyframe_distance_threshold_);
    local_mapping_window_size_ = this->declare_parameter<int>("local_mapping_window_size", local_mapping_window_size_);
    local_mapping_stride_ = this->declare_parameter<int>("local_mapping_stride", local_mapping_stride_);
    local_mapping_enabled_ = this->declare_parameter<bool>("enable_local_mapping", local_mapping_enabled_);
    local_ba_iterations_ = this->declare_parameter<int>("local_ba_iterations", local_ba_iterations_);
    local_ba_use_reprojection_ = this->declare_parameter<bool>("local_ba_use_reprojection", local_ba_use_reprojection_);
    local_ba_pose_only_ = this->declare_parameter<bool>("local_ba_pose_only", local_ba_pose_only_);
    local_ba_every_keyframe_ = this->declare_parameter<bool>("local_mapping_every_keyframe", local_ba_every_keyframe_);
    local_ba_free_kfs_ = this->declare_parameter<int>("local_ba_free_kfs", local_ba_free_kfs_);
    local_ba_init_min_keyframes_ = this->declare_parameter<int>("local_ba_init_min_keyframes", local_ba_init_min_keyframes_);
    local_ba_init_min_baseline_m_ = this->declare_parameter<double>("local_ba_init_min_baseline_m", local_ba_init_min_baseline_m_);
    local_ba_init_min_mean_flow_px_ = this->declare_parameter<double>("local_ba_init_min_mean_flow_px", local_ba_init_min_mean_flow_px_);
    local_ba_min_window_kfs_ = this->declare_parameter<int>("local_ba_min_window_kfs", local_ba_min_window_kfs_);
    local_ba_min_anchored_landmarks_ = this->declare_parameter<int>("local_ba_min_anchored_landmarks", local_ba_min_anchored_landmarks_);
    local_ba_min_reprojection_edges_ = this->declare_parameter<int>("local_ba_min_reprojection_edges", local_ba_min_reprojection_edges_);
    local_ba_min_obs_per_landmark_window_ =
        this->declare_parameter<int>("local_ba_min_obs_per_landmark_window", local_ba_min_obs_per_landmark_window_);
    local_ba_min_landmark_parallax_px_ =
        this->declare_parameter<double>("local_ba_min_landmark_parallax_px", local_ba_min_landmark_parallax_px_);
    local_ba_max_initial_reproj_err_px_ =
        this->declare_parameter<double>("local_ba_max_initial_reproj_err_px", local_ba_max_initial_reproj_err_px_);
    local_ba_max_tail_step_m_ =
        this->declare_parameter<double>("local_ba_max_tail_step_m", local_ba_max_tail_step_m_);
    local_ba_pose_edge_translation_scale_ =
        this->declare_parameter<double>("local_ba_pose_edge_translation_scale", local_ba_pose_edge_translation_scale_);
    local_ba_reprojection_info_scale_ =
        this->declare_parameter<double>("local_ba_reprojection_info_scale", local_ba_reprojection_info_scale_);
    local_ba_rollback_cooldown_kfs_ =
        this->declare_parameter<int>("local_ba_rollback_cooldown_kfs", local_ba_rollback_cooldown_kfs_);
    local_ba_disable_reprojection_after_rollbacks_ =
        this->declare_parameter<int>("local_ba_disable_reprojection_after_rollbacks", local_ba_disable_reprojection_after_rollbacks_);
    full_ba_on_loop_closure_ = this->declare_parameter<bool>("full_ba_on_loop_closure", full_ba_on_loop_closure_);
    full_ba_iterations_ = this->declare_parameter<int>("full_ba_iterations", full_ba_iterations_);
    vins_min_keyframes_for_init_ = this->declare_parameter<int>("vins.min_keyframes_for_init", vins_min_keyframes_for_init_);
    planar_motion_constraint_in_mono_ =
        this->declare_parameter<bool>("mono.planar_motion_constraint", planar_motion_constraint_in_mono_);

    local_ba_enable_height_prior_ = this->declare_parameter<bool>("local_ba_enable_height_prior", local_ba_enable_height_prior_);
    local_ba_height_prior_value_ = this->declare_parameter<double>("local_ba_height_prior_value", local_ba_height_prior_value_);
    local_ba_height_prior_stddev_ = this->declare_parameter<double>("local_ba_height_prior_stddev", local_ba_height_prior_stddev_);
    landmark_match_ratio_test_ = this->declare_parameter<double>("landmark_match_ratio_test", landmark_match_ratio_test_);
    landmark_min_observations_ = this->declare_parameter<int>("landmark_min_observations", landmark_min_observations_);
    landmark_prune_stride_ = this->declare_parameter<int>("landmark_prune_stride", landmark_prune_stride_);
    landmark_min_depth_ = this->declare_parameter<double>("landmark_min_depth", landmark_min_depth_);
    landmark_max_depth_ = this->declare_parameter<double>("landmark_max_depth", landmark_max_depth_);
    landmark_min_triangulation_parallax_px_ =
        this->declare_parameter<int>("landmark_min_triangulation_parallax_px", landmark_min_triangulation_parallax_px_);
    local_landmark_track_depth_ =
        this->declare_parameter<int>("local_landmark_track_depth", local_landmark_track_depth_);

    const double fx = this->declare_parameter<double>("camera.fx", camera_matrix_.at<double>(0, 0));
    const double fy = this->declare_parameter<double>("camera.fy", camera_matrix_.at<double>(1, 1));
    const double cx = this->declare_parameter<double>("camera.cx", camera_matrix_.at<double>(0, 2));
    const double cy = this->declare_parameter<double>("camera.cy", camera_matrix_.at<double>(1, 2));
    camera_matrix_.at<double>(0, 0) = fx;
    camera_matrix_.at<double>(1, 1) = fy;
    camera_matrix_.at<double>(0, 2) = cx;
    camera_matrix_.at<double>(1, 2) = cy;

    camera_model_name_ = this->declare_parameter<std::string>("camera.model", camera_model_name_);
    camera_distortion_[0] = this->declare_parameter<double>("camera.distortion.k1", 0.0);
    camera_distortion_[1] = this->declare_parameter<double>("camera.distortion.k2", 0.0);
    camera_distortion_[2] = this->declare_parameter<double>("camera.distortion.k3", 0.0);
    camera_distortion_[3] = this->declare_parameter<double>("camera.distortion.k4", 0.0);

    try
    {
        sensor::CameraIntrinsics intrinsics{fx, fy, cx, cy};
        camera_model_ = sensor::createCameraModel(camera_model_name_, intrinsics, camera_distortion_);
        RCLCPP_INFO(this->get_logger(), "Camera model selected: %s", camera_model_->name().c_str());
    }
    catch (const std::exception &e)
    {
        RCLCPP_WARN(this->get_logger(), "%s. Falling back to pinhole camera model.", e.what());
        camera_model_name_ = "pinhole";
        sensor::CameraIntrinsics intrinsics{fx, fy, cx, cy};
        camera_model_ = sensor::createCameraModel(camera_model_name_, intrinsics, Eigen::Vector4d::Zero());
    }

    optimizer_name_ = this->declare_parameter<std::string>("optimizer", optimizer_name_);
    try
    {
        optimizer_ = core::createOptimizerBackend(optimizer_name_);
        RCLCPP_INFO(this->get_logger(), "Optimizer selected: %s", optimizer_->name().c_str());
    }
    catch (const std::exception &e)
    {
        RCLCPP_WARN(this->get_logger(), "%s. Falling back to g2o optimizer.", e.what());
        optimizer_name_ = "g2o";
        optimizer_ = core::createOptimizerBackend(optimizer_name_);
    }

    // Removed tracking logic parameters (moved to SystemBase)
    max_optimization_jump_m_ =
        this->declare_parameter<double>("optimizer.max_tail_jump_m", 2.0);


    global_optimization_requested_ = false;
    local_optimization_requested_ = false;
    run_optimization_thread_ = true;
    optimization_thread_ = std::thread(&GraphSlam::runOptimizationLoop, this);
    
    run_mapping_thread_ = true;
    mapping_thread_ = std::thread(&GraphSlam::runMappingLoop, this);

    // System initialization moved to initializeSystem()

    RCLCPP_INFO(this->get_logger(),
                "graph slam started (keyframe_dist=%.2f, local_window=%d, local_ba_iter=%d)",
                keyframe_distance_threshold_,
                local_mapping_window_size_,
                local_ba_iterations_);
    RCLCPP_INFO(this->get_logger(),
                "motion priors: enabled=%s source=%s",
                use_motion_priors_ ? "true" : "false",
                motion_prior_source_.c_str());
    RCLCPP_INFO(this->get_logger(),
                "pure mono planar constraint: %s",
                usePlanarConstraint() ? "enabled" : "disabled");
    RCLCPP_INFO(this->get_logger(),
                "optimizer safeguards: jump_guard=%.2fm",
                max_optimization_jump_m_);
}

void GraphSlam::initializeSystem()
{
    std::string system_plugin = this->declare_parameter<std::string>("system_plugin", "slam::plugins::SystemMonoImu");
    try {
        system_ = system_loader_.createSharedInstance(system_plugin);
        auto self_ptr = std::dynamic_pointer_cast<GraphSlam>(shared_from_this());
        system_->initialize(shared_from_this(), system_plugin, self_ptr);
        RCLCPP_INFO(this->get_logger(), "Successfully loaded system plugin: %s", system_plugin.c_str());
        if (graph_) {
            graph_->setPlanarMotionConstraint(usePlanarConstraint());
            if (mode_ == "mono_imu") {
                RCLCPP_INFO(this->get_logger(), "Mono-IMU mode: Metric scale will be initialized dynamically via VINS initializer.");
            }
        }
    } catch (pluginlib::PluginlibException& ex) {
        RCLCPP_ERROR(this->get_logger(), "The system plugin failed to load. Error: %s", ex.what());
        throw std::runtime_error("Failed to load system plugin");
    }
}








void GraphSlam::initializeVisualizer()
{
    try
    {
        visualizer_ = std::make_unique<Visualizer>(shared_from_this(), graph_);
        visualizer_initialized_ = true;
        RCLCPP_INFO(this->get_logger(), "Visualizer initialized successfully");
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize Visualizer: %s", e.what());
    }
}

void GraphSlam::updateVisualization()
{
    if (!visualizer_initialized_)
    {
        RCLCPP_WARN(this->get_logger(), "Visualizer not initialized. Skipping visualization update.");
        return;
    }

    try
    {
        graph_->setGraphChanged(true);
        visualizer_->triggerUpdate();
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "Error during visualization update: %s", e.what());
    }
}

void GraphSlam::poseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    RCLCPP_INFO(this->get_logger(), "Received PoseWithCovarianceStamped: x: %.2f, y: %.2f, z: %.2f",
                msg->pose.pose.position.x,
                msg->pose.pose.position.y,
                msg->pose.pose.position.z);
}

void GraphSlam::process_data(SensorData &sensor_data)
{
    TrackingQueueItem item;
    item.image = sensor_data.image;
    item.imu_buffer = sensor_data.imu_buffer;
    if (sensor_data.odometry) {
        item.odom_pose = sensor_data.odometry->pose.pose;
        item.has_odom = true;
    } else {
        item.has_odom = false;
    }
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        tracking_queue_.push(item);
    }
    queue_cv_.notify_one();
}

void GraphSlam::process_image_frame(sensor_msgs::msg::Image::ConstSharedPtr image,
                                    const std::vector<sensor_msgs::msg::Imu>& imu_buffer)
{
    TrackingQueueItem item;
    item.image = image;
    item.imu_buffer = imu_buffer;
    item.has_odom = false;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        tracking_queue_.push(item);
    }
    queue_cv_.notify_one();
}

void GraphSlam::runMappingLoop()
{
    while (run_mapping_thread_)
    {
        TrackingQueueItem item;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]() { return !tracking_queue_.empty() || !run_mapping_thread_; });
            
            if (!run_mapping_thread_) break;
            if (tracking_queue_.empty()) continue;
            
            item = tracking_queue_.front();
            tracking_queue_.pop();
        }
        
        process_queued_item(item);
    }
}

void GraphSlam::process_queued_item(const TrackingQueueItem& item)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    
    slam::core::SensorFrame frame;
    if (item.image) {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(item.image, sensor_msgs::image_encodings::BGR8);
        frame.image = cv_ptr->image.clone();
        frame.stamp = item.image->header.stamp;
    }
    frame.imu_measurements = item.imu_buffer;
    if (item.has_odom) {
        nav_msgs::msg::Odometry odom;
        odom.pose.pose = item.odom_pose;
        if (item.image) {
            odom.header = item.image->header;
        }
        frame.wheel_odom = odom;
    }
    
    if (system_) {
        system_->processSensorFrame(frame);
    }
}

void GraphSlam::updateCameraCalibration(const sensor_msgs::msg::CameraInfo& camera_info)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (camera_info.k.size() != 9)
    {
        RCLCPP_WARN(this->get_logger(), "camera_info ignored: invalid K matrix size");
        return;
    }

    const double fx = camera_info.k[0];
    const double fy = camera_info.k[4];
    const double cx = camera_info.k[2];
    const double cy = camera_info.k[5];

    if (fx <= 1e-9 || fy <= 1e-9)
    {
        RCLCPP_WARN(this->get_logger(), "camera_info ignored: invalid focal lengths");
        return;
    }

    camera_matrix_.at<double>(0, 0) = fx;
    camera_matrix_.at<double>(1, 1) = fy;
    camera_matrix_.at<double>(0, 2) = cx;
    camera_matrix_.at<double>(1, 2) = cy;

    try
    {
        sensor::CameraIntrinsics intrinsics{fx, fy, cx, cy};
        camera_model_ = sensor::createCameraModel(camera_model_name_, intrinsics, camera_distortion_);
    }
    catch (const std::exception &e)
    {
        RCLCPP_WARN(this->get_logger(), "Failed to update GraphSlam camera model: %s", e.what());
    }



    RCLCPP_INFO(this->get_logger(),
                "GraphSlam calibration updated from /camera_info (fx=%.2f fy=%.2f cx=%.2f cy=%.2f)",
                fx, fy, cx, cy);

    if (system_) {
        slam::sensor::CameraIntrinsics new_intrinsics{fx, fy, cx, cy};
        system_->setCameraIntrinsics(new_intrinsics);
    }
}

double GraphSlam::quaternionAngularDistanceRad(const geometry_msgs::msg::Quaternion& a,
                                               const geometry_msgs::msg::Quaternion& b)
{
    tf2::Quaternion qa(a.x, a.y, a.z, a.w);
    tf2::Quaternion qb(b.x, b.y, b.z, b.w);
    qa.normalize();
    qb.normalize();
    const double dot = std::clamp(
        qa.x() * qb.x() + qa.y() * qb.y() + qa.z() * qb.z() + qa.w() * qb.w(),
        -1.0,
        1.0);
    const double abs_dot = std::abs(dot);
    return 2.0 * std::acos(std::clamp(abs_dot, -1.0, 1.0));
}


// ─────────────────────────────────────────────────────────────────────────────
// UNIFIED MEASUREMENT EDGE SYSTEM
// ─────────────────────────────────────────────────────────────────────────────
// All sensor constraints (odometry, VO, IMU, GPS, etc.) flow through this
// single entry point. The information matrix is computed based on source type
// and confidence, allowing future sensors to be added without modifying this
// method.

void GraphSlam::submitMeasurement(const MeasurementEdgeConfig& config)
{
    if (!config.isValid()) {
        RCLCPP_WARN(this->get_logger(),
                    "Invalid measurement edge config (keyframe IDs: %d -> %d)",
                    config.from_keyframe_id, config.to_keyframe_id);
        return;
    }

    if (!std::isfinite(config.relative_transform.translation.x) || !std::isfinite(config.relative_transform.translation.y) || !std::isfinite(config.relative_transform.translation.z)) {
        RCLCPP_WARN(this->get_logger(),
                    "Rejected measurement edge with non-finite transform: %d -> %d (%s)",
                    config.from_keyframe_id,
                    config.to_keyframe_id,
                    MeasurementEdgeConfig::sourceToString(config.source));
        return;
    }

    tf2::Quaternion q(config.relative_transform.rotation.x,
                      config.relative_transform.rotation.y,
                      config.relative_transform.rotation.z,
                      config.relative_transform.rotation.w);
    if (q.length2() < 1e-12) {
        RCLCPP_WARN(this->get_logger(),
                    "Rejected measurement edge with invalid quaternion: %d -> %d (%s)",
                    config.from_keyframe_id,
                    config.to_keyframe_id,
                    MeasurementEdgeConfig::sourceToString(config.source));
        return;
    }
    q.normalize();

    auto sanitized_config = config;
    sanitized_config.relative_transform.rotation = tf2ToMsg(q);

    // Determine edge type based on measurement source
    GraphEdge::Type edge_type;
    switch (sanitized_config.source) {
        case MeasurementSource::WHEEL_ODOMETRY:
            edge_type = GraphEdge::Type::ODOMETRY;            break;
        case MeasurementSource::VISUAL_ODOMETRY:
            edge_type = GraphEdge::Type::VISUAL_ODOMETRY;     break;
        case MeasurementSource::LOOP_CLOSURE:
            edge_type = GraphEdge::Type::LOOP_CLOSURE;        break;
        case MeasurementSource::IMU_PREINTEGRATION:
            edge_type = GraphEdge::Type::IMU_PREINTEGRATION; break;
        case MeasurementSource::EXTERNAL_ODOMETRY:
            edge_type = GraphEdge::Type::ODOMETRY;           break;
        default:
            edge_type = GraphEdge::Type::ODOMETRY;            break;
    }

    // Compute information matrix (weight) from source and confidence
    const auto information_matrix = MeasurementWeightCalculator::compute(sanitized_config);

    // Create and add the edge to the graph
    auto edge = std::make_shared<GraphEdge>(
        edge_type,
        sanitized_config.from_keyframe_id,
        sanitized_config.to_keyframe_id,
        sanitized_config.relative_transform,
        information_matrix);
    
    if (sanitized_config.source == MeasurementSource::IMU_PREINTEGRATION && sanitized_config.imu_data.has_value()) {
        RCLCPP_INFO(this->get_logger(), "Adding IMU_PREINTEGRATION edge %d -> %d", sanitized_config.from_keyframe_id, sanitized_config.to_keyframe_id);
        edge->setImuPreintegration(sanitized_config.imu_data.value());
        edge->setRawImuSamples(sanitized_config.raw_imu_samples);
    }

    graph_->addEdge(edge);
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience Builders
// ─────────────────────────────────────────────────────────────────────────────
// These wrap common measurement types into MeasurementEdgeConfig format.
// Simplifies the call sites — new code should use these or addMeasurementEdge
// directly, never the deprecated addXxxEdge() methods.



bool GraphSlam::usePlanarConstraint() const
{
    return planar_motion_constraint_in_mono_;
}

void GraphSlam::initializePlanarReferenceFromPose(const geometry_msgs::msg::Pose& pose)
{
    if (planar_reference_initialized_)
    {
        return;
    }

    planar_reference_z_ = pose.position.z;
    tf2::Quaternion q = msgToTF2(pose.orientation);
    q.normalize();
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    planar_reference_roll_rad_ = roll;
    planar_reference_pitch_rad_ = pitch;
    planar_reference_initialized_ = true;

    RCLCPP_INFO(this->get_logger(),
                "Planar reference initialized: z=%.3f roll=%.2fdeg pitch=%.2fdeg",
                planar_reference_z_,
                planar_reference_roll_rad_ * 180.0 / M_PI,
                planar_reference_pitch_rad_ * 180.0 / M_PI);
}

void GraphSlam::enforcePlanarPose(geometry_msgs::msg::Pose& pose) const
{
    if (!usePlanarConstraint() || !planar_reference_initialized_)
    {
        return;
    }

    pose.position.z = planar_reference_z_;
    
    // In ROS body frame (base_link / Z-up), turning on the road is rotation around Z (Yaw).
    // Project quaternion onto pure Z-axis rotation by zeroing out X (roll) and Y (pitch).
    pose.orientation.x = 0.0;
    pose.orientation.y = 0.0;
    tf2::Quaternion q(0.0, 0.0, pose.orientation.z, pose.orientation.w);
    if (q.length2() > 1e-6) {
        q.normalize();
        pose.orientation.z = q.z();
        pose.orientation.w = q.w();
    } else {
        pose.orientation.z = 0.0;
        pose.orientation.w = 1.0;
    }
}



void GraphSlam::enforcePlanarTransform(geometry_msgs::msg::Transform& tf) const
{
    if (!usePlanarConstraint())
    {
        return;
    }

    tf.translation.z = 0.0;
    
    // In ROS body frame (base_link / Z-up), turning on the road is rotation around Z (Yaw).
    // Project quaternion onto pure Z-axis rotation by zeroing out X (roll) and Y (pitch).
    tf.rotation.x = 0.0;
    tf.rotation.y = 0.0;
    tf2::Quaternion q(0.0, 0.0, tf.rotation.z, tf.rotation.w);
    if (q.length2() > 1e-6) {
        q.normalize();
        tf.rotation.z = q.z();
        tf.rotation.w = q.w();
    } else {
        tf.rotation.z = 0.0;
        tf.rotation.w = 1.0;
    }
}

void GraphSlam::optimizePoseGraph(const std::string& reason,
                                  int trigger_from_keyframe_id,
                                  int trigger_to_keyframe_id)
{
    // Hold the main mutex for the entire optimization so the
    // background thread cannot race against process_image_frame / odomCallback
    // which also write to graph_, landmarks_, and vision_integrated_pose_.
    // recursive_mutex is safe here: synchronizeVisionStateWithGraphTail() also
    // acquires it but is called at the end of this function.
    std::unique_lock<std::recursive_mutex> lock(mutex_);

    if (!optimizer_)
    {
        RCLCPP_WARN(this->get_logger(), "No optimizer configured. Skipping global optimization.");
        return;
    }

    auto pre_opt_nodes = graph_->getNodes();
    std::unordered_map<int, geometry_msgs::msg::Pose> pre_opt_poses;
    pre_opt_poses.reserve(pre_opt_nodes.size());
    for (const auto& [id, node] : pre_opt_nodes)
    {
        if (node)
        {
            pre_opt_poses.emplace(id, node->getPose());
        }
    }

    // Record the tail pose BEFORE optimization so we can measure the correction
    // applied to the most-recent keyframe.
    const geometry_msgs::msg::Pose pre_opt_tail = graph_->getLastNodePosition();

    // Hold mutex during the entire optimization to prevent data races on graph_
    bool opt_success = optimizer_->optimizePoseGraph(*graph_, global_ba_iterations_);

    if (!opt_success)
    {
        return;
    }

    if (usePlanarConstraint())
    {
        for (const auto& [id, node] : graph_->getNodes())
        {
            if (!node)
            {
                continue;
            }
            auto planar_pose = node->getPose();
            enforcePlanarPose(planar_pose);
            graph_->updateNodePose(id, planar_pose);
        }
    }

    // Measure how far the tail keyframe moved as a result of this optimization.
    const geometry_msgs::msg::Pose post_opt_tail = graph_->getLastNodePosition();
    const double tail_dx = post_opt_tail.position.x - pre_opt_tail.position.x;
    const double tail_dy = post_opt_tail.position.y - pre_opt_tail.position.y;
    const double tail_dz = post_opt_tail.position.z - pre_opt_tail.position.z;
    const double tail_jump_m = std::sqrt(tail_dx * tail_dx + tail_dy * tail_dy + tail_dz * tail_dz);

    RCLCPP_INFO(this->get_logger(),
                "Pose graph optimized: optimizer=%s nodes=%zu edges=%zu loops=%d tail_jump=%.3fm",
                optimizer_->name().c_str(),
                graph_->getNodes().size(),
                graph_->getEdges().size(),
                loop_closure_count_,
                tail_jump_m);

    // Hard safety: if correction is too large, rollback the full optimization.
    // Exception: Do not rollback if this is the "VINS Initialized" / "Global Alignment Triggered" global alignment,
    // because that intentionally rotates the entire graph by 90 degrees to align with gravity.
    if (reason != "VINS Initialized" && reason != "Global Alignment Triggered" && tail_jump_m > max_optimization_jump_m_)
    {
        for (const auto& [id, pose] : pre_opt_poses)
        {
            graph_->updateNodePose(id, pose);
        }

        if (reason.rfind("loop_closure", 0) == 0 &&
            trigger_from_keyframe_id >= 0 && trigger_to_keyframe_id >= 0)
        {
            const bool removed = graph_->removeLatestEdge(trigger_from_keyframe_id,
                                                          trigger_to_keyframe_id,
                                                          GraphEdge::Type::LOOP_CLOSURE);
            accepted_loop_edges_.erase(makeDirectedEdgeKey(trigger_from_keyframe_id,
                                                           trigger_to_keyframe_id));
            if (loop_closure_count_ > 0)
            {
                --loop_closure_count_;
            }

            RCLCPP_WARN(this->get_logger(),
                        "Optimization rollback: triggering loop %d -> %d rejected (%s)",
                        trigger_from_keyframe_id,
                        trigger_to_keyframe_id,
                        removed ? "edge removed" : "edge not found");
        }

        RCLCPP_WARN(this->get_logger(),
                    "Optimization jump too large (%.2fm > %.2fm threshold). "
                    "Optimization result discarded and graph restored.",
                    tail_jump_m, max_optimization_jump_m_);
        return;
    }

    for (const auto &[id, node] : graph_->getNodes())
    {
        if (!node)
        {
            continue;
        }

        map::Keyframe keyframe_record;
        keyframe_record.id = id;
        keyframe_record.pose = node->getPose();
        keyframe_record.stamp_ns = node->getTimestamp().nanoseconds();
        map_store_.upsertKeyframe(keyframe_record);
    }

    if (full_ba_on_loop_closure_ && reason.rfind("loop_closure", 0) == 0)
    {
        optimizeFullBundleAdjustment();
    }


}

void GraphSlam::optimizeFullBundleAdjustment()
{
    if (graph_->getNodes().size() < 3)
    {
        return;
    }

    std::vector<int> all_ids;
    all_ids.reserve(graph_->getNodes().size());
    for (const auto& [id, node] : graph_->getNodes())
    {
        if (node)
        {
            all_ids.push_back(id);
        }
    }
    std::sort(all_ids.begin(), all_ids.end());

    const int iterations = std::max(local_ba_iterations_, full_ba_iterations_);

    RCLCPP_INFO(this->get_logger(),
                "Running optional full BA on loop closure (keyframes=%zu, iterations=%d)",
                all_ids.size(),
                iterations);

    optimizeLocalPoseGraph(&all_ids, iterations);
}

void GraphSlam::optimizeLocalPoseGraph(const std::vector<int>* custom_window_ids, int custom_iterations)
{
    // Same data-race fix as optimizePoseGraph.
    std::unique_lock<std::recursive_mutex> lock(mutex_);

    std::vector<int> target_ids;
    if (custom_window_ids)
    {
        target_ids = *custom_window_ids;
    }
    else
    {
        target_ids.assign(recent_keyframe_ids_.begin(), recent_keyframe_ids_.end());
    }
    const int iterations = (custom_iterations > 0) ? custom_iterations : local_ba_iterations_;

    if ((!custom_window_ids && !local_mapping_enabled_) || target_ids.size() < 2)
    {
        return;
    }

    std::vector<int> window_ids;
    window_ids.reserve(target_ids.size());
    for (int id : target_ids)
    {
        if (graph_->getNode(id))
        {
            window_ids.push_back(id);
        }
    }

    if (window_ids.size() < 2)
    {
        return;
    }

    if (window_ids.size() < static_cast<size_t>(std::max(2, local_ba_min_window_kfs_)))
    {
        return;
    }

    if (!local_ba_initialized_)
    {
        const auto first_node = graph_->getNode(window_ids.front());
        const auto tail_node = graph_->getNode(window_ids.back());
        if (!first_node || !tail_node)
        {
            return;
        }

        const auto& p0 = first_node->getPose().position;
        const auto& p1 = tail_node->getPose().position;
        const double dx = p1.x - p0.x;
        const double dy = p1.y - p0.y;
        const double dz = p1.z - p0.z;
        const double baseline_m = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double mean_flow_px = latest_vo_result_ ? latest_vo_result_->mean_flow_px : 0.0;

        const bool enough_kfs = window_ids.size() >= static_cast<size_t>(std::max(2, local_ba_init_min_keyframes_));

        if (!enough_kfs)
        {
            return;
        }

        local_ba_initialized_ = true;
        RCLCPP_INFO(this->get_logger(),
                    "Local BA initialized: kfs=%zu baseline=%.2fm mean_flow=%.2fpx",
                    window_ids.size(),
                    baseline_m,
                    mean_flow_px);
    }

    // VO quality gates removed to ensure Local BA runs regularly and prevents drift!

    const int tail_id_for_gate = window_ids.back();
    if (tail_id_for_gate <= local_ba_skip_until_kf_id_)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                             "Local BA cooldown active: tail=%d skip_until=%d",
                             tail_id_for_gate,
                             local_ba_skip_until_kf_id_);
        return;
    }

    // In mono_imu mode, local bundle adjustment uses TightlyCoupledLocalBA (below at line ~1150)
    // when camera-base reprojection is available, optimizing 3D landmarks + IMU preintegration.

    // Capture window poses before optimization for rollback check.
    std::unordered_map<int, geometry_msgs::msg::Pose> pre_opt_window_poses;
    pre_opt_window_poses.reserve(window_ids.size());
    for (int id : window_ids)
    {
        auto node = graph_->getNode(id);
        if (node)
        {
            pre_opt_window_poses.emplace(id, node->getPose());
        }
    }

    std::unordered_set<int> window_set(window_ids.begin(), window_ids.end());

    g2o::SparseOptimizer optimizer;
    auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto block_solver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
    optimizer.setAlgorithm(algorithm);

    auto *camera = new g2o::CameraParameters(camera_matrix_.at<double>(0, 0),
                                             Eigen::Vector2d(camera_matrix_.at<double>(0, 2), camera_matrix_.at<double>(1, 2)),
                                             0.0);
    camera->setId(0);
    optimizer.addParameter(camera);

    // ORB-SLAM-style incremental local BA:
    // Fix all KFs except the most recent local_ba_free_kfs_.
    // Older fixed KFs act as stable 3D anchors for landmark positions,
    // preventing monocular scale drift when the window is large.
    const int n_window = static_cast<int>(window_ids.size());
    const int n_free   = std::min(std::max(1, local_ba_free_kfs_), n_window - 1); // free ≥1, always fix ≥1
    const int n_fixed  = n_window - n_free;
    std::unordered_set<int> fixed_kf_ids;
    std::unordered_set<int> free_kf_ids;
    fixed_kf_ids.reserve(static_cast<size_t>(n_fixed));
    free_kf_ids.reserve(static_cast<size_t>(n_free));

    int pose_vertex_count = 0;
    int pose_vertex_idx = 0;
    for (int id : window_ids)
    {
        auto node = graph_->getNode(id);
        if (!node)
        {
            pose_vertex_idx++;
            continue;
        }

        const auto &pose = node->getPose();
        Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
        Eigen::Vector3d t(pose.position.x, pose.position.y, pose.position.z);

        auto *vertex = new g2o::VertexSE3Expmap();
        vertex->setId(id);
        vertex->setEstimate(g2o::SE3Quat(q, t));
        const bool is_fixed_kf = (pose_vertex_idx < n_fixed);
        vertex->setFixed(is_fixed_kf);  // fix oldest n_fixed, free newest n_free
        optimizer.addVertex(vertex);
        if (is_fixed_kf)
        {
            fixed_kf_ids.insert(id);
        }
        else
        {
            free_kf_ids.insert(id);
        }
        pose_vertex_count++;
        pose_vertex_idx++;
    }

    int graph_edge_count = 0;
    for (const auto &edge : graph_->getEdges())
    {
        if (window_set.find(edge->getFromId()) == window_set.end() || window_set.find(edge->getToId()) == window_set.end())
        {
            continue;
        }

        // Do not add IMU edges to g2o! g2o uses EdgeSE3Expmap which expects a visual odometry transform.
        // IMU edges don't use the relative_transform field, so adding them constraints nodes to Identity (0 motion).
        if (edge->getType() == GraphEdge::Type::IMU_PREINTEGRATION)
        {
            continue;
        }

        auto *from = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(edge->getFromId()));
        auto *to = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(edge->getToId()));
        if (!from || !to)
        {
            continue;
        }

        // WEAKNESS2-FIX: We must use the true measurement (which now correctly
        // stores the base-frame relative transform AND the injected IMU rotation)
        // instead of recovering the measurement from the graph poses. Recovering
        // from graph poses causes the optimizer to anchor against drifted BA states!
        const auto &rel_tf = edge->getRelativeTransform();
        Eigen::Quaterniond rel_q(rel_tf.rotation.w, rel_tf.rotation.x, rel_tf.rotation.y, rel_tf.rotation.z);
        Eigen::Vector3d rel_t(rel_tf.translation.x, rel_tf.translation.y, rel_tf.translation.z);
        const g2o::SE3Quat measurement(rel_q, rel_t);

        auto *g2o_edge = new g2o::EdgeSE3Expmap();
        g2o_edge->setVertex(0, from);
        g2o_edge->setVertex(1, to);
        g2o_edge->setMeasurement(measurement);
        g2o_edge->setInformation(edge->getInformationMatrix());
        optimizer.addEdge(g2o_edge);
        graph_edge_count++;
    }

    constexpr int landmark_vertex_offset = 1000000;
    std::unordered_map<int, int> landmark_to_vertex_id;
    int landmark_vertex_count = 0;
    int reprojection_edge_count = 0;
    int anchored_landmark_count = 0;
    int rejected_unanchored_landmark_count = 0;
    int rejected_low_parallax_landmark_count = 0;
    int rejected_high_residual_obs_count = 0;

    bool can_use_reprojection = false;
    Eigen::Isometry3d base_T_camera = Eigen::Isometry3d::Identity();
    if (local_ba_use_reprojection_)
    {
        try
        {
            const auto t_base_cam_msg = tf_buffer_->lookupTransform(base_frame_id_, camera_frame_id_, tf2::TimePointZero);
            tf2::Transform tf_base_camera;
            tf2::fromMsg(t_base_cam_msg.transform, tf_base_camera);

            tf2::Matrix3x3 basis = tf_base_camera.getBasis();
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 3; ++c)
                {
                    base_T_camera.linear()(r, c) = basis[r][c];
                }
            }
            const auto origin = tf_base_camera.getOrigin();
            base_T_camera.translation() = Eigen::Vector3d(origin.x(), origin.y(), origin.z());
            can_use_reprojection = true;
        }
        catch (const tf2::TransformException& ex)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                                 "Local BA reprojection disabled: missing base<->camera TF (%s)",
                                 ex.what());
        }
    }

    if (mode_ == "mono_imu" && can_use_reprojection)
    {
        std::vector<vins::LocalBaLandmark> ba_landmarks;
        int anchored_landmark_count = 0;
        int rejected_unanchored_landmark_count = 0;
        int rejected_low_parallax_landmark_count = 0;

        for (auto &[landmark_id, landmark] : map_manager_->getLandmarks())
        {
            if (landmark.bad) continue;

            int observations_in_window = 0;
            int observations_in_fixed = 0;
            int observations_in_free = 0;
            std::vector<cv::Point2f> fixed_pixels;
            std::vector<cv::Point2f> free_pixels;
            
            vins::LocalBaLandmark ba_lm;
            ba_lm.id = landmark_id;
            ba_lm.position = landmark.position;

            for (const auto &observation : landmark.observations)
            {
                if (window_set.find(observation.keyframe_id) != window_set.end())
                {
                    observations_in_window++;
                    ba_lm.observations.push_back({observation.keyframe_id, Eigen::Vector2d(observation.pixel.x, observation.pixel.y)});

                    if (fixed_kf_ids.find(observation.keyframe_id) != fixed_kf_ids.end())
                    {
                        observations_in_fixed++;
                        fixed_pixels.push_back(observation.pixel);
                    }
                    if (free_kf_ids.find(observation.keyframe_id) != free_kf_ids.end())
                    {
                        observations_in_free++;
                        free_pixels.push_back(observation.pixel);
                    }
                }
            }

            if (observations_in_window < std::max(2, local_ba_min_obs_per_landmark_window_)) continue;

            if (local_ba_pose_only_ && (observations_in_fixed == 0 || observations_in_free == 0))
            {
                rejected_unanchored_landmark_count++;
                continue;
            }

            double max_cross_parallax_sq = 0.0;
            for (const auto& pf : fixed_pixels)
            {
                for (const auto& pr : free_pixels)
                {
                    const double ddx = static_cast<double>(pf.x - pr.x);
                    const double ddy = static_cast<double>(pf.y - pr.y);
                    max_cross_parallax_sq = std::max(max_cross_parallax_sq, ddx * ddx + ddy * ddy);
                }
            }
            if (std::sqrt(max_cross_parallax_sq) < std::max(1.0, local_ba_min_landmark_parallax_px_))
            {
                rejected_low_parallax_landmark_count++;
                continue;
            }

            if (local_ba_pose_only_) anchored_landmark_count++;

            ba_landmarks.push_back(ba_lm);
        }

        if (!ba_landmarks.empty() && (!local_ba_pose_only_ || anchored_landmark_count >= std::max(2, local_ba_min_anchored_landmarks_)))
        {
            double huber_delta = 4.0;
            double reproj_info = 0.25 * std::max(0.05, local_ba_reprojection_info_scale_);
            
            bool success = vins::TightlyCoupledLocalBA::optimizeWindow(
                *graph_, window_ids, fixed_kf_ids, ba_landmarks,
                camera_matrix_, base_T_camera, reproj_info, huber_delta,
                local_ba_enable_height_prior_, local_ba_height_prior_value_, local_ba_height_prior_stddev_);

            if (success) {
                for (const auto& ba_lm : ba_landmarks) {
                    map_manager_->getLandmarks()[ba_lm.id].position = ba_lm.position;
                }
                for (int id : window_ids) {
                    auto node = graph_->getNode(id);
                    if (node) {
                        map::Keyframe kf;
                        kf.id = id;
                        kf.pose = node->getPose();
                        kf.stamp_ns = node->getTimestamp().nanoseconds();
                        map_store_.upsertKeyframe(kf);
                    }
                }
                RCLCPP_INFO(this->get_logger(), "[TightlyCoupled Local BA] lm_in_window=%zu", ba_landmarks.size());
            } else {
                RCLCPP_WARN(this->get_logger(), "[TightlyCoupled Local BA] Failed or diverged.");
            }

            // If VINS metric scale is not yet initialized, run global pose graph
            // optimization so early unscaled VO frames stay globally consistent until
            // processGlobalAlignment() triggers. Once metric scale is initialized,
            // restrict local keyframe updates strictly to the sliding window!
            if (!graph_->isMetricScaleInitialized() && optimizer_) {
                optimizer_->optimizePoseGraph(*graph_, local_ba_iterations_);
            }
            return;
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "[TightlyCoupled Local BA] Skipped (immature geometry: %zu landmarks, %d anchored). Falling back to pose-graph optimization.", ba_landmarks.size(), anchored_landmark_count);
            if (!graph_->isMetricScaleInitialized() && optimizer_) {
                optimizer_->optimizePoseGraph(*graph_, local_ba_iterations_);
            }
            return;
        }
    }
    else if (mode_ == "mono_imu" && optimizer_)
    {
        // Fallback if camera-base reprojection TF is not available
        if (!graph_->isMetricScaleInitialized()) {
            optimizer_->optimizePoseGraph(*graph_, local_ba_iterations_);
        }
        return;
    }

    if (can_use_reprojection)
    {
        for (auto &[landmark_id, landmark] : map_manager_->getLandmarks())
        {
            if (landmark.bad)
            {
                continue;
            }

            int observations_in_window = 0;
            int observations_in_fixed = 0;
            int observations_in_free = 0;
            std::vector<cv::Point2f> fixed_pixels;
            std::vector<cv::Point2f> free_pixels;
            for (const auto &observation : landmark.observations)
            {
                if (window_set.find(observation.keyframe_id) != window_set.end())
                {
                    observations_in_window++;
                    if (fixed_kf_ids.find(observation.keyframe_id) != fixed_kf_ids.end())
                    {
                        observations_in_fixed++;
                        fixed_pixels.push_back(observation.pixel);
                    }
                    if (free_kf_ids.find(observation.keyframe_id) != free_kf_ids.end())
                    {
                        observations_in_free++;
                        free_pixels.push_back(observation.pixel);
                    }
                }
            }

            if (observations_in_window < std::max(2, local_ba_min_obs_per_landmark_window_))
            {
                continue;
            }

            // Critical monocular stabilization:
            // In pose-only BA, only keep landmarks that connect fixed↔free KFs.
            // Landmarks observed only in free KFs can drag the whole free cluster
            // and re-introduce jumpy gauge-like motion.
            if (local_ba_pose_only_ && (observations_in_fixed == 0 || observations_in_free == 0))
            {
                rejected_unanchored_landmark_count++;
                continue;
            }

            // Low-parallax landmarks are weakly observable for translation and
            // are a common source of unstable monocular BA updates.
            double max_cross_parallax_sq = 0.0;
            for (const auto& pf : fixed_pixels)
            {
                for (const auto& pr : free_pixels)
                {
                    const double ddx = static_cast<double>(pf.x - pr.x);
                    const double ddy = static_cast<double>(pf.y - pr.y);
                    max_cross_parallax_sq = std::max(max_cross_parallax_sq, ddx * ddx + ddy * ddy);
                }
            }
            if (std::sqrt(max_cross_parallax_sq) < std::max(1.0, local_ba_min_landmark_parallax_px_))
            {
                rejected_low_parallax_landmark_count++;
                continue;
            }

            if (local_ba_pose_only_)
            {
                anchored_landmark_count++;
            }

            const int vertex_id = landmark_vertex_offset + landmark_id;
            auto *landmark_vertex = new g2o::VertexPointXYZ();
            landmark_vertex->setId(vertex_id);
            landmark_vertex->setEstimate(landmark.position);
            // Pose-only BA: fix landmarks so only KF poses are optimized.
            // This eliminates the monocular translation gauge freedom that causes
            // the optimizer to shift the entire free-KF cluster by N keyframe steps.
            // When pose_only=false (full BA), Schur marginalization is used for speed.
            landmark_vertex->setFixed(local_ba_pose_only_);
            landmark_vertex->setMarginalized(!local_ba_pose_only_);
            optimizer.addVertex(landmark_vertex);

            landmark_to_vertex_id[landmark_id] = vertex_id;
            landmark_vertex_count++;

            for (const auto &observation : landmark.observations)
            {
                if (window_set.find(observation.keyframe_id) == window_set.end())
                {
                    continue;
                }

                auto *pose_vertex = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(observation.keyframe_id));
                auto *point_vertex = dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertex(vertex_id));
                if (!pose_vertex || !point_vertex)
                {
                    continue;
                }

                auto *projection_edge = new EdgeProjectXYZ2UVBasePose(
                    camera_matrix_.at<double>(0, 0),
                    camera_matrix_.at<double>(1, 1),
                    camera_matrix_.at<double>(0, 2),
                    camera_matrix_.at<double>(1, 2),
                    base_T_camera);
                projection_edge->setVertex(0, point_vertex);
                projection_edge->setVertex(1, pose_vertex);
                projection_edge->setMeasurement(Eigen::Vector2d(observation.pixel.x, observation.pixel.y));

                // Reject obvious outliers before optimization (depth/initial residual gate).
                const g2o::SE3Quat w_T_b = pose_vertex->estimate();
                Eigen::Isometry3d w_T_b_iso = Eigen::Isometry3d::Identity();
                w_T_b_iso.linear() = w_T_b.rotation().toRotationMatrix();
                w_T_b_iso.translation() = w_T_b.translation();
                const Eigen::Isometry3d w_T_c = w_T_b_iso * base_T_camera;
                const Eigen::Vector3d p_c = w_T_c.inverse() * point_vertex->estimate();
                if (!p_c.allFinite() || p_c.z() <= 1e-6)
                {
                    rejected_high_residual_obs_count++;
                    continue;
                }

                const double u = camera_matrix_.at<double>(0, 0) * (p_c.x() / p_c.z()) + camera_matrix_.at<double>(0, 2);
                const double v = camera_matrix_.at<double>(1, 1) * (p_c.y() / p_c.z()) + camera_matrix_.at<double>(1, 2);
                const double du = u - observation.pixel.x;
                const double dv = v - observation.pixel.y;
                const double kMaxInitialReprojErrPx = std::max(2.0, local_ba_max_initial_reproj_err_px_);
                if ((du * du + dv * dv) > (kMaxInitialReprojErrPx * kMaxInitialReprojErrPx))
                {
                    rejected_high_residual_obs_count++;
                    continue;
                }

                const double reproj_info = 0.25 * std::max(0.05, local_ba_reprojection_info_scale_);
                projection_edge->setInformation(reproj_info * Eigen::Matrix2d::Identity());

                auto *kernel = new g2o::RobustKernelHuber();
                kernel->setDelta(2.4477);
                projection_edge->setRobustKernel(kernel);

                optimizer.addEdge(projection_edge);
                reprojection_edge_count++;
            }
        }
    }

    if (pose_vertex_count < 2 || (graph_edge_count == 0 && reprojection_edge_count < 10))
    {
        return;
    }

    // Robustness gate for monocular pose-only BA.
    // If anchors are sparse, reprojection terms can drag the free tail cluster.
    if (can_use_reprojection && local_ba_pose_only_)
    {
        // Diagnostic: always show landmark population so we can verify the fix
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "[Local BA] lm_in_window=%d anchored=%d unanchored_rej=%d low_parallax_rej=%d "
                             "reproj_edges=%d  (need anchored>=%d edges>=%d)",
                             (anchored_landmark_count + rejected_unanchored_landmark_count +
                              rejected_low_parallax_landmark_count),
                             anchored_landmark_count,
                             rejected_unanchored_landmark_count,
                             rejected_low_parallax_landmark_count,
                             reprojection_edge_count,
                             local_ba_min_anchored_landmarks_,
                             local_ba_min_reprojection_edges_);

        if (anchored_landmark_count < local_ba_min_anchored_landmarks_ ||
            reprojection_edge_count < local_ba_min_reprojection_edges_)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                                 "Local BA skipped (immature geometry): anchored_lm=%d (<%d), reproj_edges=%d (<%d)",
                                 anchored_landmark_count,
                                 local_ba_min_anchored_landmarks_,
                                 reprojection_edge_count,
                                 local_ba_min_reprojection_edges_);
            return;
        }
    }

    optimizer.initializeOptimization();
    lock.unlock();
    optimizer.optimize(iterations);
    lock.lock();

    // ── Rollback check BEFORE writing to graph ──────────────────────────────
    // Measure the delta on the window's newest free vertex (window tail)
    // directly from the optimizer result — never from the graph which hasn't
    // been written yet.  This avoids a race where a newer KF inserted between
    // the BA signal and execution is already the graph tail, making the jump
    // appear zero and bypassing the rollback.
    {
        const int tail_id = window_ids.back();
        auto *tail_vertex = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(tail_id));
        if (tail_vertex)
        {
            Eigen::Vector3d t_post = tail_vertex->estimate().translation();
            const auto &pre_tail_pose = pre_opt_window_poses.at(tail_id);
            double dx = t_post.x() - pre_tail_pose.position.x;
            double dy = t_post.y() - pre_tail_pose.position.y;
            double dz = t_post.z() - pre_tail_pose.position.z;
            double tail_jump_m = std::sqrt(dx*dx + dy*dy + dz*dz);

            // Translation step clamp: limit BA translation correction to keep
            // updates stable under poor observability.
            const double tail_step_cap_m = std::max(0.05, local_ba_max_tail_step_m_);
            if (tail_jump_m > tail_step_cap_m)
            {
                const double scale = tail_step_cap_m / std::max(1e-9, tail_jump_m);
                for (int id : free_kf_ids)
                {
                    auto* v = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(id));
                    auto it = pre_opt_window_poses.find(id);
                    if (!v || it == pre_opt_window_poses.end())
                    {
                        continue;
                    }

                    const auto& pre = it->second;
                    const Eigen::Vector3d t_pre(pre.position.x, pre.position.y, pre.position.z);
                    const g2o::SE3Quat est = v->estimate();
                    const Eigen::Vector3d t_new = t_pre + scale * (est.translation() - t_pre);
                    v->setEstimate(g2o::SE3Quat(est.rotation(), t_new));
                }

                t_post = tail_vertex->estimate().translation();
                dx = t_post.x() - pre_tail_pose.position.x;
                dy = t_post.y() - pre_tail_pose.position.y;
                dz = t_post.z() - pre_tail_pose.position.z;
                tail_jump_m = std::sqrt(dx*dx + dy*dy + dz*dz);

                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                                     "Local BA translation step clamped: tail_jump=%.2fm cap=%.2fm",
                                     tail_jump_m,
                                     tail_step_cap_m);
            }

            constexpr double kJumpGateEps = 1e-3;
            if (tail_jump_m > (max_optimization_jump_m_ + kJumpGateEps))
            {
                RCLCPP_WARN(this->get_logger(),
                            "Local BA rollback: tail jump %.2fm exceeds threshold %.2fm",
                            tail_jump_m,
                            max_optimization_jump_m_);

                local_ba_skip_until_kf_id_ =
                    std::max(local_ba_skip_until_kf_id_, tail_id + std::max(1, local_ba_rollback_cooldown_kfs_));

                local_ba_rollback_count_++;
                if (local_ba_use_reprojection_ &&
                    local_ba_disable_reprojection_after_rollbacks_ > 0 &&
                    local_ba_rollback_count_ >= local_ba_disable_reprojection_after_rollbacks_)
                {
                    local_ba_use_reprojection_ = false;
                    RCLCPP_WARN(this->get_logger(),
                                "Local BA reprojection auto-disabled after %d rollbacks (stability circuit breaker).",
                                local_ba_rollback_count_);
                }

                return;  // graph untouched — nothing to restore
            }
        }
    }

    // ── Commit optimized poses to graph ─────────────────────────────────────
    for (int id : window_ids)
    {
        auto *vertex = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(id));
        if (!vertex)
        {
            continue;
        }

        const g2o::SE3Quat est = vertex->estimate();
        const Eigen::Vector3d t = est.translation();
        const Eigen::Quaterniond q = est.rotation();

        geometry_msgs::msg::Pose optimized_pose;
        optimized_pose.position.x = t.x();
        optimized_pose.position.y = t.y();
        optimized_pose.position.z = t.z();
        optimized_pose.orientation.x = q.x();
        optimized_pose.orientation.y = q.y();
        optimized_pose.orientation.z = q.z();
        optimized_pose.orientation.w = q.w();

        enforcePlanarPose(optimized_pose);

        graph_->updateNodePose(id, optimized_pose);

        map::Keyframe keyframe_record;
        keyframe_record.id = id;
        keyframe_record.pose = optimized_pose;
        auto optimized_node = graph_->getNode(id);
        keyframe_record.stamp_ns = optimized_node ? optimized_node->getTimestamp().nanoseconds() : 0;
        map_store_.upsertKeyframe(keyframe_record);
    }



    for (auto &[landmark_id, vertex_id] : landmark_to_vertex_id)
    {
        auto *vertex = dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertex(vertex_id));
        if (!vertex)
        {
            continue;
        }
        // Only write back when landmarks were actually free (full BA mode).
        if (!local_ba_pose_only_)
        {
            map_manager_->getLandmarks()[landmark_id].position = vertex->estimate();

            map::Landmark map_landmark;
            map_landmark.id = map_manager_->getLandmarks()[landmark_id].id;
            map_landmark.position = map_manager_->getLandmarks()[landmark_id].position;
            map_landmark.bad = map_manager_->getLandmarks()[landmark_id].bad;
            map_landmark.observation_count = map_manager_->getLandmarks()[landmark_id].observation_count;
            map_store_.upsertLandmark(map_landmark);
        }
    }

    // RCLCPP_INFO(this->get_logger(),
    //             "Local BA done (window_kf=%zu, fixed=%d free=%d, landmarks=%d, reproj_edges=%d, pose_only=%s, anchored_lm=%d, lm_unanchored_rej=%d, obs_rej=%d)",
    //             window_ids.size(),
    //             n_fixed,
    //             n_free,
    //             landmark_vertex_count,
    //             reprojection_edge_count,
    //             local_ba_pose_only_ ? "true" : "false",
    //             anchored_landmark_count,
    //             rejected_unanchored_landmark_count,
    //             rejected_low_parallax_landmark_count,
    //             rejected_high_residual_obs_count);
    // (pose_only: landmarks were fixed, no writeback needed)
    (void)rejected_low_parallax_landmark_count;
}



geometry_msgs::msg::Pose GraphSlam::applyTransform(const geometry_msgs::msg::Pose &base,
                                                   const geometry_msgs::msg::Transform &delta)
{
    // delta.translation is expressed in the LOCAL (previous camera) frame.
    // Rotate it into the world frame using the current pose orientation before
    // accumulating into the world position.
    tf2::Quaternion q_base(base.orientation.x, base.orientation.y,
                           base.orientation.z, base.orientation.w);
    const tf2::Vector3 local_t(delta.translation.x,
                               delta.translation.y,
                               delta.translation.z);
    const tf2::Vector3 world_t = tf2::quatRotate(q_base, local_t);

    geometry_msgs::msg::Pose out = base;
    out.position.x = base.position.x + world_t.x();
    out.position.y = base.position.y + world_t.y();
    out.position.z = base.position.z + world_t.z();

    tf2::Quaternion q_delta(delta.rotation.x, delta.rotation.y,
                            delta.rotation.z, delta.rotation.w);
    tf2::Quaternion q_out = q_base * q_delta;
    q_out.normalize();

    out.orientation.x = q_out.x();
    out.orientation.y = q_out.y();
    out.orientation.z = q_out.z();
    out.orientation.w = q_out.w();
    return out;
}

void GraphSlam::printGraph()
{
    RCLCPP_INFO(this->get_logger(), "Nodes in the graph:");

    for (const auto &[id, node] : graph_->getNodes())
    {
        RCLCPP_INFO(this->get_logger(), "Node ID: %d", node->getId());
        RCLCPP_INFO(this->get_logger(), "Pose: [x: %.2f, y: %.2f, z: %.2f]",
                    node->getPose().position.x,
                    node->getPose().position.y,
                    node->getPose().position.z);

        tf2::Quaternion q = msgToTF2(node->getPose().orientation);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        RCLCPP_INFO(this->get_logger(), "Orientation: [roll: %.2f, pitch: %.2f, yaw: %.2f]",
                    roll, pitch, yaw);
        graph_->setGraphChanged(true);
    }

    RCLCPP_INFO(this->get_logger(), "Edges in the graph:");

    for (const auto &edge : graph_->getEdges())
    {
        RCLCPP_INFO(this->get_logger(), "Edge from Node %d to Node %d",
                    edge->getFromId(),
                    edge->getToId());

        const auto &transform = edge->getRelativeTransform();
        RCLCPP_INFO(this->get_logger(), "Transform: [x: %.2f, y: %.2f, z: %.2f]",
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z);

        tf2::Quaternion q = msgToTF2(transform.rotation);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        RCLCPP_INFO(this->get_logger(), "Rotation: [roll: %.2f, pitch: %.2f, yaw: %.2f]",
                    roll, pitch, yaw);
    }

    RCLCPP_INFO(this->get_logger(),
                "Landmarks in map: %zu (thread-safe-store=%zu)",
                map_manager_->getLandmarks().size(),
                map_store_.landmarkCount());
}

tf2::Quaternion GraphSlam::msgToTF2(const geometry_msgs::msg::Quaternion &msg)
{
    return tf2::Quaternion(msg.x, msg.y, msg.z, msg.w);
}

geometry_msgs::msg::Quaternion GraphSlam::tf2ToMsg(const tf2::Quaternion &tf2)
{
    geometry_msgs::msg::Quaternion msg;
    msg.x = tf2.x();
    msg.y = tf2.y();
    msg.z = tf2.z();
    msg.w = tf2.w();
    return msg;
}


void GraphSlam::signalOptimization(bool global,
                                   const char* reason,
                                   int from_keyframe_id,
                                   int to_keyframe_id)
{
    {
        std::lock_guard<std::mutex> lock(optimization_mutex_);
        const bool already_pending = global ? global_optimization_requested_ : local_optimization_requested_;
        const uint64_t request_id  = ++optimization_request_count_;

        if (global) {
            global_optimization_requested_ = true;
            pending_global_reason_ = reason ? reason : "unspecified";
        } else {
            local_optimization_requested_ = true;
            pending_local_reason_ = reason ? reason : "unspecified";
            pending_local_from_keyframe_id_ = from_keyframe_id;
            pending_local_to_keyframe_id_ = to_keyframe_id;
        }

        RCLCPP_DEBUG(this->get_logger(),
                     "Optimization requested: id=%lu type=%s reason=%s edge=%d->%d already_pending=%s",
                     static_cast<unsigned long>(request_id),
                     global ? "global" : "local",
                     reason ? reason : "unspecified",
                     from_keyframe_id,
                     to_keyframe_id,
                     already_pending ? "yes" : "no");
    }
    optimization_cv_.notify_one();
}

void GraphSlam::runOptimizationLoop()
{
    RCLCPP_INFO(this->get_logger(), "Background optimization thread started");
    while (run_optimization_thread_) {
        bool do_global = false;
        bool do_local = false;
        std::string global_reason;
        std::string local_reason;
        int local_from_id = -1;
        int local_to_id = -1;
        uint64_t run_id = 0;

        {
            std::unique_lock<std::mutex> lock(optimization_mutex_);
            optimization_cv_.wait_for(lock, std::chrono::milliseconds(500), [this]() {
                return !run_optimization_thread_ || global_optimization_requested_ || local_optimization_requested_;
            });

            if (!run_optimization_thread_) break;

            if (global_optimization_requested_) {
                do_global = true;
                global_optimization_requested_ = false;
                global_reason = pending_global_reason_.empty() ? "unspecified" : pending_global_reason_;
                pending_global_reason_.clear();
            }
            if (local_optimization_requested_) {
                do_local = true;
                local_optimization_requested_ = false;
                local_reason = pending_local_reason_.empty() ? "unspecified" : pending_local_reason_;
                local_from_id = pending_local_from_keyframe_id_;
                local_to_id = pending_local_to_keyframe_id_;
                pending_local_reason_.clear();
                pending_local_from_keyframe_id_ = -1;
                pending_local_to_keyframe_id_ = -1;
            }

            run_id = ++optimization_run_count_;
        }

        if (do_global) {
            RCLCPP_INFO(this->get_logger(),
                        "Optimization worker: run=%lu executing global reason=%s",
                        static_cast<unsigned long>(run_id),
                        global_reason.c_str());
            optimizePoseGraph(global_reason);
            updateVisualization();
        }
        
        if (do_local) {
            RCLCPP_DEBUG(this->get_logger(),
                         "Optimization worker: run=%lu executing local reason=%s edge=%d->%d",
                         static_cast<unsigned long>(run_id),
                         local_reason.c_str(), local_from_id, local_to_id);
            {
                std::lock_guard<std::recursive_mutex> lock(mutex_);
                recent_keyframe_ids_.clear();
                if (local_from_id >= 0 && local_to_id >= local_from_id) {
                    for (int id = local_from_id; id <= local_to_id; ++id) {
                        recent_keyframe_ids_.push_back(id);
                    }
                }
            }
            optimizeLocalPoseGraph();
            updateVisualization();
        }
    }
}

GraphSlam::~GraphSlam()
{
    run_optimization_thread_ = false;
    run_mapping_thread_ = false;
    optimization_cv_.notify_all();
    queue_cv_.notify_all();
    
    if (optimization_thread_.joinable()) {
        optimization_thread_.join();
    }
    
    if (mapping_thread_.joinable()) {
        mapping_thread_.join();
    }
}
}
