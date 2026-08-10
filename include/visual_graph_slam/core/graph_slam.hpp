// graph_slam.hpp

#ifndef GRAPH_SLAM_HPP
#define GRAPH_SLAM_HPP

#include <rclcpp/rclcpp.hpp>
#include <pluginlib/class_loader.hpp>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <memory>
#include <optional>
#include <deque>
#include <queue>
#include <string>
#include <mutex>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <Eigen/Dense>
#include "visual_graph_slam/core/graph.hpp"
#include "visual_graph_slam/slam_visualizer.hpp"
#include "visual_graph_slam/sensor_data.hpp"
#include "visual_graph_slam/core/measurement_edge.hpp"
#include "visual_graph_slam/plugins/system.hpp"
#include "visual_graph_slam/map/map_manager.hpp"
#include "visual_graph_slam/backend/optimizer_backend.hpp"
#include "visual_graph_slam/sensor/camera_model.hpp"
#include "visual_graph_slam/map/thread_safe_map_store.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include "visual_graph_slam/vision_node.hpp" 
#include "visual_graph_slam/vins/imu_preintegrator.hpp"
#include "visual_graph_slam/vins/vins_initializer.hpp"

namespace slam {

class GraphSlam : public rclcpp::Node, public slam::core::GraphInterface {
public:
    GraphSlam();
    std::shared_ptr<Graph> getGraph() override { return graph_; }
    std::shared_ptr<slam::map::MapManager> getMapManager() override { return map_manager_; }
    void process_data(SensorData& sensor_data);
    void process_image_frame(sensor_msgs::msg::Image::ConstSharedPtr image,
                             const std::vector<sensor_msgs::msg::Imu>& imu_buffer = {});
    virtual ~GraphSlam();
    void updateCameraCalibration(const sensor_msgs::msg::CameraInfo& camera_info);
    void initializeVisualizer();
    void initializeSystem();


private:
    // Map and landmarks are now in MapManager

    void poseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    double quaternionAngularDistanceRad(const geometry_msgs::msg::Quaternion& a, const geometry_msgs::msg::Quaternion& b);
    
    /// Unified constraint-builder: Add any measurement as a pose-graph edge.
    /// Handles odometry, VO, IMU, external sensors, loop closures.
    /// Information matrix is auto-computed from source type + confidence.
    void submitMeasurement(const MeasurementEdgeConfig& config) override;
    
    // Legacy methods (deprecated, will be removed)
    // Use addMeasurementEdge() instead for all new code

    bool usePlanarConstraint() const;
    void initializePlanarReferenceFromPose(const geometry_msgs::msg::Pose& pose);
    void enforcePlanarPose(geometry_msgs::msg::Pose& pose) const;
    void enforcePlanarTransform(geometry_msgs::msg::Transform& tf) const;
    void printGraph();
    void updateVisualization();
    void optimizePoseGraph(const std::string& reason = "unspecified",
                           int trigger_from_keyframe_id = -1,
                           int trigger_to_keyframe_id = -1);
    void optimizeLocalPoseGraph(const std::vector<int>* custom_window_ids = nullptr,
                                int custom_iterations = -1);
    void optimizeFullBundleAdjustment();
    // Removed tracking logic and map logic
    static geometry_msgs::msg::Pose applyTransform(const geometry_msgs::msg::Pose& base,
                                                   const geometry_msgs::msg::Transform& delta);

    // Helper functions for quaternion operations
    static tf2::Quaternion msgToTF2(const geometry_msgs::msg::Quaternion& msg);
    static geometry_msgs::msg::Quaternion tf2ToMsg(const tf2::Quaternion& tf2);

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
    std::shared_ptr<Graph> graph_;
    std::unique_ptr<Visualizer> visualizer_;
    geometry_msgs::msg::Pose current_pose_;
    geometry_msgs::msg::Pose initial_pose_;
    rclcpp::Time current_time_;
    int current_node_id_{0};
    bool first_odom_received_{false};
    bool visualizer_initialized_{false};

    sensor_msgs::msg::Image::ConstSharedPtr latest_image_;
    std::optional<sensor_msgs::msg::Imu>     latest_imu_;
    std::vector<sensor_msgs::msg::Imu>       keyframe_imu_buffer_; ///< Accumulates IMU samples between keyframes for backend factors
    std::optional<VisionNode::FrontendResult> latest_vo_result_;  ///< most recent VO estimate

    double keyframe_distance_threshold_{0.5};

    int local_mapping_window_size_{15};
    int local_mapping_stride_{5};
    bool local_mapping_enabled_{true};
    int local_ba_iterations_{12};
    bool local_ba_use_reprojection_{true};
    bool local_ba_every_keyframe_{true};
    bool local_ba_pose_only_{true};  ///< Fix all landmarks, optimize only KF poses — eliminates monocular gauge freedom
    int local_ba_free_kfs_{3};    ///< How many newest KFs to keep free in local BA (older are fixed as landmark anchors)
    bool local_ba_initialized_{false};             ///< Warmup state: local BA starts only after stable initialization
    int local_ba_init_min_keyframes_{12};       ///< Minimum keyframes before local BA is enabled
    double local_ba_init_min_baseline_m_{0.18};   ///< Minimum baseline between oldest/newest local KFs before BA starts
    double local_ba_init_min_mean_flow_px_{1.2}; ///< Minimum frontend mean optical flow before BA starts
    int local_ba_min_window_kfs_{6};            ///< Skip local BA until this many keyframes are in the local window
    int local_ba_min_anchored_landmarks_{80};    ///< Minimum fixed↔free landmarks required for pose-only reprojection BA
    int local_ba_min_reprojection_edges_{200};    ///< Minimum reprojection edges required before local BA is trusted
    int local_ba_min_obs_per_landmark_window_{3}; ///< Minimum observations per landmark within BA window
    double local_ba_min_landmark_parallax_px_{8.0}; ///< Minimum fixed↔free landmark parallax to keep in BA
    double local_ba_max_initial_reproj_err_px_{8.0}; ///< Max initial reprojection error per observation for BA inclusion
    double local_ba_max_tail_step_m_{0.60};          ///< Max translation update on BA tail per solve (step clamp)
    double local_ba_pose_edge_translation_scale_{1.4};  ///< Scale factor for VO translation information in local BA
    double local_ba_reprojection_info_scale_{0.6};      ///< Scale factor for reprojection edge information in local BA
    int local_ba_rollback_cooldown_kfs_{20};           ///< Skip local BA for N keyframes after a rollback event
    int local_ba_skip_until_kf_id_{-1};                ///< Local BA disabled until this keyframe id (inclusive)
    int local_ba_rollback_count_{0};                  ///< Count rollback events in this run
    int local_ba_disable_reprojection_after_rollbacks_{2};  ///< Auto-disable reprojection BA after N rollbacks
    
    // Height Prior parameters
    bool local_ba_enable_height_prior_{false};
    double local_ba_height_prior_value_{0.0};
    double local_ba_height_prior_stddev_{0.01};

    bool full_ba_on_loop_closure_{false};
    int full_ba_iterations_{25};
    int global_ba_iterations_{15};
    int gtsam_local_call_counter_{0};   ///< Counts local-opt calls in mono_imu mode (used for throttling)
    int gtsam_local_call_interval_{5};  ///< Run GTSAM every N local-opt triggers in mono_imu mode
    int vins_min_keyframes_for_init_{20};
    double landmark_match_ratio_test_{0.75};
    int landmark_min_observations_{2};
    int landmark_prune_stride_{30};
    double landmark_min_depth_{0.1};
    double landmark_max_depth_{80.0};
    int landmark_min_triangulation_parallax_px_{3};
    int local_landmark_track_depth_{4};  ///< How many past KFs to match for landmark tracking (enables fixed↔free spanning)

    cv::Mat camera_matrix_;
    std::string camera_model_name_{"pinhole"};
    Eigen::Vector4d camera_distortion_{Eigen::Vector4d::Zero()};
    std::unique_ptr<sensor::ICameraModel> camera_model_{nullptr};
    int loop_closure_count_{0};
    std::deque<int> recent_keyframe_ids_;
    std::string optimizer_name_{"g2o"};
    std::unique_ptr<core::OptimizerBackend> optimizer_{nullptr};

    std::recursive_mutex mutex_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::string base_frame_id_;
    std::string camera_frame_id_;
    std::string imu_frame_id_;
    int frame_count_{0};
    bool imu_extrinsics_initialized_{false};
    Eigen::Quaterniond imu_extrinsic_q_;
    Eigen::Vector3d imu_extrinsic_t_;
    map::ThreadSafeMapStore map_store_;
    
    // Removed tracking logic

    std::shared_ptr<slam::map::MapManager> map_manager_;
    pluginlib::ClassLoader<slam::core::System> system_loader_;
    std::shared_ptr<slam::core::System> system_;

    std::string mode_{"mono_imu_wheel"};
    bool planar_motion_constraint_in_mono_{true};
    bool planar_reference_initialized_{false};
    double planar_reference_z_{0.0};
    double planar_reference_roll_rad_{0.0};
    double planar_reference_pitch_rad_{0.0};
    bool use_motion_priors_{true};
    std::string motion_prior_source_{"wheel_odometry"};
    double keyframe_time_threshold_sec_{0.35};

    // Maximum metres the most-recent keyframe is allowed to move as a result of
    // a global pose-graph optimisation.  If the jump exceeds this threshold the
    // live tracking pose is NOT snapped to the new tail — preventing sudden
    // visible jumps.
    double max_optimization_jump_m_{2.0};

    rclcpp::Time last_keyframe_time_;
    std::unordered_set<int64_t> accepted_loop_edges_;

    // Snapshot of keypoints/descriptors captured at processFrame time,
    // so addNode() gets features from the correct frame (not a later one).
    std::vector<cv::KeyPoint> pending_keyframe_keypoints_;
    cv::Mat                   pending_keyframe_descriptors_;


    // Removed VINS logic

    void runOptimizationLoop();
    void signalOptimization(bool global,
                            const char* reason = nullptr,
                            int from_keyframe_id = -1,
                            int to_keyframe_id = -1) override;
    void alignGraphWithGravity();

    std::thread optimization_thread_;
    std::mutex optimization_mutex_;
    std::condition_variable optimization_cv_;
    bool global_optimization_requested_;
    bool local_optimization_requested_;
    uint64_t optimization_request_count_{0};
    uint64_t optimization_run_count_{0};
    std::string pending_global_reason_;
    std::string pending_local_reason_;
    int pending_local_from_keyframe_id_{-1};
    int pending_local_to_keyframe_id_{-1};
    std::atomic<bool> run_optimization_thread_;

    // Thread Decoupling: Tracking vs Mapping (Tier 4)
    struct TrackingQueueItem {
        sensor_msgs::msg::Image::ConstSharedPtr image;
        std::vector<sensor_msgs::msg::Imu> imu_buffer;
        geometry_msgs::msg::Pose odom_pose;
        bool has_odom;
    };
    std::queue<TrackingQueueItem> tracking_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread mapping_thread_;
    std::atomic<bool> run_mapping_thread_{false};
    
    void runMappingLoop();
    void process_queued_item(const TrackingQueueItem& item);

    geometry_msgs::msg::Pose last_keyframe_vision_pose_;
    geometry_msgs::msg::Pose last_keyframe_odom_pose_;

    // Per-frame IMU gyro-only delta rotation for injection into VO relative transform.
    // Computed each frame in process_queued_item(); consumed in frontendCallback().
    // Only valid in mono_imu mode when at least 2 IMU samples are available.
    Eigen::Quaterniond imu_delta_q_for_vo_{Eigen::Quaterniond::Identity()};
    bool imu_delta_q_valid_{false};
};

} // namespace slam

#endif // GRAPH_SLAM_HPP
