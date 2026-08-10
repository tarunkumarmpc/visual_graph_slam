#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <rclcpp/rclcpp.hpp>

#include "visual_graph_slam/core/graph.hpp"

namespace slam::map {

struct LandmarkObservation {
    int keyframe_id{-1};
    int keypoint_index{-1};
    cv::Point2f pixel;
};

struct Landmark {
    int id{-1};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    int created_keyframe_id{-1};
    int last_seen_keyframe_id{-1};
    int observation_count{0};
    bool bad{false};
    std::vector<LandmarkObservation> observations;
};

class MapManager {
public:
    using SharedPtr = std::shared_ptr<MapManager>;

    MapManager(rclcpp::Logger logger);

    void addLandmark(const Landmark& lm);
    void addObservation(int landmark_id, const LandmarkObservation& obs);
    
    Landmark* getLandmark(int id);
    const Landmark* getLandmark(int id) const;
    
    int getLandmarkIdFromObservation(int keyframe_id, int keypoint_index) const;
    
    std::unordered_map<int, Landmark>& getLandmarks() { return landmarks_; }
    const std::unordered_map<int, Landmark>& getLandmarks() const { return landmarks_; }

    static int64_t makeObservationKey(int keyframe_id, int keypoint_index);

    void pruneLandmarkMap(int current_keyframe_id, int local_track_depth, int min_observations);

    // Creates new landmarks by triangulating points from the latest keyframe.
    // camera_matrix: 3×3 intrinsic matrix (K) used to build correct projection
    // matrices P = K * [R|t] for triangulation.
    void updateLandmarkMapForLatestKeyframe(
        std::shared_ptr<slam::Graph> graph,
        int latest_keyframe_id,
        const std::deque<int>& recent_keyframe_ids,
        const cv::Mat& camera_matrix,
        double match_ratio_test,
        int min_triangulation_parallax_px,
        double min_depth,
        double max_depth
    );

private:
    rclcpp::Logger logger_;
    
    std::unordered_map<int, Landmark> landmarks_;
    std::unordered_map<int64_t, int> observation_to_landmark_;
    int next_landmark_id_{0};

    bool triangulateMatchToWorld(
        const Eigen::Isometry3d& pose_from,
        const Eigen::Isometry3d& pose_to,
        const cv::Point2f& point_from,
        const cv::Point2f& point_to,
        const cv::Mat& camera_matrix,
        Eigen::Vector3d& point_world) const;
};

} // namespace slam::map
