#include "visual_graph_slam/map/map_manager.hpp"

#include <opencv2/calib3d.hpp>

namespace slam::map {

MapManager::MapManager(rclcpp::Logger logger) : logger_(logger) {}

void MapManager::addLandmark(const Landmark& lm) {
    landmarks_[lm.id] = lm;
}

void MapManager::addObservation(int landmark_id, const LandmarkObservation& obs) {
    landmarks_[landmark_id].observations.push_back(obs);
    landmarks_[landmark_id].observation_count++;
    landmarks_[landmark_id].last_seen_keyframe_id = std::max(landmarks_[landmark_id].last_seen_keyframe_id, obs.keyframe_id);
    observation_to_landmark_[makeObservationKey(obs.keyframe_id, obs.keypoint_index)] = landmark_id;
}

Landmark* MapManager::getLandmark(int id) {
    auto it = landmarks_.find(id);
    if (it != landmarks_.end()) return &it->second;
    return nullptr;
}

const Landmark* MapManager::getLandmark(int id) const {
    auto it = landmarks_.find(id);
    if (it != landmarks_.end()) return &it->second;
    return nullptr;
}

int MapManager::getLandmarkIdFromObservation(int keyframe_id, int keypoint_index) const {
    auto it = observation_to_landmark_.find(makeObservationKey(keyframe_id, keypoint_index));
    if (it != observation_to_landmark_.end()) {
        return it->second;
    }
    return -1;
}

int64_t MapManager::makeObservationKey(int keyframe_id, int keypoint_index) {
    return (static_cast<int64_t>(keyframe_id) << 32) | static_cast<uint32_t>(keypoint_index);
}

bool MapManager::triangulateMatchToWorld(
    const Eigen::Isometry3d& pose_from,
    const Eigen::Isometry3d& pose_to,
    const cv::Point2f& point_from,
    const cv::Point2f& point_to,
    const cv::Mat& camera_matrix,
    Eigen::Vector3d& point_world) const
{
    const double fx = camera_matrix.at<double>(0, 0);
    const double fy = camera_matrix.at<double>(1, 1);
    const double cx = camera_matrix.at<double>(0, 2);
    const double cy = camera_matrix.at<double>(1, 2);

    const double xn1 = (point_from.x - cx) / fx;
    const double yn1 = (point_from.y - cy) / fy;
    const double xn2 = (point_to.x - cx) / fx;
    const double yn2 = (point_to.y - cy) / fy;

    // camera_T_world for each pose
    const Eigen::Isometry3d camera_T_world_from = pose_from.inverse();
    const Eigen::Isometry3d camera_T_world_to   = pose_to.inverse();

    // [R | t] blocks (3×4)
    const Eigen::Matrix<double, 3, 4> Rt_from = camera_T_world_from.matrix().block<3, 4>(0, 0);
    const Eigen::Matrix<double, 3, 4> Rt_to   = camera_T_world_to.matrix().block<3, 4>(0, 0);

    // Linear triangulation (DLT) on normalized rays: build well-conditioned 4×4 system A * X_h = 0
    Eigen::Matrix4d A;
    A.row(0) = xn1 * Rt_from.row(2) - Rt_from.row(0);
    A.row(1) = yn1 * Rt_from.row(2) - Rt_from.row(1);
    A.row(2) = xn2 * Rt_to.row(2)   - Rt_to.row(0);
    A.row(3) = yn2 * Rt_to.row(2)   - Rt_to.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d p_homo = svd.matrixV().col(3);

    if (std::abs(p_homo(3)) < 1e-6) {
        return false;
    }

    point_world = p_homo.head<3>() / p_homo(3);

    // Accept only points with positive depth in both cameras
    const Eigen::Vector3d p_camera_from = camera_T_world_from * point_world;
    const Eigen::Vector3d p_camera_to   = camera_T_world_to   * point_world;

    return (p_camera_from.z() > 0.1 && p_camera_to.z() > 0.1);
}

void MapManager::updateLandmarkMapForLatestKeyframe(
    std::shared_ptr<slam::Graph> graph,
    int latest_keyframe_id,
    const std::deque<int>& recent_keyframe_ids,
    const cv::Mat& camera_matrix,
    double match_ratio_test,
    int min_triangulation_parallax_px,
    double min_depth,
    double max_depth)
{
    if (recent_keyframe_ids.size() < 2) return;

    auto latest_node = graph->getNode(latest_keyframe_id);
    if (!latest_node) return;

    const auto& latest_keypoints = latest_node->getKeypoints();
    const auto& latest_descriptors = latest_node->getDescriptors();
    if (latest_keypoints.empty() || latest_descriptors.empty()) return;

    // Convert latest pose to Isometry3d
    Eigen::Isometry3d latest_pose_iso = Eigen::Isometry3d::Identity();
    const auto& p1 = latest_node->getPose();
    latest_pose_iso.translation() = Eigen::Vector3d(p1.position.x, p1.position.y, p1.position.z);
    latest_pose_iso.linear() = Eigen::Quaterniond(p1.orientation.w, p1.orientation.x, p1.orientation.y, p1.orientation.z).toRotationMatrix();

    // Match against older keyframes
    int new_landmarks_count = 0;
    for (auto it = recent_keyframe_ids.rbegin(); it != recent_keyframe_ids.rend(); ++it) {
        int older_id = *it;
        if (older_id >= latest_keyframe_id) continue;

        auto older_node = graph->getNode(older_id);
        if (!older_node) continue;

        const auto& older_keypoints = older_node->getKeypoints();
        const auto& older_descriptors = older_node->getDescriptors();
        if (older_keypoints.empty() || older_descriptors.empty()) continue;
        if (older_descriptors.type() != latest_descriptors.type()) continue;

        Eigen::Isometry3d older_pose_iso = Eigen::Isometry3d::Identity();
        const auto& p0 = older_node->getPose();
        older_pose_iso.translation() = Eigen::Vector3d(p0.position.x, p0.position.y, p0.position.z);
        older_pose_iso.linear() = Eigen::Quaterniond(p0.orientation.w, p0.orientation.x, p0.orientation.y, p0.orientation.z).toRotationMatrix();

        int norm = (latest_descriptors.type() == CV_8U) ? cv::NORM_HAMMING : cv::NORM_L2;
        cv::BFMatcher matcher(norm, false);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(older_descriptors, latest_descriptors, knn_matches, 2);

        for (const auto& pair : knn_matches) {
            if (pair.size() < 2 || pair[0].distance >= match_ratio_test * pair[1].distance) continue;

            int query_idx = pair[0].queryIdx;
            int train_idx = pair[0].trainIdx;

            int existing_landmark_id = getLandmarkIdFromObservation(older_id, query_idx);

            if (existing_landmark_id >= 0) {
                // If the train_idx doesn't belong to any landmark, link it
                if (getLandmarkIdFromObservation(latest_keyframe_id, train_idx) < 0) {
                    addObservation(existing_landmark_id, {latest_keyframe_id, train_idx, latest_keypoints[train_idx].pt});
                }
            } else {
                if (getLandmarkIdFromObservation(latest_keyframe_id, train_idx) < 0) {
                    const cv::Point2f pt1 = older_keypoints[query_idx].pt;
                    const cv::Point2f pt2 = latest_keypoints[train_idx].pt;
                    
                    if (cv::norm(pt1 - pt2) < min_triangulation_parallax_px) continue;

                    Eigen::Vector3d point_world;
                    if (triangulateMatchToWorld(older_pose_iso, latest_pose_iso, pt1, pt2,
                                               camera_matrix, point_world)) {
                        // Depth check
                        Eigen::Vector3d p_camera_to = latest_pose_iso.inverse() * point_world;
                        if (p_camera_to.z() >= min_depth && p_camera_to.z() <= max_depth) {
                            Landmark lm;
                            lm.id = next_landmark_id_++;
                            lm.position = point_world;
                            lm.created_keyframe_id = older_id;
                            addLandmark(lm);

                            addObservation(lm.id, {older_id, query_idx, pt1});
                            addObservation(lm.id, {latest_keyframe_id, train_idx, pt2});
                            new_landmarks_count++;
                        }
                    }
                }
            }
        }
    }
    RCLCPP_DEBUG(logger_, "Created %d new landmarks for keyframe %d", new_landmarks_count, latest_keyframe_id);
}

void MapManager::pruneLandmarkMap(int current_keyframe_id, int local_track_depth, int min_observations) {
    int pruned_count = 0;
    for (auto it = landmarks_.begin(); it != landmarks_.end();) {
        if (it->second.bad || (current_keyframe_id - it->second.last_seen_keyframe_id > local_track_depth && it->second.observation_count < min_observations)) {
            for (const auto& obs : it->second.observations) {
                observation_to_landmark_.erase(makeObservationKey(obs.keyframe_id, obs.keypoint_index));
            }
            it = landmarks_.erase(it);
            pruned_count++;
        } else {
            ++it;
        }
    }
    if (pruned_count > 0) {
        RCLCPP_DEBUG(logger_, "Pruned %d bad/stale landmarks", pruned_count);
    }
}

} // namespace slam::map
