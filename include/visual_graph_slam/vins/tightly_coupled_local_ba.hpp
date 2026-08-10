#ifndef VISUAL_GRAPH_SLAM_VINS_TIGHTLY_COUPLED_LOCAL_BA_HPP
#define VISUAL_GRAPH_SLAM_VINS_TIGHTLY_COUPLED_LOCAL_BA_HPP

#include <vector>
#include <unordered_set>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "visual_graph_slam/core/graph.hpp"

namespace slam::vins {

struct LocalBaObservation {
    int keyframe_id;
    Eigen::Vector2d pixel;
};

struct LocalBaLandmark {
    int id;
    Eigen::Vector3d position;
    std::vector<LocalBaObservation> observations;
};

class TightlyCoupledLocalBA {
public:
    static bool optimizeWindow(
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
        double height_prior_stddev
    );
};

} // namespace slam::vins

#endif // VISUAL_GRAPH_SLAM_VINS_TIGHTLY_COUPLED_LOCAL_BA_HPP
