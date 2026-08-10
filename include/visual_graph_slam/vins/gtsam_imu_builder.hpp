#ifndef VISUAL_GRAPH_SLAM_VINS_GTSAM_IMU_BUILDER_HPP
#define VISUAL_GRAPH_SLAM_VINS_GTSAM_IMU_BUILDER_HPP

#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include "visual_graph_slam/core/graph_edge.hpp"

namespace slam::vins {

class GtsamImuBuilder {
public:
    /**
     * @brief Centralized builder to generate Preintegrated IMU Measurements (PIM) from a GraphEdge.
     * Ensures identical gravity frame and covariance scaling between Global and Local optimizations.
     *
     * @param edge The graph edge containing raw_samples.
     * @param from_bias The starting IMU bias for the integration.
     * @param gravity_magnitude Gravity magnitude (defaults to 9.81, aligned to Z).
     * @param acc_covariance_multiplier Multiplier to loosen metric acceleration constraints (defaults to 1000.0).
     * @return gtsam::PreintegratedImuMeasurements
     */
    static gtsam::PreintegratedImuMeasurements buildPIM(
        const slam::GraphEdge& edge,
        const gtsam::imuBias::ConstantBias& from_bias,
        double gravity_magnitude = 9.81,
        double acc_covariance_multiplier = 1000.0);
};

} // namespace slam::vins

#endif // VISUAL_GRAPH_SLAM_VINS_GTSAM_IMU_BUILDER_HPP
