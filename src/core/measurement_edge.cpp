// measurement_edge.cpp
//
// Unified Constraint Edge Implementation

#include "visual_graph_slam/core/measurement_edge.hpp"

#include <algorithm>
#include <cmath>

namespace slam {

// ─────────────────────────────────────────────────────────────────────────────
// Information Matrix (Confidence-based Weighting)
// ─────────────────────────────────────────────────────────────────────────────

Eigen::Matrix<double, 6, 6> MeasurementWeightCalculator::compute(
    const MeasurementEdgeConfig& config)
{
    // If custom covariance provided, use it directly
    if (config.has_covariance_override) {
        return config.covariance_override;
    }

    const double base_weight = computeBaseWeight(config.source);
    double effective_weight  = base_weight;

    // Apply confidence scaling (inlier ratio or covariance quality)
    // Only for methods that produce confidence scores
    switch (config.source) {
        case MeasurementSource::VISUAL_ODOMETRY:
            // LK optical flow produces many dense correspondences → higher base weight
            // then scale by inlier ratio
            effective_weight = base_weight * config.confidence;
            break;

        case MeasurementSource::LOOP_CLOSURE:
            // Loop closure is already geometrically verified
            // but we still discount based on how many inliers survived RANSAC
            effective_weight = base_weight * config.confidence;
            break;

        case MeasurementSource::EXTERNAL_ODOMETRY:
            // External sources (GPS, LiDAR odom) are typically high-confidence
            // but can degrade in certain environments
            effective_weight = base_weight * config.confidence;
            break;

        case MeasurementSource::IMU_PREINTEGRATION:
            // IMU is biased by cumulative errors; short-term high, long-term low
            // Confidence here represents freshness: 1.0 = current frame, <1.0 = stale
            effective_weight = base_weight * (1.0 + 9.0 * config.confidence);
            break;

        case MeasurementSource::WHEEL_ODOMETRY:
            // Wheel odometry is the baseline; always 1.0 unless explicitly tuned
            break;

        default:
            break;
    }

    // Clamp to reasonable range to prevent numerical issues
    effective_weight = std::max(0.1, std::min(effective_weight, 100.0));

    Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Identity() * effective_weight;
    if (config.source == MeasurementSource::VISUAL_ODOMETRY) {
        // Anisotropic weighting for monocular VO:
        // In GTSAM (Pose3) and g2o (SE3Quat), tangent vector variables are ordered:
        // Indices 0, 1, 2: Rotation (roll around X, pitch around Y, yaw around Z in base_link)
        // Indices 3, 4, 5: Translation (X forward, Y left, Z up in base_link)
        // Monocular visual tracking provides strong constraints on rotation (0,1,2)
        // and lateral/vertical translation (4,5), but has higher uncertainty along forward depth/scale (3).
        info.diagonal() << effective_weight * 2.0,  // 0: roll (rot X)
                           effective_weight * 2.0,  // 1: pitch (rot Y)
                           effective_weight * 2.0,  // 2: yaw (rot Z)
                           effective_weight * 0.25, // 3: trans X (forward depth/scale uncertainty)
                           effective_weight * 1.0,  // 4: trans Y (lateral left)
                           effective_weight * 1.0;  // 5: trans Z (vertical up/down)
    }
    return info;
}

double MeasurementWeightCalculator::computeBaseWeight(MeasurementSource source)
{
    switch (source) {
        case MeasurementSource::VISUAL_ODOMETRY:
            // [CURRENT] Hybrid LK optical flow + Essential matrix
            // ~100–500 dense tracked points per frame
            // Very robust to noise and low-texture regions
            // Base weight: 20.0
            // Scaled by confidence (inlier ratio): 6–20 after scaling
            return 20.0;

        case MeasurementSource::LOOP_CLOSURE:
            // [CURRENT] Place recognition + geometric verification
            // Sparse but high-quality long-range constraints
            // Crucial for drift correction
            // Base weight: 5.0
            return 5.0;

        // Future extensions (zero overhead, ready when needed)
        case MeasurementSource::WHEEL_ODOMETRY:
            // Prior anchor for monocular BA stability.
            // Keep below VO weight, but high enough to constrain translation drift.
            return 6.0;
        case MeasurementSource::IMU_PREINTEGRATION:
            return 5.0;  // High short-term, accumulates drift
        case MeasurementSource::EXTERNAL_ODOMETRY:
            return 10.0; // GPS/LiDAR: tuned per deployment

        default:
            return 1.0;
    }
}

}  // namespace slam
