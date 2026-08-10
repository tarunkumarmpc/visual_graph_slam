// measurement_edge.hpp
//
// Unified Constraint / Edge Abstraction
// ─────────────────────────────────────────────────────────────────────────────
// 
// This module decouples measurement types (VO, odometry, IMU, GPS, LiDAR) from
// the logic of constructing pose-graph edges. All measurements flow through a
// single `MeasurementEdgeConfig` struct, which specifies:
//
//   1. The measurement data (transform, source type)
//   2. Confidence / quality metrics (inliers, covariance)
//   3. An extensible `Source` enum for future sensors
//
// The graph's job is just to instantiate edges with the right information
// matrix, type, and weight — not to know about each sensor individually.

#ifndef VISUAL_GRAPH_SLAM_MEASUREMENT_EDGE_HPP
#define VISUAL_GRAPH_SLAM_MEASUREMENT_EDGE_HPP

#include <cstdint>
#include <geometry_msgs/msg/transform.hpp>
#include <Eigen/Dense>
#include <vector>
#include <optional>
#include <sensor_msgs/msg/imu.hpp>
#include "visual_graph_slam/vins/imu_preintegrator.hpp"

namespace slam {

// ─────────────────────────────────────────────────────────────────────────────
// Measurement Source Enum (Vision-Based System)
// ─────────────────────────────────────────────────────────────────────────────
// Current: VISUAL_ODOMETRY, LOOP_CLOSURE
// Future:  WHEEL_ODOMETRY, IMU_PREINTEGRATION, EXTERNAL_ODOMETRY (optional)
enum class MeasurementSource {
    VISUAL_ODOMETRY,          ///< [CURRENT] Hybrid LK optical flow + Essential matrix
    LOOP_CLOSURE,             ///< [CURRENT] Place recognition + geometric verification
    
    // Future extensions (zero overhead if not used)
    WHEEL_ODOMETRY,           ///< [FUTURE] Incremental wheel encoder
    IMU_PREINTEGRATION,       ///< [FUTURE] IMU measurement integration
    EXTERNAL_ODOMETRY,        ///< [FUTURE] RTK-GPS, LiDAR odometry, etc.
};

// ─────────────────────────────────────────────────────────────────────────────
// Measurement Edge Configuration
// ─────────────────────────────────────────────────────────────────────────────
/**
 * @brief Unified measurement -> edge spec
 *
 * This struct encapsulates all information needed to create a pose-graph edge
 * from any measurement source. The graph builder queries the source type and
 * confidence to construct the appropriate information matrix and edge type.
 */
struct MeasurementEdgeConfig {
    // ── Pose graph connectivity ───────────────────────────────────────────────
    int from_keyframe_id{-1};
    int to_keyframe_id{-1};

    // ── Measured relative transform ───────────────────────────────────────────
    geometry_msgs::msg::Transform relative_transform{};

    // ── Source and quality metadata ───────────────────────────────────────────
    MeasurementSource source{MeasurementSource::WHEEL_ODOMETRY};

    /// Confidence [0, 1]: inlier ratio for pose estimators, 
    /// or covariance-based quality for dead-reckoning.
    double confidence{1.0};

    /// Number of inliers in RANSAC / consistency filtering.
    /// -1 = not applicable (e.g., for direct wheel odometry).
    int inliers{-1};

    /// Total points / correspondences in estimation.
    /// -1 = not applicable.
    int total_points{-1};

    /// Optional per-measurement information-matrix override.
    /// Only used when has_covariance_override is true.
    Eigen::Matrix<double, 6, 6> covariance_override{};
    bool has_covariance_override{false};
    
    /// Whether to include this measurement in optimization.
    bool enabled{true};

    // ── IMU Specific Data ─────────────────────────────────────────────────────
    std::optional<vins::ImuPreintegrator::PreintegratedData> imu_data{std::nullopt};
    std::vector<sensor_msgs::msg::Imu> raw_imu_samples{};

    // ──────────────────────────────────────────────────────────────────────────
    // Validation and string utilities
    // ──────────────────────────────────────────────────────────────────────────
    bool isValid() const {
        return from_keyframe_id >= 0 && to_keyframe_id >= 0 && enabled;
    }

    static const char* sourceToString(MeasurementSource src) {
        switch (src) {
            case MeasurementSource::WHEEL_ODOMETRY:      return "wheel_odom";
            case MeasurementSource::VISUAL_ODOMETRY:     return "visual_odom";
            case MeasurementSource::IMU_PREINTEGRATION:  return "imu_preint";
            case MeasurementSource::EXTERNAL_ODOMETRY:   return "external_odom";
            case MeasurementSource::LOOP_CLOSURE:        return "loop_closure";
            default:                                     return "unknown";
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Information Matrix Calculator
// ─────────────────────────────────────────────────────────────────────────────
/**
 * @brief Compute information matrix (pose-graph weight) from measurement metadata
 *
 * This is where sensor fusion tuning lives. Each measurement source gets a
 * base weight, then scaled by confidence. Future IMU/GPS modules can extend
 * this with their own covariance propagation logic.
 */
class MeasurementWeightCalculator {
public:
    /**
     * Compute the 6×6 information matrix for a measurement edge.
     *
     * @param config  Measurement metadata (source, confidence, inliers)
     * @return        6×6 positive-definite information matrix (weight)
     *
     * Semantic:
     *   - Wheel odometry        = 1.0 × I           (baseline)
     *   - Visual odometry (LK)  = 15–30 × conf × I  (dense, robust when confident)
     *   - Visual odometry (Feat)= 6–10 × conf × I   (sparser, noisier)
     *   - IMU preintegration    = adaptive          (short-term, high-freq)
     *   - External (GPS)        = high (fixed frame reference)
     *   - Loop closure          = 5 × I             (drift-correcting, long-range)
     */
    static Eigen::Matrix<double, 6, 6> compute(const MeasurementEdgeConfig& config);

private:
    static double computeBaseWeight(MeasurementSource source);
};

}  // namespace slam

#endif  // VISUAL_GRAPH_SLAM_MEASUREMENT_EDGE_HPP
