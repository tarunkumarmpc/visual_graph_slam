#ifndef VISUAL_GRAPH_SLAM_VINS_VINS_INITIALIZER_HPP
#define VISUAL_GRAPH_SLAM_VINS_VINS_INITIALIZER_HPP

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "visual_graph_slam/core/graph.hpp"
#include "visual_graph_slam/vins/imu_preintegrator.hpp"

namespace slam::vins {

class VinsInitializer {
public:
    struct VinsConfig {
        double stationary_velocity_threshold = 0.10; // [vo-units/s] Below this, vehicle is stationary
        bool use_scale_prior_fallback = true;        // Enable modular fallback when moving at constant velocity
        double default_scale_prior = 1.0;            // Expected scale prior (metric_displacement = s * visual_displacement)
        // Scale acceptance bounds — tune per mode:
        //   KITTI outdoor car (10-30 m/s, kf ~1.5m):  min=0.3, max=30.0
        //   Indoor drone (0.5-3 m/s, kf ~0.5m):       min=0.1, max=5.0
        // Defaults are intentionally wide; the gravity magnitude check [9.5,10.5] m/s²
        // is the primary filter for bad solutions.
        double scale_prior_min = 0.1;    // Minimum accepted scale (reject degenerate near-zero)
        double scale_prior_max = 50.0;   // Maximum accepted scale (reject blown-up solutions)
    };

    enum class MotionRegime {
        STATIONARY,
        DYNAMIC_EXCITATION,
        CONSTANT_VELOCITY_FALLBACK
    };

    struct InitializationResult {
        bool success = false;
        double scale = 1.0;
        Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81);
        std::vector<Eigen::Vector3d> velocities;
        Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
        Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
        MotionRegime regime = MotionRegime::STATIONARY;
    };

    VinsInitializer();

    void setConfig(const VinsConfig& config) { config_ = config; }
    const VinsConfig& getConfig() const { return config_; }

    /**
     * @brief Attempt to initialize scale and gravity from a sequence of keyframes.
     * @param keyframes Ordered list of keyframes with visual poses and preintegrated IMU data.
     * @return Initialization result.
     */
    InitializationResult align(const std::vector<std::shared_ptr<GraphNode>>& keyframes,
                               const std::vector<ImuPreintegrator::PreintegratedData>& imu_data);

    bool estimateGyroBias(const std::vector<std::shared_ptr<GraphNode>>& keyframes,
                          const std::vector<ImuPreintegrator::PreintegratedData>& imu_data,
                          Eigen::Vector3d& gyro_bias);

    bool estimateGravityAndScale(const std::vector<std::shared_ptr<GraphNode>>& keyframes,
                                 const std::vector<ImuPreintegrator::PreintegratedData>& imu_data,
                                 double& scale, Eigen::Vector3d& gravity,
                                 MotionRegime& detected_regime);
                                 
    bool refineGravity(const std::vector<std::shared_ptr<GraphNode>>& keyframes,
                       const std::vector<ImuPreintegrator::PreintegratedData>& imu_data,
                       Eigen::Vector3d& gravity,
                       double& scale);

    MotionRegime evaluateMotionRegime(const std::vector<std::shared_ptr<GraphNode>>& keyframes,
                                      const std::vector<ImuPreintegrator::PreintegratedData>& imu_data,
                                      const Eigen::MatrixXd& A,
                                      const Eigen::VectorXd& singular_values,
                                      double estimated_scale) const;

private:
    VinsConfig config_;
};

} // namespace slam::vins

#endif // VISUAL_GRAPH_SLAM_VINS_VINS_INITIALIZER_HPP
