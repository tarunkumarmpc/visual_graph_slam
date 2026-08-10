#include "visual_graph_slam/vins/gtsam_imu_builder.hpp"
#include <algorithm>

namespace slam::vins {

gtsam::PreintegratedImuMeasurements GtsamImuBuilder::buildPIM(
    const slam::GraphEdge& edge,
    const gtsam::imuBias::ConstantBias& from_bias,
    double gravity_magnitude,
    double acc_covariance_multiplier)
{
    // Build PreintegrationParams
    auto pim_params = gtsam::PreintegrationParams::MakeSharedU(gravity_magnitude);
    
    // Calculate covariances based on the raw samples if available
    double acc_var = 1e-4;
    double gyro_var = 1e-4;
    
    const auto& pim_data = edge.getImuPreintegration();
    // GTSAM's PreintegrationParams expects the continuous-time sensor noise 
    // density (sigma^2) on the diagonal.
    acc_var  = (pim_data.accel_noise * pim_data.accel_noise) * acc_covariance_multiplier;
    gyro_var = (pim_data.gyro_noise * pim_data.gyro_noise);

    pim_params->accelerometerCovariance = gtsam::Matrix33::Identity() * acc_var;
    pim_params->gyroscopeCovariance = gtsam::Matrix33::Identity() * gyro_var;
    pim_params->integrationCovariance = gtsam::Matrix33::Identity() * 1e-8;

    gtsam::PreintegratedImuMeasurements pim(pim_params, from_bias);

    // Re-integrate raw samples to correctly build delta_R, delta_v, delta_p, and their Jacobians around from_bias
    const auto& raw_samples = edge.getRawImuSamples();
    std::cout << "[GtsamImuBuilder] Integrating edge with " << raw_samples.size() << " IMU samples" << std::endl;
    if (raw_samples.size() >= 2) {
        for (std::size_t si = 0; si + 1 < raw_samples.size(); ++si) {
            const auto& s1 = raw_samples[si];
            const auto& s2 = raw_samples[si + 1];
            
            const double dt_step = (s2.header.stamp.sec + s2.header.stamp.nanosec * 1e-9) -
                                   (s1.header.stamp.sec + s1.header.stamp.nanosec * 1e-9);
                                   
            if (dt_step <= 0.0 || dt_step > 1.0) continue;

            const Eigen::Vector3d a_mid(
                0.5 * (s1.linear_acceleration.x + s2.linear_acceleration.x),
                0.5 * (s1.linear_acceleration.y + s2.linear_acceleration.y),
                0.5 * (s1.linear_acceleration.z + s2.linear_acceleration.z));
            const Eigen::Vector3d g_mid(
                0.5 * (s1.angular_velocity.x + s2.angular_velocity.x),
                0.5 * (s1.angular_velocity.y + s2.angular_velocity.y),
                0.5 * (s1.angular_velocity.z + s2.angular_velocity.z));

            pim.integrateMeasurement(a_mid, g_mid, dt_step);
        }
    } else if (raw_samples.size() == 1) {
        // Fallback when only 1 sample is available (e.g. 10 Hz IMU with 10 Hz cameras).
        // We integrate this single measurement for the entire preintegrated dt.
        const auto& pim_data = edge.getImuPreintegration();
        if (pim_data.dt > 1e-6) {
            const auto& s = raw_samples[0];
            const Eigen::Vector3d a(s.linear_acceleration.x, s.linear_acceleration.y, s.linear_acceleration.z);
            const Eigen::Vector3d g(s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z);
            pim.integrateMeasurement(a, g, pim_data.dt);
        }
    } else {
        // Fallback when zero raw samples are available.
        // We cannot reconstruct specific force from delta_v because it includes rotation.
        // The safest honest fallback is zero specific force and zero angular velocity.
        const auto& pim_data = edge.getImuPreintegration();
        if (pim_data.dt > 1e-6) {
            pim.integrateMeasurement(
                Eigen::Vector3d::Zero(),   // zero specific force
                Eigen::Vector3d::Zero(),   // zero angular velocity
                pim_data.dt);
        }
    }

    return pim;
}

} // namespace slam::vins
