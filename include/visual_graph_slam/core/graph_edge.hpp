#ifndef GRAPH_EDGE_HPP
#define GRAPH_EDGE_HPP

#include <geometry_msgs/msg/transform.hpp>
#include <Eigen/Dense>
#include <vector>
#include <sensor_msgs/msg/imu.hpp>
#include "visual_graph_slam/vins/imu_preintegrator.hpp"

namespace slam {

class GraphEdge {
public:
    enum class Type {
        ODOMETRY,         ///< Wheel / IMU odometry constraint
        VISUAL_ODOMETRY,  ///< Vision-based (LK/Essential) pose constraint
        LOOP_CLOSURE,     ///< VPR + geometric verification constraint
        IMU_PREINTEGRATION ///< High-frequency IMU integration constraint
    };

    GraphEdge(Type type, int from_id, int to_id, 
              const geometry_msgs::msg::Transform& relative_transform,
              const Eigen::Matrix<double, 6, 6>& information_matrix);

    Type getType() const;
    int getFromId() const;
    int getToId() const;
    geometry_msgs::msg::Transform getRelativeTransform() const;
    /// allows alignGraphWithGravity to rescale pre-init edge translations.
    void setRelativeTransform(const geometry_msgs::msg::Transform& tf) {
        relative_transform_ = tf;
    }
    Eigen::Matrix<double, 6, 6> getInformationMatrix() const;

    void setImuPreintegration(const vins::ImuPreintegrator::PreintegratedData& data) {
        imu_data_ = data;
    }
    const vins::ImuPreintegrator::PreintegratedData& getImuPreintegration() const {
        return imu_data_;
    }

    void setRawImuSamples(const std::vector<sensor_msgs::msg::Imu>& samples) {
        raw_imu_samples_ = samples;
    }
    const std::vector<sensor_msgs::msg::Imu>& getRawImuSamples() const {
        return raw_imu_samples_;
    }

private:
    Type type_;
    int from_id_;
    int to_id_;
    geometry_msgs::msg::Transform relative_transform_;
    Eigen::Matrix<double, 6, 6> information_matrix_;
    vins::ImuPreintegrator::PreintegratedData imu_data_;
    std::vector<sensor_msgs::msg::Imu> raw_imu_samples_;
};

} // namespace slam

#endif // GRAPH_EDGE_HPP
