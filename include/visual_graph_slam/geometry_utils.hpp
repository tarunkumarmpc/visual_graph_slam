#ifndef VISUAL_GRAPH_SLAM_GEOMETRY_UTILS_HPP
#define VISUAL_GRAPH_SLAM_GEOMETRY_UTILS_HPP

#include <Eigen/Geometry>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

namespace slam::utils {

/**
 * @brief Convert OpenCV R and t matrices to a ROS geometry_msgs Transform.
 *
 * OpenCV recoverPose convention: X_cam2 = R * X_cam1 + t
 * i.e. [R|t] transforms points FROM cam1 frame INTO cam2 frame.
 * We want the INVERSE — T_{cam1←cam2} — which gives:
 *   translation = -R^T * t   (camera 2 centre expressed in cam1 frame)
 *   rotation    = R^T         (cam2 orientation relative to cam1)
 * This is the standard VO relative-pose representation consumed by applyTransform.
 */
inline geometry_msgs::msg::Transform rtToTransform(const cv::Mat& R, const cv::Mat& t) {
    geometry_msgs::msg::Transform tf;

    // OpenCV recoverPose returns [R|t] such that X_2 = R * X_1 + t.
    // This transforms points FROM cam1 TO cam2.
    // The actual pose of cam2 relative to cam1 (the camera's motion) is the inverse:
    // T_{c1<-c2} = [R^T | -R^T * t]
    Eigen::Matrix3d eigen_R;
    cv::cv2eigen(R, eigen_R);
    Eigen::Vector3d eigen_t;
    cv::cv2eigen(t, eigen_t);

    Eigen::Matrix3d R_inv = eigen_R.transpose();
    Eigen::Vector3d t_inv = -R_inv * eigen_t;

    tf.translation.x = t_inv.x();
    tf.translation.y = t_inv.y();
    tf.translation.z = t_inv.z();

    Eigen::Quaterniond q(R_inv);
    q.normalize();

    tf.rotation.x = q.x();
    tf.rotation.y = q.y();
    tf.rotation.z = q.z();
    tf.rotation.w = q.w();

    return tf;
}

/**
 * @brief Convert OpenCV R and t matrices to a ROS geometry_msgs Pose.
 */
inline geometry_msgs::msg::Pose rtToPose(const cv::Mat& R, const cv::Mat& t) {
    geometry_msgs::msg::Pose pose;
    
    // Position
    pose.position.x = t.at<double>(0);
    pose.position.y = t.at<double>(1);
    pose.position.z = t.at<double>(2);

    // Orientation
    Eigen::Matrix3d eigen_R;
    cv::cv2eigen(R, eigen_R);
    Eigen::Quaterniond q(eigen_R);
    q.normalize();

    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();
    pose.orientation.w = q.w();

    return pose;
}

} // namespace slam::utils

#endif // VISUAL_GRAPH_SLAM_GEOMETRY_UTILS_HPP
