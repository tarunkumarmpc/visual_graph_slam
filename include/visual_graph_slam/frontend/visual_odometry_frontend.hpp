// visual_odometry_frontend.hpp
//
// Hybrid Visual Odometry Frontend
// ─────────────────────────────────────────────────────────────────────────────
// Pipeline per frame:
//   1. Lucas-Kanade pyramid optical flow   — tracks existing 2-D points
//   2. Forward-backward consistency check  — eliminates bad tracks cheaply
//   3. Automatic re-detection              — goodFeaturesToTrack into empty areas
//   4. Pose estimation
//      a. Essential-matrix from LK tracks  (primary path, ≥12 tracks)
//      b. Descriptor BF-match + Essential  (fallback when tracks collapse)
//   5. Metric scale anchoring              — scale unit-t by |t_odom| if provided
//   6. Confidence score                    — inlier ratio, used by graph for I-matrix
// ─────────────────────────────────────────────────────────────────────────────

#ifndef VISUAL_GRAPH_SLAM_FRONTEND_VISUAL_ODOMETRY_FRONTEND_HPP
#define VISUAL_GRAPH_SLAM_FRONTEND_VISUAL_ODOMETRY_FRONTEND_HPP

#include <memory>
#include <optional>
#include <vector>
#include <deque>
#include <functional>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <Eigen/Dense>

#include "visual_graph_slam/frontend/feature_pipeline.hpp"

namespace slam::frontend {

class VisualOdometryFrontend {
public:
    // ------------------------------------------------------------------
    // Public types
    // ------------------------------------------------------------------
    enum class Method { NONE, LK_ESSENTIAL, FEATURE_ESSENTIAL };

    struct VOResult {
        bool   valid{false};
        Method method{Method::NONE};
        geometry_msgs::msg::Transform relative_transform{};
        int    tracked_points{0};   ///< total correspondences fed to estimator
        int    inliers{0};          ///< RANSAC inliers
        double inlier_ratio{0.0};   ///< inliers / tracked_points
        double confidence{0.0};     ///< [0, 1] — drives information-matrix weight
        std::string failure_reason{"none"};
        double mean_flow_px{0.0};
        int fallback_match_count{0};
        int reference_age{1};
        double unit_sphere_parallax{0.0};
    };

    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------
    /**
     * @param camera_matrix      3×3 intrinsic matrix (CV_64F).
     * @param feature_pipeline   Owned feature detector/descriptor (for fallback
     *                           and frame initialisation).
     * @param min_tracked_pts    Minimum tracks before switching to fallback path.
     * @param reinit_threshold   Re-detect new corners when alive tracks fall below
     *                           this number.
     * @param max_corners        Cap on corners kept by goodFeaturesToTrack.
     */
    explicit VisualOdometryFrontend(cv::Mat                           camera_matrix,
                                    std::unique_ptr<FeaturePipeline>  feature_pipeline,
                                    int    min_tracked_pts   = 80,
                                    int    reinit_threshold  = 40,
                                    int    max_corners       = 500,
                                    double min_flow_px       = 1.5);  ///< min mean LK flow (px) before motion estimated

    // ------------------------------------------------------------------
    // Primary API
    // ------------------------------------------------------------------
    /**
     * @brief Process one camera frame and estimate relative motion.
     *
     * @param image_bgr_or_gray  New frame (BGR or gray accepted).
     * @param odom_displacement  Optional metric translation magnitude from wheel
     *                           odometry.  When provided, used to scale the
     *                           unit-normalised Essential-matrix translation so
     *                           the result is in real-world metres.
     * @param imu_rotation       Optional relative rotation predicted by IMU preintegration.
     * @return  VOResult         Empty (valid=false) on the very first frame or
     *                           when estimation fails.
     */
    VOResult track(const cv::Mat&          image_bgr_or_gray,
                   std::optional<double>   odom_displacement = std::nullopt,
                   const std::optional<Eigen::Matrix3d>& imu_rotation = std::nullopt);

    bool isInitialized() const { return prev_state_.has_value(); }

    /// Keypoints extracted from the CURRENT frame (usable for keyframe creation).
    const std::vector<cv::KeyPoint>& currentKeypoints()   const;
    /// Descriptors extracted from the CURRENT frame.
    const cv::Mat&                   currentDescriptors() const;

    /// Clear all internal state (e.g., on tracking loss or re-localisation).
    void reset();

    void setCameraMatrix(const cv::Mat& camera_matrix) {
        camera_matrix_ = camera_matrix.clone();
    }

    /// Unproject 2D pixel coordinates onto 3D unit sphere bearing vectors
    static void unprojectToUnitSphere(const std::vector<cv::Point2f>& pixels,
                                      const cv::Mat& camera_matrix,
                                      std::vector<cv::Point3f>& unit_sphere_vectors);

    /// Compute average chordal parallax across tracked features on the unit sphere,
    /// compensated by rotation delta_R (or identity if nullopt).
    static double computeUnitSphereParallax(const std::vector<cv::Point2f>& pts_prev,
                                            const std::vector<cv::Point2f>& pts_curr,
                                            const cv::Mat& camera_matrix,
                                            const std::optional<Eigen::Matrix3d>& delta_R = std::nullopt);

private:
    // ------------------------------------------------------------------
    // Internal state for one frame
    // ------------------------------------------------------------------
    struct FrameState {
        cv::Mat                     image_gray;
        std::vector<cv::Point2f>    tracked_pts;   ///< live tracked 2-D points
        std::vector<cv::KeyPoint>   keypoints;     ///< full ORB/SIFT keypoints
        cv::Mat                     descriptors;   ///< corresponding descriptors
    };

    // ------------------------------------------------------------------
    // Sub-algorithms
    // ------------------------------------------------------------------

    /// Lucas-Kanade with forward-backward error filter.
    /// Returns the tracked positions in the *current* frame.
    /// status_out[i] == 1 means prev_pts[i] survived.
    std::vector<cv::Point2f> runLKFlow(
        const cv::Mat&                  prev_gray,
        const cv::Mat&                  curr_gray,
        const std::vector<cv::Point2f>& prev_pts,
        std::vector<uchar>&             status_out,
        const std::optional<Eigen::Matrix3d>& imu_rotation = std::nullopt) const;

    /// Essential-matrix + recoverPose.  R/t are in CV_64F.
    bool estimatePoseEssential(const std::vector<cv::Point2f>& pts_prev,
                               const std::vector<cv::Point2f>& pts_curr,
                               cv::Mat& R_out,
                               cv::Mat& t_out,
                               int&     inliers_out) const;

    /// Re-detect ShiTomasi corners into areas not covered by existing tracks.
    void reinitTracks(const cv::Mat& gray);


    static double computeConfidence(Method method, int inliers, int total);

    // ------------------------------------------------------------------
    // Members
    // ------------------------------------------------------------------
    cv::Mat                            camera_matrix_;
    std::unique_ptr<FeaturePipeline>  feature_pipeline_;
    int                                min_tracked_pts_;
    int                                reinit_threshold_;
    int                                max_corners_;
    double                             min_flow_px_;    ///< reject motion if mean LK flow < this (stationary guard)

    std::optional<FrameState>          prev_state_;
    FrameState                         current_state_;
    std::deque<FrameState>             good_reference_states_;
    std::size_t                        temporal_fallback_window_;
};

}  // namespace slam::frontend

#endif  // VISUAL_GRAPH_SLAM_FRONTEND_VISUAL_ODOMETRY_FRONTEND_HPP
