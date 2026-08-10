// visual_odometry_frontend.cpp
//
// Hybrid LK Optical Flow + Feature Matching Visual Odometry
// ─────────────────────────────────────────────────────────────────────────────

#include "visual_graph_slam/frontend/visual_odometry_frontend.hpp"
#include "visual_graph_slam/geometry_utils.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <rclcpp/rclcpp.hpp>

namespace slam::frontend {

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────

VisualOdometryFrontend::VisualOdometryFrontend(
    cv::Mat                           camera_matrix,
    std::unique_ptr<FeaturePipeline> feature_pipeline,
    int                               min_tracked_pts,
    int                               reinit_threshold,
    int                               max_corners,
    double                            min_flow_px)
    : camera_matrix_(std::move(camera_matrix))
    , feature_pipeline_(std::move(feature_pipeline))
    , min_tracked_pts_(min_tracked_pts)
    , reinit_threshold_(reinit_threshold)
    , max_corners_(max_corners)
    , min_flow_px_(min_flow_px)
    , temporal_fallback_window_(4)
{}

// ─────────────────────────────────────────────────────────────────────────────
// Unit Sphere Helpers
// ─────────────────────────────────────────────────────────────────────────────

void VisualOdometryFrontend::unprojectToUnitSphere(
    const std::vector<cv::Point2f>& pixels,
    const cv::Mat& camera_matrix,
    std::vector<cv::Point3f>& unit_sphere_vectors)
{
    unit_sphere_vectors.clear();
    unit_sphere_vectors.reserve(pixels.size());
    const double fx = camera_matrix.at<double>(0, 0);
    const double fy = camera_matrix.at<double>(1, 1);
    const double cx = camera_matrix.at<double>(0, 2);
    const double cy = camera_matrix.at<double>(1, 2);

    for (const auto& pt : pixels) {
        const double xn = (pt.x - cx) / fx;
        const double yn = (pt.y - cy) / fy;
        const double norm = std::sqrt(xn * xn + yn * yn + 1.0);
        unit_sphere_vectors.emplace_back(static_cast<float>(xn / norm),
                                         static_cast<float>(yn / norm),
                                         static_cast<float>(1.0 / norm));
    }
}

double VisualOdometryFrontend::computeUnitSphereParallax(
    const std::vector<cv::Point2f>& pts_prev,
    const std::vector<cv::Point2f>& pts_curr,
    const cv::Mat& camera_matrix,
    const std::optional<Eigen::Matrix3d>& delta_R)
{
    if (pts_prev.empty() || pts_prev.size() != pts_curr.size()) {
        return 0.0;
    }

    std::vector<cv::Point3f> u_prev, u_curr;
    unprojectToUnitSphere(pts_prev, camera_matrix, u_prev);
    unprojectToUnitSphere(pts_curr, camera_matrix, u_curr);

    double total_parallax = 0.0;
    for (std::size_t i = 0; i < u_prev.size(); ++i) {
        Eigen::Vector3d v_p(u_prev[i].x, u_prev[i].y, u_prev[i].z);
        if (delta_R.has_value()) {
            v_p = delta_R->transpose() * v_p;
        }
        Eigen::Vector3d v_c(u_curr[i].x, u_curr[i].y, u_curr[i].z);
        total_parallax += (v_c - v_p).norm();
    }
    return total_parallax / static_cast<double>(u_prev.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// Primary tracking entry point
// ─────────────────────────────────────────────────────────────────────────────

VisualOdometryFrontend::VOResult VisualOdometryFrontend::track(
    const cv::Mat&        image_bgr_or_gray,
    std::optional<double> odom_displacement,
    const std::optional<Eigen::Matrix3d>& imu_rotation)
{
    struct AttemptOutcome {
        VOResult result;
        std::vector<cv::Point2f> tracks_for_current;
    };

    const auto makeFrameCopy = [](const FrameState& source) {
        FrameState copy = source;
        copy.image_gray = source.image_gray.clone();
        copy.descriptors = source.descriptors.clone();
        if (copy.tracked_pts.empty()) {
            copy.tracked_pts.reserve(copy.keypoints.size());
            for (const auto& kp : copy.keypoints) {
                copy.tracked_pts.push_back(kp.pt);
            }
        }
        return copy;
    };

    // ── 1. Grayscale conversion ───────────────────────────────────────────────
    if (image_bgr_or_gray.channels() > 1) {
        cv::cvtColor(image_bgr_or_gray, current_state_.image_gray, cv::COLOR_BGR2GRAY);
    } else {
        current_state_.image_gray = image_bgr_or_gray.clone();
    }

    // ── 2. Always extract keypoints/descriptors for keyframe use ─────────────
    feature_pipeline_->extract(current_state_.image_gray,
                               current_state_.keypoints,
                               current_state_.descriptors);

    // ── 3. First frame: seed tracked points and return empty result ───────────
    if (!prev_state_.has_value()) {
        current_state_.tracked_pts.clear();
        for (const auto& kp : current_state_.keypoints) {
            current_state_.tracked_pts.push_back(kp.pt);
        }
        prev_state_ = makeFrameCopy(current_state_);
        VOResult result;
        result.failure_reason = "bootstrap";
        result.tracked_points = static_cast<int>(current_state_.tracked_pts.size());
        return result;
    }

    auto attemptFromReference = [&](const FrameState& reference, int reference_age) -> AttemptOutcome {
        AttemptOutcome outcome;
        outcome.result.reference_age = reference_age;

        std::vector<cv::Point2f> reference_points = reference.tracked_pts;
        if (reference_points.empty()) {
            reference_points.reserve(reference.keypoints.size());
            for (const auto& kp : reference.keypoints) {
                reference_points.push_back(kp.pt);
            }
        }

        std::vector<uchar> lk_status;
        const std::vector<cv::Point2f> tracked_now =
            runLKFlow(reference.image_gray,
                      current_state_.image_gray,
                      reference_points,
                      lk_status,
                      imu_rotation);

        std::vector<cv::Point2f> pts_prev_good, pts_curr_good;
        pts_prev_good.reserve(lk_status.size());
        pts_curr_good.reserve(lk_status.size());
        for (std::size_t i = 0; i < lk_status.size(); ++i) {
            if (lk_status[i]) {
                pts_prev_good.push_back(reference_points[i]);
                pts_curr_good.push_back(tracked_now[i]);
            }
        }

        outcome.tracks_for_current = pts_curr_good;
        outcome.result.tracked_points = static_cast<int>(pts_curr_good.size());

        double mean_flow_px = 0.0;
        if (!pts_prev_good.empty()) {
            double sum_sq = 0.0;
            for (std::size_t i = 0; i < pts_prev_good.size(); ++i) {
                const double dx = pts_curr_good[i].x - pts_prev_good[i].x;
                const double dy = pts_curr_good[i].y - pts_prev_good[i].y;
                sum_sq += dx * dx + dy * dy;
            }
            mean_flow_px = std::sqrt(sum_sq / static_cast<double>(pts_prev_good.size()));
        }
        outcome.result.mean_flow_px = mean_flow_px;

        if (!pts_prev_good.empty() && mean_flow_px < min_flow_px_) {
            outcome.result.failure_reason = "stationary_guard";
            return outcome;
        }

        const int n_good = static_cast<int>(pts_curr_good.size());
        outcome.result.failure_reason = (n_good >= min_tracked_pts_) ? "lk_essential_failed" : "insufficient_lk_tracks";

        if (n_good >= min_tracked_pts_) {
            cv::Mat R, t;
            int inliers = 0;
            if (estimatePoseEssential(pts_prev_good, pts_curr_good, R, t, inliers)) {
                // Scale the unit-vector translation to metric using odom_displacement.
                // MONO mode: wheel odometry is the authoritative metric scale source.
                //   → caller passes a valid odom_displacement; t is scaled here.
                // MONO_IMU mode: the GTSAM ImuFactor is the authoritative scale source.
                //   → caller (CameraModule::preProcess) passes std::nullopt, so this
                //     block is skipped and the VO edge stays as a unit-vector constraint.
                // Both modes are correct: the scaling is determined by what the caller
                // passes, not by conditional logic inside the frontend.
                if (odom_displacement.has_value() && *odom_displacement > 1e-4) {
                    const double clamped = std::clamp(*odom_displacement, 0.05, 3.0);
                    t = t * clamped;
                } else {
                    const double trace_r = R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2);
                    const double theta = std::acos(std::clamp((trace_r - 1.0) / 2.0, -1.0, 1.0));
                    if (theta > 0.015 && mean_flow_px < 25.0) {
                        t = t * std::clamp(mean_flow_px / 25.0, 0.1, 1.0);
                    }
                }
                outcome.result.valid = true;
                outcome.result.method = Method::LK_ESSENTIAL;
                outcome.result.inliers = inliers;
                outcome.result.tracked_points = static_cast<int>(n_good);
                outcome.result.inlier_ratio = (n_good > 0)
                    ? static_cast<double>(inliers) / static_cast<double>(n_good)
                    : 0.0;
                // IMU rotation is used only for LK seeding and parallax compensation (above).
                // The VO-recovered R is the authoritative rotation for the transform edge.
                // In mono_imu mode, the GTSAM ImuFactor refines rotation independently.
                outcome.result.relative_transform = slam::utils::rtToTransform(R, t);
                outcome.result.confidence = computeConfidence(Method::LK_ESSENTIAL, inliers, n_good);
                outcome.result.failure_reason = "none";
                if (imu_rotation.has_value()) {
                    outcome.result.unit_sphere_parallax = computeUnitSphereParallax(
                        pts_prev_good, pts_curr_good, camera_matrix_, imu_rotation);
                } else if (!R.empty()) {
                    Eigen::Matrix3d R_vo;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            R_vo(r, c) = R.at<double>(r, c);
                        }
                    }
                    outcome.result.unit_sphere_parallax = computeUnitSphereParallax(
                        pts_prev_good, pts_curr_good, camera_matrix_, R_vo);
                }
                return outcome;
            }
        }

        if (!reference.descriptors.empty() && !current_state_.descriptors.empty() &&
            reference.descriptors.type() == current_state_.descriptors.type())
        {
            const int norm = (current_state_.descriptors.type() == CV_8U)
                             ? cv::NORM_HAMMING : cv::NORM_L2;
            cv::BFMatcher bf_matcher(norm, false);

            std::vector<std::vector<cv::DMatch>> knn_matches;
            bf_matcher.knnMatch(reference.descriptors,
                                current_state_.descriptors,
                                knn_matches, 2);

            std::vector<cv::Point2f> m_prev, m_curr;
            m_prev.reserve(knn_matches.size());
            m_curr.reserve(knn_matches.size());

            for (const auto& pair : knn_matches) {
                if (pair.size() < 2) { continue; }
                if (pair[0].distance < 0.75f * pair[1].distance) {
                    m_prev.push_back(reference.keypoints[pair[0].queryIdx].pt);
                    m_curr.push_back(current_state_.keypoints[pair[0].trainIdx].pt);
                }
            }

            outcome.result.fallback_match_count = static_cast<int>(m_prev.size());

            cv::Mat R, t;
            int inliers = 0;
            if (static_cast<int>(m_prev.size()) >= 20 &&
                estimatePoseEssential(m_prev, m_curr, R, t, inliers))
            {
                // Same GPS metric scaling as LK path.
                if (odom_displacement.has_value() && *odom_displacement > 1e-4) {
                    const double clamped = std::clamp(*odom_displacement, 0.05, 3.0);
                    t = t * clamped;
                } else {
                    const double trace_r = R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2);
                    const double theta = std::acos(std::clamp((trace_r - 1.0) / 2.0, -1.0, 1.0));
                    if (theta > 0.015 && mean_flow_px < 25.0) {
                        t = t * std::clamp(mean_flow_px / 25.0, 0.1, 1.0);
                    }
                }
                outcome.result.valid = true;
                outcome.result.method = Method::FEATURE_ESSENTIAL;
                outcome.result.tracked_points = static_cast<int>(m_prev.size());
                outcome.result.inliers = inliers;
                outcome.result.inlier_ratio = (!m_prev.empty())
                    ? static_cast<double>(inliers) / static_cast<double>(m_prev.size())
                    : 0.0;
                // Same as LK path: VO R is authoritative, IMU owns rotation in the graph.
                outcome.result.relative_transform = slam::utils::rtToTransform(R, t);
                outcome.result.confidence = computeConfidence(Method::FEATURE_ESSENTIAL,
                                                              inliers,
                                                              static_cast<int>(m_prev.size()));
                outcome.result.failure_reason = "none";
                if (imu_rotation.has_value()) {
                    outcome.result.unit_sphere_parallax = computeUnitSphereParallax(
                        m_prev, m_curr, camera_matrix_, imu_rotation);
                } else if (!R.empty()) {
                    Eigen::Matrix3d R_vo;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            R_vo(r, c) = R.at<double>(r, c);
                        }
                    }
                    outcome.result.unit_sphere_parallax = computeUnitSphereParallax(
                        m_prev, m_curr, camera_matrix_, R_vo);
                }
                return outcome;
            }
        }

        if (outcome.result.fallback_match_count > 0) {
            outcome.result.failure_reason = "feature_fallback_failed";
        }
        return outcome;
    };

    std::vector<std::reference_wrapper<const FrameState>> candidates;
    for (auto it = good_reference_states_.rbegin();
         it != good_reference_states_.rend() && candidates.size() < temporal_fallback_window_;
         ++it) {
        candidates.emplace_back(*it);
    }
    if (candidates.empty() && prev_state_.has_value()) {
        candidates.emplace_back(*prev_state_);
    }

    AttemptOutcome best_failure;
    bool have_failure = false;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        auto outcome = attemptFromReference(candidates[index].get(), static_cast<int>(index + 1));
        if (outcome.result.valid) {
            current_state_.tracked_pts = std::move(outcome.tracks_for_current);
            if (static_cast<int>(current_state_.tracked_pts.size()) < reinit_threshold_) {
                reinitTracks(current_state_.image_gray);
            }

            auto safe_copy = makeFrameCopy(current_state_);
            prev_state_ = safe_copy;
            good_reference_states_.push_back(safe_copy);
            while (good_reference_states_.size() > temporal_fallback_window_) {
                good_reference_states_.pop_front();
            }
            return outcome.result;
        }

        if (!have_failure || outcome.result.tracked_points > best_failure.result.tracked_points) {
            best_failure = std::move(outcome);
            have_failure = true;
        }
    }

    current_state_.tracked_pts = have_failure ? std::move(best_failure.tracks_for_current)
                                              : std::vector<cv::Point2f>{};
    if (static_cast<int>(current_state_.tracked_pts.size()) < reinit_threshold_) {
        reinitTracks(current_state_.image_gray);
    }
    prev_state_ = makeFrameCopy(current_state_);
    return have_failure ? best_failure.result : VOResult{};
}

// ─────────────────────────────────────────────────────────────────────────────
// LK Optical Flow — with forward-backward consistency filter & IMU seeding
// ─────────────────────────────────────────────────────────────────────────────

std::vector<cv::Point2f> VisualOdometryFrontend::runLKFlow(
    const cv::Mat&                  prev_gray,
    const cv::Mat&                  curr_gray,
    const std::vector<cv::Point2f>& prev_pts,
    std::vector<uchar>&             status_out,
    const std::optional<Eigen::Matrix3d>& imu_rotation) const
{
    if (prev_pts.empty()) {
        status_out.clear();
        return {};
    }

    std::vector<cv::Point2f> curr_pts_guess = prev_pts;
    int optflow_flags = 0;
    if (imu_rotation.has_value()) {
        std::vector<cv::Point3f> u_prev;
        unprojectToUnitSphere(prev_pts, camera_matrix_, u_prev);
        const double fx = camera_matrix_.at<double>(0, 0);
        const double fy = camera_matrix_.at<double>(1, 1);
        const double cx = camera_matrix_.at<double>(0, 2);
        const double cy = camera_matrix_.at<double>(1, 2);

        for (std::size_t i = 0; i < u_prev.size(); ++i) {
            Eigen::Vector3d v_p(u_prev[i].x, u_prev[i].y, u_prev[i].z);
            // imu_rotation is R_{c_t}^{c_0} (body kinematics integration).
            // To map a point from c_0 to c_t, we need its transpose (inverse).
            Eigen::Vector3d v_rot = imu_rotation->transpose() * v_p;
            if (v_rot.z() > 1e-4) {
                const double xn = v_rot.x() / v_rot.z();
                const double yn = v_rot.y() / v_rot.z();
                curr_pts_guess[i] = cv::Point2f(static_cast<float>(fx * xn + cx),
                                                static_cast<float>(fy * yn + cy));
            }
        }
        optflow_flags = cv::OPTFLOW_USE_INITIAL_FLOW;
    }

    // Forward pass: prev → current
    std::vector<cv::Point2f> curr_pts = curr_pts_guess;
    std::vector<uchar>  fwd_status;
    std::vector<float>  fwd_err;
    cv::calcOpticalFlowPyrLK(
        prev_gray, curr_gray,
        prev_pts, curr_pts,
        fwd_status, fwd_err,
        cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.005),
        optflow_flags);

    // Backward pass: current → prev (consistency check)
    std::vector<cv::Point2f> prev_pts_back = prev_pts;
    int bwd_flags = 0;
    if (imu_rotation.has_value()) {
        std::vector<cv::Point3f> u_curr;
        unprojectToUnitSphere(curr_pts, camera_matrix_, u_curr);
        const double fx = camera_matrix_.at<double>(0, 0);
        const double fy = camera_matrix_.at<double>(1, 1);
        const double cx = camera_matrix_.at<double>(0, 2);
        const double cy = camera_matrix_.at<double>(1, 2);

        for (std::size_t i = 0; i < u_curr.size(); ++i) {
            Eigen::Vector3d v_c(u_curr[i].x, u_curr[i].y, u_curr[i].z);
            // map point from c_t to c_0 using the direct rotation matrix.
            Eigen::Vector3d v_rot = (*imu_rotation) * v_c;
            if (v_rot.z() > 1e-4) {
                const double xn = v_rot.x() / v_rot.z();
                const double yn = v_rot.y() / v_rot.z();
                prev_pts_back[i] = cv::Point2f(static_cast<float>(fx * xn + cx),
                                               static_cast<float>(fy * yn + cy));
            }
        }
        bwd_flags = cv::OPTFLOW_USE_INITIAL_FLOW;
    }

    std::vector<uchar>  bwd_status;
    std::vector<float>  bwd_err;
    cv::calcOpticalFlowPyrLK(
        curr_gray, prev_gray,
        curr_pts, prev_pts_back,
        bwd_status, bwd_err,
        cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.005),
        bwd_flags);

    // Accept track only when both passes succeed and FB error < 0.5 pixel
    constexpr float kFBThreshold = 0.5f;
    status_out.assign(prev_pts.size(), 0);

    for (std::size_t i = 0; i < prev_pts.size(); ++i) {
        if (!fwd_status[i] || !bwd_status[i]) { continue; }
        const float dx = prev_pts[i].x - prev_pts_back[i].x;
        const float dy = prev_pts[i].y - prev_pts_back[i].y;
        if ((dx * dx + dy * dy) < (kFBThreshold * kFBThreshold)) {
            status_out[i] = 1;
        }
    }

    return curr_pts;
}

// ─────────────────────────────────────────────────────────────────────────────
// Essential Matrix + recoverPose on Normalized Unit Sphere
// ─────────────────────────────────────────────────────────────────────────────

bool VisualOdometryFrontend::estimatePoseEssential(
    const std::vector<cv::Point2f>& pts_prev,
    const std::vector<cv::Point2f>& pts_curr,
    cv::Mat& R_out,
    cv::Mat& t_out,
    int&     inliers_out) const
{
    if (pts_prev.size() < 5 || pts_curr.size() < 5) {
        return false;
    }

    std::vector<cv::Point3f> u_prev, u_curr;
    unprojectToUnitSphere(pts_prev, camera_matrix_, u_prev);
    unprojectToUnitSphere(pts_curr, camera_matrix_, u_curr);

    std::vector<cv::Point2f> norm_prev, norm_curr;
    norm_prev.reserve(u_prev.size());
    norm_curr.reserve(u_curr.size());
    for (std::size_t i = 0; i < u_prev.size(); ++i) {
        norm_prev.emplace_back(u_prev[i].x / u_prev[i].z, u_prev[i].y / u_prev[i].z);
        norm_curr.emplace_back(u_curr[i].x / u_curr[i].z, u_curr[i].y / u_curr[i].z);
    }

    cv::Mat inlier_mask;
    cv::Mat K_eye = cv::Mat::eye(3, 3, CV_64F);
    const double fx = camera_matrix_.at<double>(0, 0);
    const double norm_thresh = 1.2 / std::max(100.0, fx);

    const cv::Mat E = cv::findEssentialMat(
        norm_prev, norm_curr, K_eye,
        cv::RANSAC,
        0.9999,      // confidence
        norm_thresh, // normalized reprojection threshold
        inlier_mask);

    if (E.empty()) { return false; }

    inliers_out = cv::recoverPose(
        E, norm_prev, norm_curr, K_eye,
        R_out, t_out, inlier_mask);

    return (inliers_out >= 20 && static_cast<double>(inliers_out) >= 0.40 * static_cast<double>(pts_prev.size()));
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-detect ShiTomasi corners into regions not already tracked
// ─────────────────────────────────────────────────────────────────────────────

void VisualOdometryFrontend::reinitTracks(const cv::Mat& gray)
{
    const int existing = static_cast<int>(current_state_.tracked_pts.size());
    const int needed   = max_corners_ - existing;
    if (needed <= 0) { return; }

    // Build exclusion mask: suppress a 7-px radius around each live track
    cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1) * 255;
    for (const auto& pt : current_state_.tracked_pts) {
        cv::circle(mask, pt, 7, cv::Scalar(0), -1);
    }

    std::vector<cv::Point2f> new_corners;
    cv::goodFeaturesToTrack(gray, new_corners,
                            needed,
                            0.01,   // quality level
                            15.0,   // min distance between corners
                            mask);

    for (const auto& c : new_corners) {
        current_state_.tracked_pts.push_back(c);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Confidence score  [0, 1]
// ─────────────────────────────────────────────────────────────────────────────

double VisualOdometryFrontend::computeConfidence(
    Method method, int inliers, int total)
{
    if (total <= 0) { return 0.0; }
    const double ratio = static_cast<double>(inliers) / static_cast<double>(total);
    switch (method) {
        // LK gives dense, high-quality correspondences → reward high inlier ratio
        case Method::LK_ESSENTIAL:      return std::min(1.0, ratio * 1.2);
        // Descriptor matching is sparser and noisier → slightly discounted
        case Method::FEATURE_ESSENTIAL: return std::min(1.0, ratio * 0.8);
        default:                        return 0.0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Accessors
// ─────────────────────────────────────────────────────────────────────────────

const std::vector<cv::KeyPoint>& VisualOdometryFrontend::currentKeypoints() const
{
    return current_state_.keypoints;
}

const cv::Mat& VisualOdometryFrontend::currentDescriptors() const
{
    return current_state_.descriptors;
}

void VisualOdometryFrontend::reset()
{
    prev_state_.reset();
    current_state_ = {};
    good_reference_states_.clear();
}

}  // namespace slam::frontend
