#include "visual_graph_slam/vision_node.hpp"
#include "visual_graph_slam/geometry_utils.hpp"
#include "visual_graph_slam/frontend/visual_odometry_frontend.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <opencv2/calib3d.hpp>

namespace fs = std::filesystem;

VisionNode::VisionNode(const VisionNodeConfig& config, rclcpp::Logger logger, rclcpp::Clock::SharedPtr clock)
    : loop_detector_(nullptr),
      feature_pipeline_(nullptr),
      matcher_(cv::NORM_HAMMING, false),
      temporal_loop_gap_(config.temporal_loop_gap),
      vpr_query_max_results_(config.vpr_query_max_results),
      loop_score_threshold_(config.loop_score_threshold),
      ratio_test_threshold_(config.ratio_test_threshold),
      vpr_backend_name_(config.vpr_backend_name),
      detector_name_(config.detector_name),
      vocabulary_path_(config.vocabulary_path),
      vo_min_tracked_pts_(config.vo_min_tracked_pts),
      vo_reinit_threshold_(config.vo_reinit_threshold),
      vo_max_corners_(config.vo_max_corners),
      vo_min_flow_px_(config.vo_min_flow_px),
      camera_matrix_(config.camera_matrix.clone()),
      camera_model_name_(config.camera_model_name),
      camera_distortion_(config.camera_distortion),
      camera_model_(nullptr),
      logger_(logger),
      clock_(clock) {

    if (vocabulary_path_.empty()) {
        try {
            vocabulary_path_ =
                ament_index_cpp::get_package_share_directory("visual_graph_slam") + "/voc/orb_bow_vocab.yml.gz";
        } catch (const std::exception &) {
            vocabulary_path_ = "orb_bow_vocab.yml.gz";
        }
    }

    try {
        loop_detector_ = slam::loop_closure::createLoopDetector(vpr_backend_name_);
        RCLCPP_INFO(logger_, "VPR backend selected: %s", loop_detector_->name().c_str());
    } catch (const std::exception& e) {
        RCLCPP_WARN(logger_, "%s. Falling back to ORB BoW backend.", e.what());
        vpr_backend_name_ = "orb_bow";
        loop_detector_ = slam::loop_closure::createLoopDetector(vpr_backend_name_);
    }

    try {
        feature_pipeline_ = slam::frontend::createFeaturePipeline(detector_name_);
        RCLCPP_INFO(logger_, "Frontend detector strategy selected: %s", feature_pipeline_->name().c_str());
    } catch (const std::exception& e) {
        RCLCPP_WARN(logger_, "%s. Falling back to ORB.", e.what());
        detector_name_ = "orb";
        feature_pipeline_ = slam::frontend::createFeaturePipeline("orb");
    }

    try {
        slam::sensor::CameraIntrinsics intrinsics{
            camera_matrix_.at<double>(0, 0), camera_matrix_.at<double>(1, 1),
            camera_matrix_.at<double>(0, 2), camera_matrix_.at<double>(1, 2)
        };
        camera_model_ = slam::sensor::createCameraModel(camera_model_name_, intrinsics, camera_distortion_);
        RCLCPP_INFO(logger_, "Vision camera model selected: %s", camera_model_->name().c_str());
    } catch (const std::exception& e) {
        RCLCPP_WARN(logger_, "%s. Falling back to pinhole camera model.", e.what());
        camera_model_name_ = "pinhole";
        slam::sensor::CameraIntrinsics intrinsics{
            camera_matrix_.at<double>(0, 0), camera_matrix_.at<double>(1, 1),
            camera_matrix_.at<double>(0, 2), camera_matrix_.at<double>(1, 2)
        };
        camera_model_ = slam::sensor::createCameraModel(camera_model_name_, intrinsics, Eigen::Vector4d::Zero());
    }

    RCLCPP_INFO(logger_, "Initializing visual frontend and vocabulary...");

    auto vo_pipeline = slam::frontend::createFeaturePipeline(detector_name_);
    vo_frontend_ = std::make_unique<slam::frontend::VisualOdometryFrontend>(
        camera_matrix_.clone(),
        std::move(vo_pipeline),
        vo_min_tracked_pts_,
        vo_reinit_threshold_,
        vo_max_corners_,
        vo_min_flow_px_);

    RCLCPP_INFO(logger_,
                "VO frontend ready (min_tracked=%d, reinit=%d, max_corners=%d, min_flow_px=%.2f)",
                vo_min_tracked_pts_, vo_reinit_threshold_, vo_max_corners_, vo_min_flow_px_);

    (void)ensure_vocabulary_ready();
}

void VisionNode::updateCameraCalibration(const sensor_msgs::msg::CameraInfo& camera_info) {
    if (camera_info.k.size() != 9) {
        RCLCPP_WARN(logger_, "camera_info ignored: invalid K matrix size");
        return;
    }

    const double fx = camera_info.k[0];
    const double fy = camera_info.k[4];
    const double cx = camera_info.k[2];
    const double cy = camera_info.k[5];

    if (fx <= 1e-9 || fy <= 1e-9) {
        RCLCPP_WARN(logger_, "camera_info ignored: invalid focal lengths");
        return;
    }

    // Check whether the intrinsics actually changed before rebuilding anything.
    // camera_info often just confirms the values already loaded from YAML.
    // Recreating vo_frontend_ wipes prev_state_ and all tracked points, causing
    // one dropped VO frame — avoid this when the values are unchanged.
    const double cur_fx = camera_matrix_.at<double>(0, 0);
    const double cur_fy = camera_matrix_.at<double>(1, 1);
    const double cur_cx = camera_matrix_.at<double>(0, 2);
    const double cur_cy = camera_matrix_.at<double>(1, 2);
    constexpr double kIntrinsicChangeTol = 0.5;  // pixels
    const bool intrinsics_changed =
        std::abs(fx - cur_fx) > kIntrinsicChangeTol ||
        std::abs(fy - cur_fy) > kIntrinsicChangeTol ||
        std::abs(cx - cur_cx) > kIntrinsicChangeTol ||
        std::abs(cy - cur_cy) > kIntrinsicChangeTol;

    camera_matrix_.at<double>(0, 0) = fx;
    camera_matrix_.at<double>(1, 1) = fy;
    camera_matrix_.at<double>(0, 2) = cx;
    camera_matrix_.at<double>(1, 2) = cy;

    try {
        slam::sensor::CameraIntrinsics intrinsics{fx, fy, cx, cy};
        camera_model_ = slam::sensor::createCameraModel(camera_model_name_, intrinsics, camera_distortion_);
    } catch (const std::exception& e) {
        RCLCPP_WARN(logger_, "camera model update failed: %s", e.what());
        return;
    }

    if (intrinsics_changed) {
        // Intrinsics genuinely changed — must rebuild VO frontend with new K matrix.
        auto vo_pipeline = slam::frontend::createFeaturePipeline(detector_name_);
        vo_frontend_ = std::make_unique<slam::frontend::VisualOdometryFrontend>(
            camera_matrix_.clone(),
            std::move(vo_pipeline),
            vo_min_tracked_pts_,
            vo_reinit_threshold_,
            vo_max_corners_,
            vo_min_flow_px_);
        RCLCPP_INFO(logger_,
                    "Vision calibration updated from /camera_info (fx=%.2f fy=%.2f cx=%.2f cy=%.2f) — VO frontend rebuilt.",
                    fx, fy, cx, cy);
    } else {
        RCLCPP_INFO(logger_,
                    "Vision calibration updated from /camera_info (fx=%.2f fy=%.2f cx=%.2f cy=%.2f) — intrinsics unchanged, VO state preserved.",
                    fx, fy, cx, cy);
    }
}

bool VisionNode::load_or_create_vocabulary() {
    if (vocabulary_path_.empty()) {
        RCLCPP_WARN(logger_, "No vocabulary path provided. Loop closure VPR will stay disabled.");
        return false;
    }
    const std::string vocabulary_path = vocabulary_path_;
    fs::path vocab_dir = fs::path(vocabulary_path).parent_path();
    if (!fs::exists(vocab_dir)) {
        fs::create_directories(vocab_dir);
        RCLCPP_INFO(logger_, "Created vocabulary directory: %s", vocab_dir.string().c_str());
    }

    if (fs::exists(vocabulary_path)) {
        try {
            RCLCPP_INFO(logger_, "Attempting to load vocabulary from: %s", vocabulary_path.c_str());
            bool loaded;
            {
                std::lock_guard<std::mutex> lock(vpr_mutex_);
                loaded = loop_detector_->loadVocabulary(vocabulary_path);
            }
            if (loaded) {
                std::size_t vocab_size;
                {
                    std::lock_guard<std::mutex> lock(vpr_mutex_);
                    vocab_size = loop_detector_->vocabularySize();
                }
                RCLCPP_INFO(logger_,
                            "Loaded VPR vocabulary from: %s (%zu words).",
                            vocabulary_path.c_str(),
                            vocab_size);
                return true;
            }
            RCLCPP_WARN(logger_,
                        "Vocabulary file exists but produced empty vocabulary. "
                        "Loop closure VPR will be unavailable until a valid vocabulary is provided.");
            return false;
        } catch (const std::exception &e) {
            RCLCPP_WARN(logger_,
                        "Failed to load vocabulary: %s. "
                        "Loop closure VPR will be unavailable until a valid vocabulary is provided.",
                        e.what());
            return false;
        } catch (...) {
            RCLCPP_WARN(logger_,
                        "Failed to load vocabulary due to unknown exception. "
                        "Loop closure VPR will be unavailable until a valid vocabulary is provided.");
            return false;
        }
    } else {
        RCLCPP_WARN(logger_, "Vocabulary file not found: %s. Loop closure VPR will stay disabled.", vocabulary_path.c_str());
        return false;
    }
}

bool VisionNode::ensure_vocabulary_ready() {
    if (vocabulary_initialized_) {
        return vocabulary_available_;
    }

    if (!vocabulary_load_started_) {
        vocabulary_load_started_ = true;
        RCLCPP_INFO(logger_,
                    "Starting asynchronous VPR vocabulary load from: %s",
                    vocabulary_path_.c_str());
        vocabulary_load_future_ = std::async(std::launch::async, [this]() {
            return load_or_create_vocabulary();
        });
        return false;
    }

    if (vocabulary_load_future_.valid() &&
        vocabulary_load_future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
        vocabulary_available_ = vocabulary_load_future_.get();
        vocabulary_initialized_ = true;
        if (vocabulary_available_) {
            std::size_t vocab_size;
            {
                std::lock_guard<std::mutex> lock(vpr_mutex_);
                vocab_size = loop_detector_->vocabularySize();
            }
            RCLCPP_INFO(logger_,
                        "VPR vocabulary is ready (%zu words).",
                        vocab_size);
        } else {
            RCLCPP_WARN(logger_,
                        "VPR vocabulary load failed; loop closure remains disabled.");
        }
        return vocabulary_available_;
    }

    RCLCPP_INFO_THROTTLE(logger_, *clock_, 3000,
                         "VPR vocabulary is still loading in background...");
    return false;
}



VisionNode::FrontendResult VisionNode::processFrame(
    const cv::Mat& image_bgr_or_gray,
    std::optional<double> odom_displacement)
{
    FrontendResult result;

    // Run the hybrid LK-flow + Essential-matrix VO frontend.
    // odom_displacement anchors the unit-normalised translation to metric scale.
    const auto vo = vo_frontend_->track(image_bgr_or_gray, odom_displacement);

    result.tracked_points = vo.tracked_points;
    result.inliers = vo.inliers;
    result.inlier_ratio = vo.inlier_ratio;
    result.confidence = vo.confidence;
    result.failure_reason = vo.failure_reason;
    result.mean_flow_px = vo.mean_flow_px;
    result.fallback_match_count = vo.fallback_match_count;
    result.reference_age = vo.reference_age;

    if (!vo.valid) {
        return result;  // first frame or estimation failure
    }

    result.pose_estimated     = true;
    result.relative_transform = vo.relative_transform;

    using VOMethod = slam::frontend::VisualOdometryFrontend::Method;
    switch (vo.method) {
        case VOMethod::LK_ESSENTIAL:
            result.method = FrontendResult::Method::LK_ESSENTIAL;      break;
        case VOMethod::FEATURE_ESSENTIAL:
            result.method = FrontendResult::Method::FEATURE_ESSENTIAL;  break;
        default:
            result.method = FrontendResult::Method::NONE;               break;
    }

    return result;
}

void VisionNode::addKeyframe(int keyframe_id,
                             const cv::Mat& image_bgr_or_gray,
                             const geometry_msgs::msg::Pose& pose,
                             const rclcpp::Time& timestamp) {
    KeyframeRecord keyframe;
    keyframe.id = keyframe_id;
    keyframe.pose = pose;
    keyframe.timestamp = timestamp;
    keyframe.image_gray = toGray(image_bgr_or_gray);

    extractFeatures(keyframe.image_gray, keyframe.keypoints, keyframe.descriptors);

    if (!keyframe.descriptors.empty() && ensure_vocabulary_ready()) {
            bool success;
            std::size_t vocab_size;
            {
                std::lock_guard<std::mutex> lock(vpr_mutex_);
                success = loop_detector_->addKeyframe(keyframe_id, keyframe.descriptors);
                vocab_size = loop_detector_->vocabularySize();
            }
            if (success) {
            keyframe.db_entry_id = keyframe_id;
        } else {
            RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000,
                                 "VPR index add failed for keyframe %d (desc_rows=%d, vocab_words=%zu)",
                                 keyframe_id,
                                 keyframe.descriptors.rows,
                                 vocab_size);
        }
    }

    keyframes_[keyframe_id] = keyframe;
}

std::optional<VisionNode::LoopClosureResult> VisionNode::detectLoopClosureWithVpr(int current_keyframe_id) {
    if (!ensure_vocabulary_ready()) {
        RCLCPP_INFO_THROTTLE(logger_, *clock_, 3000,
                             "Loop closure check skipped: VPR vocabulary not ready yet.");
        return std::nullopt;
    }

    auto current_it = keyframes_.find(current_keyframe_id);
    if (current_it == keyframes_.end()) {
        return std::nullopt;
    }

    const auto& current = current_it->second;
    if (current.descriptors.empty()) {
        RCLCPP_INFO_THROTTLE(logger_, *clock_, 3000,
                             "Loop closure check skipped: current keyframe %d has empty descriptors.",
                             current_keyframe_id);
        return std::nullopt;
    }

    std::optional<slam::loop_closure::VprMatchCandidate> vpr_candidate;
    {
        std::lock_guard<std::mutex> lock(vpr_mutex_);
        vpr_candidate = loop_detector_->queryBestCandidate(
            current_keyframe_id,
            current.descriptors,
            temporal_loop_gap_,
            0.0,
            static_cast<std::size_t>(std::max(1, vpr_query_max_results_)));
    }

    if (!vpr_candidate) {
        RCLCPP_INFO_THROTTLE(logger_, *clock_, 3000,
                             "No VPR loop candidate (kf=%d, indexed_kf=%zu, temporal_gap=%d, score_th=%.3f, topk=%d).",
                             current_keyframe_id,
                             keyframes_.size(),
                             temporal_loop_gap_,
                             loop_score_threshold_,
                             vpr_query_max_results_);
        return std::nullopt;
    }

    if (vpr_candidate->score < loop_score_threshold_) {
        RCLCPP_INFO_THROTTLE(logger_, *clock_, 3000,
                             "VPR best candidate below threshold (kf=%d, best=%d, score=%.4f < th=%.4f, topk=%d).",
                             current_keyframe_id,
                             vpr_candidate->matched_keyframe_id,
                             vpr_candidate->score,
                             loop_score_threshold_,
                             vpr_query_max_results_);
        return std::nullopt;
    }

    auto candidate_it = keyframes_.find(vpr_candidate->matched_keyframe_id);
    if (candidate_it == keyframes_.end()) {
        RCLCPP_INFO_THROTTLE(logger_, *clock_, 3000,
                             "VPR candidate keyframe %d not found in cache.",
                             vpr_candidate->matched_keyframe_id);
        return std::nullopt;
    }

    LoopClosureResult closure;
    closure.current_keyframe_id = current_keyframe_id;
    closure.matched_keyframe_id = vpr_candidate->matched_keyframe_id;
    closure.score = vpr_candidate->score;

    int inliers = 0;
    int matches = 0;
    if (!estimateRelativeTransformFromMatches(candidate_it->second,
                                              current,
                                              closure.relative_transform,
                                              inliers,
                                              matches)) {
        RCLCPP_INFO_THROTTLE(logger_, *clock_, 3000,
                             "VPR candidate rejected by geometry estimation (from=%d to=%d).",
                             vpr_candidate->matched_keyframe_id,
                             current_keyframe_id);
        return std::nullopt;
    }

    closure.inliers = inliers;
    closure.matches = matches;
    closure.inlier_ratio = (matches > 0)
                               ? static_cast<double>(inliers) / static_cast<double>(matches)
                               : 0.0;

    // IMPORTANT:
    // This function reports a VPR+geometry candidate. Final acceptance is
    // intentionally decided by GraphSlam::shouldAcceptLoopClosure(), which
    // applies stricter global-consistency gates (ratio, rotation, sim3,
    // distance sanity). Keep this as a candidate-level signal to avoid
    // confusing "detected" semantics.
    closure.detected = true;

    return closure;
}

void VisionNode::extractFeatures(const cv::Mat& image_bgr_or_gray,
                                 std::vector<cv::KeyPoint>& keypoints,
                                 cv::Mat& descriptors) const {
    feature_pipeline_->extract(image_bgr_or_gray, keypoints, descriptors);
}

const std::vector<cv::KeyPoint>& VisionNode::currentKeypoints() const {
    return vo_frontend_->currentKeypoints();
}

const cv::Mat& VisionNode::currentDescriptors() const {
    return vo_frontend_->currentDescriptors();
}

void VisionNode::resetFrontend() {
    if (vo_frontend_) {
        vo_frontend_->reset();
    }
}



bool VisionNode::estimateRelativeTransformFromMatches(const KeyframeRecord& from,
                                                      const KeyframeRecord& to,
                                                      geometry_msgs::msg::Transform& transform_out,
                                                      int& inliers_out,
                                                      int& matches_out) const {
    inliers_out = 0;
    matches_out = 0;

    if (from.descriptors.empty() || to.descriptors.empty()) {
        return false;
    }

    if (from.descriptors.type() != to.descriptors.type()) {
        return false;
    }

    const int norm = (from.descriptors.type() == CV_8U) ? cv::NORM_HAMMING : cv::NORM_L2;
    cv::BFMatcher matcher(norm, false);

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(from.descriptors, to.descriptors, knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    good_matches.reserve(knn_matches.size());
    for (const auto& pair : knn_matches) {
        if (pair.size() < 2) {
            continue;
        }
        if (pair[0].distance < ratio_test_threshold_ * pair[1].distance) {
            good_matches.push_back(pair[0]);
        }
    }

    matches_out = static_cast<int>(good_matches.size());
    if (good_matches.size() < 12) {
        return false;
    }

    std::vector<cv::Point2f> pts_from;
    std::vector<cv::Point2f> pts_to;
    pts_from.reserve(good_matches.size());
    pts_to.reserve(good_matches.size());
    for (const auto& m : good_matches) {
        pts_from.push_back(from.keypoints[m.queryIdx].pt);
        pts_to.push_back(to.keypoints[m.trainIdx].pt);
    }

    std::vector<cv::Point3f> u_from, u_to;
    slam::frontend::VisualOdometryFrontend::unprojectToUnitSphere(pts_from, camera_matrix_, u_from);
    slam::frontend::VisualOdometryFrontend::unprojectToUnitSphere(pts_to, camera_matrix_, u_to);

    std::vector<cv::Point2f> norm_from, norm_to;
    norm_from.reserve(u_from.size());
    norm_to.reserve(u_to.size());
    for (std::size_t i = 0; i < u_from.size(); ++i) {
        norm_from.emplace_back(u_from[i].x / u_from[i].z, u_from[i].y / u_from[i].z);
        norm_to.emplace_back(u_to[i].x / u_to[i].z, u_to[i].y / u_to[i].z);
    }

    cv::Mat inlier_mask;
    cv::Mat K_eye = cv::Mat::eye(3, 3, CV_64F);
    const double fx = camera_matrix_.at<double>(0, 0);
    const double norm_thresh = 1.5 / std::max(100.0, fx);

    cv::Mat essential = cv::findEssentialMat(norm_from, norm_to, K_eye, cv::RANSAC, 0.999, norm_thresh, inlier_mask);
    if (essential.empty()) {
        return false;
    }

    cv::Mat rotation;
    cv::Mat translation;
    inliers_out = cv::recoverPose(essential, norm_from, norm_to, K_eye, rotation, translation, inlier_mask);
    if (inliers_out < 10) {
        return false;
    }

    // Yield unit translation from Essential Matrix. Metric scaling will be
    // handled externally by GraphSlam if odom_displacement is available.

    transform_out = slam::utils::rtToTransform(rotation, translation);
    return true;
}



cv::Mat VisionNode::toGray(const cv::Mat& image_bgr_or_gray) {
    if (image_bgr_or_gray.channels() == 1) {
        return image_bgr_or_gray.clone();
    }

    cv::Mat gray;
    cv::cvtColor(image_bgr_or_gray, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

