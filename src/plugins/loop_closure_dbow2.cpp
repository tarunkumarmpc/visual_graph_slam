#include "visual_graph_slam/plugins/loop_closure.hpp"
#include "visual_graph_slam/loop_closure/loop_detector.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <stdexcept>

namespace slam::plugins {

class LoopClosureDBoW2 : public slam::core::LoopClosure {
public:
    LoopClosureDBoW2() = default;
    ~LoopClosureDBoW2() override = default;

    void initialize(rclcpp::Node::SharedPtr node, const std::string& plugin_name) override {
        node_ = node;
        logger_ = node_->get_logger();

        std::string ns = plugin_name + ".";
        std::string vocab_path = node_->declare_parameter<std::string>(ns + "vocabulary_path", "");
        temporal_gap_ = node_->declare_parameter<int>(ns + "temporal_gap", 50);
        score_threshold_ = node_->declare_parameter<double>(ns + "score_threshold", 0.05);

        RCLCPP_INFO(logger_, "Initializing LoopClosureDBoW2");

        // We use the factory we renamed in Phase 1
        detector_ = slam::loop_closure::createLoopDetector("dbow2");

        if (!vocab_path.empty()) {
            if (detector_->loadVocabulary(vocab_path)) {
                RCLCPP_INFO(logger_, "Loaded vocabulary with %zu words", detector_->vocabularySize());
                vocab_loaded_ = true;
            } else {
                RCLCPP_WARN(logger_, "Failed to load vocabulary from %s", vocab_path.c_str());
            }
        } else {
            RCLCPP_WARN(logger_, "No vocabulary path provided for DBoW2. Loop closure disabled.");
        }
    }

    bool addKeyframe(int keyframe_id, const cv::Mat& descriptors) override {
        if (!vocab_loaded_ || !detector_) return false;
        return detector_->addKeyframe(keyframe_id, descriptors);
    }

    std::optional<slam::core::LoopCandidate> detectLoop(int current_keyframe_id, const cv::Mat& current_descriptors) override {
        if (!vocab_loaded_ || !detector_) return std::nullopt;

        auto match = detector_->queryBestCandidate(
            current_keyframe_id, current_descriptors, temporal_gap_, score_threshold_, 3);

        if (match) {
            slam::core::LoopCandidate candidate;
            candidate.matched_keyframe_id = match->matched_keyframe_id;
            candidate.score = match->score;
            return candidate;
        }

        return std::nullopt;
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Logger logger_{rclcpp::get_logger("LoopClosureDBoW2")};
    std::unique_ptr<slam::loop_closure::LoopDetector> detector_;
    bool vocab_loaded_{false};
    int temporal_gap_{50};
    double score_threshold_{0.05};
};

} // namespace slam::plugins

PLUGINLIB_EXPORT_CLASS(slam::plugins::LoopClosureDBoW2, slam::core::LoopClosure)
