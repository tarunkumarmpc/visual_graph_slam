#pragma once

#include <string>
#include <optional>
#include <memory>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>

namespace slam::core {

struct LoopCandidate {
    int matched_keyframe_id;
    double score;
};

class LoopClosure {
public:
    using SharedPtr = std::shared_ptr<LoopClosure>;

    virtual ~LoopClosure() = default;

    /**
     * @brief Initialize the plugin with parameters from the lifecycle node
     */
    virtual void initialize(rclcpp::Node::SharedPtr node, const std::string& plugin_name) = 0;

    /**
     * @brief Add a keyframe's descriptors to the database
     */
    virtual bool addKeyframe(int keyframe_id, const cv::Mat& descriptors) = 0;

    /**
     * @brief Query the database for a loop closure candidate
     */
    virtual std::optional<LoopCandidate> detectLoop(int current_keyframe_id, const cv::Mat& current_descriptors) = 0;
};

} // namespace slam::core
