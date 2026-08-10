#ifndef VISUAL_GRAPH_SLAM_LOOP_CLOSURE_VPR_MODULE_HPP
#define VISUAL_GRAPH_SLAM_LOOP_CLOSURE_VPR_MODULE_HPP

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace slam::loop_closure {

struct VprMatchCandidate {
    int matched_keyframe_id{-1};
    double score{0.0};
};

class LoopDetector {
public:
    virtual ~LoopDetector() = default;

    virtual std::string name() const = 0;

    virtual bool loadVocabulary(const std::string& path) = 0;
    virtual bool createAndSaveVocabulary(
        const std::string& path,
        const std::vector<cv::Mat>& training_descriptors) = 0;

    virtual std::size_t vocabularySize() const = 0;

    virtual bool addKeyframe(int keyframe_id,
                             const cv::Mat& descriptors) = 0;

    virtual std::optional<VprMatchCandidate> queryBestCandidate(
        int current_keyframe_id,
        const cv::Mat& current_descriptors,
        int temporal_loop_gap,
        double min_score,
        std::size_t max_results = 10) = 0;
};

class DBoW2Detector final : public LoopDetector {
public:
    DBoW2Detector();
    ~DBoW2Detector() override;

    std::string name() const override;

    bool loadVocabulary(const std::string& path) override;
    bool createAndSaveVocabulary(
        const std::string& path,
        const std::vector<cv::Mat>& training_descriptors) override;

    std::size_t vocabularySize() const override;

    bool addKeyframe(int keyframe_id,
                     const cv::Mat& descriptors) override;

    std::optional<VprMatchCandidate> queryBestCandidate(
        int current_keyframe_id,
        const cv::Mat& current_descriptors,
        int temporal_loop_gap,
        double min_score,
        std::size_t max_results = 10) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class OrbBowDetector final : public LoopDetector {
public:
    OrbBowDetector();
    ~OrbBowDetector() override;

    std::string name() const override;

    bool loadVocabulary(const std::string& path) override;
    bool createAndSaveVocabulary(
        const std::string& path,
        const std::vector<cv::Mat>& training_descriptors) override;

    std::size_t vocabularySize() const override;

    bool addKeyframe(int keyframe_id,
                     const cv::Mat& descriptors) override;

    std::optional<VprMatchCandidate> queryBestCandidate(
        int current_keyframe_id,
        const cv::Mat& current_descriptors,
        int temporal_loop_gap,
        double min_score,
        std::size_t max_results = 10) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<LoopDetector> createLoopDetector(const std::string& module_name);

}  // namespace slam::loop_closure

#endif  // VISUAL_GRAPH_SLAM_LOOP_CLOSURE_VPR_MODULE_HPP
