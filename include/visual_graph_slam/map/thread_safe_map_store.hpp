#ifndef VISUAL_GRAPH_SLAM_MAP_THREAD_SAFE_MAP_STORE_HPP
#define VISUAL_GRAPH_SLAM_MAP_THREAD_SAFE_MAP_STORE_HPP

#include <cstddef>
#include <shared_mutex>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <geometry_msgs/msg/pose.hpp>

#include "visual_graph_slam/map/map_manager.hpp"

namespace slam::map {

struct Keyframe {
    int id{-1};
    geometry_msgs::msg::Pose pose{};
    int64_t stamp_ns{0};
};


struct MapSnapshot {
    std::unordered_map<int, Keyframe> keyframes;
    std::unordered_map<int, Landmark> landmarks;
    uint64_t version{0};
};

class ThreadSafeMapStore {
public:
    uint64_t upsertKeyframe(const Keyframe& keyframe) {
        std::unique_lock lock(mutex_);
        keyframes_[keyframe.id] = keyframe;
        return ++version_;
    }

    uint64_t upsertLandmark(const Landmark& landmark) {
        std::unique_lock lock(mutex_);
        landmarks_[landmark.id] = landmark;
        return ++version_;
    }

    bool removeKeyframe(int id) {
        std::unique_lock lock(mutex_);
        const auto removed = keyframes_.erase(id);
        if (removed == 0) {
            return false;
        }
        ++version_;
        return true;
    }

    bool removeLandmark(int id) {
        std::unique_lock lock(mutex_);
        const auto removed = landmarks_.erase(id);
        if (removed == 0) {
            return false;
        }
        ++version_;
        return true;
    }

    bool getKeyframe(int id, Keyframe& out) const {
        std::shared_lock lock(mutex_);
        auto it = keyframes_.find(id);
        if (it == keyframes_.end()) {
            return false;
        }
        out = it->second;
        return true;
    }

    bool getLandmark(int id, Landmark& out) const {
        std::shared_lock lock(mutex_);
        auto it = landmarks_.find(id);
        if (it == landmarks_.end()) {
            return false;
        }
        out = it->second;
        return true;
    }

    MapSnapshot snapshot() const {
        std::shared_lock lock(mutex_);
        return MapSnapshot{keyframes_, landmarks_, version_};
    }

    std::size_t keyframeCount() const {
        std::shared_lock lock(mutex_);
        return keyframes_.size();
    }

    std::size_t landmarkCount() const {
        std::shared_lock lock(mutex_);
        return landmarks_.size();
    }

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<int, Keyframe> keyframes_;
    std::unordered_map<int, Landmark> landmarks_;
    uint64_t version_{0};
};

}  // namespace slam::map

#endif  // VISUAL_GRAPH_SLAM_MAP_THREAD_SAFE_MAP_STORE_HPP
