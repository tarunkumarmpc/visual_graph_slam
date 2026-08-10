#ifndef VISUAL_GRAPH_SLAM_RUNTIME_BOUNDED_QUEUE_HPP
#define VISUAL_GRAPH_SLAM_RUNTIME_BOUNDED_QUEUE_HPP

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>

namespace slam::runtime {

enum class QueueDropPolicy {
    DROP_OLDEST,
    DROP_NEWEST
};

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity,
                          QueueDropPolicy policy = QueueDropPolicy::DROP_OLDEST)
        : capacity_(capacity), policy_(policy) {}

    bool tryPush(const T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (capacity_ == 0) {
            return false;
        }

        if (queue_.size() >= capacity_) {
            if (policy_ == QueueDropPolicy::DROP_NEWEST) {
                return false;
            }
            queue_.pop_front();
        }

        queue_.push_back(value);
        cv_.notify_one();
        return true;
    }

    std::optional<T> waitAndPop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return shutdown_ || !queue_.empty(); });

        if (queue_.empty()) {
            return std::nullopt;
        }

        T item = queue_.front();
        queue_.pop_front();
        return item;
    }

    void shutdown() {
        std::unique_lock<std::mutex> lock(mutex_);
        shutdown_ = true;
        cv_.notify_all();
    }

private:
    std::size_t capacity_;
    QueueDropPolicy policy_;
    std::deque<T> queue_;
    bool shutdown_{false};

    std::mutex mutex_;
    std::condition_variable cv_;
};

}  // namespace slam::runtime

#endif  // VISUAL_GRAPH_SLAM_RUNTIME_BOUNDED_QUEUE_HPP
