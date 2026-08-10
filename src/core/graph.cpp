#include "visual_graph_slam/core/graph.hpp"
#include <algorithm>
#include <mutex>

namespace slam {

Graph::Graph() : is_graph_changed_(false), last_node_id_(-1) {}

void Graph::addNode(std::shared_ptr<GraphNode> node) {
    std::vector<Observer> observers_snapshot;
    {
        std::unique_lock lock(mutex_);
        nodes_[node->getId()] = node;
        last_node_id_ = node->getId();
        last_node_position_ = node->getPose();
        is_graph_changed_ = true;
        observers_snapshot = observers_;
    }
    notifyObservers(observers_snapshot);
}

void Graph::addEdge(std::shared_ptr<GraphEdge> edge) {
    std::vector<Observer> observers_snapshot;
    {
        std::unique_lock lock(mutex_);
        edges_.push_back(edge);
        is_graph_changed_ = true;
        observers_snapshot = observers_;
    }
    notifyObservers(observers_snapshot);
}

bool Graph::removeLatestEdge(int from_id, int to_id, GraphEdge::Type type) {
    std::vector<Observer> observers_snapshot;
    {
        std::unique_lock lock(mutex_);
        auto it = std::find_if(edges_.rbegin(), edges_.rend(),
                               [&](const std::shared_ptr<GraphEdge>& e) {
                                   return e && e->getType() == type &&
                                          e->getFromId() == from_id &&
                                          e->getToId() == to_id;
                               });
        if (it == edges_.rend()) {
            return false;
        }

        edges_.erase(std::next(it).base());
        is_graph_changed_ = true;
        observers_snapshot = observers_;
    }

    notifyObservers(observers_snapshot);
    return true;
}

std::shared_ptr<GraphNode> Graph::getNode(int id) const {
    std::shared_lock lock(mutex_);
    auto it = nodes_.find(id);
    return (it != nodes_.end()) ? it->second : nullptr;
}

bool Graph::updateNodePose(int id, const geometry_msgs::msg::Pose& pose) {
    std::vector<Observer> observers_snapshot;
    {
        std::unique_lock lock(mutex_);
        auto it = nodes_.find(id);
        if (it == nodes_.end() || !it->second) {
            return false;
        }

        it->second->updatePose(pose);
        if (id == last_node_id_) {
            last_node_position_ = pose;
        }
        is_graph_changed_ = true;
        observers_snapshot = observers_;
    }

    notifyObservers(observers_snapshot);
    return true;
}

void Graph::addObserver(Observer observer) {
    std::unique_lock lock(mutex_);
    observers_.push_back(observer);
}

void Graph::removeObserver(Observer observer) {
    std::unique_lock lock(mutex_);
    observers_.erase(std::remove_if(observers_.begin(), observers_.end(),
        [&](const Observer& obs) { return obs.target<void(const Graph&)>() == observer.target<void(const Graph&)>(); }),
        observers_.end());
}

void Graph::notifyObservers(const std::vector<Observer>& observers_snapshot) {
    for (const auto& observer : observers_snapshot) {
        observer(*this);
    }
}

int Graph::getLastNodeId() const {
    std::shared_lock lock(mutex_);
    return last_node_id_;
}

geometry_msgs::msg::Pose Graph::getLastNodePosition() const {
    std::shared_lock lock(mutex_);
    return last_node_position_;
}

std::unordered_map<int, std::shared_ptr<GraphNode>> Graph::getNodes() const {
    std::shared_lock lock(mutex_);
    return nodes_;
}

std::vector<std::shared_ptr<GraphEdge>> Graph::getEdges() const {
    std::shared_lock lock(mutex_);
    return edges_;
}

bool Graph::isGraphChanged() const {
    std::shared_lock lock(mutex_);
    return is_graph_changed_;
}

void Graph::setGraphChanged(bool changed) {
    std::unique_lock lock(mutex_);
    is_graph_changed_ = changed;
}

bool Graph::isMetricScaleInitialized() const {
    std::shared_lock lock(mutex_);
    return metric_scale_initialized_;
}

void Graph::setMetricScaleInitialized(bool initialized) {
    std::unique_lock lock(mutex_);
    metric_scale_initialized_ = initialized;
}

bool Graph::usePlanarConstraint() const {
    std::shared_lock lock(mutex_);
    return planar_motion_constraint_;
}

void Graph::setPlanarMotionConstraint(bool enabled) {
    std::unique_lock lock(mutex_);
    planar_motion_constraint_ = enabled;
}

} // namespace slam
