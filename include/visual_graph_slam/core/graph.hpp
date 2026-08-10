#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "visual_graph_slam/core/graph_node.hpp"
#include "visual_graph_slam/core/graph_edge.hpp"
#include <unordered_map>
#include <vector>
#include <memory>
#include <functional>
#include <shared_mutex>

namespace slam {

class Graph {
public:
    using Observer = std::function<void(const Graph&)>;

    Graph();
    ~Graph() = default;

    void addNode(std::shared_ptr<GraphNode> node);
    void addEdge(std::shared_ptr<GraphEdge> edge);
    bool removeLatestEdge(int from_id, int to_id, GraphEdge::Type type);
    std::shared_ptr<GraphNode> getNode(int id) const;
    bool updateNodePose(int id, const geometry_msgs::msg::Pose& pose);

    void addObserver(Observer observer);
    void removeObserver(Observer observer);

    int getLastNodeId() const;
    geometry_msgs::msg::Pose getLastNodePosition() const;

    std::unordered_map<int, std::shared_ptr<GraphNode>> getNodes() const;
    std::vector<std::shared_ptr<GraphEdge>> getEdges() const;

    bool isGraphChanged() const;
    void setGraphChanged(bool changed);

    bool isMetricScaleInitialized() const;
    void setMetricScaleInitialized(bool initialized);

    bool usePlanarConstraint() const;
    void setPlanarMotionConstraint(bool enabled);

private:
    void notifyObservers(const std::vector<Observer>& observers_snapshot);

    mutable std::shared_mutex mutex_;
    std::unordered_map<int, std::shared_ptr<GraphNode>> nodes_;
    std::vector<std::shared_ptr<GraphEdge>> edges_;
    std::vector<Observer> observers_;
    bool is_graph_changed_;
    bool metric_scale_initialized_{false};
    int last_node_id_;
    geometry_msgs::msg::Pose last_node_position_;
    bool planar_motion_constraint_{false};
};

} // namespace slam

#endif // GRAPH_HPP
