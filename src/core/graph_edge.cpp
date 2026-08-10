#include<visual_graph_slam/core/graph_edge.hpp>

namespace slam {

GraphEdge::GraphEdge(Type type, int from_id, int to_id, 
                     const geometry_msgs::msg::Transform& relative_transform,
                     const Eigen::Matrix<double, 6, 6>& information_matrix)
    : type_(type), from_id_(from_id), to_id_(to_id),
      relative_transform_(relative_transform), information_matrix_(information_matrix) {}

GraphEdge::Type GraphEdge::getType() const {
    return type_;
}

int GraphEdge::getFromId() const {
    return from_id_;
}

int GraphEdge::getToId() const {
    return to_id_;
}

geometry_msgs::msg::Transform GraphEdge::getRelativeTransform() const {
    return relative_transform_;
}

Eigen::Matrix<double, 6, 6> GraphEdge::getInformationMatrix() const {
    return information_matrix_;
}

} // namespace slam

