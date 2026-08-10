#ifndef VISUAL_GRAPH_SLAM_BACKEND_OPTIMIZER_BACKEND_HPP
#define VISUAL_GRAPH_SLAM_BACKEND_OPTIMIZER_BACKEND_HPP

#include <memory>
#include <string>

#include "visual_graph_slam/core/graph.hpp"

namespace slam::core {

class OptimizerBackend {
public:
    virtual ~OptimizerBackend() = default;

    virtual std::string name() const = 0;
    virtual bool optimizePoseGraph(Graph& graph, int iterations) = 0;
};

class G2oOptimizerBackend final : public OptimizerBackend {
public:
    std::string name() const override;
    bool optimizePoseGraph(Graph& graph, int iterations) override;
};

class GtsamOptimizerBackend final : public OptimizerBackend {
public:
    std::string name() const override;
    bool optimizePoseGraph(Graph& graph, int iterations) override;
};

std::unique_ptr<OptimizerBackend> createOptimizerBackend(const std::string& backend_name);

}  // namespace slam::core

#endif  // VISUAL_GRAPH_SLAM_BACKEND_OPTIMIZER_BACKEND_HPP
