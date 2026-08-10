#include "visual_graph_slam/backend/optimizer_backend.hpp"
#include <pluginlib/class_list_macros.hpp>

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>

namespace {

std::string toLower(std::string input) {
    std::transform(input.begin(), input.end(), input.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return input;
}

}  // namespace

namespace slam::core {

std::string G2oOptimizerBackend::name() const {
    return "g2o";
}

bool G2oOptimizerBackend::optimizePoseGraph(Graph& graph, int iterations) {
    if (graph.getNodes().size() < 3 || graph.getEdges().empty()) {
        return false;
    }

    g2o::SparseOptimizer optimizer;
    auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto block_solver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
    optimizer.setAlgorithm(algorithm);

    for (const auto& [id, node] : graph.getNodes()) {
        const auto& pose = node->getPose();
        Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
        Eigen::Vector3d t(pose.position.x, pose.position.y, pose.position.z);

        auto* vertex = new g2o::VertexSE3Expmap();
        vertex->setId(id);
        vertex->setEstimate(g2o::SE3Quat(q, t));
        if (id == 0) {
            vertex->setFixed(true);
        }
        optimizer.addVertex(vertex);
    }

    for (const auto& edge : graph.getEdges()) {
        auto* from = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(edge->getFromId()));
        auto* to = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(edge->getToId()));
        if (!from || !to) {
            continue;
        }

        const auto& tf = edge->getRelativeTransform();
        Eigen::Quaterniond q(tf.rotation.w, tf.rotation.x, tf.rotation.y, tf.rotation.z);
        Eigen::Vector3d t(tf.translation.x, tf.translation.y, tf.translation.z);

        auto* g2o_edge = new g2o::EdgeSE3Expmap();
        g2o_edge->setVertex(0, from);
        g2o_edge->setVertex(1, to);
        g2o_edge->setMeasurement(g2o::SE3Quat(q, t));
        g2o_edge->setInformation(edge->getInformationMatrix());
        
        auto* rk = new g2o::RobustKernelHuber();
        rk->setDelta(1.0);
        g2o_edge->setRobustKernel(rk);
        
        optimizer.addEdge(g2o_edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(std::max(1, iterations));

    for (const auto& [id, node] : graph.getNodes()) {
        (void)node;
        auto* vertex = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(id));
        if (!vertex) {
            continue;
        }

        const g2o::SE3Quat est = vertex->estimate();
        const Eigen::Vector3d t = est.translation();
        const Eigen::Quaterniond q = est.rotation();

        geometry_msgs::msg::Pose optimized_pose;
        optimized_pose.position.x = t.x();
        optimized_pose.position.y = t.y();
        optimized_pose.position.z = t.z();
        optimized_pose.orientation.x = q.x();
        optimized_pose.orientation.y = q.y();
        optimized_pose.orientation.z = q.z();
        optimized_pose.orientation.w = q.w();

        graph.updateNodePose(id, optimized_pose);
    }

    return true;
}

std::unique_ptr<OptimizerBackend> createOptimizerBackend(const std::string& optimizer_name) {
    const std::string normalized = toLower(optimizer_name);
    if (normalized == "g2o") {
        return std::make_unique<G2oOptimizerBackend>();
    }

    if (normalized == "gtsam") {
#if VISUAL_GRAPH_SLAM_WITH_GTSAM
        return std::make_unique<GtsamOptimizerBackend>();
#else
        throw std::invalid_argument(
            "Unsupported optimizer: gtsam. GTSAM is not available in this build. "
            "Install GTSAM or use optimizer=g2o.");
#endif
    }

    throw std::invalid_argument("Unsupported optimizer: " + optimizer_name +
                                ". Supported: g2o"
#if VISUAL_GRAPH_SLAM_WITH_GTSAM
                                ", gtsam"
#endif
                                );
}

}  // namespace slam::core

PLUGINLIB_EXPORT_CLASS(slam::core::G2oOptimizerBackend, slam::core::OptimizerBackend)
