#include "visual_graph_slam/backend/optimizer_backend.hpp"
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <string>

namespace slam::core {

std::unique_ptr<OptimizerBackend> createOptimizerBackend(const std::string& backend_name) {
    static pluginlib::ClassLoader<slam::core::OptimizerBackend> loader("visual_graph_slam", "slam::core::OptimizerBackend");
    try {
        if (backend_name == "gtsam") {
            return std::unique_ptr<slam::core::OptimizerBackend>(
                loader.createUniqueInstance("slam::core::GtsamOptimizerBackend").release());
        } else {
            return std::unique_ptr<slam::core::OptimizerBackend>(
                loader.createUniqueInstance("slam::core::G2oOptimizerBackend").release());
        }
    } catch (const pluginlib::PluginlibException& ex) {
        RCLCPP_ERROR(rclcpp::get_logger("OptimizerFactory"), "Plugin failed to load: %s", ex.what());
        return nullptr;
    }
}

}  // namespace slam::core
