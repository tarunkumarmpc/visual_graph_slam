#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include "visual_graph_slam/core/graph_slam.hpp"
#include "visual_graph_slam/slam_visualizer.hpp"
#include "visual_graph_slam/sensor_data_manager.hpp"


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto graph_slam = std::make_shared<slam::GraphSlam>();
    graph_slam->initializeSystem();
    graph_slam->initializeVisualizer(); 
    auto sensor_data_manager = std::make_shared<slam::SensorDataManager>(graph_slam);


    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(graph_slam);
    executor.add_node(sensor_data_manager);    
    //executor.add_node(visualizer);    

    executor.spin();

    rclcpp::shutdown();
    return 0;
}