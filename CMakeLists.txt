cmake_minimum_required(VERSION 3.5)
project(visual_graph_slam)

# Set policy for OpenGL GLVND
if(POLICY CMP0072)
  cmake_policy(SET CMP0072 NEW)
endif()

option(USE_GTSAM "Use GTSAM backend for VINS features" ON)

# -----------------------------
# Dependencies
# -----------------------------
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(pluginlib REQUIRED)
# Force use of system Eigen (v3.4) over any local installs in /usr/local
if(EXISTS "/usr/share/eigen3/cmake/Eigen3Config.cmake")
  set(Eigen3_DIR "/usr/share/eigen3/cmake")
endif()
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(dbow2_vendor REQUIRED)
find_package(g2o_vendor REQUIRED)
message(STATUS "Using g2o backend")

if(USE_GTSAM)
  find_package(gtsam_vendor REQUIRED)
  # Force GTSAM search in the workspace prefix of gtsam_vendor
  get_filename_component(_gtsam_vendor_prefix "${gtsam_vendor_DIR}/../../.." ABSOLUTE)
  find_package(GTSAM CONFIG REQUIRED PATHS "${_gtsam_vendor_prefix}" NO_DEFAULT_PATH)
  
  if(GTSAM_FOUND)
    message(STATUS "GTSAM found in workspace: ${GTSAM_DIR}")
    message(STATUS "GTSAM include dir: ${GTSAM_INCLUDE_DIR}")
    if(NOT TARGET gtsam)
       message(FATAL_ERROR "GTSAM target 'gtsam' not found even though GTSAM_FOUND is true!")
    endif()
    add_definitions(-DVISUAL_GRAPH_SLAM_WITH_GTSAM=1)
    find_package(TBB REQUIRED)
    # Set RPATH to prioritize the workspace gtsam_vendor/lib
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
    list(APPEND CMAKE_INSTALL_RPATH "${_gtsam_vendor_prefix}/lib")
    # Link explicitly to the library file to avoid config mismatch issues with imported targets
    set(GTSAM_LIBRARY_FILE "${_gtsam_vendor_prefix}/lib/libgtsam.so")
  else()
    message(FATAL_ERROR "GTSAM NOT found in workspace prefix: ${_gtsam_vendor_prefix}")
  endif()
endif()

find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)

# -----------------------------
# Target
# -----------------------------
set(GRAPH_SLAM_CORE_SRC 
  src/frontend/feature_pipeline_factory.cpp
  src/frontend/feature_pipeline_orb.cpp
  src/frontend/feature_pipeline_superpoint.cpp
  src/frontend/feature_pipeline_sift.cpp
  src/frontend/feature_pipeline_custom.cpp
  src/frontend/visual_odometry_frontend.cpp
  src/core/graph_slam.cpp
  src/core/graph_node.cpp
  src/core/graph_edge.cpp
  src/core/graph.cpp
  src/core/measurement_edge.cpp
  src/map/map_manager.cpp
  src/loop_closure/loop_detector_dbow2.cpp
  src/loop_closure/loop_detector_orb_bow.cpp
  src/loop_closure/loop_detector_factory.cpp
  src/runtime/slam_visualizer.cpp
  src/runtime/sensor_data_manager.cpp
  src/frontend/vision_node.cpp
  src/vins/imu_preintegrator.cpp
  src/vins/vins_initializer.cpp
  src/vins/gtsam_imu_builder.cpp
  src/vins/tightly_coupled_local_ba.cpp
  src/backend/optimizer_factory.cpp
  src/telemetry/slam_telemetry.cpp
)

set(PLUGIN_SRC
  src/plugins/frontend_orb.cpp
  src/plugins/frontend_sift.cpp
  src/plugins/frontend_custom.cpp
  src/plugins/loop_closure_dbow2.cpp
  src/plugins/loop_closure_orb_bow.cpp
  src/plugins/optimizer_g2o.cpp
  src/plugins/optimizer_gtsam.cpp
  src/modules/camera_module.cpp
  src/modules/imu_module.cpp
  src/plugins/system_coordinator.cpp
)

add_library(visual_graph_slam_core SHARED ${GRAPH_SLAM_CORE_SRC})
add_library(visual_graph_slam_plugins SHARED ${PLUGIN_SRC})
add_executable(graph_slam src/runtime/main.cpp)

target_include_directories(visual_graph_slam_core PUBLIC
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_compile_definitions(visual_graph_slam_core PUBLIC
  VISUAL_GRAPH_SLAM_WITH_DBOW2=1
)

if(TARGET dbow2_vendor::DBoW2)
  target_link_libraries(visual_graph_slam_core dbow2_vendor::DBoW2)
else()
  target_include_directories(visual_graph_slam_core PUBLIC ${dbow2_vendor_INCLUDE_DIRS})
  target_link_libraries(visual_graph_slam_core ${dbow2_vendor_LIBRARIES})
endif()

ament_target_dependencies(visual_graph_slam_core 
  ament_index_cpp
  rclcpp
  pluginlib
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_filters
  visualization_msgs
  Eigen3
  cv_bridge
  OpenCV
  tf2
  tf2_geometry_msgs
  gtsam_vendor
  TBB
)

target_link_libraries(visual_graph_slam_core
  ${OpenCV_LIBS}
  g2o::core
  g2o::stuff
  g2o::types_sba
  g2o::solver_cholmod
)

if(USE_GTSAM AND GTSAM_FOUND)
  target_link_libraries(visual_graph_slam_core "${GTSAM_LIBRARY_FILE}" TBB::tbb)
  target_include_directories(visual_graph_slam_core PUBLIC "${GTSAM_INCLUDE_DIR}")
endif()

# Now setup executable and plugins to link to core
target_link_libraries(graph_slam visual_graph_slam_core)

target_include_directories(visual_graph_slam_plugins PUBLIC
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_compile_definitions(visual_graph_slam_plugins PUBLIC
  VISUAL_GRAPH_SLAM_WITH_DBOW2=1
)

if(TARGET dbow2_vendor::DBoW2)
  target_link_libraries(visual_graph_slam_plugins dbow2_vendor::DBoW2)
else()
  target_include_directories(visual_graph_slam_plugins PUBLIC ${dbow2_vendor_INCLUDE_DIRS})
  target_link_libraries(visual_graph_slam_plugins ${dbow2_vendor_LIBRARIES})
endif()

ament_target_dependencies(visual_graph_slam_plugins 
  ament_index_cpp
  rclcpp
  rclcpp_lifecycle
  pluginlib
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_filters
  visualization_msgs
  Eigen3
  cv_bridge
  OpenCV
  tf2
  tf2_geometry_msgs
  gtsam_vendor
  TBB
)

target_link_libraries(visual_graph_slam_plugins
  visual_graph_slam_core
  ${OpenCV_LIBS}
  g2o::core
  g2o::stuff
  g2o::types_sba
  g2o::solver_cholmod
)

if(USE_GTSAM AND GTSAM_FOUND)
  target_link_libraries(visual_graph_slam_plugins "${GTSAM_LIBRARY_FILE}" TBB::tbb)
  target_include_directories(visual_graph_slam_plugins PUBLIC "${GTSAM_INCLUDE_DIR}")
endif()

install(TARGETS
  visual_graph_slam_core
  visual_graph_slam_plugins
  DESTINATION lib
)

install(TARGETS
  graph_slam
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/gps_visualizer_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  rviz
  voc
  DESTINATION share/${PROJECT_NAME}
)

# Finalize the package configuration
pluginlib_export_plugin_description_file(visual_graph_slam plugin_description.xml)
ament_package()
