find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(GTSAM REQUIRED)
find_package(lightning_interfaces REQUIRED)
find_package(rosidl_default_runtime REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src/core/localization)

add_library(${PROJECT_NAME}.libs SHARED
        common/nav_state.cc
        common/constant.h
        common/options.cc
        common/params.cc

        io/yaml_io.cc
        io/file_io.cc

        core/lio/eskf.cc
        core/lio/laser_mapping.cc
        core/lio/lio_sam/lio_sam_mapping.cc
        core/lio/lio_sam/deskew_feature_extractor.cpp
        core/lio/lio_sam/map_optimization.cpp
        core/lio/pointcloud_preprocess.cc

        core/g2p5/g2p5_map.cc
        core/g2p5/g2p5.cc
        core/g2p5/g2p5_subgrid.cc

        core/maps/tiled_map.cc
        core/maps/tiled_map_chunk.cc

        core/localization/localization_result.cc
        core/localization/localization.cpp
        core/localization/GlobalLocalizer/GlobalLozalizer.cpp
        core/localization/GlobalLocalizer/tree_structured_parzen_estimator.cpp
        core/localization/GlobalLocalizer/ndt_omp/estimate_covariance.cpp
        core/localization/GlobalLocalizer/ndt_omp/multigrid_ndt_omp.cpp
        core/localization/GlobalLocalizer/ndt_omp/multi_voxel_grid_covariance_omp.cpp
        core/localization/MapLoader/MapLoader.cpp
        core/localization/MapLoader/utils.cpp

        utils/timer.cc
        utils/pointcloud_utils.cc

        ui/pangolin_window.cc
        ui/pangolin_window_impl.cc
        ui/ui_car.cc
        ui/ui_cloud.cc
        ui/ui_trajectory.cc

        core/system/slam.cc
        core/system/loc_system.cc

        wrapper/bag_io.cc

        runtime/mode.cc
        runtime/task.cc
        runtime/topic_input.cc
        runtime/bag_input.cc
        runtime/lightning.cc
        runtime/service.cc

        modules/mapping/mapping.cc
        modules/mapping/save_map.cc
        modules/localization/localization.cc
)

ament_target_dependencies(
        ${PROJECT_NAME}.libs
        "rclcpp"
        "std_msgs"
        "geometry_msgs"
        "sensor_msgs"
        "nav_msgs"
        "visualization_msgs"
        "std_srvs"
        "message_filters"
        "pcl_conversions"
        "tf2_ros"
        "tf2"
        "tf2_geometry_msgs"
        "tf2_eigen"
        "lightning_interfaces"
        "rosidl_default_runtime"
)

# 添加对 livox 消息类型支持的依赖
if(TARGET livox_interfaces2__rosidl_typesupport_cpp)
  target_link_libraries(${PROJECT_NAME}.libs livox_interfaces2__rosidl_typesupport_cpp)
else()
  get_property(livox_interfaces2_generated TARGET livox_interfaces2__rosidl_generator_cpp PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(${PROJECT_NAME}.libs PUBLIC ${livox_interfaces2_generated})
endif()

target_link_libraries(${PROJECT_NAME}.libs
        ${third_party_libs}
        ${PCL_LIBRARIES}
        OpenMP::OpenMP_CXX
        gtsam
)

install(TARGETS ${PROJECT_NAME}.libs
        DESTINATION lib)

add_subdirectory(app)
