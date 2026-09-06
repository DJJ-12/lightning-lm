#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include "livox_ros_driver2/msg/custom_msg.hpp"

namespace lightning::runtime {

struct BagInputProgress {
    std::uint64_t total_frames = 0;
    std::uint64_t processed_frames = 0;
};

class BagInput {
   public:
    using ImuCallback = std::function<void(const sensor_msgs::msg::Imu::SharedPtr&)>;
    using RtkPositionCallback = std::function<void(
        const sensor_msgs::msg::NavSatFix::SharedPtr&)>;
    using RtkVelocityCallback = std::function<void(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr&)>;
    using WheelOdometryCallback = std::function<void(
        const nav_msgs::msg::Odometry::SharedPtr&)>;
    using CloudCallback = std::function<void(const sensor_msgs::msg::PointCloud2::SharedPtr&)>;
    using LivoxCallback = std::function<void(const livox_ros_driver2::msg::CustomMsg::SharedPtr&)>;
    using ProgressCallback = std::function<void(const BagInputProgress&)>;
    using CancelCallback = std::function<bool()>;

    bool Run(const std::string& bag_path, const std::string& yaml_path,
             ImuCallback imu_cb, CloudCallback cloud_cb, LivoxCallback livox_cb,
             RtkPositionCallback rtk_position_cb,
             RtkVelocityCallback rtk_velocity_cb,
             WheelOdometryCallback wheel_odometry_cb,
             ProgressCallback progress_cb, CancelCallback cancel_requested);
};

}  // namespace lightning::runtime
