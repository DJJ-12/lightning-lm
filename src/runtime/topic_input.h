#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "livox_ros_driver2/msg/custom_msg.hpp"

namespace lightning::runtime {

// Owns a dedicated ROS 2 node, executor and receive thread. Every callback is
// intentionally lightweight and only forwards a standard-message shared
// pointer to Lightning's input slots.
class TopicInput {
   public:
    struct LidarReceiveInfo {
        std::uint64_t topic_sequence = 0;
        double receive_steady_sec = 0.0;
        double header_stamp = 0.0;
    };

    using ImuCallback = std::function<void(const sensor_msgs::msg::Imu::SharedPtr&)>;
    using GpsCallback = std::function<void(
        const sensor_msgs::msg::NavSatFix::SharedPtr&)>;
    using GpsOrientationCallback = std::function<void(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr&)>;
    using gpsVelocityCallback = std::function<void(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr&)>;
    using WheelOdometryCallback = std::function<void(
        const nav_msgs::msg::Odometry::SharedPtr&)>;
    using CloudCallback = std::function<void(
        const sensor_msgs::msg::PointCloud2::SharedPtr&, const LidarReceiveInfo&)>;
    using LivoxCallback = std::function<void(
        const livox_ros_driver2::msg::CustomMsg::SharedPtr&, const LidarReceiveInfo&)>;

    TopicInput() = default;
    ~TopicInput();

    bool Start(const std::string& yaml_path,
               ImuCallback imu_cb,
               CloudCallback cloud_cb,
               LivoxCallback livox_cb,
               GpsCallback gps_cb,
               GpsOrientationCallback gps_orientation_cb,
               gpsVelocityCallback gps_velocity_cb,
               WheelOdometryCallback wheel_odometry_cb);
    void SetEnabled(bool enabled, bool localization = false);
    void Shutdown();

    bool Running() const { return running_.load(); }

   private:
    void Spin();

    std::atomic_bool running_{false};
    std::atomic_bool input_enabled_{false};
    std::atomic_bool localization_input_enabled_{false};
    // SetEnabled() uses this gate to wait for an already-running lightweight
    // callback, preventing a message from the previous task entering a new one.
    std::mutex callback_gate_mutex_;

    std::uint64_t imu_received_ = 0;
    std::uint64_t cloud_received_ = 0;
    std::uint64_t livox_received_ = 0;
    std::uint64_t gps_received_ = 0;
    std::uint64_t gps_orientation_received_ = 0;
    std::uint64_t gps_velocity_received_ = 0;
    std::uint64_t wheel_odometry_received_ = 0;
    std::uint64_t lidar_topic_sequence_ = 0;

    ImuCallback imu_cb_;
    CloudCallback cloud_cb_;
    LivoxCallback livox_cb_;
    GpsCallback gps_cb_;
    GpsOrientationCallback gps_orientation_cb_;
    gpsVelocityCallback gps_velocity_cb_;
    WheelOdometryCallback wheel_odometry_cb_;

    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
    std::thread thread_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_sub_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr
        gps_orientation_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr
        gps_velocity_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_odometry_sub_;
};

}  // namespace lightning::runtime
