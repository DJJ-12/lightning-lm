#pragma once

#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "livox_ros_driver2/msg/custom_msg.hpp"

namespace lightning::runtime {

class TopicInput {
   public:
    using ImuCallback = std::function<void(const sensor_msgs::msg::Imu::SharedPtr&)>;
    using CloudCallback = std::function<void(const sensor_msgs::msg::PointCloud2::SharedPtr&)>;
    using LivoxCallback = std::function<void(const livox_ros_driver2::msg::CustomMsg::SharedPtr&)>;

    bool Start(rclcpp::Node::SharedPtr node, const std::string& yaml_path,
               ImuCallback imu_cb, CloudCallback cloud_cb, LivoxCallback livox_cb,
               bool subscribe_imu = true);
    void Stop();
    bool Running() const { return running_; }

   private:
    bool running_ = false;
    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_sub_;
};

}  // namespace lightning::runtime
