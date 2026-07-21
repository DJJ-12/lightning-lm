#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "core/localization/localization_diagnostic.h"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "livox_ros_driver2/msg/custom_msg.hpp"

namespace lightning::runtime {

// TopicInput 在程序启动时创建一次，并在独立节点、独立 executor、独立线程中接收数据。
// 回调只负责把消息交给上层队列，不执行建图、定位或服务逻辑。
class TopicInput {
   public:
    struct LidarReceiveInfo {
        std::uint64_t topic_sequence = 0;
        std::uint64_t source_sequence = 0;
        double receive_steady_sec = 0.0;
        double header_stamp = 0.0;
        double header_dt = 0.0;
        double arrival_dt = 0.0;
    };

    using ImuCallback = std::function<void(const sensor_msgs::msg::Imu::SharedPtr&)>;
    using CloudCallback = std::function<void(
        const sensor_msgs::msg::PointCloud2::SharedPtr&, const LidarReceiveInfo&)>;
    using LivoxCallback = std::function<void(
        const livox_ros_driver2::msg::CustomMsg::SharedPtr&, const LidarReceiveInfo&)>;

    TopicInput() = default;
    ~TopicInput();

    bool Start(const std::string& yaml_path,
               ImuCallback imu_cb,
               CloudCallback cloud_cb,
               LivoxCallback livox_cb);
    void Shutdown();
    bool Running() const { return running_.load(); }
    void SetEnabled(bool enabled) { input_enabled_.store(enabled); }

   private:
    void Spin();

    std::atomic_bool running_{false};
    std::atomic_bool input_enabled_{false};
    std::atomic<std::uint64_t> imu_received_{0};
    std::atomic<std::uint64_t> cloud_received_{0};
    std::atomic<std::uint64_t> livox_received_{0};
    std::uint64_t lidar_topic_sequence_ = 0;
    double last_cloud_header_stamp_ = 0.0;
    double last_cloud_receive_steady_sec_ = 0.0;
    double last_livox_header_stamp_ = 0.0;
    double last_livox_receive_steady_sec_ = 0.0;
    double max_cloud_callback_ms_ = 0.0;
    double max_livox_callback_ms_ = 0.0;
    std::uint64_t cloud_non_monotonic_stamp_count_ = 0;
    std::uint64_t livox_non_monotonic_stamp_count_ = 0;
    std::uint64_t cloud_large_header_gap_count_ = 0;
    std::uint64_t livox_large_header_gap_count_ = 0;
    std::string cloud_topic_;
    std::string livox_topic_;

    ImuCallback imu_cb_;
    CloudCallback cloud_cb_;
    LivoxCallback livox_cb_;

    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
    std::thread thread_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_sub_;
};

}  // namespace lightning::runtime
