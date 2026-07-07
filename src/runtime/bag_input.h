#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "livox_ros_driver2/msg/custom_msg.hpp"

namespace lightning::runtime {

struct BagInputProgress {
    std::uint64_t total_frames = 0;
    std::uint64_t processed_frames = 0;
};

class BagInput {
   public:
    using ImuCallback = std::function<void(const sensor_msgs::msg::Imu::SharedPtr&)>;
    using CloudCallback = std::function<void(const sensor_msgs::msg::PointCloud2::SharedPtr&)>;
    using LivoxCallback = std::function<void(const livox_ros_driver2::msg::CustomMsg::SharedPtr&)>;
    using ProgressCallback = std::function<void(const BagInputProgress&)>;

    bool Run(const std::string& bag_path, const std::string& yaml_path,
             ImuCallback imu_cb, CloudCallback cloud_cb, LivoxCallback livox_cb,
             ProgressCallback progress_cb, std::atomic_bool* cancel_flag);
};

}  // namespace lightning::runtime
