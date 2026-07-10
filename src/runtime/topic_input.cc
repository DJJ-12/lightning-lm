#include "runtime/topic_input.h"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <utility>

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

namespace lightning::runtime {
namespace {
std::int64_t NowSteadyMs() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}
}  // namespace

bool TopicInput::Start(rclcpp::Node::SharedPtr node, const std::string& yaml_path,
                       ImuCallback imu_cb, CloudCallback cloud_cb, LivoxCallback livox_cb,
                       bool subscribe_imu, double cloud_timeout_sec, TimeoutCallback timeout_cb) {
    Stop();
    if (!node) {
        LOG(ERROR) << "TopicInput failed: node is null";
        return false;
    }
    YAML::Node yaml = YAML::LoadFile(yaml_path);
    if (!yaml["common"]) {
        LOG(ERROR) << "TopicInput failed: missing common config";
        return false;
    }

    const std::string imu_topic = yaml["common"]["imu_topic"].as<std::string>();
    const std::string cloud_topic = yaml["common"]["lidar_topic"].as<std::string>();
    const std::string livox_topic = yaml["common"]["livox_lidar_topic"].as<std::string>();

    node_ = node;
    cloud_timeout_sec_ = cloud_timeout_sec;
    timeout_cb_ = std::move(timeout_cb);
    cloud_timeout_reported_ = false;
    last_cloud_time_ms_ = NowSteadyMs();

    rclcpp::QoS qos(10);
    qos.best_effort();
    qos.durability_volatile();

    if (subscribe_imu && imu_cb) {
        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, qos, [imu_cb](sensor_msgs::msg::Imu::SharedPtr msg) { imu_cb(msg); });
    }
    if (cloud_cb) {
        cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic, qos,
            [this, cloud_cb](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                MarkCloudReceived();
                cloud_cb(msg);
            });
    }
    if (livox_cb) {
        livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            livox_topic, qos,
            [this, livox_cb](livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
                MarkCloudReceived();
                livox_cb(msg);
            });
    }
    if (cloud_timeout_sec_ > 0.0 && timeout_cb_) {
        const auto check_period_ms = std::chrono::milliseconds(
            std::max<std::int64_t>(100, std::min<std::int64_t>(
                                            1000, static_cast<std::int64_t>(cloud_timeout_sec_ * 500.0))));
        cloud_timeout_timer_ = node_->create_wall_timer(
            check_period_ms, [this]() { CheckCloudTimeout(); });
    }
    running_ = true;
    LOG(INFO) << "TopicInput started. cloud=" << cloud_topic << ", livox=" << livox_topic
              << ", imu=" << (subscribe_imu ? imu_topic : std::string("disabled"))
              << ", cloud_timeout_sec=" << cloud_timeout_sec_;
    return true;
}

void TopicInput::Stop() {
    running_ = false;
    cloud_timeout_timer_.reset();
    imu_sub_.reset();
    cloud_sub_.reset();
    livox_sub_.reset();
    node_.reset();
    timeout_cb_ = nullptr;
    cloud_timeout_sec_ = 0.0;
    cloud_timeout_reported_ = false;
}

void TopicInput::MarkCloudReceived() {
    last_cloud_time_ms_ = NowSteadyMs();
    cloud_timeout_reported_ = false;
}

void TopicInput::CheckCloudTimeout() {
    if (!running_.load() || cloud_timeout_sec_ <= 0.0 || !timeout_cb_ ||
        cloud_timeout_reported_.load()) {
        return;
    }

    const std::int64_t now_ms = NowSteadyMs();
    const std::int64_t last_ms = last_cloud_time_ms_.load();
    const double elapsed_sec = static_cast<double>(now_ms - last_ms) / 1000.0;
    if (elapsed_sec < cloud_timeout_sec_) {
        return;
    }

    cloud_timeout_reported_ = true;
    std::ostringstream oss;
    oss << "lidar message timeout: no point cloud received for "
        << elapsed_sec << " seconds";
    timeout_cb_(oss.str());
}

}  // namespace lightning::runtime
