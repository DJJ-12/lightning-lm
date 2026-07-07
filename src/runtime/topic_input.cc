#include "runtime/topic_input.h"

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

namespace lightning::runtime {

bool TopicInput::Start(rclcpp::Node::SharedPtr node, const std::string& yaml_path,
                       ImuCallback imu_cb, CloudCallback cloud_cb, LivoxCallback livox_cb,
                       bool subscribe_imu) {
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
    rclcpp::QoS qos(10);
    qos.best_effort();
    qos.durability_volatile();

    if (subscribe_imu && imu_cb) {
        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, qos, [imu_cb](sensor_msgs::msg::Imu::SharedPtr msg) { imu_cb(msg); });
    }
    if (cloud_cb) {
        cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic, qos, [cloud_cb](sensor_msgs::msg::PointCloud2::SharedPtr msg) { cloud_cb(msg); });
    }
    if (livox_cb) {
        livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            livox_topic, qos, [livox_cb](livox_ros_driver2::msg::CustomMsg::SharedPtr msg) { livox_cb(msg); });
    }
    running_ = true;
    LOG(INFO) << "TopicInput started. cloud=" << cloud_topic << ", livox=" << livox_topic
              << ", imu=" << (subscribe_imu ? imu_topic : std::string("disabled"));
    return true;
}

void TopicInput::Stop() {
    imu_sub_.reset();
    cloud_sub_.reset();
    livox_sub_.reset();
    node_.reset();
    running_ = false;
}

}  // namespace lightning::runtime
