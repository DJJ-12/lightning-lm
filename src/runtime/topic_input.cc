#include "runtime/topic_input.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <exception>
#include <utility>

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include "common/localization_message_adapter.h"

namespace lightning::runtime {
namespace {

std::string NormalizeMode(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

double TopicSteadySeconds() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

TopicInput::~TopicInput() {
    LOG(INFO) << "[析构][TopicInput] 开始 this=" << this;
    Shutdown();
    LOG(INFO) << "[析构][TopicInput] 完成 this=" << this;
}

bool TopicInput::Start(const std::string& yaml_path,
                       ImuCallback imu_cb,
                       CloudCallback cloud_cb,
                       LivoxCallback livox_cb,
                       RtkInsCallback rtk_ins_cb,
                       WheelOdometryCallback wheel_odometry_cb) {
    if (running_.load()) {
        return true;
    }
    if (yaml_path.empty()) {
        LOG(ERROR) << "[Topic接收] 配置文件路径为空";
        return false;
    }
    input_enabled_.store(false, std::memory_order_release);
    imu_received_ = 0;
    cloud_received_ = 0;
    livox_received_ = 0;
    rtk_ins_received_ = 0;
    wheel_odometry_received_ = 0;
    lidar_topic_sequence_ = 0;

    YAML::Node yaml = YAML::LoadFile(yaml_path);
    if (!yaml["common"]) {
        LOG(ERROR) << "[Topic接收] 配置文件缺少 common";
        return false;
    }

    const std::string imu_topic = yaml["common"]["imu_topic"] ? yaml["common"]["imu_topic"].as<std::string>() : std::string();
    const std::string cloud_topic = yaml["common"]["lidar_topic"] ? yaml["common"]["lidar_topic"].as<std::string>() : std::string();
    const std::string livox_topic = yaml["common"]["livox_lidar_topic"] ? yaml["common"]["livox_lidar_topic"].as<std::string>() : std::string();
    const YAML::Node mapping_rtk = yaml["mapping_rtk"] ? yaml["mapping_rtk"] : YAML::Node();
    const bool mapping_rtk_enabled = mapping_rtk && mapping_rtk["enabled"] ? mapping_rtk["enabled"].as<bool>() : false;
    const YAML::Node localization = yaml["localization"];
    const YAML::Node localization_rtk = localization && localization["rtk_ins"] ? localization["rtk_ins"] : YAML::Node();
    const YAML::Node localization_eskf = localization && localization["eskf"] ? localization["eskf"] : YAML::Node();
    const std::string localization_mode = NormalizeMode(localization && localization["mode"] ? localization["mode"].as<std::string>() : "ndt_only");
    const bool localization_filter_enabled = localization_mode != "ndt_only";
    const bool localization_rtk_enabled = localization_filter_enabled && localization_rtk && localization_rtk["enabled"] ? localization_rtk["enabled"].as<bool>() : false;
    const bool rtk_ins_enabled = mapping_rtk_enabled || localization_rtk_enabled;
    const bool wheel_enabled = localization_filter_enabled && localization_eskf && localization_eskf["use_wheel_odometry"] ? localization_eskf["use_wheel_odometry"].as<bool>() : false;
    const std::string rtk_ins_topic = yaml["common"]["rtk_local_odometry_topic"] ? yaml["common"]["rtk_local_odometry_topic"].as<std::string>() : (yaml["common"]["localization_rtk_ned_odometry_topic"] ? yaml["common"]["localization_rtk_ned_odometry_topic"].as<std::string>() : std::string());
    const std::string rtk_frame_name = NormalizeMode(yaml["common"]["rtk_local_odometry_frame"] ? yaml["common"]["rtk_local_odometry_frame"].as<std::string>() : "ned");
    if (rtk_frame_name != "ned" && rtk_frame_name != "enu") {
        LOG(ERROR) << "[Topic接收][RTK] rtk_local_odometry_frame must be 'ned' or 'enu'";
        return false;
    }
    const auto rtk_input_frame = rtk_frame_name == "enu" ? localization_adapter::LocalNavigationFrame::ENU : localization_adapter::LocalNavigationFrame::NED;
    const std::string wheel_odometry_topic = yaml["common"]["wheel_odometry_topic"] ? yaml["common"]["wheel_odometry_topic"].as<std::string>() : std::string();
    if (rtk_ins_enabled && rtk_ins_topic.empty()) {
        LOG(ERROR) << "[Topic接收][RTK] mapping/localization RTK is enabled but the local odometry topic is empty";
        return false;
    }
    if (wheel_enabled && wheel_odometry_topic.empty()) {
        LOG(ERROR) << "[Topic接收][Wheel] enabled but wheel odometry topic is empty";
        return false;
    }

    imu_cb_ = std::move(imu_cb);
    cloud_cb_ = std::move(cloud_cb);
    livox_cb_ = std::move(livox_cb);
    rtk_ins_cb_ = std::move(rtk_ins_cb);
    wheel_odometry_cb_ = std::move(wheel_odometry_cb);

    node_ = std::make_shared<rclcpp::Node>("lightning_topic_input");

    // 点云在 DDS 层只保留最新帧；回调只负责交给应用层输入队列。
    rclcpp::QoS lidar_qos{rclcpp::KeepLast(1)};
    lidar_qos.best_effort();
    lidar_qos.durability_volatile();

    // IMU and RTK are not intentionally truncated at the DDS history layer.
    // BEST_EFFORT remains compatible with common sensor publishers; it cannot
    // guarantee lossless network transport, but the application never drops a
    // callback-delivered IMU/RTK message because of queue capacity.
    rclcpp::QoS auxiliary_qos{rclcpp::KeepAll()};
    auxiliary_qos.best_effort();
    auxiliary_qos.durability_volatile();

    if (imu_cb_ && !imu_topic.empty()) {
        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, auxiliary_qos,
            [this](sensor_msgs::msg::Imu::SharedPtr msg) {
                std::lock_guard<std::mutex> callback_gate(callback_gate_mutex_);
                if (!input_enabled_.load(std::memory_order_acquire)) {
                    return;
                }
                ++imu_received_;
                try {
                    imu_cb_(msg);
                } catch (const std::exception& e) {
                    LOG(ERROR) << "[Topic接收] IMU入队回调异常: " << e.what();
                } catch (...) {
                    LOG(ERROR) << "[Topic接收] IMU入队回调发生未知异常";
                }
            });
    }

    if (cloud_cb_ && !cloud_topic.empty()) {
        cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic, lidar_qos,
            [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                std::lock_guard<std::mutex> callback_gate(callback_gate_mutex_);
                if (!input_enabled_.load(std::memory_order_acquire)) {
                    return;
                }
                ++cloud_received_;
                LidarReceiveInfo info;
                info.topic_sequence = ++lidar_topic_sequence_;
                info.receive_steady_sec = TopicSteadySeconds();
                info.header_stamp = rclcpp::Time(msg->header.stamp).seconds();
                try {
                    cloud_cb_(msg, info);
                } catch (const std::exception& e) {
                    LOG(ERROR) << "[Topic接收] PointCloud2入队回调异常: " << e.what();
                } catch (...) {
                    LOG(ERROR) << "[Topic接收] PointCloud2入队回调发生未知异常";
                }
            });
    }

    if (livox_cb_ && !livox_topic.empty()) {
        livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            livox_topic, lidar_qos,
            [this](livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
                std::lock_guard<std::mutex> callback_gate(callback_gate_mutex_);
                if (!input_enabled_.load(std::memory_order_acquire)) {
                    return;
                }
                ++livox_received_;
                LidarReceiveInfo info;
                info.topic_sequence = ++lidar_topic_sequence_;
                info.receive_steady_sec = TopicSteadySeconds();
                info.header_stamp = rclcpp::Time(msg->header.stamp).seconds();
                try {
                    livox_cb_(msg, info);
                } catch (const std::exception& e) {
                    LOG(ERROR) << "[Topic接收] Livox入队回调异常: " << e.what();
                } catch (...) {
                    LOG(ERROR) << "[Topic接收] Livox入队回调发生未知异常";
                }
            });
    }


    if (rtk_ins_enabled && rtk_ins_cb_ && !rtk_ins_topic.empty()) {
        rtk_ins_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(rtk_ins_topic, auxiliary_qos, [this, rtk_input_frame](nav_msgs::msg::Odometry::SharedPtr msg) {
            std::lock_guard<std::mutex> callback_gate(callback_gate_mutex_);
            if (!input_enabled_.load(std::memory_order_acquire)) return;
            ++rtk_ins_received_;
            try {
                RtkInsMeasurement measurement;
                if (localization_adapter::LocalOdometryToRtkInsMeasurement(*msg, rtk_input_frame, &measurement)) rtk_ins_cb_(measurement);
            } catch (const std::exception& e) {
                LOG(ERROR) << "[Topic input][RTK/INS] callback exception: " << e.what();
            } catch (...) {
                LOG(ERROR) << "[Topic input][RTK/INS] unknown callback exception";
            }
        });
    }
    if (wheel_enabled && wheel_odometry_cb_ && !wheel_odometry_topic.empty()) {
        wheel_odometry_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(wheel_odometry_topic, auxiliary_qos, [this](nav_msgs::msg::Odometry::SharedPtr msg) {
            std::lock_guard<std::mutex> callback_gate(callback_gate_mutex_);
            if (!input_enabled_.load(std::memory_order_acquire)) return;
            ++wheel_odometry_received_;
            try {
                WheelOdometryMeasurement measurement;
                if (localization_adapter::OdometryToWheelMeasurement(*msg, &measurement)) wheel_odometry_cb_(measurement);
            } catch (const std::exception& e) {
                LOG(ERROR) << "[Topic input][Wheel] callback exception: " << e.what();
            } catch (...) {
                LOG(ERROR) << "[Topic input][Wheel] unknown callback exception";
            }
        });
    }

    executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(node_);
    running_ = true;
    thread_ = std::thread([this]() { Spin(); });

    LOG(INFO) << "[Topic接收] 独立节点和独立线程已启动"
              << ", cloud=" << cloud_topic
              << ", livox=" << livox_topic
              << ", imu=" << imu_topic
              << ", rtk_local_odometry=" << (rtk_ins_enabled ? rtk_ins_topic : "disabled")
              << ", rtk_local_frame=" << (rtk_ins_enabled ? rtk_frame_name : "disabled")
              << ", wheel_odometry=" << (wheel_enabled ? wheel_odometry_topic : "disabled")
              << ", cloud_qos=KEEP_LAST(1)+BEST_EFFORT+VOLATILE"
              << ", imu_rtk_qos=KEEP_ALL+BEST_EFFORT+VOLATILE"
              << ", note=application buffers do not capacity-drop IMU/RTK; "
                 "BEST_EFFORT transport itself is not lossless";
    return true;
}

void TopicInput::SetEnabled(bool enabled) {
    std::lock_guard<std::mutex> callback_gate(callback_gate_mutex_);
    input_enabled_.store(enabled, std::memory_order_release);
}

void TopicInput::Spin() {
    LOG(INFO) << "[Topic接收线程] 开始 spin, thread_id=" << std::this_thread::get_id();
    executor_->spin();
    LOG(INFO) << "[Topic接收线程] spin 已退出, thread_id=" << std::this_thread::get_id();
}

void TopicInput::Shutdown() {
    SetEnabled(false);
    const bool was_running = running_.exchange(false);
    if (!was_running && !thread_.joinable() && !node_) {
        return;
    }

    LOG(INFO) << "[Topic接收析构] [01] 请求停止独立 executor";
    if (executor_) {
        executor_->cancel();
    }

    LOG(INFO) << "[Topic接收析构] [02] 等待接收线程退出";
    if (thread_.joinable()) {
        thread_.join();
    }
    LOG(INFO) << "[Topic接收析构] [03] 接收线程已经退出";

    if (executor_ && node_) {
        try {
            executor_->remove_node(node_);
        } catch (const std::exception& e) {
            LOG(WARNING) << "[Topic接收析构] remove_node异常: " << e.what();
        }
    }

    LOG(INFO) << "[Topic接收析构] [04] 销毁订阅对象";
    imu_sub_.reset();
    cloud_sub_.reset();
    livox_sub_.reset();
    rtk_ins_sub_.reset();
    wheel_odometry_sub_.reset();
    LOG(INFO) << "[Topic接收析构] [05] 销毁输入节点和executor";
    node_.reset();
    executor_.reset();
    imu_cb_ = nullptr;
    cloud_cb_ = nullptr;
    livox_cb_ = nullptr;
    rtk_ins_cb_ = nullptr;
    wheel_odometry_cb_ = nullptr;

    LOG(INFO) << "[Topic接收] 接收线程已停止"
              << ", imu_received=" << imu_received_
              << ", cloud_received=" << cloud_received_
              << ", livox_received=" << livox_received_
              << ", rtk_ins_received=" << rtk_ins_received_
              << ", wheel_odometry_received=" << wheel_odometry_received_
              << ", lidar_topic_sequence=" << lidar_topic_sequence_;
}

}  // namespace lightning::runtime
