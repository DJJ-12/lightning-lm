#include "runtime/topic_input.h"

#include <chrono>
#include <exception>
#include <utility>

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

namespace lightning::runtime {
namespace {

std::string ReadTopic(const YAML::Node& common, const char* name) {
    if (!common) return std::string();
    const YAML::Node value = common[name];
    return value && value.IsScalar()
        ? value.as<std::string>()
        : std::string();
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
                       GpsCallback gps1_cb,
                       GpsCallback gps2_cb,
                       gpsVelocityCallback gps_velocity_cb,
                       WheelOdometryCallback wheel_odometry_cb) {
    if (running_.load()) {
        return true;
    }
    if (yaml_path.empty()) {
        LOG(ERROR) << "[Topic接收] 配置文件路径为空";
        return false;
    }
    input_enabled_.store(false, std::memory_order_release);
    localization_input_enabled_.store(false, std::memory_order_release);
    imu_received_ = 0;
    cloud_received_ = 0;
    livox_received_ = 0;
    gps1_received_ = 0;
    gps2_received_ = 0;
    gps_velocity_received_ = 0;
    wheel_odometry_received_ = 0;
    lidar_topic_sequence_ = 0;

    YAML::Node yaml = YAML::LoadFile(yaml_path);
    if (!yaml["common"]) {
        LOG(ERROR) << "[Topic接收] 配置文件缺少 common";
        return false;
    }

    const YAML::Node common = yaml["common"];
    const std::string imu_topic = ReadTopic(common, "imu_topic");
    const std::string cloud_topic = ReadTopic(common, "lidar_topic");
    const std::string livox_topic = ReadTopic(common, "livox_lidar_topic");
    const std::string gps1_topic = ReadTopic(common, "gps1_topic");
    const std::string gps2_topic = ReadTopic(common, "gps2_topic");
    const bool dual_gps_enabled =
        !gps1_topic.empty() && !gps2_topic.empty() &&
        gps1_topic != gps2_topic;
    if (!gps1_topic.empty() && gps1_topic == gps2_topic) {
        LOG(ERROR) << "[Topic input] gps1_topic and gps2_topic must differ";
        return false;
    }
    if (gps1_topic.empty() != gps2_topic.empty()) {
        LOG(WARNING) << "[Topic input] dual GPS disabled: both gps1_topic "
                        "and gps2_topic must be configured";
    }
    const std::string velocity_topic =
        ReadTopic(common, "velocity_topic");
    const std::string wheel_odometry_topic =
        ReadTopic(common, "wheel_odometry_topic");

    imu_cb_ = std::move(imu_cb);
    cloud_cb_ = std::move(cloud_cb);
    livox_cb_ = std::move(livox_cb);
    gps1_cb_ = std::move(gps1_cb);
    gps2_cb_ = std::move(gps2_cb);
    gps_velocity_cb_ = std::move(gps_velocity_cb);
    wheel_odometry_cb_ = std::move(wheel_odometry_cb);

    node_ = std::make_shared<rclcpp::Node>("lightning_topic_input");

    // 点云在 DDS 层只保留最新帧；回调只负责写应用层 latest 槽位。
    rclcpp::QoS lidar_qos{rclcpp::KeepLast(1)};
    lidar_qos.best_effort();
    lidar_qos.durability_volatile();

    // Mapping is the only path that must retain every received IMU sample.
    rclcpp::QoS imu_qos{rclcpp::KeepAll()};
    imu_qos.best_effort();
    imu_qos.durability_volatile();

    // Online localization consumes only the freshest observation from each
    // independent sensor. DDS therefore also keeps only its newest sample.
    rclcpp::QoS latest_observation_qos{rclcpp::KeepLast(1)};
    latest_observation_qos.best_effort();
    latest_observation_qos.durability_volatile();

    if (imu_cb_ && !imu_topic.empty()) {
        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, imu_qos,
            [this](sensor_msgs::msg::Imu::SharedPtr msg) {
                std::lock_guard<std::mutex> callback_gate(callback_gate_mutex_);
                if (!input_enabled_.load(std::memory_order_acquire) ||
                    localization_input_enabled_.load(
                        std::memory_order_acquire)) {
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


    // GPS input has exactly two physical sources. Lightning decides whether
    // the pair belongs to localization or Map-ENU calibration.
    if (dual_gps_enabled && gps1_cb_) {
        gps1_sub_ =
            node_->create_subscription<sensor_msgs::msg::NavSatFix>(
                gps1_topic,
                latest_observation_qos,
                [this](sensor_msgs::msg::NavSatFix::SharedPtr msg) {
                    std::lock_guard<std::mutex> callback_gate(
                        callback_gate_mutex_);
                    if (!input_enabled_.load(std::memory_order_acquire) ||
                        !localization_input_enabled_.load(
                            std::memory_order_acquire)) {
                        return;
                    }
                    ++gps1_received_;
                    try {
                        gps1_cb_(msg);
                    } catch (const std::exception& e) {
                        LOG(ERROR) << "[Topic input][GPS1] callback exception: "
                                   << e.what();
                    } catch (...) {
                        LOG(ERROR) << "[Topic input][GPS1] unknown callback exception";
                    }
                });
    }

    if (dual_gps_enabled && gps2_cb_) {
        gps2_sub_ =
            node_->create_subscription<sensor_msgs::msg::NavSatFix>(
                gps2_topic,
                latest_observation_qos,
                [this](sensor_msgs::msg::NavSatFix::SharedPtr msg) {
                    std::lock_guard<std::mutex> callback_gate(
                        callback_gate_mutex_);
                    if (!input_enabled_.load(std::memory_order_acquire) ||
                        !localization_input_enabled_.load(
                            std::memory_order_acquire)) {
                        return;
                    }
                    ++gps2_received_;
                    try {
                        gps2_cb_(msg);
                    } catch (const std::exception& e) {
                        LOG(ERROR) << "[Topic input][GPS2] callback exception: "
                                   << e.what();
                    } catch (...) {
                        LOG(ERROR) << "[Topic input][GPS2] unknown callback exception";
                    }
                });
    }

    if (gps_velocity_cb_ && !velocity_topic.empty()) {
        gps_velocity_sub_ =
            node_->create_subscription<
                geometry_msgs::msg::TwistWithCovarianceStamped>(
                velocity_topic,
                latest_observation_qos,
                [this](geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg) {
                    std::lock_guard<std::mutex> callback_gate(
                        callback_gate_mutex_);
                    if (!input_enabled_.load(std::memory_order_acquire) ||
                        !localization_input_enabled_.load(
                            std::memory_order_acquire)) {
                        return;
                    }
                    ++gps_velocity_received_;
                    try {
                        gps_velocity_cb_(msg);
                    } catch (const std::exception& e) {
                        LOG(ERROR) << "[Topic input][gps velocity] callback exception: "
                                   << e.what();
                    } catch (...) {
                        LOG(ERROR) << "[Topic input][gps velocity] unknown callback exception";
                    }
                });
    }
    if (wheel_odometry_cb_ && !wheel_odometry_topic.empty()) {
        wheel_odometry_sub_ =
            node_->create_subscription<nav_msgs::msg::Odometry>(
                wheel_odometry_topic,
                latest_observation_qos,
                [this](nav_msgs::msg::Odometry::SharedPtr msg) {
                    std::lock_guard<std::mutex> callback_gate(
                        callback_gate_mutex_);
                    if (!input_enabled_.load(std::memory_order_acquire) ||
                        !localization_input_enabled_.load(
                            std::memory_order_acquire)) {
                        return;
                    }
                    ++wheel_odometry_received_;
                    try {
                        wheel_odometry_cb_(msg);
                    } catch (const std::exception& e) {
                        LOG(ERROR) << "[Topic input][Wheel] callback exception: "
                                   << e.what();
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
              << ", gps1=" << (gps1_topic.empty() ? "disabled" : gps1_topic)
              << ", gps2=" << (gps2_topic.empty() ? "disabled" : gps2_topic)
              << ", gps_velocity=" << (velocity_topic.empty() ? "disabled" : velocity_topic)
              << ", wheel_odometry=" << (wheel_odometry_topic.empty() ? "disabled" : wheel_odometry_topic)
              << ", cloud_qos=KEEP_LAST(1)+BEST_EFFORT+VOLATILE"
              << ", imu_qos=KEEP_ALL+BEST_EFFORT+VOLATILE"
              << ", localization_observation_qos=KEEP_LAST(1)+BEST_EFFORT+VOLATILE";
    return true;
}

void TopicInput::SetEnabled(bool enabled, bool localization) {
    std::lock_guard<std::mutex> callback_gate(callback_gate_mutex_);
    localization_input_enabled_.store(
        enabled && localization, std::memory_order_release);
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
    gps1_sub_.reset();
    gps2_sub_.reset();
    gps_velocity_sub_.reset();
    wheel_odometry_sub_.reset();
    LOG(INFO) << "[Topic接收析构] [05] 销毁输入节点和executor";
    node_.reset();
    executor_.reset();
    imu_cb_ = nullptr;
    cloud_cb_ = nullptr;
    livox_cb_ = nullptr;
    gps1_cb_ = nullptr;
    gps2_cb_ = nullptr;
    gps_velocity_cb_ = nullptr;
    wheel_odometry_cb_ = nullptr;

    LOG(INFO) << "[Topic接收] 接收线程已停止"
              << ", imu_received=" << imu_received_
              << ", cloud_received=" << cloud_received_
              << ", livox_received=" << livox_received_
              << ", gps1_received=" << gps1_received_
              << ", gps2_received=" << gps2_received_
              << ", gps_velocity_received=" << gps_velocity_received_
              << ", wheel_odometry_received=" << wheel_odometry_received_
              << ", lidar_topic_sequence=" << lidar_topic_sequence_;
}

}  // namespace lightning::runtime
