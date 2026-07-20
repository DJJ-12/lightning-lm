#include "runtime/topic_input.h"

#include <exception>
#include <utility>

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

namespace lightning::runtime {

TopicInput::~TopicInput() {
    LOG(INFO) << "[析构][TopicInput] 开始 this=" << this;
    Shutdown();
    LOG(INFO) << "[析构][TopicInput] 完成 this=" << this;
}

bool TopicInput::Start(const std::string& yaml_path,
                       ImuCallback imu_cb,
                       CloudCallback cloud_cb,
                       LivoxCallback livox_cb) {
    if (running_.load()) {
        return true;
    }
    if (yaml_path.empty()) {
        LOG(ERROR) << "[Topic接收] 配置文件路径为空";
        return false;
    }

    YAML::Node yaml = YAML::LoadFile(yaml_path);
    if (!yaml["common"]) {
        LOG(ERROR) << "[Topic接收] 配置文件缺少 common";
        return false;
    }

    const std::string imu_topic = yaml["common"]["imu_topic"].as<std::string>();
    const std::string cloud_topic = yaml["common"]["lidar_topic"].as<std::string>();
    const std::string livox_topic = yaml["common"]["livox_lidar_topic"].as<std::string>();

    imu_cb_ = std::move(imu_cb);
    cloud_cb_ = std::move(cloud_cb);
    livox_cb_ = std::move(livox_cb);

    node_ = std::make_shared<rclcpp::Node>("lightning_topic_input");

    // 使用 KeepAll，应用层回调只入队；可靠性保持 BestEffort，以兼容常见雷达和 rosbag 发布端。
    rclcpp::QoS qos{rclcpp::KeepAll()};
    qos.best_effort();
    qos.durability_volatile();

    if (imu_cb_) {
        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, qos,
            [this](sensor_msgs::msg::Imu::SharedPtr msg) {
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

    if (cloud_cb_) {
        cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic, qos,
            [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                const std::uint64_t count = ++cloud_received_;
                if (count % 1000 == 0) {
                    LOG(INFO) << "[Topic接收] PointCloud2累计接收=" << count;
                }
                try {
                    cloud_cb_(msg);
                } catch (const std::exception& e) {
                    LOG(ERROR) << "[Topic接收] PointCloud2入队回调异常: " << e.what();
                } catch (...) {
                    LOG(ERROR) << "[Topic接收] PointCloud2入队回调发生未知异常";
                }
            });
    }

    if (livox_cb_) {
        livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            livox_topic, qos,
            [this](livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
                const std::uint64_t count = ++livox_received_;
                if (count % 1000 == 0) {
                    LOG(INFO) << "[Topic接收] Livox累计接收=" << count;
                }
                try {
                    livox_cb_(msg);
                } catch (const std::exception& e) {
                    LOG(ERROR) << "[Topic接收] Livox入队回调异常: " << e.what();
                } catch (...) {
                    LOG(ERROR) << "[Topic接收] Livox入队回调发生未知异常";
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
              << ", imu=" << imu_topic;
    return true;
}

void TopicInput::Spin() {
    LOG(INFO) << "[Topic接收线程] 开始 spin, thread_id=" << std::this_thread::get_id();
    executor_->spin();
    LOG(INFO) << "[Topic接收线程] spin 已退出, thread_id=" << std::this_thread::get_id();
}

void TopicInput::Shutdown() {
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

    LOG(INFO) << "[Topic接收析构] [05] 销毁输入节点和executor";
    node_.reset();
    executor_.reset();
    imu_cb_ = nullptr;
    cloud_cb_ = nullptr;
    livox_cb_ = nullptr;

    LOG(INFO) << "[Topic接收析构] [06] 完成"
              << ", imu_received=" << imu_received_.load()
              << ", cloud_received=" << cloud_received_.load()
              << ", livox_received=" << livox_received_.load();
}

}  // namespace lightning::runtime
