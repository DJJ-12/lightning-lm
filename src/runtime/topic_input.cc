#include "runtime/topic_input.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <utility>

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

namespace lightning::runtime {
namespace {

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
                       LivoxCallback livox_cb) {
    if (running_.load()) {
        return true;
    }
    if (yaml_path.empty()) {
        LOG(ERROR) << "[Topic接收] 配置文件路径为空";
        return false;
    }
    input_enabled_.store(false);

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

    // 回调只负责入队。点云继续使用 Reliable；IMU 兼容常见传感器 BestEffort 发布。
    rclcpp::QoS cloud_qos{rclcpp::KeepAll()};
    //cloud_qos.reliable();
    cloud_qos.best_effort();
    cloud_qos.durability_volatile();

    rclcpp::QoS imu_qos{rclcpp::KeepAll()};
    imu_qos.best_effort();
    imu_qos.durability_volatile();

    if (imu_cb_) {
        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, imu_qos,
            [this](sensor_msgs::msg::Imu::SharedPtr msg) {
                if (!input_enabled_.load()) {
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

    if (cloud_cb_) {
        cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic, cloud_qos,
            [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                if (!input_enabled_.load()) {
                    return;
                }
                const double callback_begin = TopicSteadySeconds();
                const std::uint64_t count = ++cloud_received_;
                LidarReceiveInfo info;
                info.topic_sequence = ++lidar_topic_sequence_;
                info.source_sequence = count;
                info.receive_steady_sec = callback_begin;
                info.header_stamp = rclcpp::Time(msg->header.stamp).seconds();
                if (last_cloud_header_stamp_ != 0.0) {
                    info.header_dt = info.header_stamp - last_cloud_header_stamp_;
                }
                if (last_cloud_receive_steady_sec_ != 0.0) {
                    info.arrival_dt = callback_begin - last_cloud_receive_steady_sec_;
                }
                if (last_cloud_header_stamp_ != 0.0 && info.header_dt <= 0.0) {
                    ++cloud_non_monotonic_stamp_count_;
                    LOG(ERROR) << std::setprecision(15)
                               << "[Topic接收诊断][PointCloud2] 时间戳不递增"
                               << ", topic_sequence=" << info.topic_sequence
                               << ", source_sequence=" << info.source_sequence
                               << ", previous_stamp=" << last_cloud_header_stamp_
                               << ", current_stamp=" << info.header_stamp
                               << ", header_dt=" << info.header_dt;
                } else if (info.header_dt > 0.15) {
                    ++cloud_large_header_gap_count_;
                    LOG(WARNING) << std::setprecision(15)
                                 << "[Topic接收诊断][PointCloud2] 回调入口已出现大时间间隔"
                                 << ", topic_sequence=" << info.topic_sequence
                                 << ", source_sequence=" << info.source_sequence
                                 << ", previous_stamp=" << last_cloud_header_stamp_
                                 << ", current_stamp=" << info.header_stamp
                                 << ", header_dt=" << info.header_dt
                                 << "; 若雷达应为10Hz，缺帧或时间戳跳变发生在本程序回调之前";
                }
                last_cloud_header_stamp_ = info.header_stamp;
                last_cloud_receive_steady_sec_ = callback_begin;

                try {
                    cloud_cb_(msg, info);
                } catch (const std::exception& e) {
                    LOG(ERROR) << "[Topic接收] PointCloud2入队回调异常: " << e.what();
                } catch (...) {
                    LOG(ERROR) << "[Topic接收] PointCloud2入队回调发生未知异常";
                }
                const double callback_ms = (TopicSteadySeconds() - callback_begin) * 1000.0;
                max_cloud_callback_ms_ = std::max(max_cloud_callback_ms_, callback_ms);
                if (callback_ms > 5.0) {
                    LOG(WARNING) << "[Topic接收诊断][PointCloud2] 接收回调耗时异常"
                                 << ", topic_sequence=" << info.topic_sequence
                                 << ", callback_ms=" << callback_ms;
                }
            });
    }

    if (livox_cb_) {
        livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            livox_topic, cloud_qos,
            [this](livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
                if (!input_enabled_.load()) {
                    return;
                }
                const double callback_begin = TopicSteadySeconds();
                const std::uint64_t count = ++livox_received_;
                LidarReceiveInfo info;
                info.topic_sequence = ++lidar_topic_sequence_;
                info.source_sequence = count;
                info.receive_steady_sec = callback_begin;
                info.header_stamp = rclcpp::Time(msg->header.stamp).seconds();
                if (last_livox_header_stamp_ != 0.0) {
                    info.header_dt = info.header_stamp - last_livox_header_stamp_;
                }
                if (last_livox_receive_steady_sec_ != 0.0) {
                    info.arrival_dt = callback_begin - last_livox_receive_steady_sec_;
                }
                if (last_livox_header_stamp_ != 0.0 && info.header_dt <= 0.0) {
                    ++livox_non_monotonic_stamp_count_;
                    LOG(ERROR) << std::setprecision(15)
                               << "[Topic接收诊断][Livox] 时间戳不递增"
                               << ", topic_sequence=" << info.topic_sequence
                               << ", source_sequence=" << info.source_sequence
                               << ", previous_stamp=" << last_livox_header_stamp_
                               << ", current_stamp=" << info.header_stamp
                               << ", header_dt=" << info.header_dt;
                } else if (info.header_dt > 0.15) {
                    ++livox_large_header_gap_count_;
                    LOG(WARNING) << std::setprecision(15)
                                 << "[Topic接收诊断][Livox] 回调入口已出现大时间间隔"
                                 << ", topic_sequence=" << info.topic_sequence
                                 << ", source_sequence=" << info.source_sequence
                                 << ", previous_stamp=" << last_livox_header_stamp_
                                 << ", current_stamp=" << info.header_stamp
                                 << ", header_dt=" << info.header_dt
                                 << "; 若雷达应为10Hz，缺帧或时间戳跳变发生在本程序回调之前";
                }
                last_livox_header_stamp_ = info.header_stamp;
                last_livox_receive_steady_sec_ = callback_begin;

                try {
                    livox_cb_(msg, info);
                } catch (const std::exception& e) {
                    LOG(ERROR) << "[Topic接收] Livox入队回调异常: " << e.what();
                } catch (...) {
                    LOG(ERROR) << "[Topic接收] Livox入队回调发生未知异常";
                }
                const double callback_ms = (TopicSteadySeconds() - callback_begin) * 1000.0;
                max_livox_callback_ms_ = std::max(max_livox_callback_ms_, callback_ms);
                if (callback_ms > 5.0) {
                    LOG(WARNING) << "[Topic接收诊断][Livox] 接收回调耗时异常"
                                 << ", topic_sequence=" << info.topic_sequence
                                 << ", callback_ms=" << callback_ms;
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
              << ", cloud_qos=KEEP_ALL+RELIABLE+VOLATILE"
              << ", imu_qos=KEEP_ALL+BEST_EFFORT+VOLATILE";
    return true;
}

void TopicInput::Spin() {
    LOG(INFO) << "[Topic接收线程] 开始 spin, thread_id=" << std::this_thread::get_id();
    executor_->spin();
    LOG(INFO) << "[Topic接收线程] spin 已退出, thread_id=" << std::this_thread::get_id();
}

void TopicInput::Shutdown() {
    input_enabled_.store(false);
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
              << ", livox_received=" << livox_received_.load()
              << ", lidar_topic_sequence=" << lidar_topic_sequence_
              << ", cloud_non_monotonic_stamp_count=" << cloud_non_monotonic_stamp_count_
              << ", livox_non_monotonic_stamp_count=" << livox_non_monotonic_stamp_count_
              << ", cloud_large_header_gap_count=" << cloud_large_header_gap_count_
              << ", livox_large_header_gap_count=" << livox_large_header_gap_count_
              << ", max_cloud_callback_ms=" << max_cloud_callback_ms_
              << ", max_livox_callback_ms=" << max_livox_callback_ms_;
}

}  // namespace lightning::runtime
