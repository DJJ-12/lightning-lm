#include "runtime/lightning.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <utility>

#include <glog/logging.h>
#include <pcl_conversions/pcl_conversions.h>
#include <yaml-cpp/yaml.h>

namespace lightning::runtime {
namespace {

double RuntimeSteadySeconds() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

Lightning::~Lightning() {
    LOG(INFO) << "[析构][Lightning] 开始 this=" << this;
    Shutdown();
    LOG(INFO) << "[析构][Lightning] 完成 this=" << this;
}

bool Lightning::Init(rclcpp::Node::SharedPtr node, const std::string& yaml_path) {
    node_ = node;
    yaml_path_ = yaml_path;
    task_.Reset(TaskState::IDLE, "lightning started");
    if (!node_ || yaml_path_.empty()) {
        return false;
    }

    mapping_map_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/lightning/mapping/map",
        rclcpp::QoS(1).reliable().transient_local());
    mapping_path_pub_ = node_->create_publisher<nav_msgs::msg::Path>(
        "/lightning/mapping/path",
        rclcpp::QoS(1).reliable().transient_local());

    YAML::Node yaml = YAML::LoadFile(yaml_path_);
    if (yaml["localization"] && yaml["localization"]["cloud_timeout_sec"]) {
        localization_cloud_timeout_sec_ = yaml["localization"]["cloud_timeout_sec"].as<double>();
    }
    if (yaml["mapping"]) {
        if (yaml["mapping"]["block_map_resolution"]) {
            save_map_options_.block_resolution = yaml["mapping"]["block_map_resolution"].as<int>();
        }
        if (yaml["mapping"]["block_map_voxel_size"]) {
            save_map_options_.block_voxel_size = yaml["mapping"]["block_map_voxel_size"].as<double>();
        }
    }

    topic_input_ = std::make_unique<TopicInput>();
    if (!topic_input_->Start(
            yaml_path_,
            [this](const sensor_msgs::msg::Imu::SharedPtr& imu) { RouteImu(imu); },
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr& cloud,
                   const TopicInput::LidarReceiveInfo& receive_info) {
                RouteCloud(cloud, receive_info);
            },
            [this](const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud,
                   const TopicInput::LidarReceiveInfo& receive_info) {
                RouteLivox(cloud, receive_info);
            })) {
        topic_input_.reset();
        return false;
    }

    LOG(INFO) << "[Lightning] 初始化完成，Topic接收节点已经常驻"
              << ", localization_timeout_sec=" << localization_cloud_timeout_sec_;
    return true;
}

void Lightning::Shutdown() {
    if (shutdown_.exchange(true)) {
        return;
    }

    std::lock_guard<std::mutex> lock(control_mutex_);
    LOG(INFO) << "[程序退出] [01] 停止在线消息路由";
    if (topic_input_) {
        topic_input_->SetEnabled(false);
    }
    online_route_ = OnlineRoute::NONE;
    task_.RequestCancel();

    LOG(INFO) << "[程序退出] [02] 停止在线工作线程";
    StopAllOnlineWorkersLocked(false);

    LOG(INFO) << "[程序退出] [03] 等待离线Bag线程退出";
    JoinOfflineThreadLocked();

    LOG(INFO) << "[程序退出] [04] 释放MappingSystem";
    if (mapping_system_) {
        mapping_system_->Stop();
    }
    ClearMappingSystemLocked();

    LOG(INFO) << "[程序退出] [05] 释放LocalizationSystem";
    ClearLocalizationSystemLocked();

    LOG(INFO) << "[程序退出] [06] 停止Topic独立executor并等待接收线程退出";
    if (topic_input_) {
        topic_input_->Shutdown();
        topic_input_.reset();
    }

    mapping_map_pub_.reset();
    mapping_path_pub_.reset();
    node_.reset();
    LOG(INFO) << "[程序退出] [07] Lightning资源释放完成";
}

bool Lightning::CanChangeModeLocked() const {
    const auto state = task_.State();
    return state != TaskState::RUNNING && state != TaskState::SAVING;
}

ServiceResult Lightning::SetMode(const std::string& mode_text) {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!CanChangeModeLocked()) {
        return {false, "task is running, finish or cancel it before changing mode"};
    }

    const Mode new_mode = ModeFromString(mode_text);
    if (!IsKnownModeName(mode_text)) {
        return {false, "unknown mode: " + mode_text};
    }

    online_route_ = OnlineRoute::NONE;
    StopAllOnlineWorkersLocked(false);
    JoinOfflineThreadLocked();
    if (mapping_system_) {
        mapping_system_->Stop();
    }
    ClearMappingSystemLocked();
    ClearLocalizationSystemLocked();
    mapping_save_path_.clear();
    localization_map_path_.clear();
    offline_bag_path_.clear();
    mode_ = new_mode;

    task_.Reset(TaskState::IDLE, ModeToString(mode_) + " mode selected");
    return {true, "mode set to " + ModeToString(mode_)};
}

TaskSnapshot Lightning::GetStatus() const {
    return task_.Snapshot();
}

Mode Lightning::CurrentMode() const {
    std::lock_guard<std::mutex> lock(control_mutex_);
    return mode_;
}

TaskSnapshot Lightning::GetOfflineMappingProgress() const {
    return task_.Snapshot();
}

void Lightning::RouteImu(const sensor_msgs::msg::Imu::SharedPtr& imu) {
    if (online_route_.load(std::memory_order_acquire) != OnlineRoute::MAPPING) {
        return;
    }
    const double now = RuntimeSteadySeconds();
    InputMessage input;
    input.sequence = ++online_sequence_;
    input.receive_steady_sec = now;
    input.header_stamp = rclcpp::Time(imu->header.stamp).seconds();
    input.type = InputType::IMU;
    input.imu = imu;
    PushMappingMessage(std::move(input));
}

void Lightning::RouteCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud,
                           const TopicInput::LidarReceiveInfo& receive_info) {
    const OnlineRoute route = online_route_.load(std::memory_order_acquire);
    if (route == OnlineRoute::NONE) {
        return;
    }

    InputMessage input;
    input.sequence = ++online_sequence_;
    input.topic_lidar_sequence = receive_info.topic_sequence;
    input.receive_steady_sec = receive_info.receive_steady_sec;
    input.header_stamp = receive_info.header_stamp;
    input.type = InputType::POINT_CLOUD2;
    input.cloud = cloud;
    if (route == OnlineRoute::MAPPING) {
        input.lidar_sequence = ++mapping_lidar_received_;
        PushMappingMessage(std::move(input));
    } else {
        input.lidar_sequence = ++localization_lidar_received_;
        PushLocalizationMessage(std::move(input));
    }
}

void Lightning::RouteLivox(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud,
                           const TopicInput::LidarReceiveInfo& receive_info) {
    const OnlineRoute route = online_route_.load(std::memory_order_acquire);
    if (route == OnlineRoute::NONE) {
        return;
    }

    InputMessage input;
    input.sequence = ++online_sequence_;
    input.topic_lidar_sequence = receive_info.topic_sequence;
    input.receive_steady_sec = receive_info.receive_steady_sec;
    input.header_stamp = receive_info.header_stamp;
    input.type = InputType::LIVOX;
    input.livox = cloud;
    if (route == OnlineRoute::MAPPING) {
        input.lidar_sequence = ++mapping_lidar_received_;
        PushMappingMessage(std::move(input));
    } else {
        input.lidar_sequence = ++localization_lidar_received_;
        PushLocalizationMessage(std::move(input));
    }
}

bool Lightning::IsLidarMessage(const InputMessage& input) {
    return input.type == InputType::POINT_CLOUD2 || input.type == InputType::LIVOX;
}

void Lightning::PushMappingMessage(InputMessage message) {
    const bool is_lidar = IsLidarMessage(message);
    const std::uint64_t lidar_sequence = message.lidar_sequence;
    std::size_t depth = 0;
    std::size_t replaced = 0;
    const bool pushed = is_lidar
        ? mapping_queue_.PushLatest(
              std::move(message),
              [](const InputMessage& input) { return Lightning::IsLidarMessage(input); },
              &depth, &replaced)
        : mapping_queue_.Push(std::move(message), &depth);
    if (!pushed) {
        if (is_lidar) {
            ++mapping_lidar_dropped_;
            LOG(ERROR) << "[数据链路诊断][在线建图] FIFO已经关闭，消息未入队"
                       << ", lidar_sequence=" << lidar_sequence;
        }
        return;
    }
    if (is_lidar) {
        ++mapping_lidar_enqueued_;
        if (replaced > 0) {
            const std::uint64_t dropped =
                mapping_lidar_dropped_.fetch_add(replaced) + replaced;
            if (dropped <= 20 || dropped % 100 == 0) {
                LOG(INFO) << "[在线建图队列] 实时模式丢弃旧点云"
                          << ", replaced=" << replaced
                          << ", dropped_total=" << dropped
                          << ", latest_lidar_sequence=" << lidar_sequence
                          << ", depth=" << depth;
            }
        }
    }
}

void Lightning::PushLocalizationMessage(InputMessage message) {
    const std::uint64_t lidar_sequence = message.lidar_sequence;
    std::size_t depth = 0;
    std::size_t replaced = 0;
    if (!localization_queue_.PushLatest(std::move(message), &depth, &replaced)) {
        ++localization_lidar_dropped_;
        LOG(ERROR) << "[数据链路诊断][在线定位] FIFO已经关闭，消息未入队"
                   << ", lidar_sequence=" << lidar_sequence;
        return;
    }
    ++localization_lidar_enqueued_;
    if (replaced > 0) {
        const std::uint64_t dropped =
            localization_lidar_dropped_.fetch_add(replaced) + replaced;
        if (dropped <= 20 || dropped % 100 == 0) {
            LOG(INFO) << "[在线定位队列] 实时模式丢弃旧帧"
                      << ", replaced=" << replaced
                      << ", dropped_total=" << dropped
                      << ", latest_lidar_sequence=" << lidar_sequence
                      << ", depth=" << depth;
        }
    }
}

void Lightning::StartOnlineMappingWorkerLocked() {
    mapping_lidar_received_ = 0;
    mapping_lidar_enqueued_ = 0;
    mapping_lidar_dropped_ = 0;
    mapping_queue_.Open();
    mapping_worker_ = std::thread([this]() { OnlineMappingWorkerLoop(); });
    online_route_.store(OnlineRoute::MAPPING, std::memory_order_release);
    if (topic_input_) {
        topic_input_->SetEnabled(true);
    }
    LOG(INFO) << "[在线建图] Topic路由已经打开，FIFO工作线程已经启动";
}

void Lightning::StopOnlineMappingWorkerLocked(bool drain) {
    if (online_route_.load() == OnlineRoute::MAPPING && topic_input_) {
        topic_input_->SetEnabled(false);
    }
    if (online_route_.load() == OnlineRoute::MAPPING) {
        online_route_ = OnlineRoute::NONE;
    } 
    const std::size_t pending = mapping_queue_.Close(drain);
    LOG(INFO) << "[在线建图] 关闭FIFO，drain=" << drain
              << ", pending=" << pending;
    if (mapping_worker_.joinable()) {
        mapping_worker_.join();
    }
}

void Lightning::OnlineMappingWorkerLoop() {
    LOG(INFO) << "[在线建图线程] 开始 thread_id=" << std::this_thread::get_id();
    std::uint64_t processed = 0;
    std::uint64_t imu_processed = 0;
    std::uint64_t lidar_processed = 0;
    std::uint64_t last_lidar_sequence = 0;
    std::uint64_t sequence_gap_count = 0;
    double max_queue_wait_ms = 0.0;
    double max_process_ms = 0.0;
    InputMessage input;
    while (mapping_queue_.WaitPop(&input) == QueuePopResult::MESSAGE) {
        const double process_begin = RuntimeSteadySeconds();
        const double queue_wait_ms = input.receive_steady_sec > 0.0
            ? (process_begin - input.receive_steady_sec) * 1000.0
            : 0.0;
        if (queue_wait_ms > max_queue_wait_ms) {
            max_queue_wait_ms = queue_wait_ms;
        }
        if (input.lidar_sequence != 0) {
            if (last_lidar_sequence != 0 && input.lidar_sequence != last_lidar_sequence + 1) {
                ++sequence_gap_count;
                LOG(WARNING) << "[数据链路诊断][在线建图] 雷达序号不连续"
                             << ", previous=" << last_lidar_sequence
                             << ", current=" << input.lidar_sequence
                             << ", header_stamp=" << input.header_stamp;
            }
            last_lidar_sequence = input.lidar_sequence;
        }

        ProcessMappingInput(input);
        const double process_ms = (RuntimeSteadySeconds() - process_begin) * 1000.0;
        if (process_ms > max_process_ms) {
            max_process_ms = process_ms;
        }
        ++processed;
        if (input.type == InputType::IMU) {
            ++imu_processed;
        }
        if (input.lidar_sequence != 0) {
            ++lidar_processed;
            if (lidar_processed % 100 == 0) {
                LOG(INFO) << "[在线建图线程] 雷达点云累计处理=" << lidar_processed
                          << ", 当前FIFO总处理=" << processed
                          << ", queue_depth=" << mapping_queue_.Size();
            }
            if (queue_wait_ms > 200.0 || process_ms > 200.0) {
                LOG(WARNING) << "[数据链路诊断][在线建图] 延迟异常"
                             << ", lidar_sequence=" << input.lidar_sequence
                             << ", header_stamp=" << input.header_stamp
                             << ", queue_wait_ms=" << queue_wait_ms
                             << ", process_ms=" << process_ms
                             << ", queue_depth=" << mapping_queue_.Size();
            }
        }
        if (processed % 1000 == 0) {
            LOG(INFO) << "[在线建图线程] FIFO累计处理(含IMU和雷达)=" << processed
                      << ", 其中IMU=" << imu_processed
                      << ", 雷达=" << lidar_processed
                      << ", queue_depth=" << mapping_queue_.Size();
        }
    }
    LOG(INFO) << "[数据链路诊断][在线建图] 工作线程退出汇总"
              << ", lidar_received=" << mapping_lidar_received_.load()
              << ", lidar_enqueued=" << mapping_lidar_enqueued_.load()
              << ", lidar_processed=" << lidar_processed
              << ", imu_processed=" << imu_processed
              << ", total_processed=" << processed
              << ", lidar_dropped=" << mapping_lidar_dropped_.load()
              << ", sequence_gap_count=" << sequence_gap_count
              << ", max_queue_wait_ms=" << max_queue_wait_ms
              << ", max_process_ms=" << max_process_ms;
    LOG(INFO) << "[在线建图线程] 退出，累计处理=" << processed;
}

void Lightning::StartOnlineLocalizationWorkerLocked() {
    localization_lidar_received_ = 0;
    localization_lidar_enqueued_ = 0;
    localization_lidar_dropped_ = 0;
    localization_queue_.Open();
    localization_worker_ = std::thread([this]() { OnlineLocalizationWorkerLoop(); });
    online_route_.store(OnlineRoute::LOCALIZATION, std::memory_order_release);
    if (topic_input_) {
        topic_input_->SetEnabled(true);
    }
    LOG(INFO) << "[在线定位] Topic路由已经打开，FIFO工作线程已经启动";
}

void Lightning::StopOnlineLocalizationWorkerLocked(bool drain) {
    if (online_route_.load() == OnlineRoute::LOCALIZATION && topic_input_) {
        topic_input_->SetEnabled(false);
    }
    if (online_route_.load() == OnlineRoute::LOCALIZATION) {
        online_route_ = OnlineRoute::NONE;
    }
    const std::size_t pending = localization_queue_.Close(drain);
    LOG(INFO) << "[在线定位] 关闭FIFO，drain=" << drain << ", pending=" << pending;
    if (localization_worker_.joinable()) {
        localization_worker_.join();
    }
}

void Lightning::OnlineLocalizationWorkerLoop() {
    LOG(INFO) << "[在线定位线程] 开始 thread_id=" << std::this_thread::get_id();
    std::uint64_t processed = 0;
    std::uint64_t pointcloud2_processed = 0;
    std::uint64_t livox_processed = 0;
    std::uint64_t ndt_executed = 0;
    std::uint64_t initialized_frames = 0;
    std::uint64_t frames_not_sent_to_ndt = 0;
    std::uint64_t last_lidar_sequence = 0;
    std::uint64_t sequence_gap_count = 0;
    std::uint64_t non_monotonic_header_count = 0;
    std::uint64_t source_large_header_gap_count = 0;
    double last_header_stamp = 0.0;
    double last_pointcloud2_stamp = 0.0;
    double last_livox_stamp = 0.0;
    double max_queue_wait_ms = 0.0;
    double max_process_ms = 0.0;
    bool timeout_reported = false;
    bool mixed_source_reported = false;
    InputMessage input;
    const auto timeout = std::chrono::duration<double>(localization_cloud_timeout_sec_);

    while (true) {
        const QueuePopResult result = localization_queue_.WaitPopFor(&input, timeout);
        if (result == QueuePopResult::CLOSED) {
            break;
        }
        if (result == QueuePopResult::TIMEOUT) {
            if (!timeout_reported) {
                HandleLocalizationTimeout();
                timeout_reported = true;
            }
            continue;
        }

        timeout_reported = false;
        const double process_begin = RuntimeSteadySeconds();
        const double queue_wait_ms = input.receive_steady_sec > 0.0
            ? (process_begin - input.receive_steady_sec) * 1000.0
            : 0.0;
        max_queue_wait_ms = std::max(max_queue_wait_ms, queue_wait_ms);

        if (last_lidar_sequence != 0 && input.lidar_sequence != last_lidar_sequence + 1) {
            ++sequence_gap_count;
            LOG(INFO) << "[在线定位输入诊断][Worker] 实时模式跳过旧帧"
                      << ", previous=" << last_lidar_sequence
                      << ", current=" << input.lidar_sequence
                      << ", topic_sequence=" << input.topic_lidar_sequence
                      << ", header_stamp=" << std::setprecision(15) << input.header_stamp;
        }
        last_lidar_sequence = input.lidar_sequence;

        const double merged_header_dt = last_header_stamp == 0.0
            ? 0.0 : input.header_stamp - last_header_stamp;
        if (last_header_stamp != 0.0 && merged_header_dt <= 0.0) {
            ++non_monotonic_header_count;
            LOG(ERROR) << std::setprecision(15)
                       << "[在线定位输入诊断][Worker] 合并后的雷达时间戳不递增"
                       << ", task_sequence=" << input.lidar_sequence
                       << ", topic_sequence=" << input.topic_lidar_sequence
                       << ", previous_stamp=" << last_header_stamp
                       << ", current_stamp=" << input.header_stamp
                       << ", header_dt=" << merged_header_dt
                       << ", source="
                       << (input.type == InputType::POINT_CLOUD2 ? "PointCloud2" : "Livox");
        }
        last_header_stamp = input.header_stamp;

        double source_header_dt = 0.0;
        if (input.type == InputType::POINT_CLOUD2) {
            ++pointcloud2_processed;
            source_header_dt = last_pointcloud2_stamp == 0.0
                ? 0.0 : input.header_stamp - last_pointcloud2_stamp;
            last_pointcloud2_stamp = input.header_stamp;
        } else {
            ++livox_processed;
            source_header_dt = last_livox_stamp == 0.0
                ? 0.0 : input.header_stamp - last_livox_stamp;
            last_livox_stamp = input.header_stamp;
        }
        if (source_header_dt > 0.15) {
            ++source_large_header_gap_count;
            LOG(WARNING) << std::setprecision(15)
                         << "[在线定位输入诊断][Worker] FIFO中相邻同源点云时间间隔过大"
                         << ", task_sequence=" << input.lidar_sequence
                         << ", topic_sequence=" << input.topic_lidar_sequence
                         << ", source="
                         << (input.type == InputType::POINT_CLOUD2 ? "PointCloud2" : "Livox")
                         << ", source_header_dt=" << source_header_dt;
        }
        if (!mixed_source_reported && pointcloud2_processed > 0 && livox_processed > 0) {
            mixed_source_reported = true;
            LOG(ERROR) << "[在线定位输入诊断][Worker] 同一定位任务同时收到PointCloud2和Livox"
                       << ", PointCloud2=" << pointcloud2_processed
                       << ", Livox=" << livox_processed
                       << "; 两路雷达消息会被合并进同一个NDT序列，请检查是否重复发布同一雷达";
        }

        if (processed < 20 || (processed + 1) % 100 == 0) {
            LOG(INFO) << std::setprecision(15)
                      << "[在线定位输入诊断][Worker] 准备送入LocalizationSystem"
                      << ", task_sequence=" << input.lidar_sequence
                      << ", topic_sequence=" << input.topic_lidar_sequence
                      << ", source="
                      << (input.type == InputType::POINT_CLOUD2 ? "PointCloud2" : "Livox")
                      << ", header_stamp=" << input.header_stamp
                      << ", merged_header_dt=" << merged_header_dt
                      << ", source_header_dt=" << source_header_dt
                      << ", queue_wait_ms=" << queue_wait_ms;
        }

        const loc::LocalizationFrameOutcome outcome = ProcessLocalizationInput(input);
        const double process_end = RuntimeSteadySeconds();
        if (outcome == loc::LocalizationFrameOutcome::NDT_EXECUTED) {
            ++ndt_executed;
        } else if (outcome == loc::LocalizationFrameOutcome::INITIALIZED_WITH_FRAME) {
            ++initialized_frames;
            const std::size_t removed = localization_queue_.RemoveIf(
                [process_end](const InputMessage& pending) {
                    return pending.receive_steady_sec <= process_end;
                });
            if (removed > 0) {
                const std::uint64_t dropped =
                    localization_lidar_dropped_.fetch_add(removed) + removed;
                LOG(INFO) << "[在线定位队列] 初始化完成，丢弃初始化期间缓存的旧帧"
                          << ", removed=" << removed
                          << ", dropped_total=" << dropped;
            }
        } else {
            ++frames_not_sent_to_ndt;
        }

        const double process_ms = (process_end - process_begin) * 1000.0;
        max_process_ms = std::max(max_process_ms, process_ms);
        ++processed;
        if (processed <= 20 || processed % 100 == 0 ||
            outcome != loc::LocalizationFrameOutcome::NDT_EXECUTED) {
            LOG(INFO) << std::setprecision(15)
                      << "[在线定位输入诊断][Worker] LocalizationSystem返回"
                      << ", task_sequence=" << input.lidar_sequence
                      << ", topic_sequence=" << input.topic_lidar_sequence
                      << ", header_stamp=" << input.header_stamp
                      << ", outcome=" << loc::LocalizationFrameOutcomeName(outcome)
                      << ", queue_wait_ms=" << queue_wait_ms
                      << ", process_ms=" << process_ms;
        }
    }
    LOG(INFO) << "[在线定位输入诊断][Worker] 退出汇总"
              << ", lidar_received=" << localization_lidar_received_.load()
              << ", lidar_enqueued=" << localization_lidar_enqueued_.load()
              << ", lidar_processed=" << processed
              << ", pointcloud2_processed=" << pointcloud2_processed
              << ", livox_processed=" << livox_processed
              << ", ndt_executed=" << ndt_executed
              << ", initialized_frames=" << initialized_frames
              << ", frames_not_sent_to_ndt=" << frames_not_sent_to_ndt
              << ", lidar_dropped=" << localization_lidar_dropped_.load()
              << ", sequence_gap_count=" << sequence_gap_count
              << ", non_monotonic_header_count=" << non_monotonic_header_count
              << ", source_large_header_gap_count=" << source_large_header_gap_count
              << ", max_queue_wait_ms=" << max_queue_wait_ms
              << ", max_process_ms=" << max_process_ms;
    LOG(INFO) << "[在线定位线程] 退出，累计处理=" << processed;
}

void Lightning::ProcessMappingInput(const InputMessage& input) {
    if (!mapping_system_) {
        return;
    }

    if (input.type == InputType::IMU) {
        mapping_system_->ProcessIMU(input.imu);
        return;
    }
    if (input.type == InputType::POINT_CLOUD2) {
        mapping_system_->ProcessCloud(input.cloud);
    } else {
        mapping_system_->ProcessCloud(input.livox);
    }

    if (mapping_system_->ConsumeMappingUpdate()) {
        PublishMappingOutputsLocked(false);
    }
}

loc::LocalizationFrameOutcome Lightning::ProcessLocalizationInput(const InputMessage& input) {
    if (!localization_system_) {
        return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
    }

    loc::LocalizationInputDiagnostic diagnostic;
    diagnostic.pipeline_sequence = input.lidar_sequence;
    diagnostic.topic_sequence = input.topic_lidar_sequence;
    diagnostic.online = input.topic_lidar_sequence != 0;
    diagnostic.topic_receive_steady_sec = input.receive_steady_sec;
    diagnostic.worker_begin_steady_sec = RuntimeSteadySeconds();
    diagnostic.header_stamp = input.header_stamp;

    if (input.type == InputType::POINT_CLOUD2) {
        return localization_system_->ProcessCloud(input.cloud, diagnostic);
    }
    if (input.type == InputType::LIVOX) {
        return localization_system_->ProcessCloud(input.livox, diagnostic);
    }
    return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
}

void Lightning::HandleLocalizationTimeout() {
    if (!localization_system_) {
        return;
    }
    const std::string message = "雷达消息超时：连续 " +
        std::to_string(localization_cloud_timeout_sec_) + " 秒没有收到点云";
    LOG(WARNING) << "[在线定位] " << message;
    localization_system_->MarkPoor(message + "; localization quality: poor");
}

void Lightning::StopAllOnlineWorkersLocked(bool drain) {
    if (topic_input_) {
        topic_input_->SetEnabled(false);
    }
    online_route_ = OnlineRoute::NONE;
    StopOnlineMappingWorkerLocked(drain);
    StopOnlineLocalizationWorkerLocked(drain);
}

void Lightning::ClearMappingSystemLocked() {
    if (!mapping_system_) {
        return;
    }
    LOG(INFO) << "[析构流程][MappingSystem] 准备释放"
              << ", generation=" << mapping_task_generation_
              << ", ptr=" << mapping_system_.get();
    mapping_system_.reset();
    LOG(INFO) << "[析构流程][MappingSystem] 已释放";
}

void Lightning::ClearLocalizationSystemLocked() {
    if (!localization_system_) {
        return;
    }
    LOG(INFO) << "[析构流程][LocalizationSystem] 准备释放 ptr=" << localization_system_.get();
    localization_system_.reset();
    LOG(INFO) << "[析构流程][LocalizationSystem] 已释放";
}

void Lightning::JoinOfflineThreadLocked() {
    if (!offline_thread_.joinable()) {
        return;
    }
    LOG(INFO) << "[离线线程] 等待Bag线程退出";
    offline_thread_.join();
    LOG(INFO) << "[离线线程] Bag线程已经退出";
}

ServiceResult Lightning::StartMapping(const std::string& save_path) {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!IsMappingMode(mode_)) {
        return {false, "start_mapping is only allowed in mapping modes"};
    }
    if (task_.State() == TaskState::RUNNING || task_.State() == TaskState::SAVING) {
        return {false, "mapping is already running"};
    }

    StopAllOnlineWorkersLocked(false);
    JoinOfflineThreadLocked();
    ClearMappingSystemLocked();
    mapping_save_path_ = save_path;
    offline_bag_path_.clear();

    if (mode_ == Mode::OFFLINE_MAPPING) {
        task_.Reset(TaskState::READY, "offline mapping ready, waiting for load_bag");
        return {true, "offline mapping ready, save_path: " + save_path};
    }

    mapping_system_ = std::make_unique<modules::MappingSystem>();
    ++mapping_task_generation_;
    LOG(INFO) << "[跨任务状态诊断][建图任务] 创建在线建图任务"
              << ", generation=" << mapping_task_generation_
              << ", MappingSystem=" << mapping_system_.get();
    modules::MappingSystemOptions mapping_options;
    mapping_options.online_input = true;
    if (!mapping_system_->Init(yaml_path_, mapping_options) || !mapping_system_->Start()) {
        ClearMappingSystemLocked();
        task_.SetFinished(false, "failed to initialize MappingSystem");
        return {false, "failed to initialize MappingSystem"};
    }

    StartOnlineMappingWorkerLocked();
    task_.Reset(TaskState::RUNNING, "online mapping running");
    return {true, "online mapping started, save_path: " + save_path};
}

ServiceResult Lightning::LoadBag(const std::string& bag_path) {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (mode_ != Mode::OFFLINE_MAPPING && mode_ != Mode::OFFLINE_LOCALIZATION) {
        return {false, "load_bag is only allowed in offline modes"};
    }
    if (task_.State() == TaskState::RUNNING || task_.State() == TaskState::SAVING) {
        return {false, "offline task is already running"};
    }

    JoinOfflineThreadLocked();
    offline_bag_path_ = bag_path;

    if (mode_ == Mode::OFFLINE_LOCALIZATION) {
        task_.Reset(TaskState::READY, "offline bag loaded, waiting for set_map_path and set_location");
        return {true, "offline localization bag loaded: " + bag_path};
    }

    if (mapping_save_path_.empty()) {
        return {false, "start_mapping has not been called"};
    }

    ClearMappingSystemLocked();
    mapping_system_ = std::make_unique<modules::MappingSystem>();
    ++mapping_task_generation_;
    LOG(INFO) << "[跨任务状态诊断][建图任务] 创建离线建图任务"
              << ", generation=" << mapping_task_generation_
              << ", MappingSystem=" << mapping_system_.get();
    modules::MappingSystemOptions mapping_options;
    mapping_options.online_input = false;
    if (!mapping_system_->Init(yaml_path_, mapping_options) || !mapping_system_->Start()) {
        ClearMappingSystemLocked();
        task_.SetFinished(false, "failed to initialize MappingSystem");
        return {false, "failed to initialize MappingSystem"};
    }

    task_.Reset(TaskState::RUNNING, "offline mapping running");
    StartBagMappingTaskLocked(bag_path);
    return {true, "offline mapping bag loaded: " + bag_path};
}

void Lightning::StartBagMappingTaskLocked(const std::string& bag_path) {
    offline_thread_ = std::thread([this, bag_path]() {
        LOG(INFO) << "[离线建图线程] 开始读取Bag";
        BagInput bag_input;
        const bool bag_ok = bag_input.Run(
            bag_path, yaml_path_,
            [this](const sensor_msgs::msg::Imu::SharedPtr& imu) {
                InputMessage input;
                input.type = InputType::IMU;
                input.imu = imu;
                ProcessMappingInput(input);
            },
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
                InputMessage input;
                input.type = InputType::POINT_CLOUD2;
                input.cloud = cloud;
                ProcessMappingInput(input);
            },
            [this](const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud) {
                InputMessage input;
                input.type = InputType::LIVOX;
                input.livox = cloud;
                ProcessMappingInput(input);
            },
            [this](const BagInputProgress& progress) {
                task_.SetProgress(progress.processed_frames, progress.total_frames, "offline mapping running");
            },
            [this]() { return task_.CancelRequested(); });

        if (task_.CancelRequested()) {
            task_.SetState(TaskState::CANCELLED, "offline mapping cancelled");
        } else if (!bag_ok) {
            task_.SetFinished(false, "offline bag mapping failed");
        } else {
            mapping_system_->Stop();
            const ServiceResult result = SaveMappingLocked(mapping_save_path_);
            task_.SetFinished(result.success, result.message);
        }
        LOG(INFO) << "[离线建图线程] 读取Bag结束";
    });
}

void Lightning::PublishMappingOutputsLocked(bool force) {
    if (!mapping_system_) {
        return;
    }
    const bool publish_map =
        mapping_map_pub_ && (force || mapping_map_pub_->get_subscription_count() > 0);
    const bool publish_path =
        mapping_path_pub_ && (force || mapping_path_pub_->get_subscription_count() > 0);
    if (!publish_map && !publish_path) {
        return;
    }

    const auto stamp = node_ ? node_->now() : rclcpp::Clock().now();
    if (publish_map) {
        CloudPtr map_base = mapping_system_->BuildCurrentMapInBaseFrame();
        if (map_base && !map_base->empty()) {
            sensor_msgs::msg::PointCloud2 msg;
            pcl::toROSMsg(*map_base, msg);
            msg.header.stamp = stamp;
            msg.header.frame_id = "map";
            mapping_map_pub_->publish(msg);
        }
    }

    if (publish_path) {
        nav_msgs::msg::Path path = mapping_system_->BuildCurrentPath();
        if (!path.poses.empty()) {
            path.header.stamp = stamp;
            path.header.frame_id = "map";
            for (auto& pose : path.poses) {
                pose.header.stamp = stamp;
                pose.header.frame_id = "map";
            }
            mapping_path_pub_->publish(path);
        }
    }
}

ServiceResult Lightning::SaveMappingLocked(const std::string& save_path) {
    if (!mapping_system_) {
        return {false, "MappingSystem is not running"};
    }
    task_.SetState(TaskState::SAVING, "saving map");
    const auto result = mapping_system_->GetResult();
    const bool ok = save_map_.Save(save_path, result, save_map_options_);
    if (ok) {
        PublishMappingOutputsLocked(true);
    }
    return {ok, ok ? "map saved" : "failed to save map"};
}

ServiceResult Lightning::FinishMapping(bool save_map) {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!IsMappingMode(mode_)) {
        return {false, "finish_mapping is only allowed in mapping modes"};
    }
    if (!mapping_system_) {
        return {false, "no mapping task"};
    }

    LOG(INFO) << "[结束建图] [01] 停止数据源";
    if (mode_ == Mode::ONLINE_MAPPING) {
        StopOnlineMappingWorkerLocked(true);
    } else {
        task_.RequestCancel();
        JoinOfflineThreadLocked();
    }

    LOG(INFO) << "[结束建图] [02] 所有建图数据处理线程已经退出";
    mapping_system_->Stop();

    ServiceResult result{true, "mapping finished without saving"};
    if (save_map) {
        result = SaveMappingLocked(mapping_save_path_);
    } else {
        PublishMappingOutputsLocked(true);
    }

    LOG(INFO) << "[结束建图] [03] 准备析构MappingSystem";
    ClearMappingSystemLocked();
    task_.SetFinished(result.success, result.message);
    LOG(INFO) << "[结束建图] [04] 完成";
    return result;
}

ServiceResult Lightning::SetMapPath(const std::string& map_path) {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!IsLocalizationMode(mode_)) {
        return {false, "set_map_path is only allowed in localization modes"};
    }
    if (task_.State() == TaskState::RUNNING || task_.State() == TaskState::SAVING) {
        return {false, "localization is already running"};
    }

    StopOnlineLocalizationWorkerLocked(false);
    JoinOfflineThreadLocked();
    ClearLocalizationSystemLocked();

    localization_system_ = std::make_unique<modules::LocalizationSystem>();
    ++localization_task_generation_;
    LOG(INFO) << "[在线定位输入诊断][定位任务] 创建LocalizationSystem"
              << ", generation=" << localization_task_generation_
              << ", mode=" << ModeToString(mode_)
              << ", ptr=" << localization_system_.get();
    if (!localization_system_->Init(yaml_path_, node_)) {
        ClearLocalizationSystemLocked();
        task_.SetFinished(false, "failed to initialize LocalizationSystem");
        return {false, "failed to initialize LocalizationSystem"};
    }
    if (!localization_system_->SetMapPath(map_path)) {
        ClearLocalizationSystemLocked();
        task_.SetFinished(false, "failed to load localization map: " + map_path);
        return {false, "failed to load localization map: " + map_path};
    }

    localization_map_path_ = map_path;
    task_.Reset(TaskState::READY, "localization map loaded, waiting for set_location");
    return {true, "localization map loaded: " + map_path};
}

ServiceResult Lightning::SetLocation(const SE3& init_pose, bool* initialized_now) {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!IsLocalizationMode(mode_)) {
        return {false, "set_location is only allowed in localization modes"};
    }
    if (!localization_system_) {
        return {false, "set_map_path has not been called"};
    }
    if (mode_ == Mode::OFFLINE_LOCALIZATION && offline_bag_path_.empty()) {
        return {false, "load_bag has not been called"};
    }
    if (mode_ == Mode::OFFLINE_LOCALIZATION &&
        (task_.State() == TaskState::RUNNING || task_.State() == TaskState::SAVING)) {
        return {false, "offline localization is already running"};
    }

    if (mode_ == Mode::ONLINE_LOCALIZATION) {
        StopOnlineLocalizationWorkerLocked(false);
    }

    bool initialized = false;
    if (!localization_system_->SetInitialGuess(init_pose, &initialized)) {
        return {false, "failed to set initial pose"};
    }
    if (initialized_now) {
        *initialized_now = initialized;
    }

    if (mode_ == Mode::OFFLINE_LOCALIZATION) {
        JoinOfflineThreadLocked();
        task_.Reset(TaskState::RUNNING, "offline localization running");
        StartBagLocalizationTaskLocked(offline_bag_path_);
        return {true, "offline localization started"};
    }

    StartOnlineLocalizationWorkerLocked();
    task_.SetState(initialized ? TaskState::RUNNING : TaskState::WAIT_CLOUD,
                   initialized ? "localization initialized" : "initial pose accepted, waiting for current cloud");
    return {true, initialized ? "localization initialized" : "initial pose accepted, waiting for current cloud"};
}

void Lightning::StartBagLocalizationTaskLocked(const std::string& bag_path) {
    offline_thread_ = std::thread([this, bag_path]() {
        LOG(INFO) << "[离线定位线程] 开始读取Bag";
        offline_localization_sequence_ = 0;
        BagInput bag_input;
        const bool bag_ok = bag_input.Run(
            bag_path, yaml_path_,
            nullptr,
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
                InputMessage input;
                input.lidar_sequence = ++offline_localization_sequence_;
                input.receive_steady_sec = RuntimeSteadySeconds();
                input.header_stamp = rclcpp::Time(cloud->header.stamp).seconds();
                input.type = InputType::POINT_CLOUD2;
                input.cloud = cloud;
                ProcessLocalizationInput(input);
            },
            [this](const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud) {
                InputMessage input;
                input.lidar_sequence = ++offline_localization_sequence_;
                input.receive_steady_sec = RuntimeSteadySeconds();
                input.header_stamp = rclcpp::Time(cloud->header.stamp).seconds();
                input.type = InputType::LIVOX;
                input.livox = cloud;
                ProcessLocalizationInput(input);
            },
            [this](const BagInputProgress& progress) {
                task_.SetProgress(progress.processed_frames, progress.total_frames, "offline localization running");
            },
            [this]() { return task_.CancelRequested(); });

        if (task_.CancelRequested()) {
            task_.SetState(TaskState::CANCELLED, "offline localization cancelled");
        } else {
            task_.SetFinished(bag_ok, bag_ok ? "offline localization finished" : "offline bag localization failed");
        }
        LOG(INFO) << "[离线定位线程] 读取Bag结束";
    });
}

ServiceResult Lightning::FinishLocalization() {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (mode_ != Mode::ONLINE_LOCALIZATION) {
        return {false, "finish_localization is only allowed in online_localization mode"};
    }
    if (!localization_system_) {
        return {false, "no online localization task"};
    }

    LOG(INFO) << "[结束定位] [01] 关闭Topic到定位队列的路由";
    StopOnlineLocalizationWorkerLocked(true);
    LOG(INFO) << "[结束定位] [02] 定位工作线程已经退出";
    ClearLocalizationSystemLocked();
    task_.SetFinished(true, "online localization finished");
    LOG(INFO) << "[结束定位] [03] 完成";
    return {true, "online localization finished"};
}

ServiceResult Lightning::GetMapPath(std::string* map_path) const {
    std::lock_guard<std::mutex> lock(control_mutex_);
    const std::string& current_map_path = IsMappingMode(mode_) ? mapping_save_path_ : localization_map_path_;
    if (map_path) {
        *map_path = current_map_path;
    }
    if (current_map_path.empty()) {
        return {false, "map path is not set"};
    }
    return {true, "map path: " + current_map_path};
}

loc::LocalizationResult Lightning::GetLocalizationQuality() const {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!localization_system_) {
        return loc::LocalizationResult();
    }
    return localization_system_->GetLatestResult();
}

ServiceResult Lightning::CancelTask() {
    std::lock_guard<std::mutex> lock(control_mutex_);
    LOG(INFO) << "[取消任务] [01] 请求停止数据源和任务线程";
    task_.RequestCancel();
    online_route_ = OnlineRoute::NONE;
    StopAllOnlineWorkersLocked(false);
    JoinOfflineThreadLocked();

    LOG(INFO) << "[取消任务] [02] 任务线程已经退出，开始释放系统对象";
    if (mapping_system_) {
        mapping_system_->Stop();
    }
    ClearMappingSystemLocked();
    ClearLocalizationSystemLocked();
    task_.SetState(TaskState::CANCELLED, "task cancelled");
    LOG(INFO) << "[取消任务] [03] 完成";
    return {true, "task cancelled"};
}

}  // namespace lightning::runtime
