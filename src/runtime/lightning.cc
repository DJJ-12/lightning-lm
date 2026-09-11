#include "runtime/lightning.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <utility>
#include <vector>

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
            [this](const sensor_msgs::msg::Imu::SharedPtr& imu) { AcceptImu(imu); },
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr& cloud,
                   const TopicInput::LidarReceiveInfo& receive_info) {
                AcceptCloud(cloud, receive_info);
            },
            [this](const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud,
                   const TopicInput::LidarReceiveInfo& receive_info) {
                AcceptLivox(cloud, receive_info);
            },
            [this](const sensor_msgs::msg::NavSatFix::SharedPtr& fix) {
                AcceptGps1(fix);
            },
            [this](const sensor_msgs::msg::NavSatFix::SharedPtr& fix) {
                AcceptGps2(fix);
            },
            [this](const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& velocity) {
                AcceptgpsVelocity(velocity);
            },
            [this](const nav_msgs::msg::Odometry::SharedPtr& odometry) {
                AcceptWheelOdometry(odometry);
            })) {
        topic_input_.reset();
        return false;
    }

    LOG(INFO) << "[Lightning] 初始化完成，Topic接收节点已经常驻";
    return true;
}

void Lightning::Shutdown() {
    if (shutdown_.exchange(true)) {
        return;
    }

    std::lock_guard<std::mutex> lock(control_mutex_);
    LOG(INFO) << "[程序退出] [01] 停止在线消息输入";
    if (topic_input_) {
        topic_input_->SetEnabled(false);
    }
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

bool Lightning::EnsureLocalizationSystemLocked() {
    if (localization_system_) return true;
    localization_system_ = std::make_unique<modules::LocalizationSystem>();
    ++localization_task_generation_;
    if (!localization_system_->Init(yaml_path_, node_)) {
        ClearLocalizationSystemLocked();
        return false;
    }
    return true;
}

bool Lightning::SetMapOriginInitialGuessLocked(
    const std::string& context) {
    if (!localization_system_) return false;

    const SE3 map_origin(
        Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    if (!localization_system_->SetInitialGuess(map_origin)) {
        LOG(ERROR) << "[" << context << "] failed to set automatic "
                      "map-origin initial pose";
        return false;
    }

    LOG(INFO) << "[" << context << "] automatic initial pose set to "
                 "map-frame identity";
    return true;
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

    // Selecting a mode never starts a worker. Online localization keeps the
    // set_map_path -> set_location sequence.
    task_.Reset(TaskState::IDLE, ModeToString(mode_) + " mode selected");
    return {true, "mode set to " + ModeToString(mode_)};
}

TaskSnapshot Lightning::GetStatus() const {
    TaskSnapshot status = task_.Snapshot();
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (IsMappingMode(mode_) && mapping_system_) {
        status.keyframe_count = mapping_system_->GetKeyframeCount();
        status.keyframe_memory_mb =
            static_cast<double>(mapping_system_->GetKeyframeCloudBytes()) /
            (1024.0 * 1024.0);
    }
    return status;
}

Mode Lightning::CurrentMode() const {
    std::lock_guard<std::mutex> lock(control_mutex_);
    return mode_;
}

TaskSnapshot Lightning::GetOfflineMappingProgress() const {
    return task_.Snapshot();
}

void Lightning::AcceptImu(const sensor_msgs::msg::Imu::SharedPtr& imu) {
    if (!imu) return;

    InputMessage input;
    input.receive_steady_sec = RuntimeSteadySeconds();
    input.header_stamp = rclcpp::Time(imu->header.stamp).seconds();
    input.type = InputType::IMU;
    input.imu = imu;

    {
        std::lock_guard<std::mutex> lock(online_input_mutex_);
        if (!online_worker_running_ || online_worker_is_localization_) return;
        pending_mapping_imu_.push_back(std::move(input));
        ++online_imu_received_;
    }
    online_input_ready_.notify_one();
}


void Lightning::AcceptGps1(
    const sensor_msgs::msg::NavSatFix::SharedPtr& fix) {
    if (!fix) return;
    InputMessage input;
    input.receive_steady_sec = RuntimeSteadySeconds();
    input.header_stamp = rclcpp::Time(fix->header.stamp).seconds();
    input.type = InputType::GPS1;
    input.gps_fix = fix;
    {
        std::lock_guard<std::mutex> lock(online_input_mutex_);
        if (!online_worker_running_ || !online_worker_is_localization_) return;
        latest_gps1_ = std::move(input);
        has_latest_gps1_ = true;
        ++online_gps1_received_;
    }
    online_input_ready_.notify_one();
}

void Lightning::AcceptGps2(
    const sensor_msgs::msg::NavSatFix::SharedPtr& fix) {
    if (!fix) return;
    InputMessage input;
    input.receive_steady_sec = RuntimeSteadySeconds();
    input.header_stamp = rclcpp::Time(fix->header.stamp).seconds();
    input.type = InputType::GPS2;
    input.gps_fix = fix;
    {
        std::lock_guard<std::mutex> lock(online_input_mutex_);
        if (!online_worker_running_ || !online_worker_is_localization_) return;
        latest_gps2_ = std::move(input);
        has_latest_gps2_ = true;
        ++online_gps2_received_;
    }
    online_input_ready_.notify_one();
}

int LocalizationInputPriority(InputType type) {
    switch (type) {
        case InputType::GPS1:
            return 0;
        case InputType::GPS2:
            return 1;
        case InputType::IMU:
            return 2;
        case InputType::gps_VELOCITY:
            return 3;
        case InputType::WHEEL_ODOMETRY:
            return 4;
        case InputType::POINT_CLOUD2:
        case InputType::LIVOX:
            return 5;
    }
    return 5;
}

void Lightning::AcceptgpsVelocity(
    const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& velocity) {
    if (!velocity) return;
    InputMessage input;
    input.receive_steady_sec = RuntimeSteadySeconds();
    input.header_stamp = rclcpp::Time(velocity->header.stamp).seconds();
    input.type = InputType::gps_VELOCITY;
    input.gps_velocity = velocity;
    {
        std::lock_guard<std::mutex> lock(online_input_mutex_);
        if (!online_worker_running_ || !online_worker_is_localization_) {
            return;
        }
        latest_gps_velocity_ = std::move(input);
        has_latest_gps_velocity_ = true;
        ++online_gps_velocity_received_;
    }
    online_input_ready_.notify_one();
}

void Lightning::AcceptWheelOdometry(
    const nav_msgs::msg::Odometry::SharedPtr& odometry) {
    if (!odometry) return;
    InputMessage input;
    input.receive_steady_sec = RuntimeSteadySeconds();
    input.header_stamp = rclcpp::Time(odometry->header.stamp).seconds();
    input.type = InputType::WHEEL_ODOMETRY;
    input.wheel_odometry = odometry;
    {
        std::lock_guard<std::mutex> lock(online_input_mutex_);
        if (!online_worker_running_ || !online_worker_is_localization_) {
            return;
        }
        latest_wheel_odometry_ = std::move(input);
        has_latest_wheel_odometry_ = true;
        ++online_wheel_odometry_received_;
    }
    online_input_ready_.notify_one();
}

void Lightning::AcceptCloud(
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud,
    const TopicInput::LidarReceiveInfo& receive_info) {
    if (!cloud) return;

    InputMessage input;
    input.topic_lidar_sequence = receive_info.topic_sequence;
    input.receive_steady_sec = receive_info.receive_steady_sec;
    input.header_stamp = receive_info.header_stamp;
    input.type = InputType::POINT_CLOUD2;
    input.cloud = cloud;
    OverwriteLatestLidar(std::move(input));
}

void Lightning::AcceptLivox(
    const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud,
    const TopicInput::LidarReceiveInfo& receive_info) {
    if (!cloud) return;

    InputMessage input;
    input.topic_lidar_sequence = receive_info.topic_sequence;
    input.receive_steady_sec = receive_info.receive_steady_sec;
    input.header_stamp = receive_info.header_stamp;
    input.type = InputType::LIVOX;
    input.livox = cloud;
    OverwriteLatestLidar(std::move(input));
}

void Lightning::OverwriteLatestLidar(InputMessage frame) {
    {
        std::lock_guard<std::mutex> lock(online_input_mutex_);
        if (!online_worker_running_) return;

        frame.lidar_sequence = ++online_lidar_received_;
        if (has_latest_lidar_) ++online_lidar_overwritten_;
        latest_lidar_ = std::move(frame);
        has_latest_lidar_ = true;
    }
    online_input_ready_.notify_one();
}

std::size_t Lightning::PendingOnlineInputCountLocked() const {
    return pending_mapping_imu_.size() +
           (has_latest_gps1_ ? 1U : 0U) +
           (has_latest_gps2_ ? 1U : 0U) +
           (has_latest_gps_velocity_ ? 1U : 0U) +
           (has_latest_wheel_odometry_ ? 1U : 0U) +
           (has_latest_lidar_ ? 1U : 0U);
}

void Lightning::ClearPendingOnlineInputLocked() {
    latest_lidar_ = InputMessage();
    has_latest_lidar_ = false;
    pending_mapping_imu_.clear();
    latest_gps1_ = InputMessage();
    has_latest_gps1_ = false;
    latest_gps2_ = InputMessage();
    has_latest_gps2_ = false;
    latest_gps_velocity_ = InputMessage();
    has_latest_gps_velocity_ = false;
    latest_wheel_odometry_ = InputMessage();
    has_latest_wheel_odometry_ = false;
}

void Lightning::StartOnlineWorkerLocked() {
    const bool mapping = mode_ == Mode::ONLINE_MAPPING;
    {
        std::lock_guard<std::mutex> lock(online_input_mutex_);
        ClearPendingOnlineInputLocked();
        online_lidar_received_ = 0;
        online_lidar_overwritten_ = 0;
        online_imu_received_ = 0;
        online_gps1_received_ = 0;
        online_gps2_received_ = 0;
        online_gps_velocity_received_ = 0;
        online_wheel_odometry_received_ = 0;
        online_worker_running_ = true;
        online_worker_is_localization_ = !mapping;
    }

    online_worker_ = std::thread([this, mapping]() {
        OnlineWorkerLoop(mapping);
    });
    if (topic_input_) topic_input_->SetEnabled(true, !mapping);
    LOG(INFO) << "[在线输入] 已启动，mode="
              << (mapping ? "mapping" : "localization")
              << ", LiDAR=latest-only"
              << (mapping
                      ? ", IMU=non-dropping queue"
                      : ", GPS1/GPS2/INS-velocity/wheel=independent latest-only slots");
}

void Lightning::StopOnlineWorkerLocked(bool drain) {
    if (topic_input_) topic_input_->SetEnabled(false);

    std::size_t pending = 0;
    bool was_running = false;
    {
        std::lock_guard<std::mutex> lock(online_input_mutex_);
        was_running = online_worker_running_;
        online_worker_running_ = false;
        online_worker_is_localization_ = false;
        pending = PendingOnlineInputCountLocked();
        if (!drain) ClearPendingOnlineInputLocked();
    }

    online_input_ready_.notify_all();
    if (online_worker_.joinable()) online_worker_.join();

    if (was_running) {
        LOG(INFO) << "[在线输入] 已停止"
                  << ", drain=" << drain
                  << ", pending_at_stop=" << pending;
    }
}

void Lightning::OnlineWorkerLoop(bool mapping) {
    if (mapping) {
        // Mapping is LiDAR-keyframe-driven. Only IMU is fed before each LiDAR
        // frame; gps and wheel observations belong to localization only.
        std::uint64_t lidar_processed = 0;
        std::uint64_t imu_processed = 0;
        while (true) {
            std::deque<InputMessage> imu_to_process;
            InputMessage lidar_to_process;
            {
                std::unique_lock<std::mutex> lock(online_input_mutex_);
                online_input_ready_.wait(lock, [this]() { return !online_worker_running_ || has_latest_lidar_; });
                if (!has_latest_lidar_) break;
                imu_to_process.swap(pending_mapping_imu_);
                lidar_to_process = std::move(latest_lidar_);
                latest_lidar_ = InputMessage();
                has_latest_lidar_ = false;
            }
            for (const InputMessage& input : imu_to_process) {
                if (input.type != InputType::IMU) continue;
                ProcessMappingInput(input);
                ++imu_processed;
            }
            ProcessMappingInput(lidar_to_process);
            ++lidar_processed;
        }
        LOG(INFO) << "[online input] mapping worker stopped"
                  << ", lidar_received=" << online_lidar_received_
                  << ", lidar_processed=" << lidar_processed
                  << ", lidar_overwritten=" << online_lidar_overwritten_
                  << ", imu_received=" << online_imu_received_
                  << ", imu_processed=" << imu_processed;
        return;
    }

    // Localization is measurement-driven. Each sensor owns one latest-only
    // slot, so slow NDT processing never blocks a DDS callback and stale
    // observations are overwritten by newer observations of the same type.
    std::uint64_t lidar_processed = 0;
    std::uint64_t auxiliary_processed = 0;
    while (true) {
        std::vector<InputMessage> ordered_inputs;
        ordered_inputs.reserve(5);
        {
            std::unique_lock<std::mutex> lock(online_input_mutex_);
            online_input_ready_.wait(lock, [this]() {
                return !online_worker_running_ ||
                       has_latest_lidar_ ||
                       has_latest_gps1_ ||
                       has_latest_gps2_ ||
                       has_latest_gps_velocity_ ||
                       has_latest_wheel_odometry_;
            });
            if (!online_worker_running_ &&
                !has_latest_lidar_ &&
                !has_latest_gps1_ &&
                !has_latest_gps2_ &&
                !has_latest_gps_velocity_ &&
                !has_latest_wheel_odometry_) {
                break;
            }

            if (has_latest_gps1_) {
                ordered_inputs.push_back(std::move(latest_gps1_));
                latest_gps1_ = InputMessage();
                has_latest_gps1_ = false;
            }
            if (has_latest_gps2_) {
                ordered_inputs.push_back(std::move(latest_gps2_));
                latest_gps2_ = InputMessage();
                has_latest_gps2_ = false;
            }
            if (has_latest_gps_velocity_) {
                ordered_inputs.push_back(std::move(latest_gps_velocity_));
                latest_gps_velocity_ = InputMessage();
                has_latest_gps_velocity_ = false;
            }
            if (has_latest_wheel_odometry_) {
                ordered_inputs.push_back(std::move(latest_wheel_odometry_));
                latest_wheel_odometry_ = InputMessage();
                has_latest_wheel_odometry_ = false;
            }
            if (has_latest_lidar_) {
                ordered_inputs.push_back(std::move(latest_lidar_));
                latest_lidar_ = InputMessage();
                has_latest_lidar_ = false;
            }
        }

        std::stable_sort(
            ordered_inputs.begin(), ordered_inputs.end(),
            [](const InputMessage& lhs, const InputMessage& rhs) {
                if (lhs.header_stamp != rhs.header_stamp) {
                    return lhs.header_stamp < rhs.header_stamp;
                }
                return LocalizationInputPriority(lhs.type) <
                       LocalizationInputPriority(rhs.type);
            });
        for (const InputMessage& input : ordered_inputs) {
            ProcessLocalizationInput(input);
            if (input.type == InputType::POINT_CLOUD2 || input.type == InputType::LIVOX) ++lidar_processed;
            else ++auxiliary_processed;
        }
    }
    LOG(INFO) << "[online input] localization worker stopped, lidar_processed=" << lidar_processed << ", auxiliary_processed=" << auxiliary_processed;
}

void Lightning::ProcessMappingInput(const InputMessage& input) {
    if (!mapping_system_) {
        return;
    }

    if (input.type == InputType::IMU) {
        mapping_system_->ProcessIMU(input.imu);
        return;
    }
    if (input.type == InputType::POINT_CLOUD2) mapping_system_->ProcessCloud(input.cloud);
    else if (input.type == InputType::LIVOX) mapping_system_->ProcessCloud(input.livox);
    else return;

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

    if (input.type == InputType::IMU) {
        localization_system_->ProcessImu(input.imu);
        return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
    }
    if (input.type == InputType::GPS1) {
        localization_system_->ProcessGps1(input.gps_fix);
        return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
    }
    if (input.type == InputType::GPS2) {
        localization_system_->ProcessGps2(input.gps_fix);
        return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
    }
    if (input.type == InputType::gps_VELOCITY) {
        localization_system_->ProcessgpsVelocity(input.gps_velocity);
        return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
    }
    if (input.type == InputType::WHEEL_ODOMETRY) {
        localization_system_->ProcessWheelOdometry(input.wheel_odometry);
        return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
    }
    if (input.type == InputType::POINT_CLOUD2) return localization_system_->ProcessCloud(input.cloud, diagnostic);
    if (input.type == InputType::LIVOX) return localization_system_->ProcessCloud(input.livox, diagnostic);
    return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
}


void Lightning::StopAllOnlineWorkersLocked(bool drain) {
    StopOnlineWorkerLocked(drain);
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

    StartOnlineWorkerLocked();
    task_.Reset(TaskState::RUNNING, "online mapping running");
    return {true, "online mapping started, save_path: " + save_path};
}

ServiceResult Lightning::LoadBag(const std::string& bag_path) {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (mode_ != Mode::OFFLINE_MAPPING && mode_ != Mode::OFFLINE_LOCALIZATION) {
        return {false,
                "load_bag is only used by offline_mapping and offline_localization"};
    }
    if (task_.State() == TaskState::RUNNING || task_.State() == TaskState::SAVING) {
        return {false, "offline task is already running"};
    }

    JoinOfflineThreadLocked();
    offline_bag_path_ = bag_path;

    if (mode_ == Mode::OFFLINE_LOCALIZATION) {
        if (!EnsureLocalizationSystemLocked()) {
            return {false, "failed to initialize localization"};
        }
        if (localization_system_->ReadyWithoutMap()) {
            if (!SetMapOriginInitialGuessLocked("offline localization")) {
                task_.Reset(TaskState::READY,
                            "offline bag loaded; failed to set zero initial pose");
                return {false, "failed to set automatic offline initial pose"};
            }
            task_.Reset(TaskState::RUNNING,
                        "offline localization running without map");
            StartBagLocalizationTaskLocked(bag_path);
            return {true,
                    "offline localization started without map: " + bag_path};
        }
        if (localization_map_path_.empty()) {
            task_.Reset(TaskState::READY,
                        "offline bag loaded, waiting for set_map_path");
            return {true, "offline localization bag loaded: " + bag_path};
        }
        if (!SetMapOriginInitialGuessLocked("offline localization")) {
            task_.Reset(TaskState::READY,
                        "offline bag and map loaded; failed to set zero initial pose");
            return {false, "failed to set automatic offline initial pose"};
        }
        task_.Reset(TaskState::RUNNING, "offline localization running");
        StartBagLocalizationTaskLocked(bag_path);
        return {true, "offline localization started: " + bag_path};
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
            nullptr,
            nullptr,
            nullptr,
            nullptr,
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
            const ServiceResult save_result = SaveMappingLocked(mapping_save_path_);
            task_.SetFinished(save_result.success, save_result.message);
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
        StopOnlineWorkerLocked(true);
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
    if (!IsLocalizationMode(mode_)) return {false, "set_map_path is only allowed in localization modes"};
    if (task_.State() == TaskState::RUNNING || task_.State() == TaskState::SAVING) return {false, "localization is already running"};
    StopOnlineWorkerLocked(false);
    JoinOfflineThreadLocked();
    ClearLocalizationSystemLocked();
    if (!EnsureLocalizationSystemLocked()) return {false, "failed to initialize LocalizationSystem"};
    if (!localization_system_->SetMapPath(map_path)) {
        ClearLocalizationSystemLocked();
        return {false, "failed to load localization map: " + map_path};
    }
    localization_map_path_ = map_path;

    // Loading a map must not start online localization or invent its initial
    // pose. Preserve the original service contract: set_location is the only
    // operation that starts the online localization worker.
    if (mode_ == Mode::ONLINE_LOCALIZATION) {
        task_.Reset(TaskState::READY,
                    "online localization map loaded, waiting for set_location");
        return {true, "localization map loaded: " + map_path};
    }

    if (mode_ == Mode::OFFLINE_LOCALIZATION && !offline_bag_path_.empty()) {
        if (!SetMapOriginInitialGuessLocked("offline localization")) {
            task_.Reset(TaskState::READY,
                        "offline bag and map loaded; failed to set zero initial pose");
            return {false, "failed to set automatic offline initial pose"};
        }
        task_.Reset(TaskState::RUNNING,
                    "offline localization running");
        StartBagLocalizationTaskLocked(offline_bag_path_);
        return {true, "localization map loaded; offline localization started"};
    }

    task_.Reset(TaskState::READY,
                "localization map loaded, waiting for offline bag");
    return {true, "localization map loaded: " + map_path};
}

ServiceResult Lightning::SetLocation(const SE3& init_pose, bool* initialized_now) {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!IsLocalizationMode(mode_)) return {false, "set_location is only allowed in localization modes"};
    if (!EnsureLocalizationSystemLocked()) return {false, "failed to initialize LocalizationSystem"};
    if (localization_system_->RequiresMap() && localization_map_path_.empty()) return {false, "set_map_path has not been called"};
    if (mode_ == Mode::OFFLINE_LOCALIZATION && offline_bag_path_.empty()) return {false, "load_bag has not been called"};
    if (mode_ == Mode::OFFLINE_LOCALIZATION && (task_.State() == TaskState::RUNNING || task_.State() == TaskState::SAVING)) return {false, "offline localization is already running"};
    if (mode_ == Mode::ONLINE_LOCALIZATION) {
        StopOnlineWorkerLocked(false);
    }
    bool initialized = false;
    if (!localization_system_->SetInitialGuess(init_pose, &initialized)) return {false, "failed to set initial pose"};
    if (initialized_now) *initialized_now = initialized;
    if (mode_ == Mode::OFFLINE_LOCALIZATION) {
        JoinOfflineThreadLocked();
        task_.Reset(TaskState::RUNNING, "offline localization running");
        StartBagLocalizationTaskLocked(offline_bag_path_);
        return {true, "offline localization started"};
    }
    StartOnlineWorkerLocked();
    task_.SetState(TaskState::RUNNING, "online localization running");
    return {true, "online localization started"};
}

void Lightning::StartBagLocalizationTaskLocked(const std::string& bag_path) {
    offline_thread_ = std::thread([this, bag_path]() {
        LOG(INFO) << "[offline localization] bag processing started";
        offline_localization_sequence_ = 0;
        BagInput bag_input;
        const bool bag_ok = bag_input.Run(
            bag_path, yaml_path_, BagInput::ImuCallback(),
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
            [this](const sensor_msgs::msg::NavSatFix::SharedPtr& fix) {
                InputMessage input;
                input.header_stamp =
                    rclcpp::Time(fix->header.stamp).seconds();
                input.type = InputType::GPS1;
                input.gps_fix = fix;
                ProcessLocalizationInput(input);
            },
            [this](const sensor_msgs::msg::NavSatFix::SharedPtr& fix) {
                InputMessage input;
                input.header_stamp =
                    rclcpp::Time(fix->header.stamp).seconds();
                input.type = InputType::GPS2;
                input.gps_fix = fix;
                ProcessLocalizationInput(input);
            },
            [this](const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& velocity) {
                InputMessage input;
                input.header_stamp =
                    rclcpp::Time(velocity->header.stamp).seconds();
                input.type = InputType::gps_VELOCITY;
                input.gps_velocity = velocity;
                ProcessLocalizationInput(input);
            },
            [this](const nav_msgs::msg::Odometry::SharedPtr& odometry) {
                InputMessage input;
                input.header_stamp =
                    rclcpp::Time(odometry->header.stamp).seconds();
                input.type = InputType::WHEEL_ODOMETRY;
                input.wheel_odometry = odometry;
                ProcessLocalizationInput(input);
            },
            [this](const BagInputProgress& progress) {
                task_.SetProgress(
                    progress.processed_frames, progress.total_frames,
                    "offline localization running");
            },
            [this]() { return task_.CancelRequested(); });
        bool task_ok = bag_ok;
        if (bag_ok && localization_system_) {
            const loc::LocalizationResult final_result =
                localization_system_->GetLatestResult();
            if (final_result.valid_) {
                LOG(INFO) << "[offline localization] final estimator result"
                          << ", stamp=" << final_result.timestamp_
                          << ", position="
                          << final_result.pose_.translation().transpose()
                          << ", message=" << final_result.message_;
            } else {
                task_ok = false;
                LOG(ERROR) << "[offline localization] bag contained no valid "
                              "localization result; check configured topics, "
                              "GNSS status and observation timestamps";
            }
        }
        if (task_.CancelRequested()) {
            task_.SetState(TaskState::CANCELLED,
                           "offline localization cancelled");
        } else {
            task_.SetFinished(
                task_ok,
                task_ok
                    ? "offline localization finished"
                    : "offline localization produced no valid estimator result");
        }
        LOG(INFO) << "[offline localization] bag processing finished";
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

    LOG(INFO) << "[结束定位] [01] 关闭Topic到定位latest槽位的输入";
    StopOnlineWorkerLocked(true);
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
