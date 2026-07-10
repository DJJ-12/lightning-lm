#include "runtime/lightning.h"

#include <chrono>

#include <glog/logging.h>
#include <pangolin/pangolin.h>
#include <yaml-cpp/yaml.h>

namespace lightning::runtime {

Lightning::~Lightning() {
    CancelTask();
    if (offline_thread_.joinable()) {
        offline_thread_.join();
    }
}

bool Lightning::Init(rclcpp::Node::SharedPtr node, const std::string& yaml_path) {
    node_ = node;
    yaml_path_ = yaml_path;
    task_.Reset(TaskState::IDLE, "lightning started");
    if (!node_ || yaml_path_.empty()) {
        return false;
    }
    YAML::Node yaml = YAML::LoadFile(yaml_path_);
    if (yaml["localization"] && yaml["localization"]["cloud_timeout_sec"]) {
        localization_cloud_timeout_sec_ = yaml["localization"]["cloud_timeout_sec"].as<double>();
    } else if (yaml["system"] && yaml["system"]["cloud_timeout_sec"]) {
        localization_cloud_timeout_sec_ = yaml["system"]["cloud_timeout_sec"].as<double>();
    }
    LOG(INFO) << "[LIGHTNING] localization cloud timeout sec = "
              << localization_cloud_timeout_sec_;
    return true;
}

bool Lightning::CanChangeModeLocked() const {
    const auto state = task_.State();
    return state != TaskState::RUNNING && state != TaskState::SAVING;
}

ServiceResult Lightning::SetMode(const std::string& mode_text) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!CanChangeModeLocked()) {
        return {false, "task is running, finish or cancel it before changing mode"};
    }

    const Mode new_mode = ModeFromString(mode_text);
    if (ModeToString(new_mode) != mode_text && mode_text != "offline" && mode_text != "online" && mode_text != "mapping" && mode_text != "loc") {
        return {false, "unknown mode: " + mode_text};
    }

    if (new_mode != mode_) {
        StopTopicInputLocked();
        ClearMappingLocked();
        ClearLocalizationLocked();
        mode_ = new_mode;
    }

    if (mode_ == Mode::LOCALIZATION) {
        task_.Reset(TaskState::IDLE, "localization mode selected, waiting for set_map_path");
    } else if (mode_ == Mode::OFFLINE_MAPPING) {
        task_.Reset(TaskState::IDLE, "offline_mapping mode selected");
    } else if (mode_ == Mode::ONLINE_MAPPING) {
        task_.Reset(TaskState::IDLE, "online_mapping mode selected");
    } else {
        task_.Reset(TaskState::IDLE, "idle mode selected");
    }
    return {true, "mode set to " + ModeToString(mode_)};
}

TaskSnapshot Lightning::GetStatus() const {
    return task_.Snapshot();
}

Mode Lightning::CurrentMode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mode_;
}

void Lightning::ClearMappingLocked() {
    if (mapping_system_) {
        mapping_system_->Reset();
    }
    mapping_system_.reset();
    mapping_save_path_.clear();
}

void Lightning::ClearLocalizationLocked() {
    if (localization_system_) {
        localization_system_->Reset();
    }
    localization_system_.reset();
    localization_map_path_.clear();
}

void Lightning::StopTopicInputLocked() {
    if (topic_input_) {
        topic_input_->Stop();
    }
    topic_input_.reset();
}

ServiceResult Lightning::StartBagMappingTask(const std::string& bag_path, const std::string& save_path) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (mode_ != Mode::OFFLINE_MAPPING) {
            return {false, "start_mapping is only allowed in offline_mapping mode for offline mapping"};
        }
        if (task_.State() == TaskState::RUNNING || task_.State() == TaskState::SAVING) {
            return {false, "offline mapping is already running"};
        }
        if (bag_path.empty() || save_path.empty()) {
            return {false, "bag_path and save_path must not be empty"};
        }
        if (offline_thread_.joinable()) {
            offline_thread_.join();
        }
        StopTopicInputLocked();
        ClearMappingLocked();
        offline_cancel_ = false;
        task_.Reset(TaskState::RUNNING, "offline mapping started");
    }

    offline_thread_ = std::thread([this, bag_path, save_path]() {
        auto mapping_system = std::make_unique<modules::MappingSystem>();
        modules::MappingSystemOptions mapping_options;
        mapping_options.online_input = false;
        if (!mapping_system->Init(yaml_path_, mapping_options) || !mapping_system->Start()) {
            task_.SetFinished(false, "failed to initialize MappingSystem");
            return;
        }

        BagInput bag_input;
        const bool bag_ok = bag_input.Run(
            bag_path, yaml_path_,
            [mapping_system_ptr = mapping_system.get()](const sensor_msgs::msg::Imu::SharedPtr& imu) {
                mapping_system_ptr->ProcessIMU(imu);
            },
            [mapping_system_ptr = mapping_system.get()](const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
                mapping_system_ptr->ProcessCloud(cloud);
            },
            [mapping_system_ptr = mapping_system.get()](const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud) {
                mapping_system_ptr->ProcessCloud(cloud);
            },
            [this](const BagInputProgress& progress) {
                task_.SetProgress(progress.processed_frames, progress.total_frames, "offline mapping running");
            },
            &offline_cancel_);

        if (!bag_ok || offline_cancel_.load()) {
            mapping_system->Reset();
            task_.SetState(TaskState::CANCELLED, "offline mapping cancelled");
            return;
        }

        task_.SetState(TaskState::SAVING, "saving offline map");
        const auto result = mapping_system->GetResult();
        const bool save_ok = save_map_.Save(save_path, result);
        // 保存完地图后，等待用户关闭UI窗口或取消任务
        if (save_ok) {
            LOG(INFO) << "Map saved successfully. Close the UI window to continue or cancel the task.";
            while (!pangolin::ShouldQuit() && !offline_cancel_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        // 等待完后再清理
        mapping_system->Reset();
        task_.SetFinished(save_ok, save_ok ? "offline mapping finished" : "failed to save offline map");
    });

    return {true, "offline mapping task accepted"};
}

TaskSnapshot Lightning::GetOfflineMappingProgress() const {
    return task_.Snapshot();
}

ServiceResult Lightning::StartMapping(const std::string& bag_path, const std::string& save_path) {
    Mode current_mode = Mode::IDLE;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_mode = mode_;
        if (current_mode == Mode::OFFLINE_MAPPING) {
            // Bag mapping has its own locking because it starts a worker thread.
        } else if (current_mode != Mode::ONLINE_MAPPING) {
            return {false, "start_mapping is only allowed in offline_mapping or online_mapping mode"};
        }
    }

    if (current_mode == Mode::OFFLINE_MAPPING) {
        return StartBagMappingTask(bag_path, save_path);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::ONLINE_MAPPING) {
        return {false, "mode changed before online mapping started"};
    }
    if (task_.State() == TaskState::RUNNING || task_.State() == TaskState::SAVING) {
        return {false, "online mapping is already running"};
    }
    if (save_path.empty()) {
        return {false, "save_path must not be empty for online mapping"};
    }
    if (!bag_path.empty()) {
        LOG(INFO) << "online mapping ignores bag_path: " << bag_path;
    }
    StopTopicInputLocked();
    ClearMappingLocked();
    mapping_save_path_ = save_path;

    mapping_system_ = std::make_unique<modules::MappingSystem>();
    modules::MappingSystemOptions mapping_options;
    mapping_options.online_input = true;
    if (!mapping_system_->Init(yaml_path_, mapping_options) || !mapping_system_->Start()) {
        ClearMappingLocked();
        task_.SetFinished(false, "failed to initialize MappingSystem");
        return {false, "failed to initialize MappingSystem"};
    }

    topic_input_ = std::make_unique<TopicInput>();
    const bool ok = topic_input_->Start(
        node_, yaml_path_,
        [this](const sensor_msgs::msg::Imu::SharedPtr& imu) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (mapping_system_) mapping_system_->ProcessIMU(imu);
        },
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (mapping_system_) mapping_system_->ProcessCloud(cloud);
        },
        [this](const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (mapping_system_) mapping_system_->ProcessCloud(cloud);
        },
        true);
    if (!ok) {
        StopTopicInputLocked();
        ClearMappingLocked();
        task_.SetFinished(false, "failed to start TopicInput");
        return {false, "failed to start TopicInput"};
    }
    task_.Reset(TaskState::RUNNING, "online mapping running");
    return {true, "online mapping started, save_path: " + save_path};
}

ServiceResult Lightning::SaveMappingLocked(const std::string& save_path) {
    if (!mapping_system_) {
        return {false, "MappingSystem is not running"};
    }
    task_.SetState(TaskState::SAVING, "saving map");
    const auto result = mapping_system_->GetResult();
    const bool ok = save_map_.Save(save_path, result);
    return {ok, ok ? "map saved" : "failed to save map"};
}

ServiceResult Lightning::FinishMapping() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::ONLINE_MAPPING) {
        return {false, "finish_mapping is only allowed in online_mapping mode"};
    }
    if (!mapping_system_) {
        return {false, "no online mapping task"};
    }
    StopTopicInputLocked();
    mapping_system_->Stop();

    ServiceResult result{false, "save_path is empty"};
    if (!mapping_save_path_.empty()) {
        result = SaveMappingLocked(mapping_save_path_);
    }
    ClearMappingLocked();
    task_.SetFinished(result.success, result.message);
    return result;
}

ServiceResult Lightning::SaveCurrentMap(const std::string& save_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::ONLINE_MAPPING || !mapping_system_) {
        return {false, "save_map is only allowed while online MappingSystem exists"};
    }
    if (save_path.empty()) {
        return {false, "save_path is empty"};
    }
    return SaveMappingLocked(save_path);
}

ServiceResult Lightning::SetMapPath(const std::string& map_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::LOCALIZATION) {
        return {false, "set_map_path is only allowed in localization mode"};
    }
    if (map_path.empty()) {
        return {false, "map_path is empty"};
    }

    StopTopicInputLocked();
    ClearLocalizationLocked();
    localization_system_ = std::make_unique<modules::LocalizationSystem>();
    if (!localization_system_->Init(yaml_path_, node_)) {
        ClearLocalizationLocked();
        task_.SetFinished(false, "failed to initialize LocalizationSystem");
        return {false, "failed to initialize LocalizationSystem"};
    }
    if (!localization_system_->SetMapPath(map_path)) {
        ClearLocalizationLocked();
        task_.SetFinished(false, "failed to set map path: " + map_path);
        return {false, "failed to set map path: " + map_path};
    }

    localization_map_path_ = map_path;
    task_.Reset(TaskState::READY, "map path set, waiting for set_location");
    return {true, "map path set: " + map_path};
}

ServiceResult Lightning::GetMapPath(std::string* map_path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (map_path) {
        *map_path = localization_map_path_;
    }
    if (localization_map_path_.empty()) {
        return {false, "map path is not set"};
    }
    return {true, "map path: " + localization_map_path_};
}

ServiceResult Lightning::SetLocation(const SE3& init_pose, bool* initialized_now) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::LOCALIZATION) {
        return {false, "set_location is only allowed in localization mode"};
    }
    if (!localization_system_) {
        return {false, "map path is not set"};
    }
    bool initialized = false;
    if (!localization_system_->SetInitialGuess(init_pose, &initialized)) {
        return {false, "failed to set initial pose"};
    }
    if (initialized_now) {
        *initialized_now = initialized;
    }

    if (!topic_input_) {
        topic_input_ = std::make_unique<TopicInput>();
        const bool ok = topic_input_->Start(
            node_, yaml_path_,
            nullptr,
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (localization_system_) {
                    localization_system_->ProcessCloud(cloud);
                }
            },
            [this](const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (localization_system_) {
                    localization_system_->ProcessCloud(cloud);
                }
            },
            false, localization_cloud_timeout_sec_,
            [this](const std::string& message) { HandleCloudTimeout(message); });
        if (!ok) {
            return {false, "failed to start TopicInput for localization"};
        }
    }
    task_.SetState(initialized ? TaskState::RUNNING : TaskState::WAIT_CLOUD,
                   initialized ? "localization initialized" : "initial pose accepted, waiting for current cloud");
    return {true, initialized ? "localization initialized" : "initial pose accepted, waiting for current cloud"};
}

loc::LocalizationResult Lightning::GetLocalizationQuality() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!localization_system_) {
        return loc::LocalizationResult();
    }
    return localization_system_->GetLatestResult();
}

void Lightning::HandleCloudTimeout(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::LOCALIZATION || !localization_system_) {
        return;
    }
    LOG(WARNING) << message;
    localization_system_->MarkPoor(message + "; localization quality: poor");
}

ServiceResult Lightning::CancelTask() {
    std::lock_guard<std::mutex> lock(mutex_);
    offline_cancel_ = true;
    StopTopicInputLocked();
    ClearMappingLocked();
    ClearLocalizationLocked();
    task_.SetState(TaskState::CANCELLED, "task cancelled");
    return {true, "task cancelled"};
}

}  // namespace lightning::runtime
