#include "runtime/lightning.h"

#include <utility>

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

namespace lightning::runtime {

Lightning::~Lightning() {
    CancelTask();
    JoinOfflineThread();
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
    }
    if (yaml["mapping"]) {
        if (yaml["mapping"]["block_map_resolution"]) {
            save_map_options_.block_resolution = yaml["mapping"]["block_map_resolution"].as<int>();
        }
        if (yaml["mapping"]["block_map_voxel_size"]) {
            save_map_options_.block_voxel_size = yaml["mapping"]["block_map_voxel_size"].as<double>();
        }
    }
    LOG(INFO) << "[LIGHTNING] localization cloud timeout sec = "
              << localization_cloud_timeout_sec_;
    LOG(INFO) << "[LIGHTNING] block map resolution = " << save_map_options_.block_resolution
              << ", voxel size = " << save_map_options_.block_voxel_size;
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
    if (!IsKnownModeName(mode_text)) {
        return {false, "unknown mode: " + mode_text};
    }

    if (new_mode != mode_) {
        StopTopicInputLocked();
        ClearMappingLocked();
        ClearLocalizationLocked();
        mapping_save_path_.clear();
        localization_map_path_.clear();
        offline_bag_path_.clear();
        mode_ = new_mode;
    }

    task_.Reset(TaskState::IDLE, ModeToString(mode_) + " mode selected");
    return {true, "mode set to " + ModeToString(mode_)};
}

TaskSnapshot Lightning::GetStatus() const {
    return task_.Snapshot();
}

Mode Lightning::CurrentMode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mode_;
}

TaskSnapshot Lightning::GetOfflineMappingProgress() const {
    return task_.Snapshot();
}

void Lightning::ClearMappingLocked() {
    if (mapping_system_) {
        mapping_system_->Reset();
    }
    mapping_system_.reset();
}

void Lightning::ClearLocalizationLocked() {
    if (localization_system_) {
        localization_system_->Reset();
    }
    localization_system_.reset();
}

void Lightning::StopTopicInputLocked() {
    if (topic_input_) {
        topic_input_->Stop();
    }
    topic_input_.reset();
}

void Lightning::JoinOfflineThread() {
    std::thread offline_thread;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (offline_thread_.joinable()) {
            offline_thread = std::move(offline_thread_);
        }
    }
    if (offline_thread.joinable()) {
        offline_thread.join();
    }
}

ServiceResult Lightning::StartMapping(const std::string& save_path) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!IsMappingMode(mode_)) {
            return {false, "start_mapping is only allowed in mapping modes"};
        }
        if (task_.State() == TaskState::RUNNING || task_.State() == TaskState::SAVING) {
            return {false, "mapping is already running"};
        }
    }

    JoinOfflineThread();

    std::lock_guard<std::mutex> lock(mutex_);
    StopTopicInputLocked();
    ClearMappingLocked();
    mapping_save_path_ = save_path;
    offline_bag_path_.clear();

    if (mode_ == Mode::OFFLINE_MAPPING) {
        task_.Reset(TaskState::READY, "offline mapping ready, waiting for load_bag");
        return {true, "offline mapping ready, save_path: " + save_path};
    }

    mapping_system_ = std::make_unique<modules::MappingSystem>();
    modules::MappingSystemOptions mapping_options;
    mapping_options.online_input = true;
    if (!mapping_system_->Init(yaml_path_, mapping_options) || !mapping_system_->Start()) {
        ClearMappingLocked();
        task_.SetFinished(false, "failed to initialize MappingSystem");
        return {false, "failed to initialize MappingSystem"};
    }

    topic_input_ = std::make_unique<TopicInput>();
    const bool input_ok = topic_input_->Start(
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
    if (!input_ok) {
        StopTopicInputLocked();
        ClearMappingLocked();
        task_.SetFinished(false, "failed to start TopicInput");
        return {false, "failed to start TopicInput"};
    }
    task_.Reset(TaskState::RUNNING, "online mapping running");
    return {true, "online mapping started, save_path: " + save_path};
}

ServiceResult Lightning::LoadBag(const std::string& bag_path) {
    Mode mode = Mode::IDLE;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        mode = mode_;
        if (mode != Mode::OFFLINE_MAPPING && mode != Mode::OFFLINE_LOCALIZATION) {
            return {false, "load_bag is only allowed in offline modes"};
        }
        if (task_.State() == TaskState::RUNNING || task_.State() == TaskState::SAVING) {
            return {false, "offline task is already running"};
        }
    }

    JoinOfflineThread();

    if (mode == Mode::OFFLINE_LOCALIZATION) {
        std::lock_guard<std::mutex> lock(mutex_);
        offline_bag_path_ = bag_path;
        task_.Reset(TaskState::READY, "offline bag loaded, waiting for set_map_path and set_location");
        return {true, "offline localization bag loaded: " + bag_path};
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (mapping_save_path_.empty()) {
            return {false, "start_mapping has not been called"};
        }
        offline_bag_path_ = bag_path;
        task_.Reset(TaskState::RUNNING, "offline mapping running");
    }
    StartBagMappingTask(bag_path);
    return {true, "offline mapping bag loaded: " + bag_path};
}

void Lightning::StartBagMappingTask(const std::string& bag_path) {
    std::lock_guard<std::mutex> start_lock(mutex_);
    offline_thread_ = std::thread([this, bag_path]() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            mapping_system_ = std::make_unique<modules::MappingSystem>();
            modules::MappingSystemOptions mapping_options;
            mapping_options.online_input = false;
            if (!mapping_system_->Init(yaml_path_, mapping_options) || !mapping_system_->Start()) {
                ClearMappingLocked();
                task_.SetFinished(false, "failed to initialize MappingSystem");
                return;
            }
        }

        BagInput bag_input;
        const bool bag_ok = bag_input.Run(
            bag_path, yaml_path_,
            [this](const sensor_msgs::msg::Imu::SharedPtr& imu) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (mapping_system_) {
                    mapping_system_->ProcessIMU(imu);
                }
            },
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (mapping_system_) {
                    mapping_system_->ProcessCloud(cloud);
                }
            },
            [this](const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (mapping_system_) {
                    mapping_system_->ProcessCloud(cloud);
                }
            },
            [this](const BagInputProgress& progress) {
                task_.SetProgress(progress.processed_frames, progress.total_frames, "offline mapping running");
            },
            [this]() { return task_.CancelRequested(); });

        if (task_.CancelRequested()) {
            task_.SetState(TaskState::CANCELLED, "offline mapping cancelled");
            return;
        }
        if (!bag_ok) {
            task_.SetFinished(false, "offline bag mapping failed");
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (task_.CancelRequested() || !mapping_system_) {
            task_.SetState(TaskState::CANCELLED, "offline mapping cancelled");
            return;
        }

        mapping_system_->Stop();
        const ServiceResult result = SaveMappingLocked(mapping_save_path_);
        ClearMappingLocked();
        task_.SetFinished(result.success, result.success ? "offline mapping finished" : result.message);
    });
}

ServiceResult Lightning::SaveMappingLocked(const std::string& save_path) {
    if (!mapping_system_) {
        return {false, "MappingSystem is not running"};
    }
    task_.SetState(TaskState::SAVING, "saving map");
    const auto result = mapping_system_->GetResult();
    const bool ok = save_map_.Save(save_path, result, save_map_options_);
    return {ok, ok ? "map saved" : "failed to save map"};
}

ServiceResult Lightning::FinishMapping(bool save_map) {
    Mode mode = Mode::IDLE;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        mode = mode_;
        if (!IsMappingMode(mode)) {
            return {false, "finish_mapping is only allowed in mapping modes"};
        }
        if (!mapping_system_) {
            return {false, "no mapping task"};
        }
        if (mode == Mode::OFFLINE_MAPPING) {
            task_.RequestCancel();
        } else {
            StopTopicInputLocked();
        }
    }

    if (mode == Mode::OFFLINE_MAPPING) {
        JoinOfflineThread();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!mapping_system_) {
        return {false, "no mapping task"};
    }

    mapping_system_->Stop();
    ServiceResult result{true, "mapping finished without saving"};
    if (save_map) {
        result = SaveMappingLocked(mapping_save_path_);
    }
    ClearMappingLocked();
    task_.SetFinished(result.success, result.message);
    return result;
}

ServiceResult Lightning::SetMapPath(const std::string& map_path) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!IsLocalizationMode(mode_)) {
            return {false, "set_map_path is only allowed in localization modes"};
        }
        if (task_.State() == TaskState::RUNNING || task_.State() == TaskState::SAVING) {
            return {false, "localization is already running"};
        }
    }

    JoinOfflineThread();

    std::lock_guard<std::mutex> lock(mutex_);
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
        task_.SetFinished(false, "failed to load localization map: " + map_path);
        return {false, "failed to load localization map: " + map_path};
    }

    localization_map_path_ = map_path;
    task_.Reset(TaskState::READY, "localization map loaded, waiting for set_location");
    return {true, "localization map loaded: " + map_path};
}

bool Lightning::StartLocalizationTopicInputLocked() {
    if (topic_input_) {
        return true;
    }

    topic_input_ = std::make_unique<TopicInput>();
    const bool input_ok = topic_input_->Start(
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
    if (!input_ok) {
        StopTopicInputLocked();
    }
    return input_ok;
}

ServiceResult Lightning::SetLocation(const SE3& init_pose, bool* initialized_now) {
    bool initialized = false;
    bool start_bag = false;
    std::string bag_path;
    {
        std::lock_guard<std::mutex> lock(mutex_);
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
        if (!localization_system_->SetInitialGuess(init_pose, &initialized)) {
            return {false, "failed to set initial pose"};
        }
        if (initialized_now) {
            *initialized_now = initialized;
        }

        if (mode_ == Mode::OFFLINE_LOCALIZATION) {
            bag_path = offline_bag_path_;
            task_.Reset(TaskState::RUNNING, "offline localization running");
            start_bag = true;
        } else {
            if (!StartLocalizationTopicInputLocked()) {
                return {false, "failed to start TopicInput for localization"};
            }
            task_.SetState(initialized ? TaskState::RUNNING : TaskState::WAIT_CLOUD,
                           initialized ? "localization initialized" : "initial pose accepted, waiting for current cloud");
        }
    }

    if (start_bag) {
        StartBagLocalizationTask(bag_path);
        return {true, "offline localization started"};
    }
    return {true, initialized ? "localization initialized" : "initial pose accepted, waiting for current cloud"};
}

void Lightning::StartBagLocalizationTask(const std::string& bag_path) {
    std::lock_guard<std::mutex> start_lock(mutex_);
    offline_thread_ = std::thread([this, bag_path]() {
        BagInput bag_input;
        const bool bag_ok = bag_input.Run(
            bag_path, yaml_path_,
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
            [this](const BagInputProgress& progress) {
                task_.SetProgress(progress.processed_frames, progress.total_frames, "offline localization running");
            },
            [this]() { return task_.CancelRequested(); });

        if (task_.CancelRequested()) {
            task_.SetState(TaskState::CANCELLED, "offline localization cancelled");
            return;
        }
        task_.SetFinished(bag_ok, bag_ok ? "offline localization finished" : "offline bag localization failed");
    });
}

ServiceResult Lightning::FinishLocalization() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::ONLINE_LOCALIZATION) {
        return {false, "finish_localization is only allowed in online_localization mode"};
    }
    if (!localization_system_) {
        return {false, "no online localization task"};
    }

    StopTopicInputLocked();
    ClearLocalizationLocked();
    task_.SetFinished(true, "online localization finished");
    return {true, "online localization finished"};
}

ServiceResult Lightning::GetMapPath(std::string* map_path) const {
    std::lock_guard<std::mutex> lock(mutex_);
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
    std::lock_guard<std::mutex> lock(mutex_);
    if (!localization_system_) {
        return loc::LocalizationResult();
    }
    return localization_system_->GetLatestResult();
}

void Lightning::HandleCloudTimeout(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::ONLINE_LOCALIZATION || !localization_system_) {
        return;
    }
    LOG(WARNING) << message;
    localization_system_->MarkPoor(message + "; localization quality: poor");
}

ServiceResult Lightning::CancelTask() {
    task_.RequestCancel();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        StopTopicInputLocked();
        ClearMappingLocked();
        ClearLocalizationLocked();
    }
    JoinOfflineThread();
    return {true, "task cancelled"};
}

}  // namespace lightning::runtime
