#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>

#include "common/eigen_types.h"
#include "core/localization/localization_result.h"
#include "modules/localization/localization.h"
#include "modules/mapping/mapping.h"
#include "modules/mapping/save_map.h"
#include "runtime/bag_input.h"
#include "runtime/mode.h"
#include "runtime/task.h"
#include "runtime/topic_input.h"

namespace lightning::runtime {

struct ServiceResult {
    bool success = false;
    std::string message;
};

class Lightning {
   public:
    Lightning() = default;
    ~Lightning();

    bool Init(rclcpp::Node::SharedPtr node, const std::string& yaml_path);

    ServiceResult SetMode(const std::string& mode);
    TaskSnapshot GetStatus() const;
    Mode CurrentMode() const;

    ServiceResult StartOfflineMapping(const std::string& bag_path, const std::string& save_path);
    TaskSnapshot GetOfflineMappingProgress() const;

    ServiceResult StartMapping(const std::string& map_id);
    ServiceResult FinishMapping(bool save_map, const std::string& save_path);
    ServiceResult SaveCurrentMap(const std::string& save_path);

    ServiceResult SetMapPath(const std::string& map_path);
    ServiceResult SetLocation(const SE3& init_pose, bool* initialized_now = nullptr);
    loc::LocalizationResult GetLocalizationQuality() const;

    ServiceResult CancelTask();

   private:
    bool CanChangeModeLocked() const;
    void ClearMappingLocked();
    void ClearLocalizationLocked();
    void StopTopicInputLocked();
    ServiceResult SaveMappingLocked(const std::string& save_path);

    rclcpp::Node::SharedPtr node_;
    std::string yaml_path_;

    mutable std::mutex mutex_;
    Mode mode_ = Mode::IDLE;
    Task task_;

    std::unique_ptr<modules::Mapping> mapping_;
    std::unique_ptr<modules::Localization> localization_;
    std::unique_ptr<TopicInput> topic_input_;
    modules::SaveMap save_map_;

    std::thread offline_thread_;
    std::atomic_bool offline_cancel_{false};
};

}  // namespace lightning::runtime
