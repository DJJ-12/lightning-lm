#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "common/eigen_types.h"
#include "core/localization/localization_result.h"
#include "modules/localizationSystem/localization_system.h"
#include "modules/mappingSystem/mapping_system.h"
#include "modules/mappingSystem/save_map.h"
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

    TaskSnapshot GetOfflineMappingProgress() const;

    ServiceResult StartMapping(const std::string& save_path);
    ServiceResult LoadBag(const std::string& bag_path);
    ServiceResult FinishMapping(bool save_map);

    ServiceResult SetMapPath(const std::string& map_path);
    ServiceResult FinishLocalization();
    ServiceResult GetMapPath(std::string* map_path) const;
    ServiceResult SetLocation(const SE3& init_pose, bool* initialized_now = nullptr);
    loc::LocalizationResult GetLocalizationQuality() const;

    ServiceResult CancelTask();

   private:
    bool CanChangeModeLocked() const;
    void ClearMappingLocked();
    void ClearLocalizationLocked();
    void StopTopicInputLocked();
    void JoinOfflineThread();
    void HandleCloudTimeout(const std::string& message);
    void StartBagMappingTask(const std::string& bag_path);
    void StartBagLocalizationTask(const std::string& bag_path);
    bool StartLocalizationTopicInputLocked();
    ServiceResult SaveMappingLocked(const std::string& save_path);
    void PublishMappingMapLocked(bool force);

    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mapping_map_pub_;
    std::string yaml_path_;
    double localization_cloud_timeout_sec_ = 5.0;

    mutable std::mutex mutex_;
    Mode mode_ = Mode::IDLE;
    Task task_;

    std::unique_ptr<modules::MappingSystem> mapping_system_;
    std::unique_ptr<modules::LocalizationSystem> localization_system_;
    std::unique_ptr<TopicInput> topic_input_;
    modules::SaveMap save_map_;
    modules::SaveMapOptions save_map_options_;
    std::string mapping_save_path_;
    std::string localization_map_path_;
    std::string offline_bag_path_;

    std::thread offline_thread_;
};

}  // namespace lightning::runtime
