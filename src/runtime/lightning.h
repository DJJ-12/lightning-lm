#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "common/eigen_types.h"
#include "core/localization/localization_diagnostic.h"
#include "core/localization/localization_result.h"
#include "modules/localizationSystem/localization_system.h"
#include "modules/mappingSystem/mapping_system.h"
#include "modules/mappingSystem/save_map.h"
#include "runtime/bag_input.h"
#include "runtime/message_queue.h"
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
    void Shutdown();

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
    enum class OnlineRoute {
        NONE,
        MAPPING,
        LOCALIZATION
    };

    enum class InputType {
        IMU,
        POINT_CLOUD2,
        LIVOX
    };

    struct InputMessage {
        std::uint64_t sequence = 0;
        std::uint64_t lidar_sequence = 0;
        std::uint64_t topic_lidar_sequence = 0;
        double receive_steady_sec = 0.0;
        double header_stamp = 0.0;
        InputType type = InputType::POINT_CLOUD2;
        sensor_msgs::msg::Imu::SharedPtr imu;
        sensor_msgs::msg::PointCloud2::SharedPtr cloud;
        livox_ros_driver2::msg::CustomMsg::SharedPtr livox;
    };

    bool CanChangeModeLocked() const;

    void RouteImu(const sensor_msgs::msg::Imu::SharedPtr& imu);
    void RouteCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud,
                    const TopicInput::LidarReceiveInfo& receive_info);
    void RouteLivox(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud,
                    const TopicInput::LidarReceiveInfo& receive_info);
    void PushMappingMessage(InputMessage message);
    void PushLocalizationMessage(InputMessage message);
    static bool IsLidarMessage(const InputMessage& input);
    std::size_t LimitMappingQueuedLidar();

    void StartOnlineMappingWorkerLocked();
    void StopOnlineMappingWorkerLocked(bool drain);
    void OnlineMappingWorkerLoop();

    void StartOnlineLocalizationWorkerLocked();
    void StopOnlineLocalizationWorkerLocked(bool drain);
    void OnlineLocalizationWorkerLoop();

    void ProcessMappingInput(const InputMessage& input);
    loc::LocalizationFrameOutcome ProcessLocalizationInput(const InputMessage& input);
    void HandleLocalizationTimeout();

    void ClearMappingSystemLocked();
    void ClearLocalizationSystemLocked();
    void JoinOfflineThreadLocked();
    void StopAllOnlineWorkersLocked(bool drain);

    void StartBagMappingTaskLocked(const std::string& bag_path);
    void StartBagLocalizationTaskLocked(const std::string& bag_path);

    ServiceResult SaveMappingLocked(const std::string& save_path);
    void PublishMappingOutputsLocked(bool force);

    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mapping_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mapping_path_pub_;
    std::string yaml_path_;
    double localization_cloud_timeout_sec_ = 5.0;

    mutable std::mutex control_mutex_;
    std::atomic_bool shutdown_{false};
    Mode mode_ = Mode::IDLE;
    Task task_;

    std::unique_ptr<TopicInput> topic_input_;
    std::atomic<OnlineRoute> online_route_{OnlineRoute::NONE};
    std::atomic<std::uint64_t> online_sequence_{0};
    std::atomic<std::uint64_t> mapping_lidar_received_{0};
    std::atomic<std::uint64_t> mapping_lidar_enqueued_{0};
    std::atomic<std::uint64_t> mapping_lidar_dropped_{0};
    std::atomic<std::uint64_t> mapping_lidar_overflow_dropped_{0};
    std::atomic<std::uint64_t> localization_lidar_received_{0};
    std::atomic<std::uint64_t> localization_lidar_enqueued_{0};
    std::atomic<std::uint64_t> localization_lidar_dropped_{0};
    std::uint64_t mapping_task_generation_ = 0;
    std::uint64_t localization_task_generation_ = 0;
    std::atomic<std::uint64_t> offline_localization_sequence_{0};

    MessageQueue<InputMessage> mapping_queue_;
    MessageQueue<InputMessage> localization_queue_;
    std::thread mapping_worker_;
    std::thread localization_worker_;

    std::unique_ptr<modules::MappingSystem> mapping_system_;
    std::unique_ptr<modules::LocalizationSystem> localization_system_;
    modules::SaveMap save_map_;
    modules::SaveMapOptions save_map_options_;
    std::string mapping_save_path_;
    std::string localization_map_path_;
    std::string offline_bag_path_;

    std::thread offline_thread_;
};

}  // namespace lightning::runtime
