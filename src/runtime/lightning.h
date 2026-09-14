#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "livox_ros_driver2/msg/custom_msg.hpp"

#include "common/eigen_types.h"
#include "core/localization/localization_diagnostic.h"
#include "core/localization/localization_result.h"
#include "modules/localizationSystem/localization_system.h"
#include "modules/mappingSystem/mapping_system.h"
#include "modules/mappingSystem/save_map.h"
#include "runtime/bag_input.h"
#include "runtime/mode.h"
#include "runtime/task.h"
#include "runtime/topic_input.h"

namespace lightning::runtime {

enum class InputType {
    IMU,
    POINT_CLOUD2,
    LIVOX,
    GPS_POSITION,
    GPS_ORIENTATION,
    gps_VELOCITY,
    WHEEL_ODOMETRY
};

struct InputMessage {
    std::uint64_t lidar_sequence = 0;
    std::uint64_t topic_lidar_sequence = 0;
    double receive_steady_sec = 0.0;
    double header_stamp = 0.0;
    InputType type = InputType::POINT_CLOUD2;
    sensor_msgs::msg::Imu::SharedPtr imu;
    sensor_msgs::msg::PointCloud2::SharedPtr cloud;
    livox_ros_driver2::msg::CustomMsg::SharedPtr livox;
    sensor_msgs::msg::NavSatFix::SharedPtr gps_fix;
    geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr gps_orientation;
    geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr gps_velocity;
    nav_msgs::msg::Odometry::SharedPtr wheel_odometry;
};

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
    bool CanChangeModeLocked() const;
    bool EnsureLocalizationSystemLocked();

    // Topic callbacks only place data into these lightweight online buffers.
    void AcceptImu(const sensor_msgs::msg::Imu::SharedPtr& imu);
    void AcceptGps(const sensor_msgs::msg::NavSatFix::SharedPtr& fix);
    void AcceptGpsOrientation(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& orientation);
    void AcceptgpsVelocity(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& velocity);
    void AcceptWheelOdometry(const nav_msgs::msg::Odometry::SharedPtr& odometry);
    void AcceptCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud,
                     const TopicInput::LidarReceiveInfo& receive_info);
    void AcceptLivox(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud,
                     const TopicInput::LidarReceiveInfo& receive_info);
    void OverwriteLatestLidar(InputMessage frame);

    void StartOnlineWorkerLocked();
    void StopOnlineWorkerLocked(bool drain);
    void OnlineWorkerLoop(bool mapping);
    std::size_t PendingOnlineInputCountLocked() const;
    void ClearPendingOnlineInputLocked();

    void ProcessMappingInput(const InputMessage& input);
    loc::LocalizationFrameOutcome ProcessLocalizationInput(const InputMessage& input);

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

    mutable std::mutex control_mutex_;
    std::atomic_bool shutdown_{false};
    Mode mode_ = Mode::IDLE;
    Task task_;

    std::unique_ptr<TopicInput> topic_input_;
    std::uint64_t mapping_task_generation_ = 0;
    std::uint64_t localization_task_generation_ = 0;
    std::atomic<std::uint64_t> offline_localization_sequence_{0};

    // The online LiDAR slot contains at most one unprocessed frame. New LiDAR
    // messages overwrite the previous pending frame while the algorithm works.
    std::mutex online_input_mutex_;
    std::condition_variable online_input_ready_;
    InputMessage latest_lidar_;
    bool has_latest_lidar_ = false;

    // Mapping retains every IMU sample. Localization-only observations use
    // one overwriteable slot per sensor so the worker consumes fresh data.
    std::deque<InputMessage> pending_mapping_imu_;
    InputMessage latest_gps_;
    bool has_latest_gps_ = false;
    InputMessage latest_gps_orientation_;
    bool has_latest_gps_orientation_ = false;
    InputMessage latest_gps_velocity_;
    bool has_latest_gps_velocity_ = false;
    InputMessage latest_wheel_odometry_;
    bool has_latest_wheel_odometry_ = false;

    bool online_worker_running_ = false;
    bool online_worker_is_localization_ = false;
    std::thread online_worker_;

    std::uint64_t online_lidar_received_ = 0;
    std::uint64_t online_lidar_overwritten_ = 0;
    std::uint64_t online_imu_received_ = 0;
    std::uint64_t online_gps_received_ = 0;
    std::uint64_t online_gps_orientation_received_ = 0;
    std::uint64_t online_gps_velocity_received_ = 0;
    std::uint64_t online_wheel_odometry_received_ = 0;

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
