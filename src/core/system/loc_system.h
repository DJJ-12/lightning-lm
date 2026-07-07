#ifndef LIGHTNING_LOC_SYSTEM_H
#define LIGHTNING_LOC_SYSTEM_H

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "common/eigen_types.h"
#include "common/imu.h"
#include "lightning_interfaces/msg/localization_pose.hpp"
#include "lightning_interfaces/srv/get_localization_quality.hpp"
#include "lightning_interfaces/srv/load_map.hpp"
#include "lightning_interfaces/srv/set_location.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"

namespace lightning {

namespace loc {
class Localization;
}

class LocSystem {
   public:
    struct Options {
        bool pub_tf_ = true;
    };

    explicit LocSystem(Options options);
    ~LocSystem();

    bool Init(const std::string& yaml_path);
    void Start();

    void ProcessIMU(const lightning::IMUPtr& imu);
    void ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud);
    void ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud);

    void Spin();

   private:
    using SetLocationService = lightning_interfaces::srv::SetLocation;
    using LoadMapService = lightning_interfaces::srv::LoadMap;
    using GetLocalizationQualityService = lightning_interfaces::srv::GetLocalizationQuality;

    Options options_;

    std::shared_ptr<loc::Localization> loc_ = nullptr;
    std::mutex loc_operation_mutex_;

    std::atomic_bool loc_started_ = false;
    std::atomic_bool map_loaded_ = false;

    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_ = nullptr;
    rclcpp::CallbackGroup::SharedPtr lidar_cb_group_ = nullptr;

    std::string cloud_topic_;
    std::string livox_topic_;
    std::string yaml_path_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_ = nullptr;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_sub_ = nullptr;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr loc_odom_pub_ = nullptr;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr loc_pose_pub_ = nullptr;
    rclcpp::Publisher<lightning_interfaces::msg::LocalizationPose>::SharedPtr loc_pose_quality_pub_ = nullptr;
    rclcpp::Service<LoadMapService>::SharedPtr load_map_srv_ = nullptr;
    rclcpp::Service<SetLocationService>::SharedPtr set_location_srv_ = nullptr;
    rclcpp::Service<GetLocalizationQualityService>::SharedPtr get_localization_quality_srv_ = nullptr;
};

}  // namespace lightning

#endif  // LIGHTNING_LOC_SYSTEM_H
