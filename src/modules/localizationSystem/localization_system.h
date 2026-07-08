#pragma once

#include <memory>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "common/eigen_types.h"
#include "core/localization/localization_result.h"
#include "lightning_interfaces/msg/localization_pose.hpp"


namespace lightning::loc { class Localization; }

namespace lightning::modules {

struct LocalizationSystemOptions {
    bool pub_tf_ = true;
};

class LocalizationSystem {
   public:
    explicit LocalizationSystem(LocalizationSystemOptions options = LocalizationSystemOptions());
    ~LocalizationSystem();

    bool Init(const std::string& yaml_path, rclcpp::Node::SharedPtr node = nullptr);
    bool SetMapPath(const std::string& map_path);
    bool SetInitialGuess(const SE3& init_pose, bool* initialized_now = nullptr);
    void ProcessCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud);
    void ProcessCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud);
    loc::LocalizationResult GetLatestResult() const;
    void Reset();

   private:
    void SetupPublishers(rclcpp::Node::SharedPtr node);
    static void PoseToQuaternionAndTranslation(const SE3& pose, Eigen::Quaterniond& q, Eigen::Vector3d& t);

    LocalizationSystemOptions options_;
    std::string yaml_path_;
    std::string map_path_;
    bool map_ready_ = false;
    bool has_initial_guess_ = false;

    std::shared_ptr<loc::Localization> loc_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr loc_odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr loc_pose_pub_;
    rclcpp::Publisher<lightning_interfaces::msg::LocalizationPose>::SharedPtr loc_pose_quality_pub_;
};

}  // namespace lightning::modules
