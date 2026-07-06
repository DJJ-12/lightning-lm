//
// Created by xiang on 25-9-12.
//

#include "core/system/loc_system.h"
#include "core/localization/localization.h"
#include "io/yaml_io.h"
#include "wrapper/ros_utils.h"
#include <iomanip>
#include <csignal>
#include <yaml-cpp/yaml.h>
#include "common/options.h"
#include "utils/timer.h"
namespace lightning {

LocSystem::LocSystem(LocSystem::Options options) : options_(options) {
    /// handle ctrl-c
    signal(SIGINT, lightning::debug::SigHandle);
}

LocSystem::~LocSystem() {
    if (loc_) {
        loc_->Finish();
    }
}

bool LocSystem::Init(const std::string &yaml_path, const std::string& map_path_override) {
    loc::Localization::Options opt;
    opt.online_mode_ = true;
    loc_ = std::make_shared<loc::Localization>(opt);

    YAML_IO yaml(yaml_path);

    YAML::Node yaml_node = YAML::LoadFile(yaml_path);
    std::string map_path;
    if (!map_path_override.empty()) {
        map_path = map_path_override;
    } else if (yaml_node["localization"] && yaml_node["localization"]["map_path"]) {
        map_path = yaml_node["localization"]["map_path"].as<std::string>();
    } else {
        map_path = yaml.GetValue<std::string>("system", "map_path");
    }

    bool pub_tf = true;
    if (yaml_node["system"] && yaml_node["system"]["pub_tf"]) {
        pub_tf = yaml_node["system"]["pub_tf"].as<bool>();
    } else if (yaml_node["pub_tf"]) {
        pub_tf = yaml_node["pub_tf"].as<bool>();
    }
    options_.pub_tf_ = pub_tf;
    LOG(INFO) << "[LOC_SYSTEM] pub_tf = " << options_.pub_tf_;

    LOG(INFO) << "online mode, creating ros2 node ... ";

    /// subscribers
    node_ = std::make_shared<rclcpp::Node>("lightning_slam");

    cloud_topic_ = yaml.GetValue<std::string>("common", "lidar_topic");
    livox_topic_ = yaml.GetValue<std::string>("common", "livox_lidar_topic");

    auto lidar_qos = rclcpp::QoS(rclcpp::KeepLast(10));
    lidar_qos.best_effort();
    lidar_qos.durability_volatile();
    // 在线定位模式下，稳定发布位姿
    lidar_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions lidar_sub_options;
    lidar_sub_options.callback_group = lidar_cb_group_;

    cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic_, lidar_qos,
        [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
            Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", false);
        },
        lidar_sub_options);

    livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
        livox_topic_, lidar_qos,
        [this](livox_ros_driver2::msg::CustomMsg::SharedPtr cloud) {
            Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", false);
        },
        lidar_sub_options);
        
    // 发布定位结果
    loc_odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>(
        "/lightning/localization/odom", 10);

    loc_pose_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/lightning/localization/pose", 10);    

    location_pose_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/lightning/localization_pose", rclcpp::QoS(10));

    set_location_srv_ = node_->create_service<SetLocationService>(
        "/lightning/set_location",
        [this](const SetLocationService::Request::SharedPtr request,
               SetLocationService::Response::SharedPtr response) {
            if (!loc_ || !map_loaded_.load()) {
                response->success = false;
                response->initialized = false;
                response->message = "localization map is not loaded";
                return;
            }

            Eigen::AngleAxisd roll_angle(request->roll, Eigen::Vector3d::UnitX());
            Eigen::AngleAxisd pitch_angle(request->pitch, Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd yaw_angle(request->yaw, Eigen::Vector3d::UnitZ());
            Eigen::Quaterniond q(yaw_angle * pitch_angle * roll_angle);
            q.normalize();
            Eigen::Vector3d t(request->x, request->y, request->z);

            const bool initialized_now = loc_->SetExternalPose(q, t);
            loc_started_ = true;

            response->success = true;
            response->initialized = initialized_now;
            response->message = initialized_now
                                     ? "robot_localizer initialized successfully"
                                     : "initial pose accepted, waiting for current cloud";
        });

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);

    loc_->SetTFCallback([this](const geometry_msgs::msg::TransformStamped& tf_msg) {
        if (options_.pub_tf_ && tf_broadcaster_) {
            tf_broadcaster_->sendTransform(tf_msg);
        }

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = tf_msg.header;
        pose_msg.pose.position.x = tf_msg.transform.translation.x;
        pose_msg.pose.position.y = tf_msg.transform.translation.y;
        pose_msg.pose.position.z = tf_msg.transform.translation.z;
        pose_msg.pose.orientation = tf_msg.transform.rotation;

        if (loc_pose_pub_) {
            loc_pose_pub_->publish(pose_msg);
        }

        if (location_pose_pub_) {
            location_pose_pub_->publish(pose_msg);
        }

        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header = tf_msg.header;
        odom_msg.child_frame_id = tf_msg.child_frame_id;
        odom_msg.pose.pose = pose_msg.pose;

        if (loc_odom_pub_) {
            loc_odom_pub_->publish(odom_msg);
        }
    });
    
    bool ret = loc_->Init(yaml_path, map_path);
    map_loaded_ = ret;
    if (ret) {
        LOG(INFO) << "online loc node has been created.";
    }

    return ret;
}

void LocSystem::SetInitPose(const SE3 &pose) {
    LOG(INFO) << "set init pose: " << pose.translation().transpose() << ", "
              << pose.unit_quaternion().coeffs().transpose();

    loc_->SetExternalPose(pose.unit_quaternion(), pose.translation());
    loc_started_ = true;
}

void LocSystem::Start() {
    loc_started_ = true;
    LOG(INFO) << "localization started";
}

void LocSystem::ProcessIMU(const IMUPtr &imu) {
    (void)imu;
}

void LocSystem::ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr &cloud) {
    if (loc_started_) {
        loc_->ProcessLidarMsg(cloud);
    }
}

void LocSystem::ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr &cloud) {
    if (loc_started_) {
        loc_->ProcessLivoxLidarMsg(cloud);
    }
}
/*
void LocSystem::Spin() {
    if (node_ != nullptr) {
        spin(node_);
    }
}
*/
void LocSystem::Spin() {
    if (node_ == nullptr) {
        return;
    }

    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), 4);

    executor.add_node(node_);
    executor.spin();
}
}  // namespace lightning
