//
// Created by xiang on 25-9-12.
//

#include "core/system/loc_system.h"
#include "core/localization/localization.h"
#include "io/yaml_io.h"
#include "wrapper/ros_utils.h"
#include <algorithm>
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

bool LocSystem::Init(const std::string &yaml_path) {
    yaml_path_ = yaml_path;
    loc::Localization::Options opt;
    opt.online_mode_ = true;
    loc_ = std::make_shared<loc::Localization>(opt);
    map_loaded_ = false;
    loc_started_ = false;

    YAML_IO yaml(yaml_path);

    YAML::Node yaml_node = YAML::LoadFile(yaml_path);
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

    loc_pose_quality_pub_ =
        node_->create_publisher<lightning_interfaces::msg::LocalizationPose>(
            "/lightning/localization/pose_with_quality", rclcpp::QoS(10));

    load_map_srv_ = node_->create_service<LoadMapService>(
        "/lightning/load_map",
        [this](const LoadMapService::Request::SharedPtr request,
               LoadMapService::Response::SharedPtr response) {
            if (!loc_) {
                response->success = false;
                response->message = "localization module is not initialized";
                return;
            }
            if (request->map_path.empty()) {
                response->success = false;
                response->message = "map_path is empty";
                return;
            }

            std::lock_guard<std::mutex> lock(loc_operation_mutex_);

            map_loaded_ = false;
            loc_started_ = false;

            const bool ok = loc_->Init(yaml_path_, request->map_path);

            map_loaded_ = ok;
            loc_started_ = false;
            response->success = ok;
            response->message = ok
                                    ? "map loaded: " + request->map_path
                                    : "failed to load map: " + request->map_path;
        });

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

            std::lock_guard<std::mutex> lock(loc_operation_mutex_);

            if (!map_loaded_.load()) {
                response->success = false;
                response->initialized = false;
                response->message = "localization map is not loaded";
                return;
            }

            const bool initialized_now = loc_->SetExternalPose(q, t);
            loc_started_ = true;

            response->success = true;
            response->initialized = initialized_now;
            response->message = initialized_now
                                     ? "robot_localizer initialized successfully"
                                     : "initial pose accepted, waiting for current cloud";
        });

    get_localization_quality_srv_ =
        node_->create_service<GetLocalizationQualityService>(
            "/lightning/get_localization_quality",
            [this](const GetLocalizationQualityService::Request::SharedPtr request,
                   GetLocalizationQualityService::Response::SharedPtr response) {
                (void)request;
                if (!loc_ || !map_loaded_.load()) {
                    response->valid = false;
                    response->reliable = false;
                    response->status = static_cast<uint8_t>(loc::LocalizationStatus::UNKNOWN);
                    response->confidence = 0.0;
                    response->x = 0.0;
                    response->y = 0.0;
                    response->z = 0.0;
                    response->roll = 0.0;
                    response->pitch = 0.0;
                    response->yaw = 0.0;
                    response->tp = 0.0;
                    response->nvtl = 0.0;
                    response->iterations = 0;
                    response->message = "localization map is not loaded";
                    return;
                }

                const auto result = loc_->GetLatestResult();
                response->valid = result.valid_;
                response->reliable = result.reliable_;
                response->status = static_cast<uint8_t>(result.status_);
                response->confidence = result.confidence_;

                const Vec3d t = result.pose_.translation();
                const Mat3d R = result.pose_.unit_quaternion().toRotationMatrix();
                const Eigen::Vector3d rpy = R.eulerAngles(0, 1, 2);

                response->x = t.x();
                response->y = t.y();
                response->z = t.z();
                response->roll = rpy.x();
                response->pitch = rpy.y();
                response->yaw = rpy.z();

                response->tp = result.tp_;
                response->nvtl = result.nvtl_;
                response->iterations = static_cast<uint32_t>(std::max(0, result.iterations_));
                response->message = result.valid_ ? result.message_ : "no localization result yet";
                if (response->message.empty()) {
                    response->message = response->reliable
                                            ? "localization reliable"
                                            : "localization not reliable";
                }
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

        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header = tf_msg.header;
        odom_msg.child_frame_id = tf_msg.child_frame_id;
        odom_msg.pose.pose = pose_msg.pose;

        if (loc_odom_pub_) {
            loc_odom_pub_->publish(odom_msg);
        }
    });

    loc_->SetResultCallback([this](const loc::LocalizationResult& result) {
        if (!loc_pose_quality_pub_ || !result.valid_) {
            return;
        }

        auto tf_msg = result.ToGeoMsg();
        lightning_interfaces::msg::LocalizationPose msg;
        msg.header = tf_msg.header;
        msg.pose.position.x = tf_msg.transform.translation.x;
        msg.pose.position.y = tf_msg.transform.translation.y;
        msg.pose.position.z = tf_msg.transform.translation.z;
        msg.pose.orientation = tf_msg.transform.rotation;
        msg.valid = result.valid_;
        msg.reliable = result.reliable_;
        msg.status = static_cast<uint8_t>(result.status_);
        msg.confidence = result.confidence_;
        msg.tp = result.tp_;
        msg.nvtl = result.nvtl_;
        msg.iterations = static_cast<uint32_t>(std::max(0, result.iterations_));
        msg.message = result.message_;
        loc_pose_quality_pub_->publish(msg);
    });
    
    LOG(INFO) << "online loc node has been created. Waiting for /lightning/load_map.";

    return true;
}

void LocSystem::Start() {
    LOG(INFO) << "localization node started, waiting for map and initial pose";
}

void LocSystem::ProcessIMU(const IMUPtr &imu) {
    (void)imu;
}

void LocSystem::ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr &cloud) {
    if (!map_loaded_.load() || !loc_started_.load()) {
        return;
    }

    std::lock_guard<std::mutex> lock(loc_operation_mutex_);

    if (map_loaded_.load() && loc_started_.load()) {
        loc_->ProcessLidarMsg(cloud);
    }
}

void LocSystem::ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr &cloud) {
    if (!map_loaded_.load() || !loc_started_.load()) {
        return;
    }

    std::lock_guard<std::mutex> lock(loc_operation_mutex_);

    if (map_loaded_.load() && loc_started_.load()) {
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
