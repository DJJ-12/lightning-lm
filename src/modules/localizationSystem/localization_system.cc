#include "modules/localizationSystem/localization_system.h"

#include <algorithm>
#include <glog/logging.h>
#include <rclcpp/node.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>

#include "core/localization/localization.h"

namespace lightning::modules {

LocalizationSystem::LocalizationSystem(LocalizationSystemOptions options) : options_(options) {}

LocalizationSystem::~LocalizationSystem() {
    LOG(INFO) << "[定位析构诊断][LocalizationSystem][01] 进入析构 this=" << this;
    Reset();
    LOG(INFO) << "[定位析构诊断][LocalizationSystem][02] 析构完成 this=" << this;
}

bool LocalizationSystem::Init(const std::string& yaml_path, rclcpp::Node::SharedPtr node) {
    Reset();
    yaml_path_ = yaml_path;

    YAML::Node yaml_node = YAML::LoadFile(yaml_path);
    if (yaml_node["system"] && yaml_node["system"]["pub_tf"]) {
        options_.pub_tf_ = yaml_node["system"]["pub_tf"].as<bool>();
    } else if (yaml_node["pub_tf"]) {
        options_.pub_tf_ = yaml_node["pub_tf"].as<bool>();
    }
    if (yaml_node["common"] && yaml_node["common"]["base_link_frame"]) {
        base_link_frame_ = yaml_node["common"]["base_link_frame"].as<std::string>();
    }
    LOG(INFO) << "[LOCALIZATION_SYSTEM] pub_tf = " << options_.pub_tf_;

    loc::Localization::Options loc_options;
    loc_options.pub_tf_ = options_.pub_tf_;
    loc_ = std::make_shared<loc::Localization>(loc_options);
    if (node) {
        SetupPublishers(node);
    }
    return true;
}

void LocalizationSystem::SetupPublishers(rclcpp::Node::SharedPtr node) {
    if (!node) {
        return;
    }
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node);
    loc_odom_pub_ = node->create_publisher<nav_msgs::msg::Odometry>("/lightning/localization/odom", 10);
    loc_pose_pub_ = node->create_publisher<geometry_msgs::msg::PoseStamped>("/lightning/localization/pose", 10);
    loc_pose_quality_pub_ = node->create_publisher<lightning_interfaces::msg::LocalizationPose>(
        "/lightning/localization/pose_with_quality", rclcpp::QoS(10));

    loc_->SetTFCallback([this](const geometry_msgs::msg::TransformStamped& tf_msg) {
        if (options_.pub_tf_ && tf_broadcaster_) {
            tf_broadcaster_->sendTransform(tf_msg);
        }
    });

    loc_->SetResultCallback([this](const loc::LocalizationResult& result) {
        if (!result.valid_) {
            return;
        }

        auto tf_msg = result.ToGeoMsg();
        tf_msg.child_frame_id = base_link_frame_;

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

        if (!loc_pose_quality_pub_) {
            return;
        }

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
}

bool LocalizationSystem::SetMapPath(const std::string& map_path) {
    if (!loc_) {
        LOG(ERROR) << "Localization is not initialized";
        return false;
    }
    if (map_path.empty()) {
        LOG(ERROR) << "map path is empty";
        return false;
    }
    map_path_ = map_path;
    map_ready_ = loc_->Init(yaml_path_, map_path_);
    has_initial_guess_ = false;
    return map_ready_;
}

void LocalizationSystem::PoseToQuaternionAndTranslation(const SE3& pose, Eigen::Quaterniond& q, Eigen::Vector3d& t) {
    q = pose.unit_quaternion();
    q.normalize();
    t = pose.translation();
}

bool LocalizationSystem::SetInitialGuess(const SE3& init_pose, bool* initialized_now) {
    if (initialized_now) {
        *initialized_now = false;
    }
    if (!loc_ || !map_ready_) {
        LOG(ERROR) << "cannot set initial guess before map path is set";
        return false;
    }
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
    PoseToQuaternionAndTranslation(init_pose, q, t);
    const bool ok = loc_->SetExternalPose(q, t);
    has_initial_guess_ = true;
    if (initialized_now) {
        *initialized_now = ok;
    }
    return true;
}

void LocalizationSystem::ProcessCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
    if (!loc_ || !map_ready_) {
        return;
    }
    loc_->ProcessLidarMsg(cloud);
}

void LocalizationSystem::ProcessCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud) {
    if (!loc_ || !map_ready_) {
        return;
    }
    loc_->ProcessLivoxLidarMsg(cloud);
}

void LocalizationSystem::MarkPoor(const std::string& message) {
    if (loc_) {
        loc_->MarkPoor(message);
    }
}

loc::LocalizationResult LocalizationSystem::GetLatestResult() const {
    if (!loc_) {
        return loc::LocalizationResult();
    }
    return loc_->GetLatestResult();
}

void LocalizationSystem::Reset() {
    LOG(INFO) << "[定位析构诊断][LocalizationSystem::Reset][01] 开始"
              << ", this=" << this << ", loc=" << loc_.get();
    if (loc_) {
        LOG(INFO) << "[定位析构诊断][LocalizationSystem::Reset][02] 调用 Localization::Finish";
        loc_->Finish();
        LOG(INFO) << "[定位析构诊断][LocalizationSystem::Reset][03] Localization::Finish 已返回";
    }
    LOG(INFO) << "[定位析构诊断][LocalizationSystem::Reset][04] 准备 reset Localization";
    loc_.reset();
    LOG(INFO) << "[定位析构诊断][LocalizationSystem::Reset][05] Localization 已 reset";
    tf_broadcaster_.reset();
    loc_odom_pub_.reset();
    loc_pose_pub_.reset();
    loc_pose_quality_pub_.reset();
    map_ready_ = false;
    has_initial_guess_ = false;
    map_path_.clear();
    LOG(INFO) << "[定位析构诊断][LocalizationSystem::Reset][06] 完成";
}

}  // namespace lightning::modules
