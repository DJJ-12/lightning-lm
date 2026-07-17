#include "modules/mappingSystem/mapping_system.h"

#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include "core/lio/laser_mapping.h"
#include "core/lio/lio_sam/lio_sam_mapping.h"
#include "core/lio/pointcloud_preprocess.h"
#include "core/lightning_math.hpp"
#include "ui/pangolin_window.h"
#include "wrapper/ros_utils.h"

namespace lightning::modules {

MappingSystem::~MappingSystem() {
    Reset();
}

bool MappingSystem::Init(const std::string& yaml_path, const MappingSystemOptions& options) {
    Reset();
    yaml_path_ = yaml_path;
    options_ = options;

    YAML::Node yaml = YAML::LoadFile(yaml_path);
    std::string frontend = "laser_mapping";
    if (yaml["system"] && yaml["system"]["frontend"]) {
        frontend = yaml["system"]["frontend"].as<std::string>();
    }
    use_lio_sam_ = frontend == "lio_sam" || frontend == "liosam";

    if (yaml["system"]) {
        if (yaml["system"]["with_ui"]) options_.with_ui = yaml["system"]["with_ui"].as<bool>();
    }
    if (yaml["common"] && yaml["common"]["base_link_frame"]) {
        base_link_frame_ = yaml["common"]["base_link_frame"].as<std::string>();
    }

    std::vector<double> base_lidar_t{0.0, 0.0, 0.0};
    std::vector<double> base_lidar_R{1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0};
    if (yaml["common"] && yaml["common"]["extrinsicBaseLidarTrans"]) {
        base_lidar_t = yaml["common"]["extrinsicBaseLidarTrans"].as<std::vector<double>>();
    }
    if (yaml["common"] && yaml["common"]["extrinsicBaseLidarRot"]) {
        base_lidar_R = yaml["common"]["extrinsicBaseLidarRot"].as<std::vector<double>>();
    }
    CHECK_EQ(base_lidar_t.size(), 3);
    CHECK_EQ(base_lidar_R.size(), 9);
    Mat3d R_base_lidar;
    R_base_lidar << base_lidar_R[0], base_lidar_R[1], base_lidar_R[2],
        base_lidar_R[3], base_lidar_R[4], base_lidar_R[5],
        base_lidar_R[6], base_lidar_R[7], base_lidar_R[8];
    Quatd q_base_lidar(R_base_lidar);
    q_base_lidar.normalize();
    T_base_lidar_ = SE3(q_base_lidar, Vec3d(base_lidar_t[0], base_lidar_t[1], base_lidar_t[2]));
    LOG(INFO) << "[Mapping][BASE_LIDAR] T_base_lidar trans=" << T_base_lidar_.translation().transpose();

    preprocess_ = std::make_shared<PointCloudPreprocess>();
    if (!preprocess_->Init(yaml_path)) {
        LOG(ERROR) << "failed to init point cloud preprocess";
        return false;
    }

    if (use_lio_sam_) {
        LioSamMapping::Options lio_options;
        lio_options.mapping_mode_ = options_.online_input
            ? LioSamMapping::MappingRuntimeMode::ONLINE_MAPPING
            : LioSamMapping::MappingRuntimeMode::OFFLINE_MAPPING;
        lio_sam_ = std::make_shared<LioSamMapping>(lio_options);
        if (!lio_sam_->Init(yaml_path)) {
            LOG(ERROR) << "failed to init LIO-SAM mapping";
            return false;
        }
    } else {
        lio_ = std::make_shared<LaserMapping>();
        if (!lio_->Init(yaml_path)) {
            LOG(ERROR) << "failed to init laser mapping";
            return false;
        }
    }

    if (options_.with_ui) {
        ui_ = std::make_shared<ui::PangolinWindow>();
        ui_->Init();
        if (use_lio_sam_) {
            lio_sam_->SetUI(ui_);
        } else {
            lio_->SetUI(ui_);
        }
    }



    return true;
}

bool MappingSystem::Start() {
    cur_kf_.reset();
    map_update_pending_ = false;
    running_ = true;
    return true;
}

void MappingSystem::Stop() {
    running_ = false;
}

void MappingSystem::Reset() {
    running_ = false;
    cur_kf_.reset();
    map_update_pending_ = false;

    // 先让算法对象断开 UI，避免 LIO-SAM / LaserMapping 析构时再次持有 UI
    if (use_lio_sam_ && lio_sam_) {
        lio_sam_->SetUI(nullptr);
    }
    if (!use_lio_sam_ && lio_) {
        lio_->SetUI(nullptr);
    }

    // 先取出来，再 reset 成员，避免 shared_ptr 析构顺序混乱
    auto ui = ui_;
    ui_.reset();

    if (ui) {
        ui->Quit();
    }


    lio_sam_.reset();
    lio_.reset();
    preprocess_.reset();
}

void MappingSystem::ProcessIMU(const sensor_msgs::msg::Imu::SharedPtr& imu) {
    if (!running_ || !imu) {
        return;
    }
    IMUPtr input = std::make_shared<IMU>();
    input->timestamp = ToSec(imu->header.stamp);
    input->angular_velocity = Vec3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
    input->linear_acceleration = Vec3d(imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z);
    input->orientation = Quatd(imu->orientation.w, imu->orientation.x, imu->orientation.y, imu->orientation.z);
    ProcessIMU(input);
}

void MappingSystem::ProcessIMU(const IMUPtr& imu) {
    if (!running_ || !imu) {
        return;
    }
    if (use_lio_sam_ && lio_sam_) {
        lio_sam_->ProcessIMU(imu);
    } else if (lio_) {
        lio_->ProcessIMU(imu);
    }
}

bool MappingSystem::BuildInputCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud, CloudPtr& input) {
    if (!preprocess_ || !cloud) {
        return false;
    }
    CloudPtr pcl_input(new PointCloudType);
    preprocess_->Process(cloud, pcl_input);
    if (use_lio_sam_) {
        input = pcl_input;
        return input && !input->empty();
    }

    CloudPtr input_base(new PointCloudType);
    pcl::transformPointCloud(*pcl_input, *input_base, T_base_lidar_.matrix().cast<float>());
    input_base->header = pcl_input->header;
    input_base->header.frame_id = base_link_frame_;
    input = input_base;
    return input && !input->empty();
}

bool MappingSystem::BuildInputCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud, CloudPtr& input) {
    if (!preprocess_ || !cloud) {
        return false;
    }
    CloudPtr pcl_input(new PointCloudType);
    preprocess_->Process(cloud, pcl_input);
    if (use_lio_sam_) {
        input = pcl_input;
        return input && !input->empty();
    }

    CloudPtr input_base(new PointCloudType);
    pcl::transformPointCloud(*pcl_input, *input_base, T_base_lidar_.matrix().cast<float>());
    input_base->header = pcl_input->header;
    input_base->header.frame_id = base_link_frame_;
    input = input_base;
    return input && !input->empty();
}

void MappingSystem::ProcessCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
    if (!running_) {
        return;
    }
    CloudPtr input;
    if (!BuildInputCloud(cloud, input)) {
        return;
    }

    Keyframe::Ptr kf;
    if (use_lio_sam_ && lio_sam_) {
        lio_sam_->ProcessPointCloud2(input);
        if (!lio_sam_->Run()) {
            return;
        }
        kf = lio_sam_->GetKeyframe();
    } else if (lio_) {
        lio_->ProcessPointCloud2(input);
        if (!lio_->Run()) {
            return;
        }
        kf = lio_->GetKeyframe();
    }
    HandleProcessedKeyframe(kf);
}

void MappingSystem::ProcessCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud) {
    if (!running_) {
        return;
    }
    CloudPtr input;
    if (!BuildInputCloud(cloud, input)) {
        return;
    }

    Keyframe::Ptr kf;
    if (use_lio_sam_ && lio_sam_) {
        lio_sam_->ProcessPointCloud2(input);
        if (!lio_sam_->Run()) {
            return;
        }
        kf = lio_sam_->GetKeyframe();
    } else if (lio_) {
        lio_->ProcessPointCloud2(input);
        if (!lio_->Run()) {
            return;
        }
        kf = lio_->GetKeyframe();
    }
    HandleProcessedKeyframe(kf);
}

void MappingSystem::HandleProcessedKeyframe(const Keyframe::Ptr& kf) {
    if (!kf || kf == cur_kf_) {
        return;
    }
    cur_kf_ = kf;
    map_update_pending_ = true;
    if (ui_) {
        ui_->UpdateKF(cur_kf_);
    }
}

CloudPtr MappingSystem::BuildCurrentMapInBaseFrame() {
    std::vector<Keyframe::Ptr> keyframes;
    if (use_lio_sam_ && lio_sam_) {
        lio_sam_->SyncOptimizedKeyframePoses();
        keyframes = lio_sam_->GetAllKeyframes();
    } else if (lio_) {
        keyframes = lio_->GetAllKeyframes();
    }
    if (keyframes.empty()) {
        return nullptr;
    }

    CloudPtr map_base(new PointCloudType());
    const Eigen::Matrix4f T_base_lidar = T_base_lidar_.matrix().cast<float>();

    for (const auto& kf : keyframes) {
        if (!kf) {
            continue;
        }
        CloudPtr cloud = kf->GetCloud();
        if (!cloud || cloud->empty()) {
            continue;
        }

        Eigen::Matrix4f pose = kf->GetOptPose().matrix().cast<float>();
        if (use_lio_sam_) {
            pose = T_base_lidar * pose;
        }

        CloudPtr cloud_base(new PointCloudType());
        pcl::transformPointCloud(*cloud, *cloud_base, pose);
        *map_base += *cloud_base;
    }

    if (map_base->empty()) {
        return nullptr;
    }
    map_base->header.frame_id = "map";
    map_base->height = 1;
    map_base->width = map_base->size();
    map_base->is_dense = false;
    return map_base;
}

bool MappingSystem::ConsumeMapUpdate() {
    const bool pending = map_update_pending_;
    map_update_pending_ = false;
    return pending;
}

MappingSystemResult MappingSystem::GetResult() {
    MappingSystemResult result;
    result.T_base_lidar = T_base_lidar_;
    if (use_lio_sam_ && lio_sam_) {
        lio_sam_->SyncOptimizedKeyframePoses();
        result.keyframes = lio_sam_->GetAllKeyframes();
        result.global_map = lio_sam_->GetGlobalMap(true);
        result.global_map_is_lidar_frame = true;
    } else if (lio_) {
        result.keyframes = lio_->GetAllKeyframes();
        result.global_map = lio_->GetGlobalMap(true);
        result.global_map_is_lidar_frame = false;
    }

    result.valid = result.global_map && !result.global_map->empty() && !result.keyframes.empty();
    return result;
}

}  // namespace lightning::modules
