#include "core/lio_sam/lio_sam_mapping.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <string>
#include <utility>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <yaml-cpp/yaml.h>

#include "ui/pangolin_window.h"
#include "core/lightning_math.hpp"
#include "wrapper/ros_utils.h"

#include "core/lio_sam/featureExtraction.h"
#include "core/lio_sam/map_optimization.h"

namespace {

template <typename T>
void SetParamOverride(std::vector<rclcpp::Parameter>& params, const std::string& name, const T& value) {
    params.erase(std::remove_if(params.begin(), params.end(),
                                [&](const rclcpp::Parameter& p) { return p.get_name() == name; }),
                 params.end());
    params.emplace_back(name, value);
}

lightning::SO3 RpyToSO3(double roll, double pitch, double yaw) {
    Eigen::Matrix3d rot =
        (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    return lightning::SO3(Eigen::Quaterniond(rot));
}

}  // namespace

namespace lightning {

LioSamMapping::LioSamMapping() : LioSamMapping(Options()) {}

LioSamMapping::LioSamMapping(Options options) : options_(options) {}

LioSamMapping::~LioSamMapping() {
    LOG(INFO) << "[LioSamMapping] destroy begin"
              << ", this=" << this
              << ", mapOptimization=" << map_optimization_.get()
              << ", map_opt_executed=" << diagnostic_map_optimization_executed_
              << ", map_opt_skipped=" << diagnostic_map_optimization_skipped_
              << ", keyframes=" << all_keyframes_.size();
    feature_extraction_.reset();
    map_optimization_.reset();
}

bool LioSamMapping::Init(const std::string& config_yaml) {
    LOG(INFO) << "init lio-sam mapping from " << config_yaml;
    if (!LoadParamsFromYAML(config_yaml)) {
        return false;
    }

    if (!rclcpp::ok()) {
        LOG(ERROR) << "rclcpp context is not initialized; start LIO-SAM through run_lightning";
        return false;
    }

    feature_extraction_ = std::make_unique<::FeatureExtractor>(node_options_);
    map_optimization_ = std::make_unique<::mapOptimization>(node_options_);

    LOG(INFO) << "[璺ㄤ换鍔＄姸鎬佽瘖鏂璢[LioSamMapping] 鍒濆鍖栨柊瀵硅薄"
              << ", this=" << this
              << ", mapOptimization=" << map_optimization_.get()
              << ", online=" << IsOnlineMapping();
    LOG(INFO) << "[LIO_SAM_FRONTEND] frontend=featureExtraction";

    return true;
}

bool LioSamMapping::LoadParamsFromYAML(const std::string& yaml_path) {
    try {
        const YAML::Node yaml = YAML::LoadFile(yaml_path);
        const YAML::Node common = yaml["common"];
        const YAML::Node params = yaml["lio_sam"];
        if (!params) {
            LOG(ERROR) << "lio_sam config section is missing";
            return false;
        }

        std::vector<rclcpp::Parameter> overrides;

        SetParamOverride(overrides, "useImuHeadingInitialization", params["useImuHeadingInitialization"].as<bool>());
        SetParamOverride(overrides, "useImuAccelRollPitchInitialization",
                         params["useImuAccelRollPitchInitialization"].as<bool>());
        SetParamOverride(overrides, "N_SCAN", common["N_SCAN"].as<int>());
        SetParamOverride(overrides, "Horizon_SCAN", common["Horizon_SCAN"].as<int>());
        SetParamOverride(overrides, "lidarType",
                         common["sensor"].as<std::string>() == "livox" ? 1 : 2);
        SetParamOverride(overrides, "downsampleRate", params["downsampleRate"].as<int>());
        SetParamOverride(overrides, "lidarMinRange", common["lidarMinRange"].as<double>());
        SetParamOverride(overrides, "lidarMaxRange", common["lidarMaxRange"].as<double>());
        SetParamOverride(overrides, "imuAccNoise", common["imuAccNoise"].as<double>());
        SetParamOverride(overrides, "imuGyrNoise", common["imuGyrNoise"].as<double>());
        SetParamOverride(overrides, "imuAccBiasN", common["imuAccBiasN"].as<double>());
        SetParamOverride(overrides, "imuGyrBiasN", common["imuGyrBiasN"].as<double>());
        SetParamOverride(overrides, "imuGravity", params["imuGravity"].as<double>());
        SetParamOverride(overrides, "imuRPYWeight", params["imuRPYWeight"].as<double>());
        SetParamOverride(overrides, "extrinsicRot", params["extrinsicRot"].as<std::vector<double>>());
        SetParamOverride(overrides, "extrinsicRPY", params["extrinsicRPY"].as<std::vector<double>>());
        SetParamOverride(overrides, "edgeThreshold", params["edgeThreshold"].as<double>());
        SetParamOverride(overrides, "surfThreshold", params["surfThreshold"].as<double>());
        SetParamOverride(overrides, "edgeFeatureMinValidNum", params["edgeFeatureMinValidNum"].as<int>());
        SetParamOverride(overrides, "surfFeatureMinValidNum", params["surfFeatureMinValidNum"].as<int>());
        SetParamOverride(overrides, "odometrySurfLeafSize", params["odometrySurfLeafSize"].as<double>());
        SetParamOverride(overrides, "mappingCornerLeafSize", params["mappingCornerLeafSize"].as<double>());
        SetParamOverride(overrides, "mappingSurfLeafSize", params["mappingSurfLeafSize"].as<double>());
        SetParamOverride(overrides, "z_tollerance", params["z_tollerance"].as<double>());
        SetParamOverride(overrides, "rotation_tollerance", params["rotation_tollerance"].as<double>());
        SetParamOverride(overrides, "numberOfCores", params["numberOfCores"].as<int>());
        const double configured_mapping_process_interval =
            params["mappingProcessInterval"].as<double>();
        SetParamOverride(overrides, "mappingProcessInterval",
                         IsOnlineMapping() ? 0.0 : configured_mapping_process_interval);
        LOG(INFO) << "[鍦ㄧ嚎寤哄浘閫愬抚淇] mappingProcessInterval="
                  << (IsOnlineMapping() ? 0.0 : configured_mapping_process_interval)
                  << ", configured=" << configured_mapping_process_interval
                  << ", online=" << IsOnlineMapping();
        SetParamOverride(overrides, "isOnlineMapping", IsOnlineMapping());
        SetParamOverride(overrides, "maxOptimizationIterations",
                         params["maxOptimizationIterations"]
                             ? params["maxOptimizationIterations"].as<int>()
                             : 30);
        SetParamOverride(overrides, "onlineMaxOptimizationIterations",
                         params["onlineMaxOptimizationIterations"]
                             ? params["onlineMaxOptimizationIterations"].as<int>()
                             : 10);
        SetParamOverride(overrides, "onlineLimitOptimizationPoints",
                         params["onlineLimitOptimizationPoints"]
                             ? params["onlineLimitOptimizationPoints"].as<bool>()
                             : true);
        SetParamOverride(overrides, "onlineMaxSurfOptimizationPoints",
                         params["onlineMaxSurfOptimizationPoints"]
                             ? params["onlineMaxSurfOptimizationPoints"].as<int>()
                             : 4000);
        SetParamOverride(overrides, "onlineMaxCornerOptimizationPoints",
                         params["onlineMaxCornerOptimizationPoints"]
                             ? params["onlineMaxCornerOptimizationPoints"].as<int>()
                             : 900);
        SetParamOverride(overrides, "debugTiming",
                         params["debugTiming"] ? params["debugTiming"].as<bool>() : false);
        SetParamOverride(overrides, "surroundingkeyframeAddingDistThreshold",
                         common["surroundingkeyframeAddingDistThreshold"].as<double>());
        SetParamOverride(overrides, "surroundingkeyframeAddingAngleThreshold",
                         common["surroundingkeyframeAddingAngleThreshold"].as<double>());
        SetParamOverride(overrides, "surroundingKeyframeDensity", common["surroundingKeyframeDensity"].as<double>());
        SetParamOverride(overrides, "surroundingKeyframeSearchRadius",
                         common["surroundingKeyframeSearchRadius"].as<double>());
        SetParamOverride(overrides, "loopClosureEnableFlag", params["loopClosureEnableFlag"].as<bool>());
        SetParamOverride(overrides, "surroundingKeyframeSize", params["surroundingKeyframeSize"].as<int>());
        SetParamOverride(overrides, "historyKeyframeSearchRadius", params["historyKeyframeSearchRadius"].as<double>());
        SetParamOverride(overrides, "historyKeyframeSearchTimeDiff",
                         params["historyKeyframeSearchTimeDiff"].as<double>());
        SetParamOverride(overrides, "historyKeyframeSearchNum", params["historyKeyframeSearchNum"].as<int>());
        SetParamOverride(overrides, "historyKeyframeFitnessScore", params["historyKeyframeFitnessScore"].as<double>());
        node_options_ = rclcpp::NodeOptions();
        node_options_.use_intra_process_comms(true);
        node_options_.parameter_overrides(overrides);
        return true;
    } catch (const std::exception& e) {
        LOG(ERROR) << "failed to load LIO-SAM params from " << yaml_path << ": " << e.what();
        return false;
    }
}

bool LioSamMapping::Run(::LioSamCloudInfo& cloud_info) {
    if (!feature_extraction_ || !map_optimization_) {
        return false;
    }
    if (!feature_extraction_->Run(cloud_info)) {
        return false;
    }

    if (!map_optimization_->Run(cloud_info)) {
        return false;
    }
    if (!map_optimization_->LastRunExecuted()) {
        ++diagnostic_map_optimization_skipped_;
        if (diagnostic_map_optimization_skipped_ <= 10 ||
            diagnostic_map_optimization_skipped_ % 100 == 0) {
            LOG(WARNING) << "[鍓嶇璺冲抚璇婃柇][LioSamMapping] 鏈抚鏈骇鐢熸柊浣嶅Э锛屼笉鏇存柊鐘舵€佸拰UI"
                         << ", this=" << this
                         << ", mapOptimization=" << map_optimization_.get()
                         << ", lidar_stamp=" << std::setprecision(14)
                         << cloud_info.timestamp
                         << ", map_opt_executed=" << diagnostic_map_optimization_executed_
                         << ", map_opt_skipped=" << diagnostic_map_optimization_skipped_;
        }
        return false;
    }
    ++diagnostic_map_optimization_executed_;

    const float* transform = map_optimization_->TransformTobeMapped();
    state_.timestamp_ = cloud_info.timestamp;
    state_.pos_ = Vec3d(transform[3], transform[4], transform[5]);
    state_.rot_ = RpyToSO3(transform[0], transform[1], transform[2]);
    state_.pose_is_ok_ = map_optimization_->mappingPoseReliable;
    state_.lidar_odom_reliable_ = map_optimization_->mappingPoseReliable;

    current_frame_id_ = cloud_info.frame_id.empty() ? std::string("lidar") : cloud_info.frame_id;
    scan_undistort_ = cloud_info.cloud_deskewed;
    if (scan_undistort_) {
        scan_undistort_->header.stamp = static_cast<std::uint64_t>(std::llround(state_.timestamp_ * 1e9));
        scan_undistort_->header.frame_id = current_frame_id_;
        scan_undistort_->height = 1;
        scan_undistort_->width = scan_undistort_->size();
        scan_undistort_->is_dense = true;
    }
    recent_cloud_ = scan_undistort_;

    if (ui_) {
        ui_->UpdateNavState(state_);
        if (scan_undistort_) {
            ui_->UpdateScan(scan_undistort_, state_.GetPose());
        }
    }

    MakeLightningKeyframeIfNeeded();
    return true;
}

bool LioSamMapping::MakeLightningKeyframeIfNeeded() {
    if (!map_optimization_ || !map_optimization_->CreatedNewKeyframe() ||
        map_optimization_->KeyPoseSize() == 0 ||
        map_optimization_->KeyPoseSize() <= map_keyframe_count_) {
        return false;
    }
    PointCloudType::Ptr raw_cloud = map_optimization_->LatestRawCloudKeyFrame();

    CloudPtr cloud(new PointCloudType());
    if (raw_cloud) {
        *cloud = *raw_cloud;
    }
    cloud->header.frame_id = current_frame_id_;
    cloud->header.stamp = static_cast<std::uint64_t>(std::llround(state_.timestamp_ * 1e9));
    cloud->height = 1;
    cloud->width = cloud->size();
    cloud->is_dense = true;

    auto kf = std::make_shared<Keyframe>(kf_id_++, cloud, state_);
    kf->SetLIOPose(state_.GetPose());
    kf->SetOptPose(state_.GetPose());
    kf->SetState(state_);

    all_keyframes_.push_back(kf);
    last_kf_ = kf;
    map_keyframe_count_ = map_optimization_->KeyPoseSize();
    map_optimization_->ClearCreatedNewKeyframe();
    return true;
}

void LioSamMapping::SyncLightningKeyframePoses() {
    if (!map_optimization_) {
        return;
    }

    const size_t n = std::min(all_keyframes_.size(), map_optimization_->KeyPoseSize());
    for (size_t i = 0; i < n; ++i) {
        const auto pose = map_optimization_->KeyPose(i);
        SE3 opt_pose(
            RpyToSO3(pose.roll, pose.pitch, pose.yaw),
            Vec3d(pose.x, pose.y, pose.z));

        all_keyframes_[i]->SetLIOPose(opt_pose);
        all_keyframes_[i]->SetOptPose(opt_pose);
    }
}

void LioSamMapping::SyncOptimizedKeyframePoses() {
    SyncLightningKeyframePoses();
}

CloudPtr LioSamMapping::GetGlobalMap(bool use_lio_pose, bool use_voxel, float res) {
    CloudPtr global_map(new PointCloudType);

    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(res, res, res);

    for (auto& kf : all_keyframes_) {
        CloudPtr cloud = kf->GetCloud();
        CloudPtr cloud_filter(new PointCloudType);

        if (use_voxel) {
            voxel.setInputCloud(cloud);
            voxel.filter(*cloud_filter);
        } else {
            cloud_filter = cloud;
        }

        CloudPtr cloud_trans(new PointCloudType);
        if (use_lio_pose) {
            pcl::transformPointCloud(*cloud_filter, *cloud_trans, kf->GetLIOPose().matrix());
        } else {
            pcl::transformPointCloud(*cloud_filter, *cloud_trans, kf->GetOptPose().matrix());
        }
        *global_map += *cloud_trans;
    }

    CloudPtr global_map_filtered(new PointCloudType);
    if (use_voxel) {
        voxel.setInputCloud(global_map);
        voxel.filter(*global_map_filtered);
    } else {
        global_map_filtered = global_map;
    }

    global_map_filtered->height = 1;
    global_map_filtered->width = global_map_filtered->size();
    global_map_filtered->is_dense = false;
    return global_map_filtered;
}

}  // namespace lightning
