#include "core/lio/lio_sam/lio_sam_mapping.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <utility>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <yaml-cpp/yaml.h>

#include "ui/pangolin_window.h"
#include "core/lightning_math.hpp"
#include "wrapper/ros_utils.h"

#include "core/lio/lio_sam/deskew_feature_extractor.h"
#include "core/lio/lio_sam/map_optimization.h"

namespace {

template <typename T>
void SetParamOverride(std::vector<rclcpp::Parameter>& params, const std::string& name, const T& value) {
    params.erase(std::remove_if(params.begin(), params.end(),
                                [&](const rclcpp::Parameter& p) { return p.get_name() == name; }),
                 params.end());
    params.emplace_back(name, value);
}

builtin_interfaces::msg::Time ToRosStamp(double timestamp) {
    builtin_interfaces::msg::Time stamp;
    const double clamped = std::max(0.0, timestamp);
    stamp.sec = static_cast<int32_t>(std::floor(clamped));
    stamp.nanosec = static_cast<uint32_t>(std::llround((clamped - stamp.sec) * 1e9));
    if (stamp.nanosec >= 1000000000U) {
        ++stamp.sec;
        stamp.nanosec -= 1000000000U;
    }
    return stamp;
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
    deskew_feature_extractor_.reset();
    map_optimization_.reset();
    frontend_cloud_info_.reset();
    if (owns_rclcpp_context_ && rclcpp::ok()) {
        rclcpp::shutdown();
    }
}

bool LioSamMapping::Init(const std::string& config_yaml) {
    LOG(INFO) << "init lio-sam mapping from " << config_yaml;
    if (!LoadParamsFromYAML(config_yaml)) {
        return false;
    }

    if (!rclcpp::ok()) {
        int argc = 1;
        const char* argv[] = {"lightning_lio_sam_offline"};
        rclcpp::init(argc, argv);
        owns_rclcpp_context_ = true;
    }

    frontend_cloud_info_ = std::make_unique<::LioSamCloudInfo>();
    deskew_feature_extractor_ = std::make_unique<::DeskewFeatureExtractor>(node_options_);
    map_optimization_ = std::make_unique<::mapOptimization>(node_options_);

    LOG(INFO) << "[LIO_SAM_FRONTEND] frontend=deskew_feature_extractor";

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
        SetParamOverride(overrides, "mappingProcessInterval", params["mappingProcessInterval"].as<double>());
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

void LioSamMapping::ProcessIMU(const IMUPtr& input) {
    sensor_msgs::msg::Imu imu;
    imu.header.stamp = ToRosStamp(input->timestamp);
    imu.angular_velocity.x = input->angular_velocity.x();
    imu.angular_velocity.y = input->angular_velocity.y();
    imu.angular_velocity.z = input->angular_velocity.z();
    imu.linear_acceleration.x = input->linear_acceleration.x();
    imu.linear_acceleration.y = input->linear_acceleration.y();
    imu.linear_acceleration.z = input->linear_acceleration.z();
    imu.orientation.x = input->orientation.x();
    imu.orientation.y = input->orientation.y();
    imu.orientation.z = input->orientation.z();
    imu.orientation.w = input->orientation.w();
    const double timestamp = input->timestamp;
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "lio-sam imu loop back, clear buffer";
        imu_buffer_.clear();
    }

    last_timestamp_imu_ = timestamp;
    imu_buffer_.push_back(imu);
}

void LioSamMapping::ProcessPointCloud2(CloudPtr cloud) {
    const double timestamp = math::ToSec(cloud->header.stamp);

    CloudPtr frontend_cloud(new PointCloudType());
    frontend_cloud->header = cloud->header;
    frontend_cloud->reserve(cloud->size());

    // PointCloudPreprocess is the single lidar-format adapter in lightning.
    // Its unified PointType::time contract is milliseconds:
    //   Livox offset_time(ns) / 1e6 -> ms
    //   Ouster t(ns) / 1e6 -> ms
    //   RoboSense/Hesai absolute seconds - header seconds, then * 1e3 -> ms
    //   Velodyne/CX16 raw point time * fasterlio.time_scale -> ms
    // Native LIO-SAM uses per-point relative time in seconds, so this wrapper
    // performs exactly one fixed ms -> s conversion. Do not auto-detect units here;
    // set fasterlio.time_scale correctly for each lidar/bag.
    float min_source_time_ms = std::numeric_limits<float>::max();
    float max_source_time_ms = 0.0f;
    float max_point_time = 0.0f;
    for (const auto& source : cloud->points) {
        min_source_time_ms = std::min(min_source_time_ms, static_cast<float>(source.time));
        max_source_time_ms = std::max(max_source_time_ms, static_cast<float>(source.time));

        PointType point = source;
        point.time = static_cast<float>(source.time * 1e-3);
        max_point_time = std::max(max_point_time, static_cast<float>(point.time));
        frontend_cloud->push_back(point);
    }

    static int lio_sam_time_log_count = 0;
    if (!cloud->empty() && (++lio_sam_time_log_count <= 10 || lio_sam_time_log_count % 100 == 0)) {
        LOG(INFO) << "[LIO_SAM_TIME] preprocess_time_ms_min=" << min_source_time_ms
                  << " preprocess_time_ms_max=" << max_source_time_ms
                  << " final_time_sec_max=" << max_point_time
                  << " stamp=" << std::setprecision(14) << timestamp;
    }

    frontend_cloud->height = 1;
    frontend_cloud->width = frontend_cloud->size();
    frontend_cloud->is_dense = true;

    const double scan_duration = static_cast<double>(max_point_time);

    std::lock_guard<std::mutex> lock(mtx_buffer_);
    if (timestamp < last_timestamp_lidar_) {
        LOG(ERROR) << "lio-sam lidar loop back, clear buffer";
        lidar_buffer_.clear();
        time_buffer_.clear();
        scan_duration_buffer_.clear();
        frame_id_buffer_.clear();
        lidar_pushed_ = false;

    }
    last_timestamp_lidar_ = timestamp;
    lidar_buffer_.push_back(frontend_cloud);
    time_buffer_.push_back(timestamp);
    scan_duration_buffer_.push_back(scan_duration);
    frame_id_buffer_.push_back(cloud->header.frame_id);
}

bool LioSamMapping::SyncPackages() {
    std::lock_guard<std::mutex> lock(mtx_buffer_);

    if (lidar_buffer_.empty()) {
        return false;
    }
    if (imu_buffer_.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed_) {
        measures_ = SyncedPackage();
        measures_.cloud = lidar_buffer_.front();
        measures_.lidar_begin_time = time_buffer_.front();
        measures_.frame_id = frame_id_buffer_.front();

        const double scan_duration = scan_duration_buffer_.front();

        lidar_begin_time_ = measures_.lidar_begin_time;
        lidar_end_time_ = measures_.lidar_begin_time + scan_duration;

        measures_.lidar_end_time = lidar_end_time_;
        lidar_pushed_ = true;
        lo::lidar_time_interval = scan_duration;
    }

    if (last_timestamp_imu_ < lidar_end_time_) {
        return false;
    }

    /*** push imu_ data, and pop from imu_ buffer ***/
    while (imu_buffer_.size() >= 2 && ToSec(imu_buffer_[1].header.stamp) < lidar_begin_time_ - 0.05) {
        imu_buffer_.pop_front();
    }

    if (ToSec(imu_buffer_.front().header.stamp) > lidar_begin_time_) {
        LOG(WARNING) << "LIO-SAM IMU does not cover scan begin, drop lidar scan. scan="
                     << std::setprecision(14) << lidar_begin_time_
                     << ", first imu=" << ToSec(imu_buffer_.front().header.stamp);
        lidar_buffer_.pop_front();
        time_buffer_.pop_front();
        scan_duration_buffer_.pop_front();
        frame_id_buffer_.pop_front();
        lidar_pushed_ = false;
        return false;
    }

    measures_.imus.clear();
    const double imu_collect_end = lidar_end_time_ + 0.01;
    bool imu_covers_scan_end = false;
    for (const auto& imu : imu_buffer_) {
        const double imu_time = ToSec(imu.header.stamp);
        measures_.imus.push_back(imu);
        if (imu_time >= lidar_end_time_) {
            imu_covers_scan_end = true;
        }
        if (imu_time >= imu_collect_end) {
            break;
        }
    }

    if (measures_.imus.empty() || !imu_covers_scan_end) {
        return false;
    }

    while (imu_buffer_.size() >= 2 && ToSec(imu_buffer_[1].header.stamp) <= lidar_end_time_) {
        imu_buffer_.pop_front();
    }

    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    scan_duration_buffer_.pop_front();
    frame_id_buffer_.pop_front();
    lidar_pushed_ = false;
    return true;
}

bool LioSamMapping::Run() {
    if (!map_optimization_ || !frontend_cloud_info_ || !deskew_feature_extractor_) {
        return false;
    }

    if (!SyncPackages()) {
        return false;
    }

    LioSamCloudInfo& cloud_info = *frontend_cloud_info_;
    if (!deskew_feature_extractor_->Run(
            measures_.cloud,
            measures_.imus,
            measures_.lidar_begin_time,
            measures_.lidar_end_time,
            measures_.frame_id,
            true,
            cloud_info)) {
        return false;
    }

    // No external ESKF/DR odometry is used here.
    // This matches native LIO-SAM mapping behavior: updateInitialGuess()
    // uses IMU rotation increment when odom_available is false.
    cloud_info.odom_available = false;
    cloud_info.initial_guess_x = 0.0f;
    cloud_info.initial_guess_y = 0.0f;
    cloud_info.initial_guess_z = 0.0f;
    cloud_info.initial_guess_roll = 0.0f;
    cloud_info.initial_guess_pitch = 0.0f;
    cloud_info.initial_guess_yaw = 0.0f;

    if (!map_optimization_->Run(cloud_info)) {
        return false;
    }

    const float* transform = map_optimization_->TransformTobeMapped();
    state_.timestamp_ = measures_.lidar_end_time;
    state_.pos_ = Vec3d(transform[3], transform[4], transform[5]);
    state_.rot_ = RpyToSO3(transform[0], transform[1], transform[2]);
    state_.pose_is_ok_ = map_optimization_->mappingPoseReliable;
    state_.lidar_odom_reliable_ = map_optimization_->mappingPoseReliable;

    scan_undistort_ = cloud_info.cloud_deskewed;
    if (scan_undistort_) {
        scan_undistort_->header.stamp = static_cast<std::uint64_t>(std::llround(state_.timestamp_ * 1e9));
        scan_undistort_->header.frame_id = measures_.frame_id;
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
    cloud->header.frame_id = measures_.frame_id;
    cloud->header.stamp = static_cast<std::uint64_t>(std::llround(state_.timestamp_ * 1e9));
    if (raw_cloud) {
        *cloud = *raw_cloud;
    }
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
