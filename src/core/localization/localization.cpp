#include "core/localization/localization.h"

#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/time.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "core/lightning_math.hpp"
#include "ui/pangolin_window.h"

namespace lightning::loc {
namespace {

double SteadySeconds() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void AppendXYZCloud(const pcl::PointCloud<pcl::PointXYZ>& src, PointCloudType& dst) {
    dst.reserve(dst.size() + src.size());
    for (const auto& src_pt : src.points) {
        PointType dst_pt;
        dst_pt.x = src_pt.x;
        dst_pt.y = src_pt.y;
        dst_pt.z = src_pt.z;
        dst_pt.intensity = 0.0f;
        dst_pt.ring = 0;
        dst_pt.time = 0.0;
        dst.push_back(dst_pt);
    }
}

}  // namespace

Localization::Localization(Options options) : options_(options) {}

bool Localization::Init(const std::string& yaml_path, const std::string& global_map_path) {
    Finish();

    std::lock_guard<std::mutex> loc_lock(localizer_mutex_);
    UL lock(global_mutex_);

    YAML::Node yaml_node = YAML::LoadFile(yaml_path);
    options_.with_ui_ = false;
    LOG(INFO) << "[LOCALIZATION_CORE] pub_tf = " << options_.pub_tf_;
    if (yaml_node["system"] && yaml_node["system"]["with_ui"]) {
        options_.with_ui_ = yaml_node["system"]["with_ui"].as<bool>();
    }
    if (yaml_node["common"] && yaml_node["common"]["base_link_frame"]) {
        base_link_frame_ = yaml_node["common"]["base_link_frame"].as<std::string>();
    }

    std::vector<double> base_lidar_xyz_rpy_deg{
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (yaml_node["extrinsicBaseLidar"]) {
        base_lidar_xyz_rpy_deg =
            yaml_node["extrinsicBaseLidar"].as<std::vector<double>>();
    } else if (yaml_node["common"] &&
               yaml_node["common"]["extrinsicBaseLidar"]) {
        base_lidar_xyz_rpy_deg =
            yaml_node["common"]["extrinsicBaseLidar"]
                .as<std::vector<double>>();
    }
    CHECK_EQ(base_lidar_xyz_rpy_deg.size(), 6U);
    options_.T_base_lidar_ =
        math::XyzRpyDegreesToSE3(base_lidar_xyz_rpy_deg);
    T_base_lidar_matrix_f_ = options_.T_base_lidar_.matrix().cast<float>();
    LOG(INFO) << "[BASE_LIDAR] T_base_lidar trans="
              << options_.T_base_lidar_.translation().transpose()
              << ", rpy_deg="
              << Vec3d(base_lidar_xyz_rpy_deg[3],
                       base_lidar_xyz_rpy_deg[4],
                       base_lidar_xyz_rpy_deg[5]).transpose();

    if (yaml_node["localization"] && yaml_node["localization"]["quality"]) {
        const auto quality = yaml_node["localization"]["quality"];
        if (quality["excellent_threshold"]) {
            quality_thresholds_.excellent_threshold =
                quality["excellent_threshold"].as<double>();
        }
        if (quality["good_threshold"]) {
            quality_thresholds_.good_threshold = quality["good_threshold"].as<double>();
        }
        if (quality["fair_threshold"]) {
            quality_thresholds_.fair_threshold = quality["fair_threshold"].as<double>();
        }
        if (quality["excellent_iteration_threshold"]) {
            quality_thresholds_.excellent_iteration_threshold =
                quality["excellent_iteration_threshold"].as<int>();
        }
    }

    robot_localizer::NdtCovarianceOptions covariance_options;
    if (yaml_node["localization"] && yaml_node["localization"]["ndt_covariance"]) {
        const YAML::Node covariance = yaml_node["localization"]["ndt_covariance"];
        if (covariance["information_scale"]) covariance_options.information_scale = covariance["information_scale"].as<double>();
        if (covariance["min_information_eigenvalue"]) covariance_options.min_information_eigenvalue = covariance["min_information_eigenvalue"].as<double>();
        if (covariance["max_information_eigenvalue"]) covariance_options.max_information_eigenvalue = covariance["max_information_eigenvalue"].as<double>();
        if (covariance["min_translation_std"]) covariance_options.min_translation_std = covariance["min_translation_std"].as<double>();
        if (covariance["max_translation_std"]) covariance_options.max_translation_std = covariance["max_translation_std"].as<double>();
        if (covariance["min_rotation_std_deg"]) covariance_options.min_rotation_std_rad = covariance["min_rotation_std_deg"].as<double>() * M_PI / 180.0;
        if (covariance["max_rotation_std_deg"]) covariance_options.max_rotation_std_rad = covariance["max_rotation_std_deg"].as<double>() * M_PI / 180.0;
    }

    const std::string metadata_path =
        global_map_path + "/BlockMap/pointcloud_map_metadata.yaml";
    const std::string pcd_directory = global_map_path + "/BlockMap/pointcloud_map";

    localizer_.MapReset();
    localizer_.SetStaticMap(metadata_path, pcd_directory);
    localizer_.SetQualityThresholds(quality_thresholds_);
    localizer_.SetCovarianceOptions(covariance_options);
    localizer_.ResetLocalizationState();

    map_loaded_ = true;
    localization_inited_ = false;
    has_pending_initial_pose_ = false;
    init_in_progress_ = false;
    pending_initial_pose_ = Eigen::Matrix4d::Identity();
    latest_pose_ = Eigen::Matrix4d::Identity();
    latest_cloud_timestamp_ = 0.0;
    has_last_processed_cloud_timestamp_ = false;
    last_processed_cloud_timestamp_ = 0.0;
    has_last_callback_start_steady_sec_ = false;
    last_callback_start_steady_sec_ = 0.0;
    {
        std::lock_guard<std::mutex> cloud_lock(current_cloud_mutex_);
        latest_cloud_.reset();
    }
    {
        UL lock_result(loc_result_mutex_);
        loc_result_ = LocalizationResult();
    }
    if (options_.with_ui_) {
        ui_ = std::make_shared<ui::PangolinWindow>();
        if (!ui_->Init()) {
            LOG(ERROR) << "[LOCALIZATION_UI] failed to create Pangolin window";
            ui_.reset();
            return false;
        }
        LoadTargetMapForUI(global_map_path);
    }

    return true;
}

void Localization::LoadTargetMapForUI(const std::string& global_map_path) {
    CloudPtr target_map(new PointCloudType());
    namespace fs = std::filesystem;
    std::string map_source = "BlockMap";
    int block_file_count = 0;
    const fs::path block_map_dir = fs::path(global_map_path) / "BlockMap" / "pointcloud_map";
    if (fs::exists(block_map_dir) && fs::is_directory(block_map_dir)) {
        for (const auto& entry : fs::directory_iterator(block_map_dir)) {
            const std::string ext = entry.path().extension().string();
            if (!entry.is_regular_file() || (ext != ".pcd" && ext != ".PCD")) {
                continue;
            }
            pcl::PointCloud<pcl::PointXYZ> block_cloud;
            if (pcl::io::loadPCDFile(entry.path().string(), block_cloud) != 0) {
                LOG(WARNING) << "[LOCALIZATION_UI] failed to load BlockMap pcd: "
                             << entry.path().string();
                continue;
            }
            AppendXYZCloud(block_cloud, *target_map);
            ++block_file_count;
        }
    }

    if (target_map->empty()) {
        map_source = "global.pcd";
        const std::string global_pcd_path = global_map_path + "/global.pcd";
        if (pcl::io::loadPCDFile(global_pcd_path, *target_map) != 0 || target_map->empty()) {
            LOG(WARNING) << "[LOCALIZATION_UI] failed to load map for UI from BlockMap or global.pcd under: "
                         << global_map_path;
            return;
        }
    }

    target_map->height = 1;
    target_map->width = target_map->size();
    target_map->is_dense = false;

    std::map<int, CloudPtr> ui_map;
    ui_map.emplace(0, target_map);
    ui_->UpdatePointCloudGlobal(ui_map);
    LOG(INFO) << "[LOCALIZATION_UI] target map loaded for UI from " << map_source
              << ", block_files=" << block_file_count
              << ", points=" << target_map->size();
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Localization::ConvertToBaseCloud(
    const sensor_msgs::msg::PointCloud2& msg,
    const Mat4f& T_base_lidar_matrix_f,
    const std::string& base_link_frame) const {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(msg, *cloud);

    std::vector<int> nan_indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, nan_indices);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_base(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(
        *cloud, *cloud_base, T_base_lidar_matrix_f);
    cloud_base->header = cloud->header;
    cloud_base->header.frame_id = base_link_frame;
    cloud_base->width = cloud_base->size();
    cloud_base->height = 1;
    cloud_base->is_dense = false;
    return cloud_base;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Localization::ConvertToBaseCloud(
    const livox_ros_driver2::msg::CustomMsg& msg,
    const Mat4f& T_base_lidar_matrix_f,
    const std::string& base_link_frame) const {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->header.stamp = rclcpp::Time(msg.header.stamp).nanoseconds();
    cloud->header.frame_id = msg.header.frame_id;
    cloud->reserve(msg.points.size());
    for (const auto& pt : msg.points) {
        pcl::PointXYZ xyz;
        xyz.x = pt.x;
        xyz.y = pt.y;
        xyz.z = pt.z;
        cloud->push_back(xyz);
    }

    std::vector<int> nan_indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, nan_indices);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_base(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(
        *cloud, *cloud_base, T_base_lidar_matrix_f);
    cloud_base->header = cloud->header;
    cloud_base->header.frame_id = base_link_frame;
    cloud_base->width = cloud_base->size();
    cloud_base->height = 1;
    cloud_base->is_dense = false;
    return cloud_base;
}

LocalizationFrameOutcome Localization::ProcessLidarMsg(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg,
    const LocalizationInputDiagnostic& diagnostic) {
    const double callback_start_steady_sec = SteadySeconds();
    bool map_loaded = false;
    Mat4f T_base_lidar_matrix_f;
    std::string base_link_frame;
    double arrival_dt = 0.0;

    {
        UL lock(global_mutex_);
        map_loaded = map_loaded_;
        T_base_lidar_matrix_f = T_base_lidar_matrix_f_;
        base_link_frame = base_link_frame_;
        if (map_loaded) {
            if (has_last_callback_start_steady_sec_) {
                arrival_dt = callback_start_steady_sec - last_callback_start_steady_sec_;
            }
            last_callback_start_steady_sec_ = callback_start_steady_sec;
            has_last_callback_start_steady_sec_ = true;
        }
    }

    ++diagnostic_input_frames_;
    if (!map_loaded) {
        ++diagnostic_map_not_loaded_;
        LOG(ERROR) << "[在线定位输入诊断][Localization] 地图未加载，丢弃PointCloud2"
                   << ", sequence=" << diagnostic.pipeline_sequence;
        return LocalizationFrameOutcome::MAP_NOT_LOADED;
    }

    LocCloudFrame frame;
    const double convert_start_steady_sec = SteadySeconds();
    frame.cloud = ConvertToBaseCloud(*msg, T_base_lidar_matrix_f, base_link_frame);
    frame.convert_ms = (SteadySeconds() - convert_start_steady_sec) * 1000.0;
    diagnostic_max_convert_ms_ = std::max(diagnostic_max_convert_ms_, frame.convert_ms);
    frame.timestamp = rclcpp::Time(msg->header.stamp).seconds();
    frame.callback_start_steady_sec = callback_start_steady_sec;
    frame.arrival_dt = arrival_dt;
    frame.raw_points = frame.cloud->size();
    frame.message_points = static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height);
    frame.frame_id = msg->header.frame_id;
    frame.diagnostic = diagnostic;
    if (frame.cloud->empty()) {
        ++diagnostic_empty_after_convert_;
        LOG(ERROR) << "[Localization] PointCloud2 convert produced empty cloud"
                   << ", sequence=" << diagnostic.pipeline_sequence
                   << ", message_points=" << frame.message_points;
        return LocalizationFrameOutcome::EMPTY_AFTER_CONVERT;
    }
    if (diagnostic.pipeline_sequence <= 20 || diagnostic.pipeline_sequence % 100 == 0) {
        LOG(INFO) << std::setprecision(15)
                  << "[在线定位输入诊断][Localization] PointCloud2转换完成"
                  << ", sequence=" << diagnostic.pipeline_sequence
                  << ", topic_sequence=" << diagnostic.topic_sequence
                  << ", header_stamp=" << frame.timestamp
                  << ", arrival_dt_ms=" << frame.arrival_dt * 1000.0
                  << ", message_points=" << frame.message_points
                  << ", converted_points=" << frame.raw_points
                  << ", convert_ms=" << frame.convert_ms
                  << ", frame_id=" << frame.frame_id;
    }
    return HandleCloudFrame(frame);
}

LocalizationFrameOutcome Localization::ProcessLivoxLidarMsg(
    const livox_ros_driver2::msg::CustomMsg::SharedPtr msg,
    const LocalizationInputDiagnostic& diagnostic) {
    const double callback_start_steady_sec = SteadySeconds();
    bool map_loaded = false;
    Mat4f T_base_lidar_matrix_f;
    std::string base_link_frame;
    double arrival_dt = 0.0;
    {
        UL lock(global_mutex_);
        map_loaded = map_loaded_;
        T_base_lidar_matrix_f = T_base_lidar_matrix_f_;
        base_link_frame = base_link_frame_;
        if (map_loaded) {
            if (has_last_callback_start_steady_sec_) {
                arrival_dt = callback_start_steady_sec - last_callback_start_steady_sec_;
            }
            last_callback_start_steady_sec_ = callback_start_steady_sec;
            has_last_callback_start_steady_sec_ = true;
        }
    }

    ++diagnostic_input_frames_;
    if (!map_loaded) {
        ++diagnostic_map_not_loaded_;
        LOG(ERROR) << "[在线定位输入诊断][Localization] 地图未加载，丢弃Livox"
                   << ", sequence=" << diagnostic.pipeline_sequence;
        return LocalizationFrameOutcome::MAP_NOT_LOADED;
    }

    LocCloudFrame frame;
    const double convert_start_steady_sec = SteadySeconds();
    frame.cloud = ConvertToBaseCloud(*msg, T_base_lidar_matrix_f, base_link_frame);
    frame.convert_ms = (SteadySeconds() - convert_start_steady_sec) * 1000.0;
    diagnostic_max_convert_ms_ = std::max(diagnostic_max_convert_ms_, frame.convert_ms);
    frame.timestamp = rclcpp::Time(msg->header.stamp).seconds();
    frame.callback_start_steady_sec = callback_start_steady_sec;
    frame.arrival_dt = arrival_dt;
    frame.raw_points = frame.cloud->size();
    frame.message_points = msg->points.size();
    frame.frame_id = msg->header.frame_id;
    frame.diagnostic = diagnostic;
    if (frame.cloud->empty()) {
        ++diagnostic_empty_after_convert_;
        LOG(ERROR) << "[Localization] Livox convert produced empty cloud"
                   << ", sequence=" << diagnostic.pipeline_sequence
                   << ", message_points=" << frame.message_points;
        return LocalizationFrameOutcome::EMPTY_AFTER_CONVERT;
    }
    if (diagnostic.pipeline_sequence <= 20 || diagnostic.pipeline_sequence % 100 == 0) {
        LOG(INFO) << std::setprecision(15)
                  << "[在线定位输入诊断][Localization] Livox转换完成"
                  << ", sequence=" << diagnostic.pipeline_sequence
                  << ", topic_sequence=" << diagnostic.topic_sequence
                  << ", header_stamp=" << frame.timestamp
                  << ", arrival_dt_ms=" << frame.arrival_dt * 1000.0
                  << ", message_points=" << frame.message_points
                  << ", converted_points=" << frame.raw_points
                  << ", convert_ms=" << frame.convert_ms
                  << ", frame_id=" << frame.frame_id;
    }
    return HandleCloudFrame(frame);
}

LocalizationFrameOutcome Localization::HandleCloudFrame(const LocCloudFrame& frame) {
    {
        std::lock_guard<std::mutex> cloud_lock(current_cloud_mutex_);
        latest_cloud_ = frame.cloud;
        latest_cloud_timestamp_ = frame.timestamp;
        latest_cloud_diagnostic_ = frame.diagnostic;
    }

    bool initialized = false;
    bool has_init_pose = false;
    bool initializing = false;
    {
        UL lock(global_mutex_);
        initialized = localization_inited_;
        has_init_pose = has_pending_initial_pose_;
        initializing = init_in_progress_;
    }

    if (!initialized) {
        if (has_init_pose && !initializing) {
            const double init_begin = SteadySeconds();
            const bool initialized_now = TryInitializeWithCurrentCloud();
            const double init_ms = (SteadySeconds() - init_begin) * 1000.0;
            if (initialized_now) {
                ++diagnostic_initialized_frames_;
                LOG(INFO) << std::setprecision(15)
                          << "[Localization][Initialization] current frame initialized"
                          << ", sequence=" << frame.diagnostic.pipeline_sequence
                          << ", topic_sequence=" << frame.diagnostic.topic_sequence
                          << ", header_stamp=" << frame.timestamp
                          << ", points=" << frame.cloud->size()
                          << ", init_ms=" << init_ms;
                return LocalizationFrameOutcome::INITIALIZED_WITH_FRAME;
            }
        }
        ++diagnostic_waiting_initial_pose_;
        return LocalizationFrameOutcome::WAITING_INITIAL_POSE;
    }

    return ProcessLocalizationCloud(frame);
}

LocalizationFrameOutcome Localization::ProcessLocalizationCloud(const LocCloudFrame& frame) {
    XYZCloud::Ptr current_cloud(new XYZCloud);

    const double voxel_start_steady_sec = SteadySeconds();
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setLeafSize(2.0f, 2.0f, 2.0f);
    voxel_grid.setInputCloud(frame.cloud);
    voxel_grid.filter(*current_cloud);
    const double voxel_ms = (SteadySeconds() - voxel_start_steady_sec) * 1000.0;
    diagnostic_max_voxel_ms_ = std::max(diagnostic_max_voxel_ms_, voxel_ms);

    if (current_cloud->empty()) {
        ++diagnostic_empty_after_voxel_;
        LOG(ERROR) << "[Localization] voxel filter produced empty cloud"
                   << ", sequence=" << frame.diagnostic.pipeline_sequence
                   << ", input_points=" << frame.cloud->size();
        return LocalizationFrameOutcome::EMPTY_AFTER_VOXEL;
    }

    std::lock_guard<std::mutex> loc_lock(localizer_mutex_);

    Eigen::Matrix4d previous_pose = Eigen::Matrix4d::Identity();
    bool has_previous_pose = false;
    double previous_timestamp = 0.0;
    {
        UL lock(global_mutex_);
        if (!localization_inited_ || init_in_progress_) {
            ++diagnostic_state_not_ready_;
            return LocalizationFrameOutcome::STATE_NOT_READY;
        }
        previous_pose = latest_pose_;
        has_previous_pose = has_last_processed_cloud_timestamp_;
        previous_timestamp = last_processed_cloud_timestamp_;
    }

    const double ndt_begin_steady_sec = SteadySeconds();
    const double topic_to_ndt_ms = frame.diagnostic.topic_receive_steady_sec > 0.0
        ? (ndt_begin_steady_sec - frame.diagnostic.topic_receive_steady_sec) * 1000.0
        : 0.0;
    diagnostic_max_topic_to_ndt_ms_ = std::max(diagnostic_max_topic_to_ndt_ms_, topic_to_ndt_ms);

    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    XYZCloud::Ptr cloud_reg(new XYZCloud);
    robot_localizer::LocalizationQuality quality;

    // After the one-time absolute initialization, NDT keeps its own
    // last/last-last pose history as the next-frame initial guess. EKF output
    // is deliberately not fed back here; NDT is an independent observation
    // source for the fusion filter.
    const bool reliable = localizer_.RegisterFrame(
        current_cloud, cloud_reg, pose, quality,
        frame.diagnostic.pipeline_sequence, frame.timestamp);
    const double ndt_ms = (SteadySeconds() - ndt_begin_steady_sec) * 1000.0;
    diagnostic_max_ndt_ms_ = std::max(diagnostic_max_ndt_ms_, ndt_ms);
    ++diagnostic_ndt_frames_;

    LocalizationResult res;
    res.timestamp_ = frame.timestamp;
    res.valid_ = true;
    res.localization_valid_ = reliable;
    res.pose_ = Matrix4dToSE3(pose);
    res.confidence_ = quality.transform_probability;
    res.status_ = reliable ? LocalizationStatus::GOOD : LocalizationStatus::FAIL;
    res.reliable_ = quality.is_reliable;
    res.tp_ = quality.transform_probability;
    res.nvtl_ = quality.nearest_voxel_likelihood;
    res.iterations_ = quality.iteration_num;
    res.pose_covariance_ = quality.pose_covariance;
    res.covariance_valid_ = quality.covariance_valid;
    res.message_ = reliable ? "localization reliable" : "localization not reliable";

    const double header_dt = has_previous_pose ? frame.timestamp - previous_timestamp : 0.0;
    if (has_previous_pose && header_dt <= 0.0) {
        ++diagnostic_non_monotonic_header_;
    }
    const Eigen::Vector3d translation_delta =
        pose.block<3, 1>(0, 3) - previous_pose.block<3, 1>(0, 3);
    const double translation_jump = translation_delta.norm();
    const Eigen::Matrix3d relative_rotation =
        previous_pose.block<3, 3>(0, 0).transpose() * pose.block<3, 3>(0, 0);
    const double yaw_jump = std::atan2(relative_rotation(1, 0), relative_rotation(0, 0));

    {
        UL lock(global_mutex_);
        latest_pose_ = pose;
        last_processed_cloud_timestamp_ = frame.timestamp;
        has_last_processed_cloud_timestamp_ = true;
    }

    const double total_ms = (SteadySeconds() - frame.callback_start_steady_sec) * 1000.0;
    if (frame.diagnostic.pipeline_sequence <= 20 ||
        frame.diagnostic.pipeline_sequence % 20 == 0 ||
        header_dt <= 0.0 || translation_jump > 1.0 || std::abs(yaw_jump) > 0.35) {
        LOG(INFO) << std::setprecision(15)
                  << "[Localization][NDT] frame result"
                  << ", sequence=" << frame.diagnostic.pipeline_sequence
                  << ", topic_sequence=" << frame.diagnostic.topic_sequence
                  << ", online=" << frame.diagnostic.online
                  << ", header_stamp=" << frame.timestamp
                  << ", header_dt=" << header_dt
                  << ", arrival_dt_ms=" << frame.arrival_dt * 1000.0
                  << ", message_points=" << frame.message_points
                  << ", converted_points=" << frame.raw_points
                  << ", voxel_points=" << current_cloud->size()
                  << ", convert_ms=" << frame.convert_ms
                  << ", voxel_ms=" << voxel_ms
                  << ", topic_to_ndt_ms=" << topic_to_ndt_ms
                  << ", ndt_ms=" << ndt_ms
                  << ", total_ms=" << total_ms
                  << ", pose_x=" << pose(0, 3)
                  << ", pose_y=" << pose(1, 3)
                  << ", pose_z=" << pose(2, 3)
                  << ", translation_jump=" << translation_jump
                  << ", yaw_jump_rad=" << yaw_jump
                  << ", TP=" << quality.transform_probability
                  << ", NVTL=" << quality.nearest_voxel_likelihood
                  << ", iterations=" << quality.iteration_num
                  << ", reliable=" << quality.is_reliable;
    }
    PublishResult(res);
    return LocalizationFrameOutcome::NDT_EXECUTED;
}

bool Localization::TryInitializeWithCurrentCloud() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;
    double timestamp = 0.0;
    LocalizationInputDiagnostic diagnostic;
    {
        std::lock_guard<std::mutex> cloud_lock(current_cloud_mutex_);
        input_cloud = latest_cloud_;
        timestamp = latest_cloud_timestamp_;
        diagnostic = latest_cloud_diagnostic_;
    }

    std::lock_guard<std::mutex> loc_lock(localizer_mutex_);

    Eigen::Matrix4d init_guess = Eigen::Matrix4d::Identity();
    {
        UL lock(global_mutex_);
        if (!map_loaded_ ||
            !has_pending_initial_pose_ ||
            localization_inited_ ||
            init_in_progress_) {
            return false;
        }
        init_in_progress_ = true;
        init_guess = pending_initial_pose_;
    }

    Eigen::Matrix4d aligned_pose = Eigen::Matrix4d::Identity();
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(
        new pcl::PointCloud<pcl::PointXYZ>());
    robot_localizer::LocalizationQuality quality;

    const double global_init_begin = SteadySeconds();
    localizer_.GetInitPose(
        init_guess, aligned_pose, input_cloud, output_cloud, quality);
    const double global_init_ms = (SteadySeconds() - global_init_begin) * 1000.0;
    LOG(INFO) << std::setprecision(15)
              << "[在线定位输入诊断][Initialization] 全局定位算法返回"
              << ", sequence=" << diagnostic.pipeline_sequence
              << ", topic_sequence=" << diagnostic.topic_sequence
              << ", header_stamp=" << timestamp
              << ", input_points=" << input_cloud->size()
              << ", output_points=" << output_cloud->size()
              << ", global_init_ms=" << global_init_ms
              << ", pose_x=" << aligned_pose(0, 3)
              << ", pose_y=" << aligned_pose(1, 3)
              << ", pose_z=" << aligned_pose(2, 3)
              << ", TP=" << quality.transform_probability
              << ", NVTL=" << quality.nearest_voxel_likelihood
              << ", iterations=" << quality.iteration_num
              << ", reliable=" << quality.is_reliable;

    {
        UL lock(global_mutex_);
        latest_pose_ = aligned_pose;
        localization_inited_ = true;
        has_pending_initial_pose_ = false;
        init_in_progress_ = false;
    }

    LocalizationResult res;
    res.timestamp_ = timestamp;
    res.valid_ = true;
    res.localization_valid_ = quality.is_reliable;
    res.pose_ = Matrix4dToSE3(aligned_pose);
    res.confidence_ = quality.transform_probability;
    res.status_ = quality.is_reliable ? LocalizationStatus::GOOD : LocalizationStatus::FAIL;
    res.reliable_ = quality.is_reliable;
    res.tp_ = quality.transform_probability;
    res.nvtl_ = quality.nearest_voxel_likelihood;
    res.iterations_ = quality.iteration_num;
    res.pose_covariance_ = quality.pose_covariance;
    res.covariance_valid_ = quality.covariance_valid;
    res.message_ = "localization initialized: " + quality.quality_level;

    PublishResult(res);
    return true;
}

void Localization::Finish() {
    LOG(INFO) << "[Localization::Finish] begin"
              << ", this=" << this
              << ", thread_id=" << std::this_thread::get_id()
              << ", ui=" << ui_.get()
              << ", input_frames=" << diagnostic_input_frames_
              << ", map_not_loaded=" << diagnostic_map_not_loaded_
              << ", empty_after_convert=" << diagnostic_empty_after_convert_
              << ", waiting_initial_pose=" << diagnostic_waiting_initial_pose_
              << ", initialized_frames=" << diagnostic_initialized_frames_
              << ", empty_after_voxel=" << diagnostic_empty_after_voxel_
              << ", state_not_ready=" << diagnostic_state_not_ready_
              << ", ndt_frames=" << diagnostic_ndt_frames_
              << ", non_monotonic_header=" << diagnostic_non_monotonic_header_
              << ", max_convert_ms=" << diagnostic_max_convert_ms_
              << ", max_voxel_ms=" << diagnostic_max_voxel_ms_
              << ", max_ndt_ms=" << diagnostic_max_ndt_ms_
              << ", max_topic_to_ndt_ms=" << diagnostic_max_topic_to_ndt_ms_;
    if (ui_) {
        LOG(INFO) << "[定位析构诊断][Localization::Finish][02] 调用 PangolinWindow::Quit";
        ui_->Quit();
        LOG(INFO) << "[Localization::Finish] UI quit returned";
        LOG(INFO) << "[定位析构诊断][Localization::Finish][04] 准备 reset PangolinWindow";
        ui_.reset();
        LOG(INFO) << "[定位析构诊断][Localization::Finish][05] PangolinWindow 已 reset";
    }

    {
        std::lock_guard<std::mutex> loc_lock(localizer_mutex_);
        localizer_.MapReset();
    }

    {
        UL lock(global_mutex_);
        map_loaded_ = false;
        localization_inited_ = false;
        has_pending_initial_pose_ = false;
        init_in_progress_ = false;
        latest_cloud_timestamp_ = 0.0;
        latest_cloud_diagnostic_ = LocalizationInputDiagnostic();
        has_last_processed_cloud_timestamp_ = false;
        last_processed_cloud_timestamp_ = 0.0;
        has_last_callback_start_steady_sec_ = false;
        last_callback_start_steady_sec_ = 0.0;
    }
    {
        std::lock_guard<std::mutex> cloud_lock(current_cloud_mutex_);
        latest_cloud_.reset();
    }
    LOG(INFO) << "[定位析构诊断][Localization::Finish][06] 完成";
}

bool Localization::SetExternalPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
    Eigen::Matrix4d init_guess = Eigen::Matrix4d::Identity();
    init_guess.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    init_guess.block<3, 1>(0, 3) = t;

    {
        std::lock_guard<std::mutex> cloud_lock(current_cloud_mutex_);
        latest_cloud_.reset();
        latest_cloud_timestamp_ = 0.0;
        latest_cloud_diagnostic_ = LocalizationInputDiagnostic();
    }

    {
        std::lock_guard<std::mutex> loc_lock(localizer_mutex_);
        {
            UL lock(global_mutex_);
            pending_initial_pose_ = init_guess;
            has_pending_initial_pose_ = true;
            localization_inited_ = false;
            init_in_progress_ = false;
            latest_pose_ = Eigen::Matrix4d::Identity();
            has_last_processed_cloud_timestamp_ = false;
            last_processed_cloud_timestamp_ = 0.0;
        }
        localizer_.ResetLocalizationState();
    }

    {
        UL lock_result(loc_result_mutex_);
        loc_result_ = LocalizationResult();
        loc_result_.valid_ = true;
        loc_result_.status_ = LocalizationStatus::INITIALIZING;
        loc_result_.pose_ = Matrix4dToSE3(init_guess);
        loc_result_.message_ = "initial pose accepted, waiting for next cloud";
    }

    LOG(INFO) << "[SET_LOCATION] accepted new initial pose x=" << init_guess(0, 3)
              << ", y=" << init_guess(1, 3)
              << ", z=" << init_guess(2, 3);
    return true;
}

void Localization::PublishResult(const LocalizationResult& result) {
    {
        UL lock_result(loc_result_mutex_);
        loc_result_ = result;
    }

    std::string base_link_frame;
    {
        UL lock(global_mutex_);
        base_link_frame = base_link_frame_;
    }

    if (tf_callback_) {
        auto tf_msg = result.ToGeoMsg();
        tf_msg.child_frame_id = base_link_frame;
        tf_callback_(tf_msg);
    }

    if (result_callback_) {
        result_callback_(result);
    }
}

void Localization::UpdateVisualization(const LocalizationResult& result) {
    if (!ui_) return;
    ui_->UpdateNavState(result.ToNavState());
}

void Localization::UpdategpsObservationVisualization(
    const Eigen::Vector2d& position_map) {
    if (!ui_) return;
    ui_->UpdategpsPosition(position_map);
}

void Localization::UpdateNdtObservationVisualization(
    const Eigen::Vector2d& position_map) {
    if (!ui_) return;
    ui_->UpdateNdtPosition(position_map);
}

SE3 Localization::Matrix4dToSE3(const Eigen::Matrix4d& pose) {
    Mat3d rotation = pose.block<3, 3>(0, 0);
    Quatd q(rotation);
    q.normalize();
    return SE3(q, pose.block<3, 1>(0, 3));
}

void Localization::SetTFCallback(Localization::TFCallback&& callback) {
    tf_callback_ = std::move(callback);
}

void Localization::SetResultCallback(Localization::ResultCallback&& callback) {
    result_callback_ = std::move(callback);
}

void Localization::MarkPoor(const std::string& message) {
    LocalizationResult result;
    {
        UL lock_result(loc_result_mutex_);
        result = loc_result_;
    }
    result.valid_ = true;
    result.localization_valid_ = false;
    result.status_ = LocalizationStatus::FAIL;
    result.confidence_ = 0.0;
    result.reliable_ = false;
    result.tp_ = 0.0;
    result.nvtl_ = 0.0;
    result.iterations_ = 0;
    result.message_ = message.empty() ? "localization poor" : message;
    {
        UL lock_result(loc_result_mutex_);
        loc_result_ = result;
    }
}

LocalizationResult Localization::GetLatestResult() const {
    UL lock_result(loc_result_mutex_);
    return loc_result_;
}

}  // namespace lightning::loc
