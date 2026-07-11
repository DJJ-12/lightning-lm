#include "core/localization/localization.h"

#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/time.hpp>
#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>

#include "ui/pangolin_window.h"

namespace lightning::loc {

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

    std::vector<double> base_lidar_t{0.0, 0.0, 0.0};
    std::vector<double> base_lidar_R{1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0};
    if (yaml_node["extrinsicBaseLidarTrans"]) {
        base_lidar_t = yaml_node["extrinsicBaseLidarTrans"].as<std::vector<double>>();
    } else if (yaml_node["common"] && yaml_node["common"]["extrinsicBaseLidarTrans"]) {
        base_lidar_t = yaml_node["common"]["extrinsicBaseLidarTrans"].as<std::vector<double>>();
    }
    if (yaml_node["extrinsicBaseLidarRot"]) {
        base_lidar_R = yaml_node["extrinsicBaseLidarRot"].as<std::vector<double>>();
    } else if (yaml_node["common"] && yaml_node["common"]["extrinsicBaseLidarRot"]) {
        base_lidar_R = yaml_node["common"]["extrinsicBaseLidarRot"].as<std::vector<double>>();
    }
    CHECK_EQ(base_lidar_t.size(), 3);
    CHECK_EQ(base_lidar_R.size(), 9);

    Mat3d R_base_lidar;
    R_base_lidar << base_lidar_R[0], base_lidar_R[1], base_lidar_R[2],
        base_lidar_R[3], base_lidar_R[4], base_lidar_R[5],
        base_lidar_R[6], base_lidar_R[7], base_lidar_R[8];
    Quatd q_base_lidar(R_base_lidar);
    q_base_lidar.normalize();
    options_.T_base_lidar_ =
        SE3(q_base_lidar, Vec3d(base_lidar_t[0], base_lidar_t[1], base_lidar_t[2]));
    T_base_lidar_matrix_f_ = options_.T_base_lidar_.matrix().cast<float>();
    LOG(INFO) << "[BASE_LIDAR] T_base_lidar trans="
              << options_.T_base_lidar_.translation().transpose();

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

    const std::string metadata_path =
        global_map_path + "/BlockMap/pointcloud_map_metadata.yaml";
    const std::string pcd_directory = global_map_path + "/BlockMap/pointcloud_map";

    localizer_.MapReset();
    localizer_.SetStaticMap(metadata_path, pcd_directory);
    localizer_.SetQualityThresholds(quality_thresholds_);
    localizer_.ResetLocalizationState();

    map_loaded_ = true;
    localization_inited_ = false;
    has_pending_initial_pose_ = false;
    init_in_progress_ = false;
    pending_initial_pose_ = Eigen::Matrix4d::Identity();
    latest_pose_ = Eigen::Matrix4d::Identity();
    latest_cloud_timestamp_ = 0.0;
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
        ui_->Init();
    }

    return true;
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

void Localization::ProcessLidarMsg(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    bool map_loaded = false;
    Mat4f T_base_lidar_matrix_f;
    std::string base_link_frame;
    {
        UL lock(global_mutex_);
        map_loaded = map_loaded_;
        T_base_lidar_matrix_f = T_base_lidar_matrix_f_;
        base_link_frame = base_link_frame_;
    }

    if (!map_loaded) {
        return;
    }

    LocCloudFrame frame;
    frame.cloud = ConvertToBaseCloud(*msg, T_base_lidar_matrix_f, base_link_frame);
    frame.timestamp = rclcpp::Time(msg->header.stamp).seconds();
    HandleCloudFrame(frame);
}

void Localization::ProcessLivoxLidarMsg(
    const livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
    bool map_loaded = false;
    Mat4f T_base_lidar_matrix_f;
    std::string base_link_frame;
    {
        UL lock(global_mutex_);
        map_loaded = map_loaded_;
        T_base_lidar_matrix_f = T_base_lidar_matrix_f_;
        base_link_frame = base_link_frame_;
    }

    if (!map_loaded) {
        return;
    }

    LocCloudFrame frame;
    frame.cloud = ConvertToBaseCloud(*msg, T_base_lidar_matrix_f, base_link_frame);
    frame.timestamp = rclcpp::Time(msg->header.stamp).seconds();
    HandleCloudFrame(frame);
}

void Localization::HandleCloudFrame(const LocCloudFrame& frame) {
    if (!frame.cloud || frame.cloud->empty()) {
        return;
    }

    {
        std::lock_guard<std::mutex> cloud_lock(current_cloud_mutex_);
        latest_cloud_ = frame.cloud;
        latest_cloud_timestamp_ = frame.timestamp;
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
            TryInitializeWithCurrentCloud();
        }
        return;
    }

    if (initializing) {
        return;
    }

    ProcessLocalizationCloud(frame);
}

void Localization::ProcessLocalizationCloud(const LocCloudFrame& frame) {
    if (!frame.cloud || frame.cloud->empty()) {
        return;
    }

    auto current_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setLeafSize(2.0f, 2.0f, 2.0f);
    voxel_grid.setInputCloud(frame.cloud);
    voxel_grid.filter(*current_cloud);

    if (!current_cloud || current_cloud->empty()) {
        return;
    }

    std::lock_guard<std::mutex> loc_lock(localizer_mutex_);

    {
        UL lock(global_mutex_);
        if (!localization_inited_ || init_in_progress_) {
            return;
        }
    }

    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    auto cloud_reg = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    robot_localizer::LocalizationQuality quality;

    const bool reliable = localizer_.RegisterFrame(current_cloud, cloud_reg, pose, quality);

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
    res.message_ = reliable ? "localization reliable" : "localization not reliable";

    {
        UL lock(global_mutex_);
        latest_pose_ = pose;
    }

    PublishResult(res);
}

bool Localization::TryInitializeWithCurrentCloud() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;
    double timestamp = 0.0;
    {
        std::lock_guard<std::mutex> cloud_lock(current_cloud_mutex_);
        if (!latest_cloud_ || latest_cloud_->empty()) {
            return false;
        }
        input_cloud = latest_cloud_;
        timestamp = latest_cloud_timestamp_;
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

    localizer_.GetInitPose(
        init_guess, aligned_pose, input_cloud, output_cloud, quality);

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
    res.message_ = "localization initialized: " + quality.quality_level;

    {
        UL lock_result(loc_result_mutex_);
        loc_result_ = res;
    }
    return true;
}

void Localization::Finish() {
    if (ui_) {
        ui_->Quit();
        ui_.reset();
    }

    {
        UL lock(global_mutex_);
        map_loaded_ = false;
        localization_inited_ = false;
        has_pending_initial_pose_ = false;
        init_in_progress_ = false;
        latest_cloud_timestamp_ = 0.0;
    }
    {
        std::lock_guard<std::mutex> cloud_lock(current_cloud_mutex_);
        latest_cloud_.reset();
    }
}

bool Localization::SetExternalPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
    Eigen::Matrix4d init_guess = Eigen::Matrix4d::Identity();
    init_guess.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    init_guess.block<3, 1>(0, 3) = t;

    {
        std::lock_guard<std::mutex> loc_lock(localizer_mutex_);
        {
            UL lock(global_mutex_);
            pending_initial_pose_ = init_guess;
            has_pending_initial_pose_ = true;
            localization_inited_ = false;
            init_in_progress_ = false;
            latest_pose_ = Eigen::Matrix4d::Identity();
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
    return TryInitializeWithCurrentCloud();
}

void Localization::PublishResult(const LocalizationResult& result) {
    {
        UL lock_result(loc_result_mutex_);
        loc_result_ = result;
    }

    if (ui_ && result.valid_) {
        ui_->UpdateNavState(result.ToNavState());
        ui_->UpdateRecentPose(result.pose_);
    }

    const bool pose_is_publishable = result.valid_;

    std::string base_link_frame;
    {
        UL lock(global_mutex_);
        base_link_frame = base_link_frame_;
    }

    if (pose_is_publishable && tf_callback_) {
        auto tf_msg = result.ToGeoMsg();
        tf_msg.child_frame_id = base_link_frame;
        tf_callback_(tf_msg);
    }

    if (result_callback_) {
        result_callback_(result);
    }
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
