#pragma once

#include <cstddef>
#include <mutex>
#include <functional>
#include <memory>
#include <string>
#include <cstdint>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "common/eigen_types.h"
#include "common/std_types.h"
#include "core/localization/GlobalLocalizer/GlobalLocalizer.h"
#include "core/localization/localization_diagnostic.h"
#include "core/localization/localization_result.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace lightning {
namespace ui {
class PangolinWindow;
}

namespace loc {

class Localization {
   public:
    struct Options {
        bool with_ui_ = false;
        SE3 T_base_lidar_ = SE3();
        bool pub_tf_ = false;
    };

    explicit Localization(Options options);
    ~Localization() = default;

    bool Init(const std::string& yaml_path, const std::string& global_map_path);

    LocalizationFrameOutcome ProcessLidarMsg(
        const sensor_msgs::msg::PointCloud2::SharedPtr laser_msg,
        const LocalizationInputDiagnostic& diagnostic = {});
    LocalizationFrameOutcome ProcessLivoxLidarMsg(
        const livox_ros_driver2::msg::CustomMsg::SharedPtr laser_msg,
        const LocalizationInputDiagnostic& diagnostic = {});

    bool SetExternalPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t);
    void Finish();

    using TFCallback = std::function<void(const geometry_msgs::msg::TransformStamped& odom)>;
    using ResultCallback = std::function<void(const LocalizationResult& result)>;
    using XYZCloud = pcl::PointCloud<pcl::PointXYZ>;

    void SetTFCallback(TFCallback&& callback);
    void SetResultCallback(ResultCallback&& callback);
    // The localization system owns the final estimator output. NDT results
    // reach it through ResultCallback; only the selected final result is sent
    // back here for visualization.
    void UpdateVisualization(const LocalizationResult& result);
    // Draws the pre-filter gps antenna position after WGS84/UTM/ENU -> map
    // conversion. This is visualization only and never changes localization.
    void UpdategpsObservationVisualization(
        const Eigen::Vector2d& position_map);
    // Draws every valid raw NDT output. This path is independent from EKF
    // acceptance and remains available in LiDAR-only mode.
    void UpdateNdtObservationVisualization(
        const Eigen::Vector2d& position_map);
    void MarkPoor(const std::string& message);
    LocalizationResult GetLatestResult() const;

   private:
    struct LocCloudFrame {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = nullptr;
        double timestamp = 0.0;
        double callback_start_steady_sec = 0.0;
        double arrival_dt = 0.0;
        double convert_ms = 0.0;
        size_t raw_points = 0;
        size_t message_points = 0;
        std::string frame_id;
        LocalizationInputDiagnostic diagnostic;
    };

    bool TryInitializeWithCurrentCloud();
    pcl::PointCloud<pcl::PointXYZ>::Ptr ConvertToBaseCloud(
        const sensor_msgs::msg::PointCloud2& msg,
        const Mat4f& T_base_lidar_matrix_f,
        const std::string& base_link_frame) const;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ConvertToBaseCloud(
        const livox_ros_driver2::msg::CustomMsg& msg,
        const Mat4f& T_base_lidar_matrix_f,
        const std::string& base_link_frame) const;
    LocalizationFrameOutcome HandleCloudFrame(const LocCloudFrame& frame);
    LocalizationFrameOutcome ProcessLocalizationCloud(const LocCloudFrame& frame);
    void PublishResult(const LocalizationResult& result);
    void LoadTargetMapForUI(const std::string& global_map_path);
    static SE3 Matrix4dToSE3(const Eigen::Matrix4d& pose);

    std::mutex global_mutex_;
    std::mutex localizer_mutex_;
    Options options_;

    robot_localizer::Localizer localizer_;

    bool map_loaded_ = false;
    bool localization_inited_ = false;
    bool has_pending_initial_pose_ = false;
    bool init_in_progress_ = false;

    Eigen::Matrix4d pending_initial_pose_ = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d latest_pose_ = Eigen::Matrix4d::Identity();
    Mat4f T_base_lidar_matrix_f_ = Mat4f::Identity();
    pcl::PointCloud<pcl::PointXYZ>::Ptr latest_cloud_ = nullptr;
    double latest_cloud_timestamp_ = 0.0;
    LocalizationInputDiagnostic latest_cloud_diagnostic_;
    bool has_last_processed_cloud_timestamp_ = false;
    double last_processed_cloud_timestamp_ = 0.0;
    bool has_last_callback_start_steady_sec_ = false;
    double last_callback_start_steady_sec_ = 0.0;

    std::mutex current_cloud_mutex_;

    robot_localizer::QualityThresholds quality_thresholds_;

    LocalizationResult loc_result_;
    mutable std::mutex loc_result_mutex_;

    TFCallback tf_callback_;
    ResultCallback result_callback_;
    std::shared_ptr<ui::PangolinWindow> ui_ = nullptr;

    std::string base_link_frame_ = "base_link";

    std::uint64_t diagnostic_input_frames_ = 0;
    std::uint64_t diagnostic_map_not_loaded_ = 0;
    std::uint64_t diagnostic_empty_after_convert_ = 0;
    std::uint64_t diagnostic_waiting_initial_pose_ = 0;
    std::uint64_t diagnostic_initialized_frames_ = 0;
    std::uint64_t diagnostic_empty_after_voxel_ = 0;
    std::uint64_t diagnostic_state_not_ready_ = 0;
    std::uint64_t diagnostic_ndt_frames_ = 0;
    std::uint64_t diagnostic_non_monotonic_header_ = 0;
    double diagnostic_max_convert_ms_ = 0.0;
    double diagnostic_max_voxel_ms_ = 0.0;
    double diagnostic_max_ndt_ms_ = 0.0;
    double diagnostic_max_topic_to_ndt_ms_ = 0.0;
};

}  // namespace loc
}  // namespace lightning
