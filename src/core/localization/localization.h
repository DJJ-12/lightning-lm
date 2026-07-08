#pragma once

#include <mutex>
#include <functional>
#include <memory>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/std_types.h"
#include "core/localization/GlobalLocalizer/GlobalLocalizer.h"
#include "core/localization/localization_result.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/int32.hpp"

namespace lightning {
namespace ui {
class PangolinWindow;
}

namespace loc {

class Localization {
   public:
    struct Options {
        bool online_mode_ = false;
        bool with_ui_ = false;
        SE3 T_base_lidar_ = SE3();
        bool pub_tf_ = false;
        
    };

    explicit Localization(Options options);
    ~Localization() = default;

    bool Init(const std::string& yaml_path, const std::string& global_map_path);

    void ProcessLidarMsg(const sensor_msgs::msg::PointCloud2::SharedPtr laser_msg);
    void ProcessLivoxLidarMsg(const livox_ros_driver2::msg::CustomMsg::SharedPtr laser_msg);
    void ProcessIMUMsg(IMUPtr imu);

    bool SetExternalPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t);
    void Finish();

    using TFCallback = std::function<void(const geometry_msgs::msg::TransformStamped& odom)>;
    using LocStateCallback = std::function<void(const std_msgs::msg::Int32& state)>;
    using ResultCallback = std::function<void(const LocalizationResult& result)>;

    void SetTFCallback(TFCallback&& callback);
    void SetLocStateCallback(LocStateCallback&& callback);
    void SetResultCallback(ResultCallback&& callback);
    LocalizationResult GetLatestResult() const;

   private:
    struct LocCloudFrame {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = nullptr;
        double timestamp = 0.0;
    };

    bool TryInitializeWithCurrentCloud();
    pcl::PointCloud<pcl::PointXYZ>::Ptr ConvertToBaseCloud(
        const sensor_msgs::msg::PointCloud2& msg,
        const SE3& T_base_lidar,
        const std::string& base_link_frame) const;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ConvertToBaseCloud(
        const livox_ros_driver2::msg::CustomMsg& msg,
        const SE3& T_base_lidar,
        const std::string& base_link_frame) const;
    void HandleCloudFrame(const LocCloudFrame& frame);
    void LidarLocProcCloud(const LocCloudFrame& frame);
    void PublishResult(const LocalizationResult& result);
    static SE3 Matrix4dToSE3(const Eigen::Matrix4d& pose);

    std::mutex global_mutex_;
    std::mutex lidar_loc_mutex_;
    Options options_;

    robot_localizer::Localizer lidar_loc_;

    bool map_loaded_ = false;
    bool lidar_loc_inited_ = false;
    bool has_pending_initial_pose_ = false;
    bool init_in_progress_ = false;

    Eigen::Matrix4d pending_initial_pose_ = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d latest_pose_ = Eigen::Matrix4d::Identity();
    pcl::PointCloud<pcl::PointXYZ>::Ptr latest_cloud_ = nullptr;
    double latest_cloud_timestamp_ = 0.0;

    std::mutex current_cloud_mutex_;

    robot_localizer::QualityThresholds quality_thresholds_;

    LocalizationResult loc_result_;
    mutable std::mutex loc_result_mutex_;

    TFCallback tf_callback_;
    LocStateCallback loc_state_callback_;
    ResultCallback result_callback_;
    std::shared_ptr<ui::PangolinWindow> ui_ = nullptr;

    std::string base_link_frame_ = "base_link";
};

}  // namespace loc
}  // namespace lightning
