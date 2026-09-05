#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>

#include "common/eigen_types.h"
#include "core/localization/ESKF/eskf.h"
#include "core/localization/localization_diagnostic.h"
#include "core/localization/localization_result.h"
#include "lightning_interfaces/msg/localization_pose.hpp"

namespace lightning::loc { class Localization; }

namespace lightning::modules {

struct LocalizationSystemOptions { bool pub_tf_ = true; };

class LocalizationSystem {
   public:
    enum class Mode { NDT_ONLY, ESKF_FUSION, RTK_ONLY, AUTO };

    explicit LocalizationSystem(LocalizationSystemOptions options = LocalizationSystemOptions());
    ~LocalizationSystem();

    bool Init(const std::string& yaml_path, rclcpp::Node::SharedPtr node = nullptr);
    bool SetMapPath(const std::string& map_path);
    bool SetInitialGuess(const SE3& init_pose, bool* initialized_now = nullptr);
    loc::LocalizationFrameOutcome ProcessCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud, const loc::LocalizationInputDiagnostic& diagnostic = {});
    loc::LocalizationFrameOutcome ProcessCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud, const loc::LocalizationInputDiagnostic& diagnostic = {});

    // Each standard ROS observation enters the filter independently. No RTK
    // synchronization packet or input history is maintained in this module.
    void ProcessRtkPosition(const sensor_msgs::msg::NavSatFix::SharedPtr& fix);
    void ProcessInsOrientation(const sensor_msgs::msg::Imu::SharedPtr& orientation);
    void ProcessInsVelocity(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& velocity);
    void ProcessWheelOdometry(const nav_msgs::msg::Odometry::SharedPtr& odometry);
    void ProcessImu(const sensor_msgs::msg::Imu::SharedPtr& imu);

    void MarkPoor(const std::string& message);
    loc::LocalizationResult GetLatestResult() const;
    void Reset();

    Mode GetMode() const { return mode_; }
    bool UsesLidar() const {
        return mode_ != Mode::RTK_ONLY &&
               (!lidar_topic_.empty() || !livox_lidar_topic_.empty());
    }
    bool UsesRtk() const {
        return mode_ != Mode::NDT_ONLY && !rtk_fix_topic_.empty();
    }
    bool UsesInsOrientation() const {
        return mode_ != Mode::NDT_ONLY && !rtk_orientation_topic_.empty();
    }
    bool UsesInsVelocity() const {
        return mode_ != Mode::NDT_ONLY && !rtk_velocity_topic_.empty();
    }
    bool UsesWheelOdometry() const {
        return mode_ != Mode::NDT_ONLY && !wheel_odometry_topic_.empty();
    }
    bool UsesImuAngularVelocity() const {
        return mode_ != Mode::NDT_ONLY && !imu_topic_.empty();
    }
    // LiDAR needs the map for NDT; the UI also needs it for visualization.
    bool RequiresMap() const { return UsesLidar() || with_ui_; }
    bool RequiresInitialGuess() const {
        return UsesLidar() && !(UsesRtk() && UsesInsOrientation());
    }
    bool ReadyWithoutMap() const { return !RequiresMap(); }
    static Mode ModeFromString(const std::string& value);
    static std::string ModeToString(Mode mode);

   private:
    void SetupPublishers(rclcpp::Node::SharedPtr node);
    void HandleNdtResult(const loc::LocalizationResult& result);
    bool InitializeFixedMapTransform(const YAML::Node& map_from_enu);
    bool PositionToMap(const sensor_msgs::msg::NavSatFix& fix,
                       Eigen::Vector3d* position_map,
                       Eigen::Matrix3d* covariance_map) const;
    bool OrientationToMapYaw(const sensor_msgs::msg::Imu& orientation,
                             Eigen::Matrix3d* rotation_map_tracking,
                             double* yaw_map,
                             double* variance) const;
    bool VelocityToMap(
        const geometry_msgs::msg::TwistWithCovarianceStamped& velocity,
        Eigen::Vector3d* velocity_map,
        Eigen::Matrix3d* covariance_map) const;
    void TryInitializeEskfFromGlobalObservations();
    void InitializeEskfFromNdt(const loc::LocalizationResult& ndt);
    void PublishPredictionIfAdvanced(double stamp, const std::string& message);
    loc::LocalizationResult BuildEskfResult(double stamp, const std::string& message, bool reliable) const;
    void PublishResult(const loc::LocalizationResult& result);
    void AppendPath(const loc::LocalizationResult& result);
    static std::string Normalize(std::string value);
    static Eigen::Vector3d ReadVector3(const YAML::Node& node, const Eigen::Vector3d& fallback);
    static Eigen::Array3i ReadAxisMask(const YAML::Node& node, const Eigen::Array3i& fallback);
    static double PoseYaw(const SE3& pose);

    LocalizationSystemOptions options_;
    Mode mode_ = Mode::NDT_ONLY;
    std::string yaml_path_;
    std::string map_path_;
    std::string base_link_frame_ = "base_link";
    std::string output_frame_ = "map";
    // The topic name is the only sensor switch: empty means disabled.
    std::string lidar_topic_;
    std::string livox_lidar_topic_;
    std::string imu_topic_;
    std::string rtk_fix_topic_;
    std::string rtk_orientation_topic_;
    std::string rtk_velocity_topic_;
    std::string wheel_odometry_topic_;
    bool with_ui_ = false;
    bool map_ready_ = false;
    bool has_initial_guess_ = false;
    bool manual_initial_guess_pending_ = false;
    double last_lidar_stamp_ = -1.0;

    std::shared_ptr<loc::Localization> loc_;
    mutable std::mutex filter_mutex_;
    loc::ESKF eskf_;

    Eigen::Array3i rtk_position_axes_ = Eigen::Array3i(1, 1, 0);
    Eigen::Array3i rtk_velocity_axes_ = Eigen::Array3i(1, 1, 1);
    Eigen::Array3i wheel_linear_axes_ = Eigen::Array3i(1, 0, 0);
    Eigen::Array3i wheel_angular_axes_ = Eigen::Array3i(0, 0, 1);
    Eigen::Array3i imu_angular_axes_ = Eigen::Array3i(1, 1, 1);
    Eigen::Matrix3d tracking_from_imu_rotation_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d imu_from_tracking_rotation_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d rtk_ins_lever_arm_tracking_ = Eigen::Vector3d::Zero();
    double rtk_position_sigma_xy_ = 0.30;
    double rtk_position_sigma_z_ = 1.00;
    double rtk_velocity_sigma_xy_ = 0.20;
    double rtk_velocity_sigma_z_ = 0.50;
    double rtk_velocity_covariance_scale_ = 2.0;
    double fallback_ins_yaw_sigma_rad_ =
        10.0 * 3.14159265358979323846 / 180.0;
    double wheel_linear_sigma_ = 0.20;
    double wheel_angular_sigma_ = 0.10;
    double imu_angular_sigma_ = 0.10;
    double initial_velocity_sigma_ = 3.0;
    double initial_angular_velocity_sigma_ = 1.0;
    double rtk_position_gate_chi2_ = 16.0;
    double rtk_velocity_gate_chi2_ = 16.0;
    double ins_yaw_gate_chi2_ = 9.0;
    double wheel_gate_chi2_ = 20.0;
    double imu_angular_gate_chi2_ = 16.0;
    double ndt_pose_gate_chi2_ = 20.0;

    int utm_zone_ = 0;
    bool map_from_enu_ready_ = false;
    Eigen::Matrix3d map_from_enu_rotation_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d map_from_enu_translation_ = Eigen::Vector3d::Zero();
    // INS attitude/velocity use true local ENU axes. UTM positions use grid
    // axes, so this fixed rotation applies the reference meridian convergence.
    Eigen::Matrix3d utm_from_true_enu_rotation_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d map_from_true_enu_rotation_ = Eigen::Matrix3d::Identity();

    // RTK-only initialization keeps one latest value per observation type. It
    // is state, not a pending-message queue.
    bool has_initial_position_ = false;
    double initial_position_stamp_ = 0.0;
    Eigen::Vector3d initial_sensor_position_map_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d initial_position_covariance_map_ = Eigen::Matrix3d::Identity();
    bool has_initial_yaw_ = false;
    double initial_yaw_stamp_ = 0.0;
    Eigen::Matrix3d initial_orientation_map_ = Eigen::Matrix3d::Identity();
    double initial_yaw_variance_ = 1.0;
    bool has_initial_velocity_ = false;
    double initial_velocity_stamp_ = 0.0;
    Eigen::Vector3d initial_sensor_velocity_map_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d initial_velocity_covariance_map_ = Eigen::Matrix3d::Identity();
    double initial_observation_max_dt_ = 0.05;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr loc_odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr loc_pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr loc_path_pub_;
    rclcpp::Publisher<lightning_interfaces::msg::LocalizationPose>::SharedPtr loc_pose_quality_pub_;
    mutable std::mutex result_mutex_;
    loc::LocalizationResult latest_result_;
    nav_msgs::msg::Path path_;
};

}  // namespace lightning::modules
