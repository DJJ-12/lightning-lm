#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/StdVector>

#include "common/eigen_types.h"
#include "common/localization_sensor_measurements.h"
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

    // RTK position comes from NavSatFix; RTK heading comes from Pose2D.theta.
    void ProcessRtkIns(const RtkInsMeasurement& measurement);
    void ProcessWheelOdometry(const WheelOdometryMeasurement& measurement);
    void ProcessImu(const sensor_msgs::msg::Imu::SharedPtr& imu);

    void MarkPoor(const std::string& message);
    loc::LocalizationResult GetLatestResult() const;
    void Reset();

    Mode GetMode() const { return mode_; }
    bool UsesLidar() const { return mode_ != Mode::RTK_ONLY; }
    bool UsesRtk() const { return mode_ != Mode::NDT_ONLY && use_rtk_position_; }
    bool UsesWheelOdometry() const { return mode_ != Mode::NDT_ONLY && use_wheel_odometry_; }
    bool UsesImuAngularVelocity() const { return mode_ != Mode::NDT_ONLY && use_imu_angular_velocity_; }
    bool RequiresMap() const { return UsesLidar(); }
    bool RequiresInitialGuess() const { return UsesLidar(); }
    bool ReadyWithoutMap() const { return mode_ == Mode::RTK_ONLY; }
    static Mode ModeFromString(const std::string& value);
    static std::string ModeToString(Mode mode);

   private:
    struct MapAlignmentPair {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        double rtk_stamp = 0.0;
        Eigen::Vector3d position_enu = Eigen::Vector3d::Zero();
        Eigen::Vector3d position_map = Eigen::Vector3d::Zero();
    };

    void SetupPublishers(rclcpp::Node::SharedPtr node);
    void HandleNdtResult(const loc::LocalizationResult& result);
    void UpdateFromRtkIns(const RtkInsMeasurement& measurement);
    bool TransformRtkInsToFilterFrame(const RtkInsMeasurement& measurement, Eigen::Vector3d* position, Eigen::Vector3d* velocity, Eigen::Matrix3d* position_covariance, Eigen::Matrix3d* velocity_covariance) const;
    bool FindClosestRtk(double stamp, RtkInsMeasurement* measurement) const;
    bool TryInitializeMapFromEnu(const loc::LocalizationResult& ndt);
    bool SolveMapFromEnuAlignment();
    void InitializeEskfFromNdt(const loc::LocalizationResult& ndt);
    void InitializeEskfFromRtkIns(const RtkInsMeasurement& measurement);
    void ApplyNdtDerivedTwist(const loc::LocalizationResult& ndt);
    loc::LocalizationResult BuildEskfResult(double stamp, const std::string& message, bool reliable) const;
    void PublishResult(const loc::LocalizationResult& result);
    void AppendPath(const loc::LocalizationResult& result);
    static std::string Normalize(std::string value);
    static Eigen::Vector3d ReadVector3(const YAML::Node& node, const Eigen::Vector3d& fallback);
    static Eigen::Array3i ReadAxisMask(const YAML::Node& node, const Eigen::Array3i& fallback);
    static double WrapAngle(double angle);
    static double PoseYaw(const SE3& pose);
    static SE3 PoseFromPositionYaw(const Eigen::Vector3d& position, double yaw, const Eigen::Matrix3d& roll_pitch_hint);

    LocalizationSystemOptions options_;
    Mode mode_ = Mode::NDT_ONLY;
    std::string yaml_path_;
    std::string map_path_;
    std::string base_link_frame_ = "base_link";
    std::string output_frame_ = "map";
    bool map_ready_ = false;
    bool has_initial_guess_ = false;
    double last_lidar_stamp_ = -1.0;

    std::shared_ptr<loc::Localization> loc_;
    mutable std::mutex filter_mutex_;
    loc::ESKF eskf_;

    bool use_rtk_position_ = true;
    bool use_rtk_velocity_ = false;
    bool use_rtk_elevation_ = false;
    bool use_course_yaw_ = false;
    bool use_wheel_odometry_ = false;
    bool use_imu_angular_velocity_ = false;
    bool use_ndt_derived_twist_ = true;
    Eigen::Array3i rtk_position_axes_ = Eigen::Array3i(1, 1, 0);
    Eigen::Array3i rtk_velocity_axes_ = Eigen::Array3i(1, 1, 1);
    Eigen::Array3i wheel_linear_axes_ = Eigen::Array3i(1, 0, 0);
    Eigen::Array3i wheel_angular_axes_ = Eigen::Array3i(0, 0, 1);
    Eigen::Array3i imu_angular_axes_ = Eigen::Array3i(1, 1, 1);
    Eigen::Vector3d rtk_ins_lever_arm_tracking_ = Eigen::Vector3d::Zero();
    double rtk_position_sigma_xy_ = 0.30;
    double rtk_position_sigma_z_ = 1.00;
    double rtk_velocity_sigma_xy_ = 0.20;
    double rtk_velocity_sigma_z_ = 0.50;
    double rtk_velocity_covariance_scale_ = 2.0;
    double course_yaw_min_speed_ = 2.0;
    double course_yaw_sigma_rad_ = 10.0 * 3.14159265358979323846 / 180.0;
    double wheel_linear_sigma_ = 0.20;
    double wheel_angular_sigma_ = 0.10;
    double imu_angular_sigma_ = 0.10;
    double ndt_twist_covariance_scale_ = 10.0;
    double initial_velocity_sigma_ = 3.0;
    double initial_angular_velocity_sigma_ = 1.0;
    double rtk_position_gate_chi2_ = 16.0;
    double rtk_velocity_gate_chi2_ = 16.0;
    double course_yaw_gate_chi2_ = 9.0;
    double wheel_gate_chi2_ = 20.0;
    double imu_angular_gate_chi2_ = 16.0;
    double ndt_pose_gate_chi2_ = 20.0;
    double ndt_twist_gate_chi2_ = 20.0;

    std::deque<RtkInsMeasurement> rtk_history_;
    std::size_t rtk_history_limit_ = 300;
    std::vector<MapAlignmentPair, Eigen::aligned_allocator<MapAlignmentPair>> map_alignment_pairs_;
    double last_alignment_rtk_stamp_ = -1.0;
    double map_alignment_max_age_ = 0.30;
    int map_alignment_min_pairs_ = 8;
    double map_alignment_min_baseline_ = 5.0;
    double map_alignment_max_rmse_ = 1.5;
    std::string map_from_enu_mode_ = "trajectory_alignment";
    bool map_from_enu_ready_ = false;
    Eigen::Matrix3d map_from_enu_rotation_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d map_from_enu_translation_ = Eigen::Vector3d::Zero();

    bool has_previous_ndt_ = false;
    loc::LocalizationResult previous_ndt_;

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
