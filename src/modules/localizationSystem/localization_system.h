#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <Eigen/Geometry>
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
#include "core/lightning_math.hpp"
#include "core/localization/EKF/ekf.h"
#include "core/localization/localization_diagnostic.h"
#include "core/localization/localization_result.h"
#include "lightning_interfaces/msg/localization_pose.hpp"

namespace lightning::loc {
class Localization;
}

namespace lightning::modules {

struct LocalizationSystemOptions {
    bool pub_tf_ = true;
};

class LocalizationSystem {
   public:
    enum class Mode { NDT_ONLY, EKF_FUSION, gps_ONLY, AUTO };

    explicit LocalizationSystem(
        LocalizationSystemOptions options = LocalizationSystemOptions());
    ~LocalizationSystem();

    bool Init(const std::string& yaml_path,
              rclcpp::Node::SharedPtr node = nullptr);
    bool SetMapPath(const std::string& map_path);
    bool SetInitialGuess(const SE3& init_pose,
                         bool* initialized_now = nullptr);

    loc::LocalizationFrameOutcome ProcessCloud(
        const sensor_msgs::msg::PointCloud2::SharedPtr& cloud,
        const loc::LocalizationInputDiagnostic& diagnostic = {});
    loc::LocalizationFrameOutcome ProcessCloud(
        const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud,
        const loc::LocalizationInputDiagnostic& diagnostic = {});

    void ProcessGps(const sensor_msgs::msg::NavSatFix::SharedPtr& fix);
    void ProcessGpsOrientation(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& orientation);
    void ProcessgpsVelocity(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& velocity);
    void ProcessWheelOdometry(
        const nav_msgs::msg::Odometry::SharedPtr& odometry);
    void ProcessImu(const sensor_msgs::msg::Imu::SharedPtr& imu);

    void MarkPoor(const std::string& message);
    loc::LocalizationResult GetLatestResult() const;
    void Reset();

    Mode GetMode() const { return mode_; }
    bool UsesLidar() const {
        return !lidar_topic_.empty() || !livox_lidar_topic_.empty();
    }
    bool UsesGpsPosition() const { return !gps_topic_.empty(); }
    bool UsesGpsOrientation() const { return !orientation_topic_.empty(); }
    bool UsesGpsPose() const {
        return UsesGpsPosition() && UsesGpsOrientation();
    }
    bool UsesGpsVelocity() const { return !velocity_topic_.empty(); }
    bool UsesWheelOdometry() const { return !wheel_odometry_topic_.empty(); }
    bool RequiresMap() const { return UsesLidar() || with_ui_; }

    static Mode ModeFromString(const std::string& value);
    static std::string ModeToString(Mode mode);

   private:
    void SetupPublishers(rclcpp::Node::SharedPtr node);
    void HandleNdtResult(const loc::LocalizationResult& result);

    void TryHandleGpsInitialization();
    void HandleGpsInitializationPair(
        const sensor_msgs::msg::NavSatFix::SharedPtr& fix,
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& orientation);
    void HandleGpsPosition(
        const sensor_msgs::msg::NavSatFix::SharedPtr& fix);
    void HandleGpsOrientation(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr&
            orientation);

    // GPS initialization is the second stage of localization initialization:
    // 1) NDT relocalizes from the user-provided rough MAP<-BODY seed;
    // 2) while the vehicle stays still, average the first N synchronized GPS
    //    positions and INS orientations and solve one fixed 3-D ENU<-MAP
    //    transform. After that, GPS position updates no longer wait for INS.
    void BeginGpsInitialization(const SE3& initial_map_body_pose);
    bool AddGpsInitializationSample(
        const Eigen::Vector3d& gps_output_position_enu,
        const Eigen::Matrix3d& rotation_enu_gps);
    bool FinishGpsInitialization();

    bool GnssToEnu(const sensor_msgs::msg::NavSatFix& fix,
                   Eigen::Vector3d* position_enu,
                   Eigen::Matrix3d* covariance_enu) const;
    bool VelocityToMap(
        const geometry_msgs::msg::TwistWithCovarianceStamped& velocity,
        Eigen::Vector2d* velocity_map,
        Eigen::Matrix2d* covariance_map) const;

    void InitializeEkfFromNdt(const loc::LocalizationResult& ndt);
    bool InitializeManualGuess(double stamp);
    void PublishPredictionIfAdvanced(double stamp,
                                     const std::string& message);
    loc::LocalizationResult BuildEkfResult(double stamp,
                                           const std::string& message,
                                           bool reliable) const;
    void PublishResult(const loc::LocalizationResult& result);
    void AppendPath(const loc::LocalizationResult& result);
    void AppendDebugPath(
        nav_msgs::msg::Path* path,
        const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr& publisher,
        double stamp, const Eigen::Vector2d& position, double yaw);

    static std::string Normalize(std::string value);
    static Eigen::Vector3d ReadVector3(const YAML::Node& node,
                                       const Eigen::Vector3d& fallback);
    static double PoseYaw(const SE3& pose);

    LocalizationSystemOptions options_;
    Mode mode_ = Mode::NDT_ONLY;
    std::string yaml_path_;
    std::string map_path_;
    std::string base_link_frame_ = "base_link";
    std::string output_frame_ = "map";

    // A non-empty topic is the sensor enable switch.
    std::string lidar_topic_;
    std::string livox_lidar_topic_;
    std::string imu_topic_;
    std::string gps_topic_;
    std::string orientation_topic_;
    std::string velocity_topic_;
    std::string wheel_odometry_topic_;

    bool with_ui_ = false;
    bool map_ready_ = false;
    bool manual_initial_guess_pending_ = false;
    SE3 manual_initial_pose_;

    std::shared_ptr<loc::Localization> loc_;
    mutable std::mutex filter_mutex_;
    loc::EKF ekf_;

    // Static GPS geometry. A is exactly the antenna point whose WGS84
    // coordinate is published by NavSatFix. The configured lever arm is its
    // coordinate in the GPS-device frame: p_G_A. T_B_G then gives:
    //   p_B_A = t_B_G + R_B_G * p_G_A.
    Eigen::Vector3d gps_antenna_position_gps_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d gps_translation_body_gps_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d gps_rotation_body_gps_rpy_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d gps_rotation_body_gps_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d gps_antenna_position_body_ = Eigen::Vector3d::Zero();
    double gps_initialization_sync_tolerance_sec_ = 0.05;
    int gps_initialization_sample_count_required_ = 10;

    double initial_position_std_ = 0.5;
    double initial_orientation_std_ =
        3.0 * 3.14159265358979323846 / 180.0;
    double initial_velocity_std_ = 2.0;
    double initial_yaw_rate_std_ = 0.5;
    double gps_position_std_x_ = 0.05;
    double gps_position_std_y_ = 0.05;
    double gps_position_std_z_ = 100.0;
    double gps_orientation_std_roll_ =
        0.2 * 3.14159265358979323846 / 180.0;
    double gps_orientation_std_pitch_ =
        0.2 * 3.14159265358979323846 / 180.0;
    double gps_orientation_std_yaw_ =
        0.5 * 3.14159265358979323846 / 180.0;
    double gps_velocity_std_x_ = 0.10;
    double gps_velocity_std_y_ = 0.10;
    double ndt_position_std_x_ = 0.10;
    double ndt_position_std_y_ = 0.10;
    double ndt_position_std_z_ = 0.20;
    double ndt_orientation_std_ =
        1.0 * 3.14159265358979323846 / 180.0;

    // Fixed transform initialized once per localization run from the first
    // reliable NDT pose + first N stationary GPS/orientation pairs:
    //   p_enu = R_enu_map * p_map + t_enu_map.
    bool gps_initial_map_body_ready_ = false;
    SE3 gps_initial_map_body_pose_;
    int gps_initialization_sample_count_ = 0;
    Eigen::Vector3d gps_initial_body_position_enu_sum_ =
        Eigen::Vector3d::Zero();
    Eigen::Vector4d gps_initial_body_quaternion_sum_ =
        Eigen::Vector4d::Zero();
    Eigen::Quaterniond gps_initial_body_reference_quaternion_ =
        Eigen::Quaterniond::Identity();
    bool gps_initial_body_reference_quaternion_ready_ = false;
    math::JsbsimWgs84Enu enu_projector_;
    bool enu_from_map_ready_ = false;
    Eigen::Matrix3d enu_from_map_rotation_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d enu_from_map_translation_ = Eigen::Vector3d::Zero();

    // Approximate synchronization is initialization-only and keeps one latest
    // unmatched GPS position and one latest INS orientation message.
    std::mutex gps_initialization_mutex_;
    sensor_msgs::msg::NavSatFix::SharedPtr pending_gps_;
    geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr
        pending_gps_orientation_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr loc_odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr loc_pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr loc_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr raw_gps_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr raw_ndt_path_pub_;
    rclcpp::Publisher<lightning_interfaces::msg::LocalizationPose>::SharedPtr
        loc_pose_quality_pub_;

    mutable std::mutex result_mutex_;
    mutable std::mutex debug_path_mutex_;
    loc::LocalizationResult latest_result_;
    nav_msgs::msg::Path path_;
    nav_msgs::msg::Path raw_gps_path_;
    nav_msgs::msg::Path raw_ndt_path_;
};

}  // namespace lightning::modules
