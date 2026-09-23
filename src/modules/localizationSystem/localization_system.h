#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

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

namespace lightning::ui {
class PangolinWindow;
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

    loc::LocalizationResult GetLatestResult() const;
    void Reset();

    Mode GetMode() const { return mode_; }
    bool UsesLidar() const {
        return !lidar_topic_.empty() || !livox_lidar_topic_.empty();
    }
    bool UsesGpsPosition() const { return !gps_topic_.empty(); }
    bool UsesGpsOrientation() const { return !orientation_topic_.empty(); }
    bool UsesGpsVelocity() const { return !velocity_topic_.empty(); }
    bool UsesWheelOdometry() const { return !wheel_odometry_topic_.empty(); }
    bool RequiresMap() const { return UsesLidar() || with_ui_; }

    static Mode ModeFromString(const std::string& value);
    static std::string ModeToString(Mode mode);

   private:
    void SetupPublishers(rclcpp::Node::SharedPtr node);
    bool InitializeVisualization(const std::string& map_path);
    void LoadMapForVisualization(const std::string& map_path);
    void HandleNdtResult(const loc::LocalizationResult& result);
    void UpdateEkfWithNdt(const loc::LocalizationResult& result);

    struct TimedPosition {
        double stamp = 0.0;
        Eigen::Vector3d position = Eigen::Vector3d::Zero();
    };

    void AddGpsCalibrationSample(double stamp,
                                 const Eigen::Vector3d& position_enu);
    void AddNdtCalibrationSample(const loc::LocalizationResult& result);
    bool TryFinishMapEnuCalibration();
    void ResetMapEnuCalibration();
    void HandleGpsPosition(double stamp,
                           const Eigen::Vector3d& position_enu,
                           const Eigen::Matrix3d& covariance_enu);
    void HandleGpsOrientation(
        const geometry_msgs::msg::TwistWithCovarianceStamped& orientation);

    void GnssToEnu(const sensor_msgs::msg::NavSatFix& fix,
                   Eigen::Vector3d& position_enu,
                   Eigen::Matrix3d& covariance_enu) const;
    void VelocityToMap(
        const geometry_msgs::msg::TwistWithCovarianceStamped& velocity,
        Eigen::Vector2d& velocity_map,
        Eigen::Matrix2d& covariance_map) const;

    bool InitializeEkfFromNdt(const loc::LocalizationResult& ndt);
    void PublishPredictionIfAdvanced(double stamp,
                                     const std::string& message);
    loc::LocalizationResult BuildEkfResult(double stamp,
                                           const std::string& message,
                                           bool reliable) const;
    void PublishResult(const loc::LocalizationResult& result);
    void AppendPath(const loc::LocalizationResult& result);
    void AppendDebugPath(
        nav_msgs::msg::Path& path,
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

    // NDT only: map registration, initial-pose alignment and raw NDT results.
    std::shared_ptr<loc::Localization> ndt_localization_;
    // All localization visualization is owned by this outer coordinator.
    std::shared_ptr<ui::PangolinWindow> ui_;
    mutable std::mutex filter_mutex_;
    loc::EKF ekf_;

    // Rotation only. NavSatFix already reports the GPS-device origin, which
    // is treated as the EKF body-position observation; no lever arm is used.
    Eigen::Vector3d gps_rotation_body_gps_rpy_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d gps_rotation_body_gps_ = Eigen::Matrix3d::Identity();

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

    // Calibration estimates p_map = Rz(yaw_map_enu) * p_enu + t_map_enu from
    // timestamp-matched GPS/NDT positions sampled over the first 50 m. MAP and
    // ENU are gravity-aligned, so roll_map_enu and pitch_map_enu stay zero.
    math::JsbsimWgs84Enu enu_projector_;
    bool map_enu_calibrated_ = false;
    Eigen::Matrix3d rotation_map_enu_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation_map_enu_ = Eigen::Vector3d::Zero();
    std::mutex map_enu_calibration_mutex_;
    std::vector<TimedPosition> gps_calibration_samples_;
    std::vector<TimedPosition> ndt_calibration_samples_;
    double calibration_travel_distance_m_ = 0.0;
    bool calibration_distance_complete_ = false;

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
