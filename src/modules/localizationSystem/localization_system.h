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
#include "core/localization/EKF/ekf.h"
#include "core/localization/localization_diagnostic.h"
#include "core/localization/localization_result.h"
#include "lightning_interfaces/msg/localization_pose.hpp"

namespace lightning::loc { class Localization; }

namespace lightning::modules {

struct LocalizationSystemOptions { bool pub_tf_ = true; };

class LocalizationSystem {
   public:
    enum class Mode { NDT_ONLY, EKF_FUSION, gps_ONLY, AUTO };

    explicit LocalizationSystem(LocalizationSystemOptions options = LocalizationSystemOptions());
    ~LocalizationSystem();

    bool Init(const std::string& yaml_path, rclcpp::Node::SharedPtr node = nullptr);
    bool SetMapPath(const std::string& map_path);
    bool SetInitialGuess(const SE3& init_pose, bool* initialized_now = nullptr);
    loc::LocalizationFrameOutcome ProcessCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud, const loc::LocalizationInputDiagnostic& diagnostic = {});
    loc::LocalizationFrameOutcome ProcessCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud, const loc::LocalizationInputDiagnostic& diagnostic = {});

    // Each standard ROS observation enters the filter independently. No gps
    // synchronization packet or input history is maintained in this module.
    void ProcessgpsPosition(const sensor_msgs::msg::NavSatFix::SharedPtr& fix);
    void ProcessgpsVelocity(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& velocity);
    void ProcessWheelOdometry(const nav_msgs::msg::Odometry::SharedPtr& odometry);
    void ProcessImu(const sensor_msgs::msg::Imu::SharedPtr& imu);

    void MarkPoor(const std::string& message);
    loc::LocalizationResult GetLatestResult() const;
    void Reset();

    Mode GetMode() const { return mode_; }
    bool UsesLidar() const {
        return !lidar_topic_.empty() || !livox_lidar_topic_.empty();
    }
    bool Usesgps() const { return !gps_topic_.empty(); }
    bool UsesgpsVelocity() const { return !velocity_topic_.empty(); }
    bool UsesWheelOdometry() const { return !wheel_odometry_topic_.empty(); }
    // LiDAR needs the map for NDT; the UI also needs it for visualization.
    bool RequiresMap() const { return UsesLidar() || with_ui_; }
    bool RequiresInitialGuess() const { return UsesLidar(); }
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
    bool VelocityToMap(
        const geometry_msgs::msg::TwistWithCovarianceStamped& velocity,
        Eigen::Vector2d* velocity_map,
        Eigen::Matrix2d* covariance_map) const;
    Eigen::Vector3d gpsLeverArmForFilter() const;
    void TryInitializeEkf();
    void InitializeEkfFromNdt(const loc::LocalizationResult& ndt);
    bool InitializeManualGuess(double stamp);
    void PublishPredictionIfAdvanced(double stamp, const std::string& message);
    loc::LocalizationResult BuildEkfResult(double stamp, const std::string& message, bool reliable) const;
    void PublishResult(const loc::LocalizationResult& result);
    void AppendPath(const loc::LocalizationResult& result);
    void AppendDebugPath(nav_msgs::msg::Path* path,
                         const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr& publisher,
                         double stamp, const Eigen::Vector2d& position,
                         double yaw);
    static std::string Normalize(std::string value);
    static Eigen::Vector3d ReadVector3(const YAML::Node& node, const Eigen::Vector3d& fallback);
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
    std::string gps_topic_;
    std::string velocity_topic_;
    std::string wheel_odometry_topic_;
    bool with_ui_ = false;
    bool map_ready_ = false;
    bool manual_initial_guess_pending_ = false;
    SE3 manual_initial_pose_;

    std::shared_ptr<loc::Localization> loc_;
    mutable std::mutex filter_mutex_;
    loc::EKF ekf_;

    Eigen::Vector3d gps_ins_lever_arm_tracking_ = Eigen::Vector3d::Zero();
    double initial_position_std_ = 0.5;
    double initial_orientation_std_ =
        3.0 * 3.14159265358979323846 / 180.0;
    double initial_velocity_std_ = 2.0;
    double initial_yaw_rate_std_ = 0.5;
    double gps_position_std_x_ = 0.05;
    double gps_position_std_y_ = 0.05;
    // Mandatory GNSS altitude uncertainty floor. The map-frame height is
    // intentionally established by NDT rather than receiver altitude.
    double gps_position_std_z_ = 100.0;
    double gps_velocity_std_x_ = 0.10;
    double gps_velocity_std_y_ = 0.10;
    double ndt_position_std_x_ = 0.10;
    double ndt_position_std_y_ = 0.10;
    double ndt_position_std_z_ = 0.20;
    double ndt_orientation_std_ =
        1.0 * 3.14159265358979323846 / 180.0;

    int utm_zone_ = 0;
    bool map_from_enu_ready_ = false;
    Eigen::Matrix3d map_from_utm_rotation_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d map_from_utm_translation_ = Eigen::Vector3d::Zero();
    // gps/INS velocity uses true local ENU axes. UTM positions use grid axes, so
    // this fixed rotation applies the reference meridian convergence.
    Eigen::Matrix3d utm_from_true_enu_rotation_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d map_from_true_enu_rotation_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d reference_gnss_utm_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d reference_gnss_map_ = Eigen::Vector3d::Zero();
    // gps-only initialization keeps one latest value per observation type. It
    // is state, not a pending-message queue.
    bool has_initial_position_ = false;
    double initial_position_stamp_ = 0.0;
    Eigen::Vector3d initial_sensor_position_map_ = Eigen::Vector3d::Zero();

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr loc_odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr loc_pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr loc_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr raw_gps_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr raw_ndt_path_pub_;
    rclcpp::Publisher<lightning_interfaces::msg::LocalizationPose>::SharedPtr loc_pose_quality_pub_;
    mutable std::mutex result_mutex_;
    mutable std::mutex debug_path_mutex_;
    loc::LocalizationResult latest_result_;
    nav_msgs::msg::Path path_;
    nav_msgs::msg::Path raw_gps_path_;
    nav_msgs::msg::Path raw_ndt_path_;
};

}  // namespace lightning::modules
