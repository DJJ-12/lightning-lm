#include "modules/localizationSystem/localization_system.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <vector>

#include <glog/logging.h>
#include <rclcpp/node.hpp>
#include <sensor_msgs/msg/nav_sat_status.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "core/lightning_math.hpp"
#include "core/localization/localization.h"

namespace lightning::modules {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;

std::string ReadTopic(const YAML::Node& common, const char* name) {
    if (!common) return std::string();
    const YAML::Node value = common[name];
    return value && value.IsScalar()
        ? value.as<std::string>()
        : std::string();
}

double CourseDegreesToEnuYaw(double course_degrees) {
    return std::atan2(
        std::sin((90.0 - course_degrees) * kDegToRad),
        std::cos((90.0 - course_degrees) * kDegToRad));
}

Eigen::Matrix3d RotationFromRollPitchYaw(
    double roll, double pitch, double yaw) {
    return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
        .toRotationMatrix();
}

bool Covariance3Valid(const Eigen::Matrix3d& covariance) {
    if (!covariance.allFinite()) return false;
    const Eigen::Matrix3d symmetric =
        0.5 * (covariance + covariance.transpose());
    return symmetric.diagonal().minCoeff() >= 0.0 &&
           symmetric.diagonal().maxCoeff() > 0.0;
}

// UTM uses grid east/north while the INS reports true local ENU. Determine the
// fixed grid convergence at the map reference by projecting a small true-north
// displacement. The returned matrix only rotates axes; UTM scale distortion is
// intentionally not applied to metric velocity observations.
bool ComputeUtmFromTrueEnuRotation(
    double latitude_deg, double longitude_deg, double altitude_m,
    int utm_zone, Eigen::Matrix3d* rotation) {
    if (!rotation) return false;
    Eigen::Vector3d reference;
    Eigen::Vector3d north_point;
    constexpr double kLatitudeStepDeg = 1e-5;
    if (!math::JsbsimWgs84Enu::ForwardUtmDegrees(
            latitude_deg, longitude_deg, altitude_m, utm_zone, &reference) ||
        !math::JsbsimWgs84Enu::ForwardUtmDegrees(
            latitude_deg + kLatitudeStepDeg, longitude_deg, altitude_m,
            utm_zone, &north_point)) {
        return false;
    }
    Eigen::Vector2d grid_north =
        (north_point - reference).head<2>();
    const double norm = grid_north.norm();
    if (!std::isfinite(norm) || norm < 1e-9) return false;
    grid_north /= norm;
    const Eigen::Vector2d grid_east(grid_north.y(), -grid_north.x());

    rotation->setIdentity();
    rotation->block<2, 1>(0, 0) = grid_east;
    rotation->block<2, 1>(0, 1) = grid_north;
    return rotation->allFinite();
}

Eigen::Matrix<double, 6, 6> NdtCovarianceToRightError(const SE3& pose, const Eigen::Matrix<double, 6, 6>& ndt_covariance) {
    const Eigen::Vector3d rpy = math::RotMtoEuler(pose.so3().matrix());
    const double roll = rpy.x();
    const double pitch = rpy.y();
    Eigen::Matrix3d euler_to_right_error;
    euler_to_right_error << 1.0, 0.0, -std::sin(pitch),
                            0.0, std::cos(roll), std::sin(roll) * std::cos(pitch),
                            0.0, -std::sin(roll), std::cos(roll) * std::cos(pitch);
    Eigen::Matrix<double, 6, 6> transform = Eigen::Matrix<double, 6, 6>::Identity();
    transform.block<3, 3>(3, 3) = euler_to_right_error;
    const Eigen::Matrix<double, 6, 6> covariance = transform * ndt_covariance * transform.transpose();
    return 0.5 * (covariance + covariance.transpose());
}

loc::ESKF::Covariance InitialEskfCovariance(const loc::LocalizationResult* pose_result, double velocity_sigma, double angular_velocity_sigma) {
    loc::ESKF::Covariance covariance = loc::ESKF::Covariance::Zero();
    if (pose_result && pose_result->covariance_valid_) covariance.block<6, 6>(0, 0) = NdtCovarianceToRightError(pose_result->pose_, pose_result->pose_covariance_);
    else {
        covariance.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        covariance.block<3, 3>(3, 3) = std::pow(20.0 * kDegToRad, 2) * Eigen::Matrix3d::Identity();
    }
    covariance.block<3, 3>(6, 6) = std::pow(velocity_sigma, 2) * Eigen::Matrix3d::Identity();
    covariance.block<3, 3>(9, 9) = std::pow(angular_velocity_sigma, 2) * Eigen::Matrix3d::Identity();
    return covariance;
}

Eigen::Matrix3d DiagonalCovariance(double sigma_xy, double sigma_z) {
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    covariance(0, 0) = sigma_xy * sigma_xy;
    covariance(1, 1) = sigma_xy * sigma_xy;
    covariance(2, 2) = sigma_z * sigma_z;
    return covariance;
}

bool ImuCovarianceValid(const std::array<double, 9>& values) {
    bool nonzero = false;
    for (double value : values) {
        if (!std::isfinite(value)) return false;
        nonzero = nonzero || std::fabs(value) > 0.0;
    }
    return nonzero && values[0] >= 0.0 && values[4] >= 0.0 && values[8] >= 0.0;
}

}  // namespace

LocalizationSystem::LocalizationSystem(LocalizationSystemOptions options) : options_(options) {}
LocalizationSystem::~LocalizationSystem() { Reset(); }

std::string LocalizationSystem::Normalize(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

LocalizationSystem::Mode LocalizationSystem::ModeFromString(const std::string& value) {
    const std::string mode = Normalize(value);
    if (mode == "eskf" || mode == "ndt_rtk_eskf" || mode == "ndt_rtk_ins_eskf" || mode == "ndt_rtk") return Mode::ESKF_FUSION;
    if (mode == "rtk_only" || mode == "rtk_ins_only") return Mode::RTK_ONLY;
    if (mode == "auto") return Mode::AUTO;
    return Mode::NDT_ONLY;
}

std::string LocalizationSystem::ModeToString(Mode mode) {
    if (mode == Mode::ESKF_FUSION) return "ndt_rtk_ins_eskf";
    if (mode == Mode::RTK_ONLY) return "rtk_ins_only";
    if (mode == Mode::AUTO) return "auto";
    return "ndt_only";
}

Eigen::Vector3d LocalizationSystem::ReadVector3(const YAML::Node& node, const Eigen::Vector3d& fallback) {
    if (!node) return fallback;
    const std::vector<double> values = node.as<std::vector<double>>();
    if (values.size() != 3) return fallback;
    return Eigen::Vector3d(values[0], values[1], values[2]);
}

Eigen::Array3i LocalizationSystem::ReadAxisMask(const YAML::Node& node, const Eigen::Array3i& fallback) {
    if (!node) return fallback;
    const std::vector<int> values = node.as<std::vector<int>>();
    if (values.size() != 3) return fallback;
    return Eigen::Array3i(values[0] != 0, values[1] != 0, values[2] != 0);
}

bool LocalizationSystem::Init(const std::string& yaml_path, rclcpp::Node::SharedPtr node) {
    Reset();
    yaml_path_ = yaml_path;
    const YAML::Node yaml = YAML::LoadFile(yaml_path);
    const YAML::Node localization = yaml["localization"];
    const YAML::Node eskf = localization && localization["eskf"] ? localization["eskf"] : YAML::Node();
    const YAML::Node rtk_ins = localization && localization["rtk_ins"] ? localization["rtk_ins"] : YAML::Node();
    const YAML::Node common = yaml["common"];
    mode_ = ModeFromString(localization && localization["mode"] ? localization["mode"].as<std::string>() : "ndt_only");
    if (yaml["system"] && yaml["system"]["pub_tf"]) options_.pub_tf_ = yaml["system"]["pub_tf"].as<bool>();
    with_ui_ = yaml["system"] && yaml["system"]["with_ui"]
        ? yaml["system"]["with_ui"].as<bool>()
        : false;
    if (common && common["base_link_frame"]) base_link_frame_ = common["base_link_frame"].as<std::string>();
    output_frame_ = localization && localization["output_frame"]
        ? localization["output_frame"].as<std::string>()
        : "map";

    // A configured topic is the sensor's only enable switch. Missing and empty
    // topic values both mean that the sensor is disabled.
    lidar_topic_ = ReadTopic(common, "lidar_topic");
    livox_lidar_topic_ = ReadTopic(common, "livox_lidar_topic");
    imu_topic_ = ReadTopic(common, "imu_topic");
    rtk_fix_topic_ = ReadTopic(common, "rtk_fix_topic");
    rtk_orientation_topic_ = ReadTopic(common, "rtk_orientation_topic");
    rtk_velocity_topic_ = ReadTopic(common, "rtk_velocity_topic");
    wheel_odometry_topic_ = ReadTopic(common, "wheel_odometry_topic");
    rtk_position_axes_ = ReadAxisMask(
        eskf["rtk_position_axes"], Eigen::Array3i(1, 1, 0));
    rtk_velocity_axes_ = ReadAxisMask(eskf["rtk_velocity_axes"], Eigen::Array3i(1, 1, 1));
    wheel_linear_axes_ = ReadAxisMask(eskf["wheel_linear_axes"], Eigen::Array3i(1, 0, 0));
    wheel_angular_axes_ = ReadAxisMask(eskf["wheel_angular_axes"], Eigen::Array3i(0, 0, 1));
    imu_angular_axes_ = ReadAxisMask(eskf["imu_angular_axes"], Eigen::Array3i(1, 1, 1));

    rtk_ins_lever_arm_tracking_ = ReadVector3(rtk_ins && rtk_ins["lever_arm_tracking"] ? rtk_ins["lever_arm_tracking"] : YAML::Node(), Eigen::Vector3d::Zero());
    if (rtk_ins && rtk_ins["fallback_position_sigma_xy"]) rtk_position_sigma_xy_ = rtk_ins["fallback_position_sigma_xy"].as<double>();
    if (rtk_ins && rtk_ins["fallback_position_sigma_z"]) rtk_position_sigma_z_ = rtk_ins["fallback_position_sigma_z"].as<double>();
    if (rtk_ins && rtk_ins["fallback_velocity_sigma_xy"]) rtk_velocity_sigma_xy_ = rtk_ins["fallback_velocity_sigma_xy"].as<double>();
    if (rtk_ins && rtk_ins["fallback_velocity_sigma_z"]) rtk_velocity_sigma_z_ = rtk_ins["fallback_velocity_sigma_z"].as<double>();
    if (rtk_ins && rtk_ins["velocity_covariance_scale"]) rtk_velocity_covariance_scale_ = rtk_ins["velocity_covariance_scale"].as<double>();
    if (rtk_ins && rtk_ins["fallback_orientation_sigma_deg"])
        fallback_ins_yaw_sigma_rad_ =
            rtk_ins["fallback_orientation_sigma_deg"].as<double>() *
            kDegToRad;
    if (rtk_ins && rtk_ins["initial_observation_max_dt"])
        initial_observation_max_dt_ =
            std::max(0.0, rtk_ins["initial_observation_max_dt"].as<double>());

    if (eskf && eskf["wheel_linear_sigma"]) wheel_linear_sigma_ = eskf["wheel_linear_sigma"].as<double>();
    if (eskf && eskf["wheel_angular_sigma"]) wheel_angular_sigma_ = eskf["wheel_angular_sigma"].as<double>();
    if (eskf && eskf["imu_angular_sigma"]) imu_angular_sigma_ = eskf["imu_angular_sigma"].as<double>();
    if (eskf && eskf["initial_velocity_sigma"]) initial_velocity_sigma_ = eskf["initial_velocity_sigma"].as<double>();
    if (eskf && eskf["initial_angular_velocity_sigma"]) initial_angular_velocity_sigma_ = eskf["initial_angular_velocity_sigma"].as<double>();
    if (eskf && eskf["ndt_pose_gate_chi2"]) ndt_pose_gate_chi2_ = eskf["ndt_pose_gate_chi2"].as<double>();
    if (eskf && eskf["rtk_position_gate_chi2"]) rtk_position_gate_chi2_ = eskf["rtk_position_gate_chi2"].as<double>();
    if (eskf && eskf["rtk_velocity_gate_chi2"]) rtk_velocity_gate_chi2_ = eskf["rtk_velocity_gate_chi2"].as<double>();
    if (eskf && eskf["ins_yaw_gate_chi2"]) ins_yaw_gate_chi2_ = eskf["ins_yaw_gate_chi2"].as<double>();
    if (eskf && eskf["wheel_gate_chi2"]) wheel_gate_chi2_ = eskf["wheel_gate_chi2"].as<double>();
    if (eskf && eskf["imu_angular_gate_chi2"]) imu_angular_gate_chi2_ = eskf["imu_angular_gate_chi2"].as<double>();

    if (yaml["lio_sam"] && yaml["lio_sam"]["extrinsicRot"]) {
        const std::vector<double> values =
            yaml["lio_sam"]["extrinsicRot"].as<std::vector<double>>();
        if (values.size() != 9) {
            LOG(ERROR) << "[LOCALIZATION_ESKF] lio_sam.extrinsicRot must contain 9 values";
            return false;
        }
        tracking_from_imu_rotation_ <<
            values[0], values[1], values[2],
            values[3], values[4], values[5],
            values[6], values[7], values[8];
    }
    if (yaml["lio_sam"] && yaml["lio_sam"]["extrinsicRPY"]) {
        const std::vector<double> values =
            yaml["lio_sam"]["extrinsicRPY"].as<std::vector<double>>();
        if (values.size() != 9) {
            LOG(ERROR) << "[LOCALIZATION_ESKF] lio_sam.extrinsicRPY must contain 9 values";
            return false;
        }
        imu_from_tracking_rotation_ <<
            values[0], values[1], values[2],
            values[3], values[4], values[5],
            values[6], values[7], values[8];
    }

    loc::ESKF::Options filter_options;
    if (eskf && eskf["body_acceleration_noise_std"]) filter_options.body_acceleration_noise_std = eskf["body_acceleration_noise_std"].as<double>();
    else if (eskf && eskf["acceleration_noise_std"]) filter_options.body_acceleration_noise_std = eskf["acceleration_noise_std"].as<double>();
    if (eskf && eskf["angular_acceleration_noise_std"]) filter_options.angular_acceleration_noise_std = eskf["angular_acceleration_noise_std"].as<double>();
    if (eskf && eskf["max_prediction_step"]) filter_options.max_prediction_step = eskf["max_prediction_step"].as<double>();
    filter_options.pose_gate_chi2 = ndt_pose_gate_chi2_;
    filter_options.position_gate_chi2 = rtk_position_gate_chi2_;
    filter_options.yaw_gate_chi2 = ins_yaw_gate_chi2_;
    filter_options.twist_gate_chi2 = wheel_gate_chi2_;
    eskf_.Configure(filter_options);

    if (UsesRtk() || UsesInsOrientation() || UsesInsVelocity()) {
        const YAML::Node map_from_enu =
            localization && localization["map_from_enu"]
                ? localization["map_from_enu"]
                : YAML::Node();
        if (!InitializeFixedMapTransform(map_from_enu)) {
            return false;
        }
    }

    // Localization also owns the map visualization. Keep it alive even when
    // no LiDAR topic is configured; in that case it loads/displays the map but
    // never receives a cloud or produces an NDT observation.
    loc::Localization::Options options;
    options.pub_tf_ = false;
    loc_ = std::make_shared<loc::Localization>(options);
    loc_->SetResultCallback(
        [this](const loc::LocalizationResult& result) {
            HandleNdtResult(result);
        });
    if (!UsesLidar()) has_initial_guess_ = true;
    if (node) SetupPublishers(node);
    LOG(INFO) << "[LOCALIZATION_SYSTEM] mode=" << ModeToString(mode_)
              << ", rtk_position=" << UsesRtk()
              << ", ins_orientation=" << UsesInsOrientation()
              << ", ins_velocity=" << UsesInsVelocity()
              << ", wheel=" << UsesWheelOdometry()
              << ", imu_omega=" << UsesImuAngularVelocity();
    return true;
}

void LocalizationSystem::SetupPublishers(rclcpp::Node::SharedPtr node) {
    if (!node) return;
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node);
    loc_odom_pub_ = node->create_publisher<nav_msgs::msg::Odometry>("/lightning/localization/odom", 10);
    loc_pose_pub_ = node->create_publisher<geometry_msgs::msg::PoseStamped>("/lightning/localization/pose", 10);
    loc_path_pub_ = node->create_publisher<nav_msgs::msg::Path>("/lightning/localization/path", rclcpp::QoS(1).reliable().transient_local());
    loc_pose_quality_pub_ = node->create_publisher<lightning_interfaces::msg::LocalizationPose>("/lightning/localization/pose_with_quality", 10);
}

bool LocalizationSystem::SetMapPath(const std::string& map_path) {
    if (!loc_ || map_path.empty()) return false;
    map_path_ = map_path;
    map_ready_ = loc_->Init(yaml_path_, map_path_);
    has_initial_guess_ = !UsesLidar();
    if (map_ready_) {
        LOG(INFO) << "[LOCALIZATION_SYSTEM] map and visualization initialized"
                  << ", path=" << map_path_
                  << ", lidar_enabled=" << UsesLidar()
                  << ", ui_enabled=" << with_ui_;
    }
    return map_ready_;
}

bool LocalizationSystem::SetInitialGuess(const SE3& init_pose, bool* initialized_now) {
    if (initialized_now) *initialized_now = false;
    if (!UsesLidar()) { has_initial_guess_ = true; return true; }
    if (!loc_ || !map_ready_) return false;
    const bool accepted = loc_->SetExternalPose(init_pose.unit_quaternion(), init_pose.translation());
    has_initial_guess_ = accepted;
    if (accepted && mode_ != Mode::NDT_ONLY) {
        std::lock_guard<std::mutex> lock(filter_mutex_);
        eskf_.Reset();
        manual_initial_guess_pending_ = true;
        has_initial_position_ = false;
        has_initial_yaw_ = false;
        has_initial_velocity_ = false;
    }
    return accepted;
}

loc::LocalizationFrameOutcome LocalizationSystem::ProcessCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud, const loc::LocalizationInputDiagnostic& diagnostic) {
    if (!UsesLidar() || !loc_ || !map_ready_) return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
    if (cloud) {
        const double stamp = rclcpp::Time(cloud->header.stamp).seconds();
        last_lidar_stamp_ = stamp;
        std::lock_guard<std::mutex> lock(filter_mutex_);
        if (eskf_.Initialized() && eskf_.PredictTo(stamp)) loc_->SetPredictionPose(eskf_.Pose());
    }
    return loc_->ProcessLidarMsg(cloud, diagnostic);
}

loc::LocalizationFrameOutcome LocalizationSystem::ProcessCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud, const loc::LocalizationInputDiagnostic& diagnostic) {
    if (!UsesLidar() || !loc_ || !map_ready_) return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
    if (cloud) {
        const double stamp = rclcpp::Time(cloud->header.stamp).seconds();
        last_lidar_stamp_ = stamp;
        std::lock_guard<std::mutex> lock(filter_mutex_);
        if (eskf_.Initialized() && eskf_.PredictTo(stamp)) loc_->SetPredictionPose(eskf_.Pose());
    }
    return loc_->ProcessLivoxLidarMsg(cloud, diagnostic);
}

void LocalizationSystem::ProcessRtkPosition(
    const sensor_msgs::msg::NavSatFix::SharedPtr& fix) {
    if (!UsesRtk() || !fix) return;

    Eigen::Vector3d position_map;
    Eigen::Matrix3d covariance_map;
    if (!PositionToMap(*fix, &position_map, &covariance_map)) return;
    const double stamp = rclcpp::Time(fix->header.stamp).seconds();

    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!eskf_.Initialized()) {
        if (!manual_initial_guess_pending_) {
            has_initial_position_ = true;
            initial_position_stamp_ = stamp;
            initial_sensor_position_map_ = position_map;
            initial_position_covariance_map_ = covariance_map;
            TryInitializeEskfFromGlobalObservations();
        }
        return;
    }

    double distance = 0.0;
    const bool accepted = eskf_.UpdatePositionWithLeverArm(
        stamp, position_map, rtk_ins_lever_arm_tracking_, covariance_map,
        rtk_position_axes_, rtk_position_gate_chi2_, &distance);
    if (accepted) {
        PublishResult(BuildEskfResult(stamp, "ESKF GNSS position update", true));
    } else {
        PublishPredictionIfAdvanced(
            stamp, "ESKF prediction; GNSS position rejected");
        LOG_EVERY_N(WARNING, 20)
            << "[LOCALIZATION_ESKF] GNSS position rejected, stamp=" << stamp
            << ", mahalanobis=" << distance;
    }
}

void LocalizationSystem::ProcessInsOrientation(
    const sensor_msgs::msg::Imu::SharedPtr& orientation) {
    if (!UsesInsOrientation() || !orientation) return;

    double yaw_map = 0.0;
    double variance = 0.0;
    Eigen::Matrix3d rotation_map_tracking;
    if (!OrientationToMapYaw(
            *orientation, &rotation_map_tracking, &yaw_map, &variance)) {
        return;
    }
    const double stamp = rclcpp::Time(orientation->header.stamp).seconds();

    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!eskf_.Initialized()) {
        if (!manual_initial_guess_pending_) {
            has_initial_yaw_ = true;
            initial_yaw_stamp_ = stamp;
            initial_orientation_map_ = rotation_map_tracking;
            initial_yaw_variance_ = variance;
            TryInitializeEskfFromGlobalObservations();
        }
        return;
    }

    double distance = 0.0;
    const bool accepted = eskf_.UpdateYaw(
        stamp, yaw_map, variance, ins_yaw_gate_chi2_, &distance);
    if (accepted) {
        PublishResult(BuildEskfResult(stamp, "ESKF INS yaw update", true));
    } else {
        PublishPredictionIfAdvanced(
            stamp, "ESKF prediction; INS yaw rejected");
        LOG_EVERY_N(WARNING, 20)
            << "[LOCALIZATION_ESKF] INS yaw rejected, stamp=" << stamp
            << ", mahalanobis=" << distance;
    }
}

void LocalizationSystem::ProcessInsVelocity(
    const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& velocity) {
    if (!UsesInsVelocity() || !velocity) return;

    Eigen::Vector3d velocity_map;
    Eigen::Matrix3d covariance_map;
    if (!VelocityToMap(*velocity, &velocity_map, &covariance_map)) return;
    const double stamp = rclcpp::Time(velocity->header.stamp).seconds();

    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!eskf_.Initialized()) {
        if (!manual_initial_guess_pending_) {
            has_initial_velocity_ = true;
            initial_velocity_stamp_ = stamp;
            initial_sensor_velocity_map_ = velocity_map;
            initial_velocity_covariance_map_ = covariance_map;
            TryInitializeEskfFromGlobalObservations();
        }
        return;
    }

    double velocity_distance = 0.0;
    const bool accepted = eskf_.UpdateMapVelocityWithLeverArm(
        stamp, velocity_map, rtk_ins_lever_arm_tracking_, covariance_map,
        rtk_velocity_axes_, rtk_velocity_gate_chi2_, &velocity_distance);

    if (accepted) {
        PublishResult(BuildEskfResult(stamp, "ESKF INS velocity update", true));
    } else {
        PublishPredictionIfAdvanced(
            stamp, "ESKF prediction; INS velocity rejected");
        LOG_EVERY_N(WARNING, 20)
            << "[LOCALIZATION_ESKF] INS velocity rejected, stamp=" << stamp
            << ", velocity_d2=" << velocity_distance;
    }
}

void LocalizationSystem::ProcessWheelOdometry(
    const nav_msgs::msg::Odometry::SharedPtr& odometry) {
    if (!UsesWheelOdometry() || !odometry) return;
    const double stamp = rclcpp::Time(odometry->header.stamp).seconds();
    const Eigen::Vector3d linear_velocity_body(
        odometry->twist.twist.linear.x,
        odometry->twist.twist.linear.y,
        odometry->twist.twist.linear.z);
    const Eigen::Vector3d angular_velocity_body(
        odometry->twist.twist.angular.x,
        odometry->twist.twist.angular.y,
        odometry->twist.twist.angular.z);
    if (!std::isfinite(stamp) || !linear_velocity_body.allFinite() ||
        !angular_velocity_body.allFinite()) {
        return;
    }

    Eigen::Matrix<double, 6, 6> covariance;
    for (int row = 0; row < 6; ++row) {
        for (int column = 0; column < 6; ++column) {
            covariance(row, column) =
                odometry->twist.covariance[row * 6 + column];
        }
    }
    if (!covariance.allFinite() ||
        covariance.diagonal().minCoeff() < 0.0 ||
        covariance.diagonal().maxCoeff() <= 0.0) {
        covariance.setZero();
        covariance.block<3, 3>(0, 0) =
            std::pow(wheel_linear_sigma_, 2) * Eigen::Matrix3d::Identity();
        covariance.block<3, 3>(3, 3) =
            std::pow(wheel_angular_sigma_, 2) * Eigen::Matrix3d::Identity();
    }

    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!eskf_.Initialized()) return;
    double distance = 0.0;
    const bool accepted = eskf_.UpdateBodyTwist(
        stamp, linear_velocity_body, angular_velocity_body, covariance,
        wheel_linear_axes_, wheel_angular_axes_, wheel_gate_chi2_, &distance);
    if (accepted) {
        PublishResult(BuildEskfResult(stamp, "ESKF wheel update", true));
    } else {
        PublishPredictionIfAdvanced(
            stamp, "ESKF prediction; wheel observation rejected");
        LOG_EVERY_N(WARNING, 20)
            << "[LOCALIZATION_ESKF] wheel observation rejected, mahalanobis="
            << distance;
    }
}

void LocalizationSystem::ProcessImu(const sensor_msgs::msg::Imu::SharedPtr& imu) {
    if (!UsesImuAngularVelocity() || !imu) return;
    const double stamp = rclcpp::Time(imu->header.stamp).seconds();
    const Eigen::Vector3d omega_imu(
        imu->angular_velocity.x,
        imu->angular_velocity.y,
        imu->angular_velocity.z);
    const Eigen::Vector3d omega = tracking_from_imu_rotation_ * omega_imu;
    if (!std::isfinite(stamp) || !omega.allFinite()) return;
    Eigen::Matrix3d covariance = std::pow(imu_angular_sigma_, 2) * Eigen::Matrix3d::Identity();
    if (ImuCovarianceValid(imu->angular_velocity_covariance)) {
        Eigen::Matrix3d covariance_imu;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                covariance_imu(row, col) =
                    imu->angular_velocity_covariance[row * 3 + col];
        covariance = tracking_from_imu_rotation_ * covariance_imu *
                     tracking_from_imu_rotation_.transpose();
    }
    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!eskf_.Initialized()) return;
    double distance = 0.0;
    const bool accepted = eskf_.UpdateAngularVelocity(stamp, omega, covariance, imu_angular_axes_, imu_angular_gate_chi2_, &distance);
    if (accepted) {
        PublishResult(BuildEskfResult(
            stamp, "ESKF IMU angular velocity update", true));
    } else {
        PublishPredictionIfAdvanced(
            stamp, "ESKF prediction; IMU angular velocity rejected");
        LOG_EVERY_N(WARNING, 50) << "[LOCALIZATION_ESKF] IMU angular velocity rejected, mahalanobis=" << distance;
    }
}

void LocalizationSystem::HandleNdtResult(const loc::LocalizationResult& result) {
    if (mode_ == Mode::NDT_ONLY) { PublishResult(result); return; }
    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!result.valid_) {
        if (eskf_.Initialized() && eskf_.PredictTo(result.timestamp_)) PublishResult(BuildEskfResult(result.timestamp_, "ESKF prediction; NDT invalid", false));
        else PublishResult(result);
        return;
    }

    bool pose_accepted = false;
    if (!eskf_.Initialized()) {
        InitializeEskfFromNdt(result);
        pose_accepted = eskf_.Initialized();
    } else {
        Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Zero();
        if (result.covariance_valid_) covariance = NdtCovarianceToRightError(result.pose_, result.pose_covariance_);
        else {
            covariance.block<3, 3>(0, 0) = 0.25 * Eigen::Matrix3d::Identity();
            covariance.block<3, 3>(3, 3) = std::pow(5.0 * kDegToRad, 2) * Eigen::Matrix3d::Identity();
        }
        double distance = 0.0;
        pose_accepted = eskf_.UpdatePose(result.timestamp_, result.pose_, covariance, ndt_pose_gate_chi2_, &distance);
        if (!pose_accepted) LOG(WARNING) << "[LOCALIZATION_ESKF] NDT pose rejected, mahalanobis=" << distance;
    }
    PublishResult(BuildEskfResult(result.timestamp_, pose_accepted ? "ESKF NDT update" : "ESKF prediction; NDT rejected", pose_accepted && result.reliable_));
}

void LocalizationSystem::InitializeEskfFromNdt(const loc::LocalizationResult& ndt) {
    const loc::ESKF::Covariance covariance = InitialEskfCovariance(&ndt, initial_velocity_sigma_, initial_angular_velocity_sigma_);
    if (eskf_.Initialize(ndt.timestamp_, ndt.pose_, Eigen::Vector3d::Zero(),
                         Eigen::Vector3d::Zero(), covariance)) {
        manual_initial_guess_pending_ = false;
    }
}

bool LocalizationSystem::InitializeFixedMapTransform(
    const YAML::Node& map_from_enu) {
    const std::array<const char*, 6> required = {
        "lat", "lon", "alt", "pitch", "roll", "yaw"};
    for (const char* key : required) {
        if (!map_from_enu || !map_from_enu[key]) {
            LOG(ERROR) << "[LOCALIZATION_ESKF] localization.map_from_enu is missing '"
                       << key << "'";
            return false;
        }
    }

    const double latitude_deg = map_from_enu["lat"].as<double>();
    const double longitude_deg = map_from_enu["lon"].as<double>();
    const double altitude_m = map_from_enu["alt"].as<double>();
    const double pitch = map_from_enu["pitch"].as<double>() * kDegToRad;
    const double roll = map_from_enu["roll"].as<double>() * kDegToRad;
    // `yaw` intentionally follows the source INS convention requested for the
    // map reference: north=0 degrees and clockwise positive (course angle).
    const double course_deg = map_from_enu["yaw"].as<double>();
    if (!std::isfinite(latitude_deg) || !std::isfinite(longitude_deg) ||
        !std::isfinite(altitude_m) || !std::isfinite(pitch) ||
        !std::isfinite(roll) || !std::isfinite(course_deg) ||
        latitude_deg < -90.0 || latitude_deg > 90.0 ||
        longitude_deg < -180.0 || longitude_deg > 180.0) {
        LOG(ERROR) << "[LOCALIZATION_ESKF] invalid fixed map reference";
        return false;
    }

    utm_zone_ = math::JsbsimWgs84Enu::UtmZoneFromLongitude(longitude_deg);
    Eigen::Vector3d reference_gnss_utm;
    if (utm_zone_ <= 0 ||
        !math::JsbsimWgs84Enu::ForwardUtmDegrees(
            latitude_deg, longitude_deg, altitude_m, utm_zone_,
            &reference_gnss_utm) ||
        !ComputeUtmFromTrueEnuRotation(
            latitude_deg, longitude_deg, altitude_m, utm_zone_,
            &utm_from_true_enu_rotation_)) {
        LOG(ERROR) << "[LOCALIZATION_ESKF] failed to project fixed map reference";
        return false;
    }

    const double yaw_enu = CourseDegreesToEnuYaw(course_deg);
    const Eigen::Matrix3d true_enu_from_tracking =
        RotationFromRollPitchYaw(roll, pitch, yaw_enu) *
        imu_from_tracking_rotation_;
    const Eigen::Matrix3d utm_from_tracking =
        utm_from_true_enu_rotation_ * true_enu_from_tracking;
    const Eigen::Vector3d reference_tracking_utm =
        reference_gnss_utm -
        utm_from_tracking * rtk_ins_lever_arm_tracking_;

    // Current mapping initializes the first tracking pose with IMU roll/pitch,
    // forces map yaw to zero, and sets its translation to zero. Therefore the
    // same six reference values are sufficient for both T_utm_tracking0 and
    // T_map_tracking0; no seventh "initial map pose" parameter is needed.
    const Eigen::Vector3d tracking_rpy =
        math::RotMtoEuler(true_enu_from_tracking);
    const Eigen::Matrix3d map_from_tracking_at_reference =
        RotationFromRollPitchYaw(tracking_rpy.x(), tracking_rpy.y(), 0.0);
    map_from_enu_rotation_ =
        map_from_tracking_at_reference * utm_from_tracking.transpose();
    map_from_enu_translation_ =
        -map_from_enu_rotation_ * reference_tracking_utm;
    map_from_true_enu_rotation_ =
        map_from_enu_rotation_ * utm_from_true_enu_rotation_;
    map_from_enu_ready_ =
        map_from_enu_rotation_.allFinite() &&
        map_from_enu_translation_.allFinite();
    if (!map_from_enu_ready_) return false;

    LOG(INFO) << "[LOCALIZATION_ESKF] fixed map<-UTM transform ready"
              << ", zone=" << utm_zone_
              << ", reference_gnss_utm=" << reference_gnss_utm.transpose()
              << ", course_deg=" << course_deg
              << ", yaw_enu_deg=" << yaw_enu / kDegToRad
              << ", translation=" << map_from_enu_translation_.transpose();
    return true;
}

bool LocalizationSystem::PositionToMap(
    const sensor_msgs::msg::NavSatFix& fix,
    Eigen::Vector3d* position_map,
    Eigen::Matrix3d* covariance_map) const {
    if (!position_map || !covariance_map || !map_from_enu_ready_) return false;
    const double stamp = rclcpp::Time(fix.header.stamp).seconds();
    if (fix.status.status == sensor_msgs::msg::NavSatStatus::STATUS_NO_FIX ||
        !std::isfinite(stamp) || !std::isfinite(fix.latitude) ||
        !std::isfinite(fix.longitude) || !std::isfinite(fix.altitude)) {
        return false;
    }

    Eigen::Vector3d position_utm;
    if (!math::JsbsimWgs84Enu::ForwardUtmDegrees(
            fix.latitude, fix.longitude, fix.altitude, utm_zone_,
            &position_utm)) {
        return false;
    }
    *position_map =
        map_from_enu_rotation_ * position_utm + map_from_enu_translation_;

    Eigen::Matrix3d covariance_true_enu;
    covariance_true_enu.setZero();
    if (fix.position_covariance_type !=
        sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN) {
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                covariance_true_enu(row, column) =
                    fix.position_covariance[row * 3 + column];
    }
    if (!Covariance3Valid(covariance_true_enu)) {
        covariance_true_enu =
            DiagonalCovariance(rtk_position_sigma_xy_, rtk_position_sigma_z_);
    }
    *covariance_map = map_from_true_enu_rotation_ * covariance_true_enu *
                      map_from_true_enu_rotation_.transpose();
    return position_map->allFinite() && covariance_map->allFinite();
}

bool LocalizationSystem::OrientationToMapYaw(
    const sensor_msgs::msg::Imu& orientation,
    Eigen::Matrix3d* rotation_map_tracking,
    double* yaw_map,
    double* variance) const {
    if (!rotation_map_tracking || !yaw_map || !variance ||
        !map_from_enu_ready_ ||
        orientation.orientation_covariance[0] < 0.0) {
        return false;
    }
    const double stamp = rclcpp::Time(orientation.header.stamp).seconds();
    Eigen::Quaterniond quaternion(
        orientation.orientation.w,
        orientation.orientation.x,
        orientation.orientation.y,
        orientation.orientation.z);
    if (!std::isfinite(stamp) || !quaternion.coeffs().allFinite() ||
        quaternion.squaredNorm() < 1e-12) {
        return false;
    }
    quaternion.normalize();
    *rotation_map_tracking =
        map_from_true_enu_rotation_ * quaternion.toRotationMatrix() *
        imu_from_tracking_rotation_;
    *yaw_map = PoseYaw(SE3(Eigen::Quaterniond(*rotation_map_tracking),
                           Eigen::Vector3d::Zero()));
    const double message_variance = orientation.orientation_covariance[8];
    *variance = std::isfinite(message_variance) && message_variance > 0.0
        ? message_variance
        : fallback_ins_yaw_sigma_rad_ * fallback_ins_yaw_sigma_rad_;
    return rotation_map_tracking->allFinite() && std::isfinite(*yaw_map) &&
           std::isfinite(*variance) &&
           *variance > 0.0;
}

bool LocalizationSystem::VelocityToMap(
    const geometry_msgs::msg::TwistWithCovarianceStamped& velocity,
    Eigen::Vector3d* velocity_map,
    Eigen::Matrix3d* covariance_map) const {
    if (!velocity_map || !covariance_map || !map_from_enu_ready_) return false;
    const double stamp = rclcpp::Time(velocity.header.stamp).seconds();
    const Eigen::Vector3d velocity_true_enu(
        velocity.twist.twist.linear.x,
        velocity.twist.twist.linear.y,
        velocity.twist.twist.linear.z);
    if (!std::isfinite(stamp) || !velocity_true_enu.allFinite()) return false;
    *velocity_map = map_from_true_enu_rotation_ * velocity_true_enu;

    Eigen::Matrix3d covariance_true_enu;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            covariance_true_enu(row, column) =
                velocity.twist.covariance[row * 6 + column];
    if (!Covariance3Valid(covariance_true_enu)) {
        covariance_true_enu =
            DiagonalCovariance(rtk_velocity_sigma_xy_, rtk_velocity_sigma_z_);
    }
    covariance_true_enu *= rtk_velocity_covariance_scale_;
    *covariance_map = map_from_true_enu_rotation_ * covariance_true_enu *
                      map_from_true_enu_rotation_.transpose();
    return velocity_map->allFinite() && covariance_map->allFinite();
}

void LocalizationSystem::TryInitializeEskfFromGlobalObservations() {
    if (eskf_.Initialized() || mode_ == Mode::NDT_ONLY ||
        manual_initial_guess_pending_ ||
        !has_initial_position_ || !has_initial_yaw_) {
        return;
    }
    if (std::fabs(initial_position_stamp_ - initial_yaw_stamp_) >
        initial_observation_max_dt_) {
        return;
    }

    double stamp = std::max(initial_position_stamp_, initial_yaw_stamp_);
    const Eigen::Vector3d tracking_position_map =
        initial_sensor_position_map_ -
        initial_orientation_map_ * rtk_ins_lever_arm_tracking_;
    Eigen::Vector3d velocity_body = Eigen::Vector3d::Zero();
    const bool use_initial_velocity =
        has_initial_velocity_ &&
        std::fabs(initial_velocity_stamp_ - stamp) <=
            initial_observation_max_dt_;
    if (use_initial_velocity) {
        velocity_body =
            initial_orientation_map_.transpose() *
            initial_sensor_velocity_map_;
        stamp = std::max(stamp, initial_velocity_stamp_);
    }

    loc::ESKF::Covariance covariance = InitialEskfCovariance(
        nullptr, initial_velocity_sigma_, initial_angular_velocity_sigma_);
    covariance.block<3, 3>(0, 0) = initial_position_covariance_map_;
    covariance(5, 5) = initial_yaw_variance_;
    if (use_initial_velocity) {
        covariance.block<3, 3>(6, 6) =
            initial_orientation_map_.transpose() *
            initial_velocity_covariance_map_ * initial_orientation_map_;
    }
    const SE3 pose(Eigen::Quaterniond(initial_orientation_map_),
                   tracking_position_map);
    if (eskf_.Initialize(stamp, pose, velocity_body,
                         Eigen::Vector3d::Zero(), covariance)) {
        if (UsesLidar()) {
            if (!loc_ || !map_ready_ ||
                !loc_->SetExternalPose(pose.unit_quaternion(),
                                       pose.translation())) {
                LOG(ERROR) << "[LOCALIZATION_ESKF] failed to pass GNSS/INS "
                              "initial pose to NDT";
                eskf_.Reset();
                return;
            }
            has_initial_guess_ = true;
        }
        LOG(INFO) << "[LOCALIZATION_ESKF] initialized from GNSS/INS"
                  << ", stamp=" << stamp
                  << ", position_map=" << tracking_position_map.transpose()
                  << ", yaw_map_deg=" << PoseYaw(pose) / kDegToRad
                  << ", initial_velocity_body=" << velocity_body.transpose();
        PublishResult(BuildEskfResult(
            stamp, "ESKF initialized from fixed-map GNSS/INS", true));
    }
}

void LocalizationSystem::PublishPredictionIfAdvanced(
    double stamp, const std::string& message) {
    // Every update function predicts first. If the observation is then gated
    // out, publish that new predicted state; do nothing for stale input whose
    // timestamp could not advance the filter.
    if (eskf_.Initialized() &&
        std::fabs(eskf_.State().stamp - stamp) <= 1e-6) {
        PublishResult(BuildEskfResult(stamp, message, false));
    }
}

loc::LocalizationResult LocalizationSystem::BuildEskfResult(double stamp, const std::string& message, bool reliable) const {
    loc::LocalizationResult result;
    if (!eskf_.Initialized()) return result;
    result.timestamp_ = stamp;
    result.valid_ = true;
    result.localization_valid_ = true;
    result.status_ = loc::LocalizationStatus::GOOD;
    result.reliable_ = reliable;
    result.confidence_ = 1.0 / (1.0 + std::sqrt(std::max(0.0, eskf_.P().block<3, 3>(0, 0).trace())));
    result.pose_ = eskf_.Pose();
    result.pose_covariance_.setZero();
    result.pose_covariance_.block<3, 3>(0, 0) = eskf_.P().block<3, 3>(0, 0);
    result.pose_covariance_.block<3, 3>(0, 3) = eskf_.P().block<3, 3>(0, 3);
    result.pose_covariance_.block<3, 3>(3, 0) = eskf_.P().block<3, 3>(3, 0);
    result.pose_covariance_.block<3, 3>(3, 3) = eskf_.P().block<3, 3>(3, 3);
    result.covariance_valid_ = true;
    result.velocity_map_ = eskf_.VelocityMap();
    result.angular_velocity_body_ = eskf_.State().angular_velocity_body;
    result.twist_covariance_ = eskf_.P().block<6, 6>(6, 6);
    result.twist_covariance_ = 0.5 * (result.twist_covariance_ + result.twist_covariance_.transpose());
    result.twist_covariance_valid_ = result.twist_covariance_.allFinite();
    result.frame_id_ = output_frame_;
    result.message_ = message;
    return result;
}

void LocalizationSystem::PublishResult(const loc::LocalizationResult& input) {
    loc::LocalizationResult result = input;
    if (result.frame_id_.empty()) result.frame_id_ = output_frame_;
    { std::lock_guard<std::mutex> lock(result_mutex_); latest_result_ = result; }
    if (!result.valid_) return;
    // Pangolin receives the same final estimator result as ROS publishers.
    // In fusion modes this is always the ESKF state, never raw NDT output.
    if (loc_) loc_->UpdateVisualization(result);
    AppendPath(result);
    geometry_msgs::msg::TransformStamped transform = result.ToGeoMsg();
    transform.header.frame_id = result.frame_id_;
    transform.child_frame_id = base_link_frame_;
    if (options_.pub_tf_ && tf_broadcaster_) tf_broadcaster_->sendTransform(transform);

    geometry_msgs::msg::PoseStamped pose;
    pose.header = transform.header;
    pose.pose.position.x = transform.transform.translation.x;
    pose.pose.position.y = transform.transform.translation.y;
    pose.pose.position.z = transform.transform.translation.z;
    pose.pose.orientation = transform.transform.rotation;
    if (loc_pose_pub_) loc_pose_pub_->publish(pose);

    nav_msgs::msg::Odometry odometry;
    odometry.header = transform.header;
    odometry.child_frame_id = base_link_frame_;
    odometry.pose.pose = pose.pose;
    const Eigen::Vector3d velocity_body = result.pose_.so3().inverse().matrix() * result.velocity_map_;
    odometry.twist.twist.linear.x = velocity_body.x();
    odometry.twist.twist.linear.y = velocity_body.y();
    odometry.twist.twist.linear.z = velocity_body.z();
    odometry.twist.twist.angular.x = result.angular_velocity_body_.x();
    odometry.twist.twist.angular.y = result.angular_velocity_body_.y();
    odometry.twist.twist.angular.z = result.angular_velocity_body_.z();
    if (result.covariance_valid_) for (int row = 0; row < 6; ++row) for (int col = 0; col < 6; ++col) odometry.pose.covariance[row * 6 + col] = result.pose_covariance_(row, col);
    if (result.twist_covariance_valid_) for (int row = 0; row < 6; ++row) for (int col = 0; col < 6; ++col) odometry.twist.covariance[row * 6 + col] = result.twist_covariance_(row, col);
    if (loc_odom_pub_) loc_odom_pub_->publish(odometry);

    if (loc_pose_quality_pub_) {
        lightning_interfaces::msg::LocalizationPose quality;
        quality.header = transform.header;
        quality.pose.position.x = pose.pose.position.x;
        quality.pose.position.y = pose.pose.position.y;
        quality.pose.position.z = pose.pose.position.z;
        quality.pose.orientation = pose.pose.orientation;
        quality.valid = result.valid_;
        quality.reliable = result.reliable_;
        quality.status = static_cast<uint8_t>(result.status_);
        quality.confidence = result.confidence_;
        quality.tp = result.tp_;
        quality.nvtl = result.nvtl_;
        quality.iterations = static_cast<uint32_t>(std::max(0, result.iterations_));
        quality.message = result.message_;
        loc_pose_quality_pub_->publish(quality);
    }
}

void LocalizationSystem::AppendPath(const loc::LocalizationResult& result) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = math::FromSec(result.timestamp_);
    pose.header.frame_id = result.frame_id_;
    pose.pose.position.x = result.pose_.translation().x();
    pose.pose.position.y = result.pose_.translation().y();
    pose.pose.position.z = result.pose_.translation().z();
    const Eigen::Quaterniond quaternion = result.pose_.unit_quaternion();
    pose.pose.orientation.x = quaternion.x();
    pose.pose.orientation.y = quaternion.y();
    pose.pose.orientation.z = quaternion.z();
    pose.pose.orientation.w = quaternion.w();
    path_.header = pose.header;
    path_.poses.push_back(pose);
    if (path_.poses.size() > 100000) path_.poses.erase(path_.poses.begin(), path_.poses.begin() + 1000);
    if (loc_path_pub_) loc_path_pub_->publish(path_);
}

void LocalizationSystem::MarkPoor(const std::string& message) {
    loc::LocalizationResult result = GetLatestResult();
    result.localization_valid_ = false;
    result.status_ = loc::LocalizationStatus::FAIL;
    result.reliable_ = false;
    result.message_ = message;
    if (result.valid_) PublishResult(result);
}

loc::LocalizationResult LocalizationSystem::GetLatestResult() const { std::lock_guard<std::mutex> lock(result_mutex_); return latest_result_; }
double LocalizationSystem::PoseYaw(const SE3& pose) { return std::atan2(pose.so3().matrix()(1, 0), pose.so3().matrix()(0, 0)); }

void LocalizationSystem::Reset() {
    if (loc_) loc_->Finish();
    loc_.reset();
    { std::lock_guard<std::mutex> lock(filter_mutex_); eskf_.Reset(); }
    lidar_topic_.clear();
    livox_lidar_topic_.clear();
    imu_topic_.clear();
    rtk_fix_topic_.clear();
    rtk_orientation_topic_.clear();
    rtk_velocity_topic_.clear();
    wheel_odometry_topic_.clear();
    with_ui_ = false;
    utm_zone_ = 0;
    map_from_enu_ready_ = false;
    map_from_enu_rotation_ = Eigen::Matrix3d::Identity();
    map_from_enu_translation_ = Eigen::Vector3d::Zero();
    utm_from_true_enu_rotation_ = Eigen::Matrix3d::Identity();
    map_from_true_enu_rotation_ = Eigen::Matrix3d::Identity();
    tracking_from_imu_rotation_ = Eigen::Matrix3d::Identity();
    imu_from_tracking_rotation_ = Eigen::Matrix3d::Identity();
    has_initial_position_ = false;
    has_initial_yaw_ = false;
    has_initial_velocity_ = false;
    initial_orientation_map_ = Eigen::Matrix3d::Identity();
    map_ready_ = false;
    has_initial_guess_ = false;
    manual_initial_guess_pending_ = false;
    last_lidar_stamp_ = -1.0;
    path_ = nav_msgs::msg::Path();
    { std::lock_guard<std::mutex> lock(result_mutex_); latest_result_ = loc::LocalizationResult(); }
    tf_broadcaster_.reset();
    loc_odom_pub_.reset();
    loc_pose_pub_.reset();
    loc_path_pub_.reset();
    loc_pose_quality_pub_.reset();
}

}  // namespace lightning::modules
