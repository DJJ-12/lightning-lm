#include "modules/localizationSystem/localization_system.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>

#include <glog/logging.h>
#include <rclcpp/node.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "core/lightning_math.hpp"
#include "core/localization/localization.h"

namespace lightning::modules {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;

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
    mode_ = ModeFromString(localization && localization["mode"] ? localization["mode"].as<std::string>() : "ndt_only");
    if (yaml["system"] && yaml["system"]["pub_tf"]) options_.pub_tf_ = yaml["system"]["pub_tf"].as<bool>();
    if (yaml["common"] && yaml["common"]["base_link_frame"]) base_link_frame_ = yaml["common"]["base_link_frame"].as<std::string>();
    output_frame_ = mode_ == Mode::RTK_ONLY ? "enu" : "map";
    if (mode_ != Mode::RTK_ONLY && localization && localization["output_frame"]) output_frame_ = localization["output_frame"].as<std::string>();

    use_rtk_position_ = rtk_ins && rtk_ins["enabled"] ? rtk_ins["enabled"].as<bool>() : true;
    use_rtk_velocity_ = eskf && eskf["use_rtk_velocity"] ? eskf["use_rtk_velocity"].as<bool>() : false;
    use_rtk_elevation_ = eskf && eskf["use_rtk_elevation"] ? eskf["use_rtk_elevation"].as<bool>() : false;
    use_course_yaw_ = eskf && eskf["use_course_yaw"] ? eskf["use_course_yaw"].as<bool>() : mode_ == Mode::RTK_ONLY;
    use_wheel_odometry_ = eskf && eskf["use_wheel_odometry"] ? eskf["use_wheel_odometry"].as<bool>() : false;
    use_imu_angular_velocity_ = eskf && eskf["use_imu_angular_velocity"] ? eskf["use_imu_angular_velocity"].as<bool>() : false;
    use_ndt_derived_twist_ = eskf && eskf["use_ndt_derived_twist"] ? eskf["use_ndt_derived_twist"].as<bool>() : true;
    rtk_position_axes_ = ReadAxisMask(eskf["rtk_position_axes"], use_rtk_elevation_ ? Eigen::Array3i(1, 1, 1) : Eigen::Array3i(1, 1, 0));
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
    if (rtk_ins && rtk_ins["history_limit"]) rtk_history_limit_ = static_cast<std::size_t>(std::max(10, rtk_ins["history_limit"].as<int>()));

    if (eskf && eskf["course_yaw_min_speed"]) course_yaw_min_speed_ = eskf["course_yaw_min_speed"].as<double>();
    if (eskf && eskf["course_yaw_sigma_deg"]) course_yaw_sigma_rad_ = eskf["course_yaw_sigma_deg"].as<double>() * kDegToRad;
    if (eskf && eskf["wheel_linear_sigma"]) wheel_linear_sigma_ = eskf["wheel_linear_sigma"].as<double>();
    if (eskf && eskf["wheel_angular_sigma"]) wheel_angular_sigma_ = eskf["wheel_angular_sigma"].as<double>();
    if (eskf && eskf["imu_angular_sigma"]) imu_angular_sigma_ = eskf["imu_angular_sigma"].as<double>();
    if (eskf && eskf["ndt_twist_covariance_scale"]) ndt_twist_covariance_scale_ = eskf["ndt_twist_covariance_scale"].as<double>();
    if (eskf && eskf["initial_velocity_sigma"]) initial_velocity_sigma_ = eskf["initial_velocity_sigma"].as<double>();
    if (eskf && eskf["initial_angular_velocity_sigma"]) initial_angular_velocity_sigma_ = eskf["initial_angular_velocity_sigma"].as<double>();
    if (eskf && eskf["ndt_pose_gate_chi2"]) ndt_pose_gate_chi2_ = eskf["ndt_pose_gate_chi2"].as<double>();
    if (eskf && eskf["rtk_position_gate_chi2"]) rtk_position_gate_chi2_ = eskf["rtk_position_gate_chi2"].as<double>();
    if (eskf && eskf["rtk_velocity_gate_chi2"]) rtk_velocity_gate_chi2_ = eskf["rtk_velocity_gate_chi2"].as<double>();
    if (eskf && eskf["course_yaw_gate_chi2"]) course_yaw_gate_chi2_ = eskf["course_yaw_gate_chi2"].as<double>();
    if (eskf && eskf["wheel_gate_chi2"]) wheel_gate_chi2_ = eskf["wheel_gate_chi2"].as<double>();
    if (eskf && eskf["imu_angular_gate_chi2"]) imu_angular_gate_chi2_ = eskf["imu_angular_gate_chi2"].as<double>();
    if (eskf && eskf["ndt_twist_gate_chi2"]) ndt_twist_gate_chi2_ = eskf["ndt_twist_gate_chi2"].as<double>();

    loc::ESKF::Options filter_options;
    if (eskf && eskf["body_acceleration_noise_std"]) filter_options.body_acceleration_noise_std = eskf["body_acceleration_noise_std"].as<double>();
    else if (eskf && eskf["acceleration_noise_std"]) filter_options.body_acceleration_noise_std = eskf["acceleration_noise_std"].as<double>();
    if (eskf && eskf["angular_acceleration_noise_std"]) filter_options.angular_acceleration_noise_std = eskf["angular_acceleration_noise_std"].as<double>();
    if (eskf && eskf["max_prediction_step"]) filter_options.max_prediction_step = eskf["max_prediction_step"].as<double>();
    filter_options.pose_gate_chi2 = ndt_pose_gate_chi2_;
    filter_options.position_gate_chi2 = rtk_position_gate_chi2_;
    filter_options.yaw_gate_chi2 = course_yaw_gate_chi2_;
    filter_options.twist_gate_chi2 = wheel_gate_chi2_;
    eskf_.Configure(filter_options);

    const YAML::Node map_from_enu = localization && localization["map_from_enu"] ? localization["map_from_enu"] : YAML::Node();
    map_from_enu_mode_ = map_from_enu && map_from_enu["mode"] ? Normalize(map_from_enu["mode"].as<std::string>()) : "trajectory_alignment";
    if (map_from_enu_mode_ == "first_ndt") map_from_enu_mode_ = "trajectory_alignment";
    if (map_from_enu && map_from_enu["alignment_max_age"]) map_alignment_max_age_ = map_from_enu["alignment_max_age"].as<double>();
    if (map_from_enu && map_from_enu["alignment_min_pairs"]) map_alignment_min_pairs_ = std::max(2, map_from_enu["alignment_min_pairs"].as<int>());
    if (map_from_enu && map_from_enu["alignment_min_baseline"]) map_alignment_min_baseline_ = map_from_enu["alignment_min_baseline"].as<double>();
    if (map_from_enu && map_from_enu["alignment_max_rmse"]) map_alignment_max_rmse_ = map_from_enu["alignment_max_rmse"].as<double>();
    if (map_from_enu_mode_ == "fixed") {
        const double yaw = (map_from_enu && map_from_enu["yaw_deg"] ? map_from_enu["yaw_deg"].as<double>() : 0.0) * kDegToRad;
        map_from_enu_rotation_ = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        map_from_enu_translation_ = ReadVector3(map_from_enu["translation"], Eigen::Vector3d::Zero());
        map_from_enu_ready_ = true;
    }

    if (UsesLidar()) {
        loc::Localization::Options options;
        options.pub_tf_ = false;
        loc_ = std::make_shared<loc::Localization>(options);
        loc_->SetResultCallback([this](const loc::LocalizationResult& result) { HandleNdtResult(result); });
    } else {
        map_ready_ = true;
        has_initial_guess_ = true;
    }
    if (node) SetupPublishers(node);
    LOG(INFO) << "[LOCALIZATION_SYSTEM] mode=" << ModeToString(mode_) << ", RTK_INS=" << UsesRtk() << ", wheel=" << UsesWheelOdometry() << ", imu_omega=" << UsesImuAngularVelocity();
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
    if (!UsesLidar()) { map_ready_ = true; return true; }
    if (!loc_ || map_path.empty()) return false;
    map_path_ = map_path;
    map_ready_ = loc_->Init(yaml_path_, map_path_);
    has_initial_guess_ = false;
    return map_ready_;
}

bool LocalizationSystem::SetInitialGuess(const SE3& init_pose, bool* initialized_now) {
    if (initialized_now) *initialized_now = false;
    if (!UsesLidar()) { has_initial_guess_ = true; return true; }
    if (!loc_ || !map_ready_) return false;
    const bool accepted = loc_->SetExternalPose(init_pose.unit_quaternion(), init_pose.translation());
    has_initial_guess_ = accepted;
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

void LocalizationSystem::ProcessRtkIns(const RtkInsMeasurement& measurement) {
    if (!UsesRtk() || !measurement.position_valid || !measurement.yaw_valid) return;
    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!rtk_history_.empty() && measurement.stamp < rtk_history_.back().stamp - 1e-6) {
        LOG_EVERY_N(WARNING, 20) << "[LOCALIZATION_RTK] out-of-order measurement dropped, stamp=" << measurement.stamp;
        return;
    }
    rtk_history_.push_back(measurement);
    while (rtk_history_.size() > rtk_history_limit_) rtk_history_.pop_front();
    UpdateFromRtkIns(measurement);
}

void LocalizationSystem::ProcessWheelOdometry(const WheelOdometryMeasurement& measurement) {
    if (!UsesWheelOdometry() || !measurement.valid) return;
    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!eskf_.Initialized()) return;
    Eigen::Matrix<double, 6, 6> covariance = measurement.covariance;
    if (!measurement.covariance_valid) {
        covariance.setZero();
        covariance.block<3, 3>(0, 0) = std::pow(wheel_linear_sigma_, 2) * Eigen::Matrix3d::Identity();
        covariance.block<3, 3>(3, 3) = std::pow(wheel_angular_sigma_, 2) * Eigen::Matrix3d::Identity();
    }
    double distance = 0.0;
    const bool accepted = eskf_.UpdateBodyTwist(measurement.stamp, measurement.linear_velocity_body, measurement.angular_velocity_body, covariance, wheel_linear_axes_, wheel_angular_axes_, wheel_gate_chi2_, &distance);
    if (accepted) {
        PublishResult(BuildEskfResult(measurement.stamp, "ESKF wheel update", true));
    } else {
        LOG_EVERY_N(WARNING, 20) << "[LOCALIZATION_ESKF] wheel observation rejected, mahalanobis=" << distance;
    }
}

void LocalizationSystem::ProcessImu(const sensor_msgs::msg::Imu::SharedPtr& imu) {
    if (!UsesImuAngularVelocity() || !imu) return;
    const double stamp = rclcpp::Time(imu->header.stamp).seconds();
    const Eigen::Vector3d omega(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
    if (!std::isfinite(stamp) || !omega.allFinite()) return;
    Eigen::Matrix3d covariance = std::pow(imu_angular_sigma_, 2) * Eigen::Matrix3d::Identity();
    if (ImuCovarianceValid(imu->angular_velocity_covariance)) {
        for (int row = 0; row < 3; ++row) for (int col = 0; col < 3; ++col) covariance(row, col) = imu->angular_velocity_covariance[row * 3 + col];
    }
    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!eskf_.Initialized()) return;
    double distance = 0.0;
    const bool accepted = eskf_.UpdateAngularVelocity(stamp, omega, covariance, imu_angular_axes_, imu_angular_gate_chi2_, &distance);
    if (!accepted) {
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

    TryInitializeMapFromEnu(result);
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
    if (pose_accepted && use_ndt_derived_twist_) ApplyNdtDerivedTwist(result);
    if (pose_accepted) { previous_ndt_ = result; has_previous_ndt_ = true; }
    PublishResult(BuildEskfResult(result.timestamp_, pose_accepted ? "ESKF NDT update" : "ESKF prediction; NDT rejected", pose_accepted && result.reliable_));
}

void LocalizationSystem::InitializeEskfFromNdt(const loc::LocalizationResult& ndt) {
    const loc::ESKF::Covariance covariance = InitialEskfCovariance(&ndt, initial_velocity_sigma_, initial_angular_velocity_sigma_);
    eskf_.Initialize(ndt.timestamp_, ndt.pose_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), covariance);
}

void LocalizationSystem::InitializeEskfFromRtkIns(const RtkInsMeasurement& measurement) {
    if (!measurement.position_valid) return;
    Eigen::Matrix3d position_covariance = measurement.position_covariance_valid ? measurement.position_covariance_enu : DiagonalCovariance(rtk_position_sigma_xy_, rtk_position_sigma_z_);
    Eigen::Matrix3d velocity_covariance = measurement.velocity_covariance_valid ? measurement.velocity_covariance_enu : DiagonalCovariance(rtk_velocity_sigma_xy_, rtk_velocity_sigma_z_);
    velocity_covariance *= rtk_velocity_covariance_scale_;
    double yaw = 0.0;
    const double horizontal_speed = measurement.velocity_enu.head<2>().norm();
    if (measurement.yaw_valid) yaw = measurement.yaw_enu;
    else if (measurement.velocity_valid && horizontal_speed >= course_yaw_min_speed_) yaw = std::atan2(measurement.velocity_enu.y(), measurement.velocity_enu.x());
    const Eigen::Matrix3d rotation_enu_body = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Vector3d body_position = measurement.position_enu - rotation_enu_body * rtk_ins_lever_arm_tracking_;
    const SE3 pose = PoseFromPositionYaw(body_position, yaw, Eigen::Matrix3d::Identity());
    const Eigen::Vector3d velocity_body = pose.so3().inverse().matrix() * measurement.velocity_enu;
    loc::ESKF::Covariance covariance = InitialEskfCovariance(nullptr, initial_velocity_sigma_, initial_angular_velocity_sigma_);
    covariance.block<3, 3>(0, 0) = position_covariance;
    covariance.block<3, 3>(6, 6) = pose.so3().inverse().matrix() * velocity_covariance * pose.so3().matrix();
    covariance(5, 5) = course_yaw_sigma_rad_ * course_yaw_sigma_rad_;
    eskf_.Initialize(measurement.stamp, pose, velocity_body, Eigen::Vector3d::Zero(), covariance);
}

void LocalizationSystem::UpdateFromRtkIns(const RtkInsMeasurement& measurement) {
    if (mode_ == Mode::RTK_ONLY && !eskf_.Initialized()) {
        InitializeEskfFromRtkIns(measurement);
        if (eskf_.Initialized()) PublishResult(BuildEskfResult(measurement.stamp, "ESKF initialized from RTK UTM ENU", true));
        return;
    }
    if (!eskf_.Initialized()) return;

    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Matrix3d position_covariance;
    Eigen::Matrix3d velocity_covariance;
    if (!TransformRtkInsToFilterFrame(measurement, &position, &velocity, &position_covariance, &velocity_covariance)) return;

    bool accepted = false;
    double position_distance = 0.0;
    double velocity_distance = 0.0;
    double yaw_distance = 0.0;
    if (measurement.position_valid && use_rtk_position_) accepted = eskf_.UpdatePositionWithLeverArm(measurement.stamp, position, rtk_ins_lever_arm_tracking_, position_covariance, rtk_position_axes_, rtk_position_gate_chi2_, &position_distance) || accepted;
    if (measurement.velocity_valid && use_rtk_velocity_) accepted = eskf_.UpdateMapVelocityWithLeverArm(measurement.stamp, velocity, rtk_ins_lever_arm_tracking_, velocity_covariance, rtk_velocity_axes_, rtk_velocity_gate_chi2_, &velocity_distance) || accepted;
    if (measurement.yaw_valid) {
        double yaw = measurement.yaw_enu;
        if (mode_ != Mode::RTK_ONLY) yaw = WrapAngle(yaw + std::atan2(map_from_enu_rotation_(1, 0), map_from_enu_rotation_(0, 0)));
        const double yaw_variance = measurement.yaw_variance > 0.0 ? measurement.yaw_variance : course_yaw_sigma_rad_ * course_yaw_sigma_rad_;
        accepted = eskf_.UpdateYaw(measurement.stamp, yaw, yaw_variance, course_yaw_gate_chi2_, &yaw_distance) || accepted;
    }
    const double horizontal_speed = velocity.head<2>().norm();
    if (use_course_yaw_ && measurement.velocity_valid && horizontal_speed >= course_yaw_min_speed_) accepted = eskf_.UpdateYaw(measurement.stamp, std::atan2(velocity.y(), velocity.x()), course_yaw_sigma_rad_ * course_yaw_sigma_rad_, course_yaw_gate_chi2_, &yaw_distance) || accepted;
    if (accepted) {
        PublishResult(BuildEskfResult(measurement.stamp, "ESKF RTK update", true));
    } else {
        LOG_EVERY_N(WARNING, 20) << "[LOCALIZATION_ESKF] RTK/INS rejected, pos_d2=" << position_distance << ", vel_d2=" << velocity_distance << ", yaw_d2=" << yaw_distance;
    }
}

bool LocalizationSystem::TransformRtkInsToFilterFrame(const RtkInsMeasurement& measurement, Eigen::Vector3d* position, Eigen::Vector3d* velocity, Eigen::Matrix3d* position_covariance, Eigen::Matrix3d* velocity_covariance) const {
    if (!position || !velocity || !position_covariance || !velocity_covariance) return false;
    const Eigen::Matrix3d position_covariance_enu = measurement.position_covariance_valid ? measurement.position_covariance_enu : DiagonalCovariance(rtk_position_sigma_xy_, rtk_position_sigma_z_);
    const Eigen::Matrix3d velocity_covariance_enu = (measurement.velocity_covariance_valid ? measurement.velocity_covariance_enu : DiagonalCovariance(rtk_velocity_sigma_xy_, rtk_velocity_sigma_z_)) * rtk_velocity_covariance_scale_;
    if (mode_ == Mode::RTK_ONLY) {
        *position = measurement.position_enu;
        *velocity = measurement.velocity_enu;
        *position_covariance = position_covariance_enu;
        *velocity_covariance = velocity_covariance_enu;
        return true;
    }
    if (!map_from_enu_ready_) return false;
    *position = map_from_enu_rotation_ * measurement.position_enu + map_from_enu_translation_;
    *velocity = map_from_enu_rotation_ * measurement.velocity_enu;
    *position_covariance = map_from_enu_rotation_ * position_covariance_enu * map_from_enu_rotation_.transpose();
    *velocity_covariance = map_from_enu_rotation_ * velocity_covariance_enu * map_from_enu_rotation_.transpose();
    return true;
}

bool LocalizationSystem::FindClosestRtk(double stamp, RtkInsMeasurement* measurement) const {
    if (!measurement || rtk_history_.empty()) return false;
    double best_age = std::numeric_limits<double>::infinity();
    const RtkInsMeasurement* best = nullptr;
    for (const auto& candidate : rtk_history_) {
        const double age = std::fabs(candidate.stamp - stamp);
        if (age < best_age) { best_age = age; best = &candidate; }
    }
    if (!best || best_age > map_alignment_max_age_) return false;
    *measurement = *best;
    return true;
}

bool LocalizationSystem::TryInitializeMapFromEnu(const loc::LocalizationResult& ndt) {
    if (map_from_enu_ready_) return true;
    if (map_from_enu_mode_ != "trajectory_alignment") return false;
    RtkInsMeasurement measurement;
    if (!FindClosestRtk(ndt.timestamp_, &measurement) || !measurement.position_valid) return false;
    if (std::fabs(measurement.stamp - last_alignment_rtk_stamp_) < 1e-6) return false;
    MapAlignmentPair pair;
    pair.rtk_stamp = measurement.stamp;
    pair.position_enu = measurement.position_enu;
    pair.position_map = ndt.pose_.translation() + ndt.pose_.so3().matrix() * rtk_ins_lever_arm_tracking_;
    map_alignment_pairs_.push_back(pair);
    last_alignment_rtk_stamp_ = measurement.stamp;
    return SolveMapFromEnuAlignment();
}

bool LocalizationSystem::SolveMapFromEnuAlignment() {
    if (static_cast<int>(map_alignment_pairs_.size()) < map_alignment_min_pairs_) return false;
    double maximum_baseline = 0.0;
    for (std::size_t i = 0; i < map_alignment_pairs_.size(); ++i) {
        for (std::size_t j = i + 1; j < map_alignment_pairs_.size(); ++j) maximum_baseline = std::max(maximum_baseline, (map_alignment_pairs_[i].position_enu.head<2>() - map_alignment_pairs_[j].position_enu.head<2>()).norm());
    }
    if (maximum_baseline < map_alignment_min_baseline_) return false;

    Eigen::Vector2d mean_enu = Eigen::Vector2d::Zero();
    Eigen::Vector2d mean_map = Eigen::Vector2d::Zero();
    for (const auto& pair : map_alignment_pairs_) { mean_enu += pair.position_enu.head<2>(); mean_map += pair.position_map.head<2>(); }
    mean_enu /= static_cast<double>(map_alignment_pairs_.size());
    mean_map /= static_cast<double>(map_alignment_pairs_.size());
    double cosine_term = 0.0;
    double sine_term = 0.0;
    for (const auto& pair : map_alignment_pairs_) {
        const Eigen::Vector2d q = pair.position_enu.head<2>() - mean_enu;
        const Eigen::Vector2d p = pair.position_map.head<2>() - mean_map;
        cosine_term += q.x() * p.x() + q.y() * p.y();
        sine_term += q.x() * p.y() - q.y() * p.x();
    }
    if (std::hypot(cosine_term, sine_term) < 1e-9) return false;
    const double yaw = std::atan2(sine_term, cosine_term);
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    translation.head<2>() = mean_map - rotation.topLeftCorner<2, 2>() * mean_enu;
    for (const auto& pair : map_alignment_pairs_) translation.z() += pair.position_map.z() - (rotation * pair.position_enu).z();
    translation.z() /= static_cast<double>(map_alignment_pairs_.size());

    double squared_error = 0.0;
    for (const auto& pair : map_alignment_pairs_) {
        const Eigen::Vector2d error = (rotation * pair.position_enu + translation - pair.position_map).head<2>();
        squared_error += error.squaredNorm();
    }
    const double rmse = std::sqrt(squared_error / static_cast<double>(map_alignment_pairs_.size()));
    if (!std::isfinite(rmse) || rmse > map_alignment_max_rmse_) {
        LOG_EVERY_N(WARNING, 10) << "[LOCALIZATION_ESKF] map_from_enu alignment waiting, pairs=" << map_alignment_pairs_.size() << ", baseline=" << maximum_baseline << ", rmse=" << rmse;
        return false;
    }
    map_from_enu_rotation_ = rotation;
    map_from_enu_translation_ = translation;
    map_from_enu_ready_ = true;
    LOG(INFO) << "[LOCALIZATION_ESKF] map_from_enu initialized from trajectory, pairs=" << map_alignment_pairs_.size() << ", yaw_deg=" << yaw / kDegToRad << ", rmse=" << rmse << ", translation=" << translation.transpose();
    return true;
}

void LocalizationSystem::ApplyNdtDerivedTwist(const loc::LocalizationResult& ndt) {
    if (!has_previous_ndt_ || !previous_ndt_.valid_ || !ndt.covariance_valid_ || !previous_ndt_.covariance_valid_) return;
    const double dt = ndt.timestamp_ - previous_ndt_.timestamp_;
    if (dt <= 1e-3 || dt > 2.0) return;
    const Eigen::Vector3d velocity_map = (ndt.pose_.translation() - previous_ndt_.pose_.translation()) / dt;
    const Eigen::Vector3d angular_velocity_body = loc::ESKF::Log(previous_ndt_.pose_.so3().matrix().transpose() * ndt.pose_.so3().matrix()) / dt;
    const Eigen::Matrix<double, 6, 6> current_covariance = NdtCovarianceToRightError(ndt.pose_, ndt.pose_covariance_);
    const Eigen::Matrix<double, 6, 6> previous_covariance = NdtCovarianceToRightError(previous_ndt_.pose_, previous_ndt_.pose_covariance_);
    const Eigen::Matrix3d velocity_covariance = ndt_twist_covariance_scale_ * (current_covariance.block<3, 3>(0, 0) + previous_covariance.block<3, 3>(0, 0)) / (dt * dt);
    const Eigen::Matrix3d angular_covariance = ndt_twist_covariance_scale_ * (current_covariance.block<3, 3>(3, 3) + previous_covariance.block<3, 3>(3, 3)) / (dt * dt);
    eskf_.UpdateMapVelocity(ndt.timestamp_, velocity_map, velocity_covariance, Eigen::Array3i(1, 1, 1), ndt_twist_gate_chi2_);
    eskf_.UpdateAngularVelocity(ndt.timestamp_, angular_velocity_body, angular_covariance, Eigen::Array3i(1, 1, 1), ndt_twist_gate_chi2_);
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
double LocalizationSystem::WrapAngle(double angle) { return std::atan2(std::sin(angle), std::cos(angle)); }
double LocalizationSystem::PoseYaw(const SE3& pose) { return std::atan2(pose.so3().matrix()(1, 0), pose.so3().matrix()(0, 0)); }

SE3 LocalizationSystem::PoseFromPositionYaw(const Eigen::Vector3d& position, double yaw, const Eigen::Matrix3d& roll_pitch_hint) {
    const Eigen::Vector3d rpy = math::RotMtoEuler(roll_pitch_hint);
    return SE3(Eigen::Quaterniond(math::RpyToRotM2(rpy.x(), rpy.y(), yaw)), position);
}

void LocalizationSystem::Reset() {
    if (loc_) loc_->Finish();
    loc_.reset();
    { std::lock_guard<std::mutex> lock(filter_mutex_); eskf_.Reset(); }
    rtk_history_.clear();
    map_alignment_pairs_.clear();
    last_alignment_rtk_stamp_ = -1.0;
    has_previous_ndt_ = false;
    map_from_enu_ready_ = false;
    map_from_enu_rotation_ = Eigen::Matrix3d::Identity();
    map_from_enu_translation_ = Eigen::Vector3d::Zero();
    map_ready_ = false;
    has_initial_guess_ = false;
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
