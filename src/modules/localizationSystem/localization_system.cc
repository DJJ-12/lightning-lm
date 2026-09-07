#include "modules/localizationSystem/localization_system.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
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

Eigen::Matrix3d MapFromTrueEnuRotation(double yaw_degrees) {
    // Explicit map calibration: standard right-handed positive yaw applied
    // to true-ENU coordinate values to obtain map coordinate values. This is
    // a map property and is unrelated to the vehicle course in /INS_data.
    return Eigen::AngleAxisd(
        yaw_degrees * kDegToRad, Eigen::Vector3d::UnitZ())
        .toRotationMatrix();
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

loc::EKF::Covariance InitialEkfCovariance(
    double position_std, double orientation_std, double velocity_std,
    double yaw_rate_std) {
    loc::EKF::Covariance covariance = loc::EKF::Covariance::Zero();
    covariance.block<3, 3>(loc::EKF::kPositionX, loc::EKF::kPositionX) =
        Eigen::Matrix3d::Identity() * position_std * position_std;
    covariance.block<3, 3>(loc::EKF::kRoll, loc::EKF::kRoll) =
        Eigen::Matrix3d::Identity() * orientation_std * orientation_std;
    covariance.block<3, 3>(loc::EKF::kVelocityX, loc::EKF::kVelocityX) =
        Eigen::Matrix3d::Identity() * velocity_std * velocity_std;
    covariance(loc::EKF::kAngularVelocityZ,
               loc::EKF::kAngularVelocityZ) =
        yaw_rate_std * yaw_rate_std;
    return covariance;
}

loc::EKF::Matrix6d FixedPoseNoise(double position_std_x,
                                  double position_std_y,
                                  double position_std_z,
                                  double orientation_std) {
    loc::EKF::Matrix6d covariance = loc::EKF::Matrix6d::Zero();
    covariance(0, 0) = position_std_x * position_std_x;
    covariance(1, 1) = position_std_y * position_std_y;
    covariance(2, 2) = position_std_z * position_std_z;
    covariance(3, 3) = orientation_std * orientation_std;
    covariance(4, 4) = orientation_std * orientation_std;
    covariance(5, 5) = orientation_std * orientation_std;
    return covariance;
}

bool IsUsableCovariance(const Eigen::Matrix2d& covariance) {
    if (!covariance.allFinite() || covariance(0, 0) <= 0.0 ||
        covariance(1, 1) <= 0.0) {
        return false;
    }
    const Eigen::Matrix2d symmetric =
        0.5 * (covariance + covariance.transpose());
    Eigen::LDLT<Eigen::Matrix2d> decomposition(symmetric);
    return decomposition.info() == Eigen::Success &&
           decomposition.isPositive();
}

bool IsUsableCovariance(const Eigen::Matrix3d& covariance) {
    if (!covariance.allFinite() || covariance(0, 0) <= 0.0 ||
        covariance(1, 1) <= 0.0 || covariance(2, 2) <= 0.0) {
        return false;
    }
    const Eigen::Matrix3d symmetric =
        0.5 * (covariance + covariance.transpose());
    Eigen::LDLT<Eigen::Matrix3d> decomposition(symmetric);
    return decomposition.info() == Eigen::Success &&
           decomposition.isPositive();
}

bool IsUsableCovariance(const loc::EKF::Matrix6d& covariance) {
    if (!covariance.allFinite()) return false;
    for (int index = 0; index < 6; ++index) {
        if (covariance(index, index) <= 0.0) return false;
    }
    const loc::EKF::Matrix6d symmetric =
        0.5 * (covariance + covariance.transpose());
    Eigen::LDLT<loc::EKF::Matrix6d> decomposition(symmetric);
    return decomposition.info() == Eigen::Success &&
           decomposition.isPositive();
}

double MahalanobisDistance(const Eigen::VectorXd& residual,
                           const Eigen::MatrixXd& covariance) {
    if (residual.size() == 0 || covariance.rows() != residual.size() ||
        covariance.cols() != residual.size() || !residual.allFinite() ||
        !covariance.allFinite()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const Eigen::MatrixXd symmetric =
        0.5 * (covariance + covariance.transpose());
    Eigen::LDLT<Eigen::MatrixXd> decomposition(symmetric);
    if (decomposition.info() != Eigen::Success ||
        !decomposition.isPositive()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const Eigen::VectorXd solved = decomposition.solve(residual);
    if (decomposition.info() != Eigen::Success || !solved.allFinite()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return residual.dot(solved);
}

void LogNdtRejectionDiagnostic(
    const loc::LocalizationResult& result,
    const loc::EKF::State& predicted,
    const loc::EKF::Matrix6d& predicted_pose_covariance,
    const loc::EKF::Matrix6d& measurement_covariance,
    double mahalanobis_total,
    bool covariance_from_hessian) {
    const Eigen::Vector3d measured_rpy =
        loc::EKF::RpyFromRotation(result.pose_.rotationMatrix());
    Eigen::Matrix<double, 6, 1> residual;
    residual.head<3>() =
        result.pose_.translation() - predicted.position_map;
    for (int axis = 0; axis < 3; ++axis) {
        residual(3 + axis) = loc::EKF::WrapAngle(
            measured_rpy(axis) - predicted.rpy_map(axis));
    }

    const loc::EKF::Matrix6d innovation_covariance =
        predicted_pose_covariance + measurement_covariance;
    const Eigen::Matrix<double, 6, 1> predicted_std =
        predicted_pose_covariance.diagonal().cwiseMax(0.0).cwiseSqrt();
    const Eigen::Matrix<double, 6, 1> measurement_std =
        measurement_covariance.diagonal().cwiseMax(0.0).cwiseSqrt();
    const Eigen::Matrix<double, 6, 1> innovation_std =
        innovation_covariance.diagonal().cwiseMax(0.0).cwiseSqrt();

    Eigen::SelfAdjointEigenSolver<loc::EKF::Matrix6d> measurement_solver(
        measurement_covariance);
    Eigen::SelfAdjointEigenSolver<loc::EKF::Matrix6d> innovation_solver(
        innovation_covariance);
    Eigen::Matrix<double, 6, 1> measurement_eigenvalues =
        Eigen::Matrix<double, 6, 1>::Constant(
            std::numeric_limits<double>::quiet_NaN());
    if (measurement_solver.info() == Eigen::Success) {
        measurement_eigenvalues = measurement_solver.eigenvalues();
    }
    Eigen::Matrix<double, 6, 1> innovation_eigenvalues =
        Eigen::Matrix<double, 6, 1>::Constant(
            std::numeric_limits<double>::quiet_NaN());
    if (innovation_solver.info() == Eigen::Success) {
        innovation_eigenvalues = innovation_solver.eigenvalues();
    }

    const Eigen::Vector2d residual_xy = residual.head<2>();
    const Eigen::Matrix2d innovation_xy =
        innovation_covariance.topLeftCorner<2, 2>();
    const Eigen::Vector2d residual_roll_pitch = residual.segment<2>(3);
    const Eigen::Matrix2d innovation_roll_pitch =
        innovation_covariance.block<2, 2>(3, 3);
    const double nis_xy =
        MahalanobisDistance(residual_xy, innovation_xy);
    const double nis_z = innovation_covariance(2, 2) > 0.0
        ? residual(2) * residual(2) / innovation_covariance(2, 2)
        : std::numeric_limits<double>::quiet_NaN();
    const double nis_roll_pitch = MahalanobisDistance(
        residual_roll_pitch, innovation_roll_pitch);
    const double nis_yaw = innovation_covariance(5, 5) > 0.0
        ? residual(5) * residual(5) / innovation_covariance(5, 5)
        : std::numeric_limits<double>::quiet_NaN();
    const double recomputed_total_nis =
        MahalanobisDistance(residual, innovation_covariance);

    // A compact line every ten rejected frames is enough to identify the bad
    // dimension without flooding a full offline-bag run with 6-D matrices.
    LOG_EVERY_N(WARNING, 10)
        << "[LOCALIZATION_EKF][NDT_REJECT_DIAGNOSTIC] stamp="
        << result.timestamp_
        << ", filter_stamp=" << predicted.stamp
        << ", observation_minus_filter_stamp="
        << result.timestamp_ - predicted.stamp
        << ", mahalanobis_total=" << mahalanobis_total
        << ", recomputed_total_nis=" << recomputed_total_nis
        << ", covariance_source="
        << (covariance_from_hessian ? "hessian" : "fixed_fallback")
        << ", predicted_position=" << predicted.position_map.transpose()
        << ", measured_position=" << result.pose_.translation().transpose()
        << ", residual_position=" << residual.head<3>().transpose()
        << ", predicted_rpy_deg="
        << (predicted.rpy_map / kDegToRad).transpose()
        << ", measured_rpy_deg=" << (measured_rpy / kDegToRad).transpose()
        << ", residual_rpy_deg="
        << (residual.tail<3>() / kDegToRad).transpose()
        << ", predicted_pose_std=" << predicted_std.transpose()
        << ", ndt_measurement_std=" << measurement_std.transpose()
        << ", innovation_std=" << innovation_std.transpose()
        << ", block_nis_xy=" << nis_xy
        << ", block_nis_z=" << nis_z
        << ", block_nis_roll_pitch=" << nis_roll_pitch
        << ", block_nis_yaw=" << nis_yaw
        << ", ndt_cov_eigenvalues=" << measurement_eigenvalues.transpose()
        << ", innovation_eigenvalues=" << innovation_eigenvalues.transpose();
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
    if (mode == "ekf" || mode == "ndt_gps_ekf" ||
        mode == "ndt_gps_ins_ekf" || mode == "ndt_gps") {
        return Mode::EKF_FUSION;
    }
    if (mode == "gps_only" || mode == "gps_ins_only") return Mode::gps_ONLY;
    if (mode == "auto") return Mode::AUTO;
    return Mode::NDT_ONLY;
}

std::string LocalizationSystem::ModeToString(Mode mode) {
    if (mode == Mode::EKF_FUSION) return "ndt_gps_ins_ekf";
    if (mode == Mode::gps_ONLY) return "gps_ins_only";
    if (mode == Mode::AUTO) return "auto";
    return "ndt_only";
}

Eigen::Vector3d LocalizationSystem::ReadVector3(const YAML::Node& node, const Eigen::Vector3d& fallback) {
    if (!node) return fallback;
    const std::vector<double> values = node.as<std::vector<double>>();
    if (values.size() != 3) return fallback;
    return Eigen::Vector3d(values[0], values[1], values[2]);
}

bool LocalizationSystem::Init(const std::string& yaml_path, rclcpp::Node::SharedPtr node) {
    Reset();
    yaml_path_ = yaml_path;
    const YAML::Node yaml = YAML::LoadFile(yaml_path);
    const YAML::Node localization = yaml["localization"];
    const YAML::Node ekf = localization && localization["ekf"]
        ? localization["ekf"]
        : YAML::Node();
    const YAML::Node gps_ins = localization && localization["gps_ins"] ? localization["gps_ins"] : YAML::Node();
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
    gps_topic_ = ReadTopic(common, "gps_topic");
    velocity_topic_ = ReadTopic(common, "velocity_topic");
    wheel_odometry_topic_ = ReadTopic(common, "wheel_odometry_topic");
    gps_ins_lever_arm_tracking_ = ReadVector3(gps_ins && gps_ins["lever_arm_tracking"] ? gps_ins["lever_arm_tracking"] : YAML::Node(), Eigen::Vector3d::Zero());

    // The EKF section is intentionally flat: every number has one physical
    // meaning and is consumed in exactly one place.
    const auto read_ekf = [&ekf](const char* key, double fallback) {
        return ekf && ekf[key] ? ekf[key].as<double>() : fallback;
    };
    loc::EKF::Options filter_options;
    filter_options.process_acceleration_std = read_ekf(
        "process_acceleration_std", filter_options.process_acceleration_std);
    filter_options.process_yaw_acceleration_std = read_ekf(
        "process_yaw_acceleration_std",
        filter_options.process_yaw_acceleration_std);
    filter_options.process_vertical_position_rate_std = read_ekf(
        "process_vertical_position_rate_std",
        filter_options.process_vertical_position_rate_std);
    filter_options.process_roll_pitch_rate_std = read_ekf(
        "process_roll_pitch_rate_std",
        filter_options.process_roll_pitch_rate_std);
    filter_options.max_prediction_step = read_ekf(
        "max_prediction_step", filter_options.max_prediction_step);
    filter_options.gps_position_gate_chi2 = read_ekf(
        "gps_position_gate_chi2", filter_options.gps_position_gate_chi2);
    filter_options.gps_velocity_gate_chi2 = read_ekf(
        "gps_velocity_gate_chi2", filter_options.gps_velocity_gate_chi2);
    filter_options.ndt_pose_gate_chi2 = read_ekf(
        "ndt_pose_gate_chi2", filter_options.ndt_pose_gate_chi2);

    initial_position_std_ = read_ekf(
        "initial_position_std", initial_position_std_);
    initial_orientation_std_ = read_ekf(
        "initial_orientation_std_deg",
        initial_orientation_std_ / kDegToRad) * kDegToRad;
    initial_velocity_std_ = read_ekf(
        "initial_velocity_std", initial_velocity_std_);
    initial_yaw_rate_std_ = read_ekf(
        "initial_yaw_rate_std", initial_yaw_rate_std_);
    gps_position_std_x_ = read_ekf(
        "gps_position_std_x", gps_position_std_x_);
    gps_position_std_y_ = read_ekf(
        "gps_position_std_y", gps_position_std_y_);
    gps_position_std_z_ = read_ekf(
        "gps_position_std_z", gps_position_std_z_);
    gps_velocity_std_x_ = read_ekf(
        "gps_velocity_std_x", gps_velocity_std_x_);
    gps_velocity_std_y_ = read_ekf(
        "gps_velocity_std_y", gps_velocity_std_y_);
    ndt_position_std_x_ = read_ekf(
        "ndt_position_std_x", ndt_position_std_x_);
    ndt_position_std_y_ = read_ekf(
        "ndt_position_std_y", ndt_position_std_y_);
    ndt_position_std_z_ = read_ekf(
        "ndt_position_std_z", ndt_position_std_z_);
    ndt_orientation_std_ = read_ekf(
        "ndt_orientation_std_deg",
        ndt_orientation_std_ / kDegToRad) * kDegToRad;
    const std::array<double, 17> standard_deviations = {
        filter_options.process_acceleration_std,
        filter_options.process_yaw_acceleration_std,
        filter_options.process_vertical_position_rate_std,
        filter_options.process_roll_pitch_rate_std,
        initial_position_std_, initial_orientation_std_,
        initial_velocity_std_, initial_yaw_rate_std_,
        gps_position_std_x_, gps_position_std_y_, gps_position_std_z_,
        gps_velocity_std_x_, gps_velocity_std_y_, ndt_position_std_x_,
        ndt_position_std_y_, ndt_position_std_z_, ndt_orientation_std_};
    const std::array<double, 3> gates = {
        filter_options.gps_position_gate_chi2,
        filter_options.gps_velocity_gate_chi2,
        filter_options.ndt_pose_gate_chi2};
    if (!std::isfinite(filter_options.max_prediction_step) ||
        filter_options.max_prediction_step <= 0.0 ||
        std::any_of(standard_deviations.begin(), standard_deviations.end(),
                    [](double value) {
                        return !std::isfinite(value) || value <= 0.0;
                    }) ||
        std::any_of(gates.begin(), gates.end(), [](double value) {
            return !std::isfinite(value) || value <= 0.0;
        })) {
        LOG(ERROR) << "[LOCALIZATION_EKF] EKF standard deviations, gates and "
                      "max_prediction_step must be finite and positive";
        return false;
    }

    ekf_.Configure(filter_options);

    if (Usesgps() || UsesgpsVelocity()) {
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
    if (node) SetupPublishers(node);
    LOG(INFO) << "[LOCALIZATION_SYSTEM] mode=" << ModeToString(mode_)
              << ", filter=3d_pose_planar_motion_12_state"
              << ", gps_position=" << Usesgps()
              << ", gps_velocity=" << UsesgpsVelocity()
              << ", wheel_input_reserved=" << UsesWheelOdometry()
              << ", imu_input_reserved=" << (!imu_topic_.empty())
              << " (not fused into localization EKF)"
              << ", gps_z_std_floor_m=" << gps_position_std_z_;
    return true;
}

void LocalizationSystem::SetupPublishers(rclcpp::Node::SharedPtr node) {
    if (!node) return;
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node);
    loc_odom_pub_ = node->create_publisher<nav_msgs::msg::Odometry>("/lightning/localization/odom", 10);
    loc_pose_pub_ = node->create_publisher<geometry_msgs::msg::PoseStamped>("/lightning/localization/pose", 10);
    loc_path_pub_ = node->create_publisher<nav_msgs::msg::Path>("/lightning/localization/path", rclcpp::QoS(1).reliable().transient_local());
    raw_gps_path_pub_ = node->create_publisher<nav_msgs::msg::Path>(
        "/lightning/localization/debug/raw_gps_path",
        rclcpp::QoS(1).reliable().transient_local());
    raw_ndt_path_pub_ = node->create_publisher<nav_msgs::msg::Path>(
        "/lightning/localization/debug/raw_ndt_path",
        rclcpp::QoS(1).reliable().transient_local());
    loc_pose_quality_pub_ = node->create_publisher<lightning_interfaces::msg::LocalizationPose>("/lightning/localization/pose_with_quality", 10);
}

bool LocalizationSystem::SetMapPath(const std::string& map_path) {
    if (!loc_ || map_path.empty()) return false;
    map_path_ = map_path;
    map_ready_ = loc_->Init(yaml_path_, map_path_);
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
    if (!UsesLidar()) {
        std::lock_guard<std::mutex> lock(filter_mutex_);
        // No observation timestamp is available here. Store the pose and
        // initialize at the first sensor timestamp instead of timestamp zero;
        // otherwise an offline bag would request a multi-year prediction.
        manual_initial_pose_ = init_pose;
        manual_initial_guess_pending_ = true;
        return true;
    }
    if (!loc_ || !map_ready_) return false;
    const bool accepted = loc_->SetExternalPose(init_pose.unit_quaternion(), init_pose.translation());
    if (accepted) {
        std::lock_guard<std::mutex> lock(filter_mutex_);
        ekf_.Reset();
        manual_initial_guess_pending_ = true;
        has_initial_position_ = false;
    }
    return accepted;
}

loc::LocalizationFrameOutcome LocalizationSystem::ProcessCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud, const loc::LocalizationInputDiagnostic& diagnostic) {
    if (!UsesLidar() || !loc_ || !map_ready_) return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
    return loc_->ProcessLidarMsg(cloud, diagnostic);
}

loc::LocalizationFrameOutcome LocalizationSystem::ProcessCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud, const loc::LocalizationInputDiagnostic& diagnostic) {
    if (!UsesLidar() || !loc_ || !map_ready_) return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
    return loc_->ProcessLivoxLidarMsg(cloud, diagnostic);
}

void LocalizationSystem::ProcessgpsPosition(
    const sensor_msgs::msg::NavSatFix::SharedPtr& fix) {
    if (!Usesgps() || !fix) return;

    Eigen::Vector3d position_map;
    Eigen::Matrix3d covariance_map;
    if (!PositionToMap(*fix, &position_map, &covariance_map)) {
        LOG_EVERY_N(WARNING, 100)
            << "[LOCALIZATION_EKF] gps fix rejected before coordinate conversion"
            << ", status=" << static_cast<int>(fix->status.status)
            << ", latitude=" << fix->latitude
            << ", longitude=" << fix->longitude
            << ", altitude=" << fix->altitude;
        return;
    }
    const double stamp = rclcpp::Time(fix->header.stamp).seconds();
    Eigen::Vector3d receiver_std_enu;
    receiver_std_enu <<
        std::sqrt(std::max(0.0, fix->position_covariance[0])),
        std::sqrt(std::max(0.0, fix->position_covariance[4])),
        std::sqrt(std::max(0.0, fix->position_covariance[8]));
    LOG_EVERY_N(INFO, 50)
        << "[LOCALIZATION_EKF][GNSS_DIAGNOSTIC] stamp=" << stamp
        << ", lat_lon_alt=" << fix->latitude << "," << fix->longitude
        << "," << fix->altitude
        << ", receiver_covariance_type="
        << static_cast<int>(fix->position_covariance_type)
        << ", receiver_std_enu=" << receiver_std_enu.transpose()
        << ", antenna_position_map=" << position_map.transpose()
        << ", observation_std_map="
        << covariance_map.diagonal().cwiseMax(0.0).cwiseSqrt().transpose()
        << ", configured_z_std_floor_m=" << gps_position_std_z_;
    AppendDebugPath(
        &raw_gps_path_, raw_gps_path_pub_, stamp, position_map.head<2>(), 0.0);
    // Green UI trajectory: the raw GNSS antenna position after coordinate
    // conversion, before initialization, prediction, gating or EKF update.
    if (loc_) loc_->UpdategpsObservationVisualization(position_map.head<2>());

    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!ekf_.Initialized()) {
        if (manual_initial_guess_pending_) {
            if (!InitializeManualGuess(stamp)) return;
        } else {
            has_initial_position_ = true;
            initial_position_stamp_ = stamp;
            initial_sensor_position_map_ = position_map;
            TryInitializeEkf();
            return;
        }
    }

    double distance = 0.0;
    const bool accepted = ekf_.UpdategpsPosition(
        stamp, position_map, gpsLeverArmForFilter(),
        covariance_map, -1.0, &distance);
    if (accepted) {
        PublishResult(BuildEkfResult(stamp, "EKF gps position update", true));
    } else {
        PublishPredictionIfAdvanced(
            stamp, "EKF prediction; gps position rejected");
        LOG_EVERY_N(WARNING, 20)
            << "[LOCALIZATION_EKF] gps position rejected, stamp=" << stamp
            << ", mahalanobis=" << distance;
    }
}

void LocalizationSystem::ProcessgpsVelocity(
    const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& velocity) {
    if (!UsesgpsVelocity() || !velocity) return;

    Eigen::Vector2d velocity_map;
    Eigen::Matrix2d covariance_map;
    if (!VelocityToMap(*velocity, &velocity_map, &covariance_map)) return;
    const double stamp = rclcpp::Time(velocity->header.stamp).seconds();

    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!ekf_.Initialized()) {
        if (!manual_initial_guess_pending_ ||
            !InitializeManualGuess(stamp)) {
            return;
        }
    }

    double distance = 0.0;
    const bool accepted = ekf_.UpdateMapVelocity(
        stamp, velocity_map, covariance_map, -1.0, &distance);
    if (accepted) {
        PublishResult(BuildEkfResult(
            stamp, "EKF gps/INS velocity update", true));
    } else {
        PublishPredictionIfAdvanced(
            stamp, "EKF prediction; gps/INS velocity rejected");
        LOG_EVERY_N(WARNING, 20)
            << "[LOCALIZATION_EKF] gps/INS velocity rejected, stamp=" << stamp
            << ", mahalanobis=" << distance;
    }
}

void LocalizationSystem::ProcessWheelOdometry(
    const nav_msgs::msg::Odometry::SharedPtr& odometry) {
    // Reserved input chain. Wheel odometry is intentionally not part of the
    // current 3-D-pose/planar-motion EKF observation model.
    (void)odometry;
}

void LocalizationSystem::ProcessImu(const sensor_msgs::msg::Imu::SharedPtr& imu) {
    // Reserved input chain. Both linear acceleration and angular velocity from
    // this device are unreliable, so localization intentionally uses neither.
    (void)imu;
}

void LocalizationSystem::HandleNdtResult(const loc::LocalizationResult& result) {
    if (result.valid_) {
        const Eigen::Vector2d ndt_position_map =
            result.pose_.translation().head<2>();
        AppendDebugPath(
            &raw_ndt_path_, raw_ndt_path_pub_, result.timestamp_,
            ndt_position_map, PoseYaw(result.pose_));

        // Yellow UI trajectory is the raw NDT output. This happens before the
        // EKF gate so a rejected NDT observation is still visible.
        if (loc_) {
            loc_->UpdateNdtObservationVisualization(ndt_position_map);
        }
    }

    // With LiDAR as the only active localization observation, preserve the
    // original NDT localization architecture: NDT is the final result and its
    // own constant-velocity pose history supplies the next initial guess.
    // There is no reason to route a single observation source through the EKF.
    if (UsesLidar() && !Usesgps() && !UsesgpsVelocity()) {
        if (result.valid_) PublishResult(result);
        return;
    }

    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!result.valid_) {
        if (ekf_.Initialized() && ekf_.PredictTo(result.timestamp_)) {
            PublishResult(BuildEkfResult(
                result.timestamp_, "EKF prediction; NDT invalid", false));
        }
        return;
    }

    bool pose_accepted = false;
    if (!ekf_.Initialized()) {
        if (!result.reliable_) {
            LOG_EVERY_N(WARNING, 20)
                << "[LOCALIZATION_EKF] waiting for first reliable NDT pose";
            return;
        }
        InitializeEkfFromNdt(result);
        pose_accepted = ekf_.Initialized();
    } else {
        loc::EKF::Matrix6d covariance = FixedPoseNoise(
            ndt_position_std_x_, ndt_position_std_y_, ndt_position_std_z_,
            ndt_orientation_std_);
        bool covariance_from_hessian = false;
        if (result.covariance_valid_ &&
            IsUsableCovariance(result.pose_covariance_)) {
            covariance = 0.5 *
                (result.pose_covariance_ + result.pose_covariance_.transpose());
            covariance_from_hessian = true;
        }
        double distance = 0.0;
        pose_accepted = ekf_.UpdateNdtPose(
            result.timestamp_, result.pose_, covariance, -1.0, &distance);
        if (!pose_accepted) {
            // UpdateNdtPose predicts before applying its gate. A rejected
            // observation therefore leaves the EKF at the exact predicted
            // state used to compute this innovation.
            LogNdtRejectionDiagnostic(
                result, ekf_.GetState(), ekf_.PoseCovariance(), covariance,
                distance, covariance_from_hessian);
        }
    }
    if (pose_accepted) {
        PublishResult(BuildEkfResult(
            result.timestamp_, "EKF NDT update", result.reliable_));
    } else {
        PublishPredictionIfAdvanced(
            result.timestamp_, "EKF prediction; NDT rejected");
    }
}

void LocalizationSystem::InitializeEkfFromNdt(
    const loc::LocalizationResult& ndt) {
    const Eigen::Vector3d position = ndt.pose_.translation();
    const Eigen::Vector3d rpy =
        loc::EKF::RpyFromRotation(ndt.pose_.rotationMatrix());
    const loc::EKF::Covariance covariance = InitialEkfCovariance(
        initial_position_std_, initial_orientation_std_, initial_velocity_std_,
        initial_yaw_rate_std_);
    if (ekf_.Initialize(
            ndt.timestamp_, position, rpy, Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero(), covariance)) {
        manual_initial_guess_pending_ = false;
    }
}

bool LocalizationSystem::InitializeFixedMapTransform(
    const YAML::Node& map_from_enu) {
    const std::array<const char*, 4> required = {
        "lat", "lon", "alt", "map_from_true_enu_yaw_deg"};
    for (const char* key : required) {
        if (!map_from_enu || !map_from_enu[key]) {
            LOG(ERROR) << "[LOCALIZATION_EKF] localization.map_from_enu is missing '"
                       << key << "'";
            return false;
        }
    }

    const double latitude_deg = map_from_enu["lat"].as<double>();
    const double longitude_deg = map_from_enu["lon"].as<double>();
    const double altitude_m = map_from_enu["alt"].as<double>();
    const double map_from_enu_yaw_deg =
        map_from_enu["map_from_true_enu_yaw_deg"].as<double>();
    if (!std::isfinite(latitude_deg) || !std::isfinite(longitude_deg) ||
        !std::isfinite(altitude_m) ||
        !std::isfinite(map_from_enu_yaw_deg) ||
        latitude_deg < -90.0 || latitude_deg > 90.0 ||
        longitude_deg < -180.0 || longitude_deg > 180.0) {
        LOG(ERROR) << "[LOCALIZATION_EKF] invalid fixed map reference";
        return false;
    }

    utm_zone_ = math::JsbsimWgs84Enu::UtmZoneFromLongitude(longitude_deg);
    if (utm_zone_ <= 0 ||
        !math::JsbsimWgs84Enu::ForwardUtmDegrees(
            latitude_deg, longitude_deg, altitude_m, utm_zone_,
            &reference_gnss_utm_) ||
        !ComputeUtmFromTrueEnuRotation(
            latitude_deg, longitude_deg, altitude_m, utm_zone_,
            &utm_from_true_enu_rotation_)) {
        LOG(ERROR) << "[LOCALIZATION_EKF] failed to project fixed map reference";
        return false;
    }

    map_from_true_enu_rotation_ =
        MapFromTrueEnuRotation(map_from_enu_yaw_deg);
    const double map_from_true_enu_yaw = std::atan2(
        map_from_true_enu_rotation_(1, 0),
        map_from_true_enu_rotation_(0, 0));

    // NavSatFix is projected into UTM grid east/north, whereas gps/INS velocity
    // uses true ENU. Remove the fixed grid convergence before applying the
    // independently calibrated map yaw.
    map_from_utm_rotation_ =
        map_from_true_enu_rotation_ *
        utm_from_true_enu_rotation_.transpose();

    // At mapping start the tracking origin is the map origin and its +X axis
    // is the map +X axis. The reference GNSS antenna therefore lies at the
    // configured tracking-frame lever arm in map coordinates.
    reference_gnss_map_ = gps_ins_lever_arm_tracking_;
    map_from_utm_translation_ =
        reference_gnss_map_ -
        map_from_utm_rotation_ * reference_gnss_utm_;
    map_from_enu_ready_ =
        map_from_utm_rotation_.allFinite() &&
        map_from_utm_translation_.allFinite() &&
        reference_gnss_utm_.allFinite() && reference_gnss_map_.allFinite();
    if (!map_from_enu_ready_) return false;

    LOG(INFO) << "[LOCALIZATION_EKF] fixed map<-UTM transform ready"
              << ", zone=" << utm_zone_
              << ", reference_gnss_utm=" << reference_gnss_utm_.transpose()
              << ", reference_gnss_map=" << reference_gnss_map_.transpose()
              << ", calibrated_map_from_true_enu_yaw_deg="
              << map_from_true_enu_yaw / kDegToRad
              << ", R_map_true_enu=["
              << map_from_true_enu_rotation_(0, 0) << ","
              << map_from_true_enu_rotation_(0, 1) << ";"
              << map_from_true_enu_rotation_(1, 0) << ","
              << map_from_true_enu_rotation_(1, 1) << "]"
              << ", R_map_utm=["
              << map_from_utm_rotation_(0, 0) << ","
              << map_from_utm_rotation_(0, 1) << ";"
              << map_from_utm_rotation_(1, 0) << ","
              << map_from_utm_rotation_(1, 1) << "]"
              << ", translation=" << map_from_utm_translation_.transpose();
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
    // Rotate the local delta around the configured reference instead of
    // repeatedly multiplying/subtracting large absolute UTM coordinates.
    *position_map =
        reference_gnss_map_ +
        map_from_utm_rotation_ * (position_utm - reference_gnss_utm_);

    Eigen::Matrix3d covariance_enu;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            covariance_enu(row, column) =
                fix.position_covariance[row * 3 + column];
        }
    }
    if (fix.position_covariance_type !=
            sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN &&
        IsUsableCovariance(covariance_enu)) {
        *covariance_map =
            map_from_true_enu_rotation_ * covariance_enu *
            map_from_true_enu_rotation_.transpose();
        *covariance_map =
            0.5 * (*covariance_map + covariance_map->transpose());
    } else {
        covariance_map->setZero();
        (*covariance_map)(0, 0) =
            gps_position_std_x_ * gps_position_std_x_;
        (*covariance_map)(1, 1) =
            gps_position_std_y_ * gps_position_std_y_;
        (*covariance_map)(2, 2) =
            gps_position_std_z_ * gps_position_std_z_;
    }

    // GNSS altitude and the LiDAR map's vertical datum/tracking origin can
    // differ far more than the receiver's millimetre-level internal standard
    // deviation. Apply the configured z standard deviation as a mandatory
    // floor even when NavSatFix reports a valid covariance. Clearing x/z and
    // y/z correlations makes the deliberately weak height observation unable
    // to distort horizontal positioning through a vendor cross term.
    const double vertical_variance = std::max(
        (*covariance_map)(2, 2),
        gps_position_std_z_ * gps_position_std_z_);
    covariance_map->row(2).setZero();
    covariance_map->col(2).setZero();
    (*covariance_map)(2, 2) = vertical_variance;
    return position_map->allFinite() && IsUsableCovariance(*covariance_map);
}

bool LocalizationSystem::VelocityToMap(
    const geometry_msgs::msg::TwistWithCovarianceStamped& velocity,
    Eigen::Vector2d* velocity_map,
    Eigen::Matrix2d* covariance_map) const {
    if (!velocity_map || !covariance_map || !map_from_enu_ready_) {
        return false;
    }
    const double stamp = rclcpp::Time(velocity.header.stamp).seconds();
    const Eigen::Vector2d velocity_enu(
        velocity.twist.twist.linear.x,
        velocity.twist.twist.linear.y);
    if (!std::isfinite(stamp) || !velocity_enu.allFinite()) return false;

    const Eigen::Matrix2d rotation =
        map_from_true_enu_rotation_.topLeftCorner<2, 2>();
    *velocity_map = rotation * velocity_enu;

    Eigen::Matrix2d covariance_enu;
    covariance_enu << velocity.twist.covariance[0],
                      velocity.twist.covariance[1],
                      velocity.twist.covariance[6],
                      velocity.twist.covariance[7];
    if (IsUsableCovariance(covariance_enu)) {
        *covariance_map =
            rotation * covariance_enu * rotation.transpose();
        *covariance_map =
            0.5 * (*covariance_map + covariance_map->transpose());
    } else {
        covariance_map->setZero();
        (*covariance_map)(0, 0) =
            gps_velocity_std_x_ * gps_velocity_std_x_;
        (*covariance_map)(1, 1) =
            gps_velocity_std_y_ * gps_velocity_std_y_;
    }
    return velocity_map->allFinite() && IsUsableCovariance(*covariance_map);
}

Eigen::Vector3d LocalizationSystem::gpsLeverArmForFilter() const {
    // The lever-arm equation depends on yaw. NDT is now the only yaw
    // observation, so a GNSS-only run must not let position residuals invent
    // a heading through a non-zero lever arm.
    return UsesLidar()
        ? gps_ins_lever_arm_tracking_
        : Eigen::Vector3d::Zero();
}

void LocalizationSystem::TryInitializeEkf() {
    if (ekf_.Initialized() ||
        manual_initial_guess_pending_ ||
        !has_initial_position_) {
        return;
    }

    const double stamp = initial_position_stamp_;
    const Eigen::Vector3d initial_rpy = Eigen::Vector3d::Zero();
    const bool lever_arm_applied = UsesLidar();
    const Eigen::Vector3d lever_arm = gpsLeverArmForFilter();
    const Eigen::Vector3d tracking_position_map =
        initial_sensor_position_map_ -
        loc::EKF::RotationFromRpy(initial_rpy) * lever_arm;
    loc::EKF::Covariance covariance = InitialEkfCovariance(
        initial_position_std_, kPi, initial_velocity_std_,
        initial_yaw_rate_std_);
    // gps initializes x/y, but its altitude is intentionally weak. Give the
    // first NDT observation enough prior uncertainty to establish map z.
    covariance(loc::EKF::kPositionZ, loc::EKF::kPositionZ) =
        gps_position_std_z_ * gps_position_std_z_;
    if (ekf_.Initialize(
            stamp, tracking_position_map, initial_rpy,
            Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), covariance)) {
        if (UsesLidar()) {
            const SE3 pose = ekf_.Pose();
            if (!loc_ || !map_ready_ ||
                !loc_->SetExternalPose(pose.unit_quaternion(),
                                       pose.translation())) {
                LOG(ERROR) << "[LOCALIZATION_EKF] failed to pass automatic "
                              "EKF initial pose to NDT";
                ekf_.Reset();
                return;
            }
        }
        LOG(INFO) << "[LOCALIZATION_EKF] initialized from gps position"
                  << ", stamp=" << stamp
                  << ", position_map=" << tracking_position_map.transpose()
                  << ", rpy_map_deg=0 0 0"
                  << ", lever_arm_applied=" << lever_arm_applied;
        PublishResult(BuildEkfResult(
            stamp,
            lever_arm_applied
                ? "3-D pose EKF initialized from gps; attitude awaits NDT"
                : "3-D pose EKF initialized from gps antenna position; "
                  "lever arm disabled because attitude is unobserved",
            false));
    }
}

bool LocalizationSystem::InitializeManualGuess(double stamp) {
    if (ekf_.Initialized()) return true;
    if (!manual_initial_guess_pending_ || !std::isfinite(stamp)) return false;
    const Eigen::Vector3d position = manual_initial_pose_.translation();
    const Eigen::Vector3d rpy = loc::EKF::RpyFromRotation(
        manual_initial_pose_.rotationMatrix());
    const loc::EKF::Covariance covariance = InitialEkfCovariance(
        initial_position_std_, initial_orientation_std_, initial_velocity_std_,
        initial_yaw_rate_std_);
    if (!ekf_.Initialize(
            stamp, position, rpy, Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero(), covariance)) {
        return false;
    }
    manual_initial_guess_pending_ = false;
    LOG(INFO) << "[LOCALIZATION_EKF] initialized from manual pose at first "
                 "observation stamp=" << stamp;
    return true;
}

void LocalizationSystem::PublishPredictionIfAdvanced(
    double stamp, const std::string& message) {
    // Every update function predicts first. If the observation is then gated
    // out, publish that new predicted state; do nothing for stale input whose
    // timestamp could not advance the filter.
    if (ekf_.Initialized() &&
        std::fabs(ekf_.GetState().stamp - stamp) <= 1e-6) {
        PublishResult(BuildEkfResult(stamp, message, false));
    }
}

loc::LocalizationResult LocalizationSystem::BuildEkfResult(
    double stamp, const std::string& message, bool reliable) const {
    loc::LocalizationResult result;
    if (!ekf_.Initialized()) return result;
    result.timestamp_ = stamp;
    result.valid_ = true;
    result.localization_valid_ = true;
    result.status_ = loc::LocalizationStatus::GOOD;
    result.reliable_ = reliable;
    result.confidence_ = 1.0 /
        (1.0 + std::sqrt(std::max(
            0.0, ekf_.P()(0, 0) + ekf_.P()(1, 1))));
    result.pose_ = ekf_.Pose();
    result.velocity_map_ = ekf_.GetState().velocity_map;
    result.angular_velocity_body_ = ekf_.GetState().angular_velocity;
    result.pose_covariance_ = ekf_.PoseCovariance();
    result.covariance_valid_ = true;
    result.twist_covariance_ = ekf_.TwistCovarianceBody();
    result.twist_covariance_valid_ = true;
    result.frame_id_ = output_frame_;
    result.message_ = message;
    return result;
}

void LocalizationSystem::PublishResult(const loc::LocalizationResult& input) {
    loc::LocalizationResult result = input;
    if (result.frame_id_.empty()) result.frame_id_ = output_frame_;
    { std::lock_guard<std::mutex> lock(result_mutex_); latest_result_ = result; }
    if (!result.valid_) return;
    // The red trajectory receives exactly the same final localization result
    // as the ROS publishers: raw NDT in LiDAR-only mode, EKF in fusion mode.
    // Raw gps and raw NDT observations use independent green/yellow paths.
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
    if (result.covariance_valid_) for (int row = 0; row < 6; ++row) for (int col = 0; col < 6; ++col) odometry.pose.covariance[row * 6 + col] = result.pose_covariance_(row, col);
    const Eigen::Vector3d velocity_body =
        result.pose_.so3().inverse().matrix() * result.velocity_map_;
    odometry.twist.twist.linear.x = velocity_body.x();
    odometry.twist.twist.linear.y = velocity_body.y();
    odometry.twist.twist.linear.z = velocity_body.z();
    odometry.twist.twist.angular.x = result.angular_velocity_body_.x();
    odometry.twist.twist.angular.y = result.angular_velocity_body_.y();
    odometry.twist.twist.angular.z = result.angular_velocity_body_.z();
    if (result.twist_covariance_valid_) {
        for (int row = 0; row < 6; ++row) {
            for (int col = 0; col < 6; ++col) {
                odometry.twist.covariance[row * 6 + col] =
                    result.twist_covariance_(row, col);
            }
        }
    }
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

void LocalizationSystem::AppendDebugPath(
    nav_msgs::msg::Path* path,
    const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr& publisher,
    double stamp, const Eigen::Vector2d& position, double yaw) {
    if (!path || !std::isfinite(stamp) || !position.allFinite() ||
        !std::isfinite(yaw)) {
        return;
    }
    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = math::FromSec(stamp);
    pose.header.frame_id = output_frame_;
    pose.pose.position.x = position.x();
    pose.pose.position.y = position.y();
    pose.pose.position.z = 0.0;
    const Eigen::Quaterniond quaternion(
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    pose.pose.orientation.x = quaternion.x();
    pose.pose.orientation.y = quaternion.y();
    pose.pose.orientation.z = quaternion.z();
    pose.pose.orientation.w = quaternion.w();

    std::lock_guard<std::mutex> lock(debug_path_mutex_);
    path->header = pose.header;
    path->poses.push_back(pose);
    if (path->poses.size() > 100000) {
        path->poses.erase(path->poses.begin(), path->poses.begin() + 1000);
    }
    if (publisher) publisher->publish(*path);
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
    { std::lock_guard<std::mutex> lock(filter_mutex_); ekf_.Reset(); }
    lidar_topic_.clear();
    livox_lidar_topic_.clear();
    imu_topic_.clear();
    gps_topic_.clear();
    velocity_topic_.clear();
    wheel_odometry_topic_.clear();
    with_ui_ = false;
    utm_zone_ = 0;
    map_from_enu_ready_ = false;
    map_from_utm_rotation_ = Eigen::Matrix3d::Identity();
    map_from_utm_translation_ = Eigen::Vector3d::Zero();
    utm_from_true_enu_rotation_ = Eigen::Matrix3d::Identity();
    map_from_true_enu_rotation_ = Eigen::Matrix3d::Identity();
    reference_gnss_utm_ = Eigen::Vector3d::Zero();
    reference_gnss_map_ = Eigen::Vector3d::Zero();
    gps_ins_lever_arm_tracking_ = Eigen::Vector3d::Zero();
    initial_position_std_ = 0.5;
    initial_orientation_std_ = 3.0 * kDegToRad;
    initial_velocity_std_ = 2.0;
    initial_yaw_rate_std_ = 0.5;
    gps_position_std_x_ = 0.05;
    gps_position_std_y_ = 0.05;
    gps_position_std_z_ = 100.0;
    gps_velocity_std_x_ = 0.10;
    gps_velocity_std_y_ = 0.10;
    ndt_position_std_x_ = 0.10;
    ndt_position_std_y_ = 0.10;
    ndt_position_std_z_ = 0.20;
    ndt_orientation_std_ = 1.0 * kDegToRad;
    has_initial_position_ = false;
    initial_position_stamp_ = 0.0;
    initial_sensor_position_map_.setZero();
    map_ready_ = false;
    manual_initial_guess_pending_ = false;
    manual_initial_pose_ = SE3();
    path_ = nav_msgs::msg::Path();
    {
        std::lock_guard<std::mutex> lock(debug_path_mutex_);
        raw_gps_path_ = nav_msgs::msg::Path();
        raw_ndt_path_ = nav_msgs::msg::Path();
    }
    { std::lock_guard<std::mutex> lock(result_mutex_); latest_result_ = loc::LocalizationResult(); }
    tf_broadcaster_.reset();
    loc_odom_pub_.reset();
    loc_pose_pub_.reset();
    loc_path_pub_.reset();
    raw_gps_path_pub_.reset();
    raw_ndt_path_pub_.reset();
    loc_pose_quality_pub_.reset();
}

}  // namespace lightning::modules
