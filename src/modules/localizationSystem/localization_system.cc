#include "modules/localizationSystem/localization_system.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <utility>
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

bool ReadTransformVector3(const YAML::Node& node,
                          Eigen::Vector3d* vector) {
    if (!vector || !node) return false;
    if (node.IsSequence() && node.size() == 3) {
        *vector = Eigen::Vector3d(
            node[0].as<double>(), node[1].as<double>(),
            node[2].as<double>());
        return vector->allFinite();
    }
    if (node.IsMap() && node["x"] && node["y"] && node["z"]) {
        *vector = Eigen::Vector3d(
            node["x"].as<double>(), node["y"].as<double>(),
            node["z"].as<double>());
        return vector->allFinite();
    }
    return false;
}

bool ReadTransformQuaternion(const YAML::Node& node,
                             Eigen::Quaterniond* quaternion) {
    if (!quaternion || !node) return false;
    if (node.IsSequence() && node.size() == 4) {
        *quaternion = Eigen::Quaterniond(
            node[3].as<double>(), node[0].as<double>(),
            node[1].as<double>(), node[2].as<double>());
    } else if (node.IsMap() && node["x"] && node["y"] && node["z"] &&
               node["w"]) {
        *quaternion = Eigen::Quaterniond(
            node["w"].as<double>(), node["x"].as<double>(),
            node["y"].as<double>(), node["z"].as<double>());
    } else {
        return false;
    }
    if (!quaternion->coeffs().allFinite() || quaternion->norm() < 1e-9) {
        return false;
    }
    quaternion->normalize();
    return true;
}

bool ReadTransformCovariance(
    const YAML::Node& node,
    Eigen::Matrix<double, 6, 6>* covariance) {
    if (!covariance) return false;
    covariance->setZero();
    if (!node) return true;
    if (!node.IsSequence() || node.size() != 36) return false;
    for (int row = 0; row < 6; ++row) {
        for (int column = 0; column < 6; ++column) {
            (*covariance)(row, column) =
                node[row * 6 + column].as<double>();
        }
    }
    if (!covariance->allFinite()) return false;
    *covariance = 0.5 * (*covariance + covariance->transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(
        *covariance);
    return solver.info() == Eigen::Success &&
           solver.eigenvalues().minCoeff() >= -1e-10;
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

bool LocalizationSystem::Init(const std::string& yaml_path,
                              rclcpp::Node::SharedPtr node) {
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
    gps1_topic_ = ReadTopic(common, "gps1_topic");
    gps2_topic_ = ReadTopic(common, "gps2_topic");
    velocity_topic_ = ReadTopic(common, "velocity_topic");
    wheel_odometry_topic_ = ReadTopic(common, "wheel_odometry_topic");
    gps1_lever_arm_tracking_ = ReadVector3(
        gps_ins && gps_ins["lever_arm_tracking"]
            ? gps_ins["lever_arm_tracking"]
            : YAML::Node(),
        Eigen::Vector3d::Zero());
    gps2_lever_arm_tracking_ = ReadVector3(
        gps_ins && gps_ins["gps2_lever_arm_tracking"]
            ? gps_ins["gps2_lever_arm_tracking"]
            : YAML::Node(),
        Eigen::Vector3d::Zero());
    const auto read_gps_ins = [&gps_ins](const char* key, double fallback) {
        return gps_ins && gps_ins[key]
            ? gps_ins[key].as<double>()
            : fallback;
    };
    dual_gps_sync_tolerance_sec_ = read_gps_ins(
        "dual_gps_sync_tolerance_sec", dual_gps_sync_tolerance_sec_);
    dual_gps_baseline_length_tolerance_m_ = read_gps_ins(
        "dual_gps_baseline_length_tolerance_m",
        dual_gps_baseline_length_tolerance_m_);

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
    filter_options.dual_gps_pose_gate_chi2 = read_ekf(
        "dual_gps_pose_gate_chi2",
        filter_options.dual_gps_pose_gate_chi2);
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
        filter_options.dual_gps_pose_gate_chi2,
        filter_options.gps_velocity_gate_chi2,
        filter_options.ndt_pose_gate_chi2};
    if (!std::isfinite(filter_options.max_prediction_step) ||
        filter_options.max_prediction_step <= 0.0 ||
        !std::isfinite(dual_gps_sync_tolerance_sec_) ||
        dual_gps_sync_tolerance_sec_ <= 0.0 ||
        !std::isfinite(dual_gps_baseline_length_tolerance_m_) ||
        dual_gps_baseline_length_tolerance_m_ <= 0.0 ||
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
    if (UsesDualGps() &&
        (gps2_lever_arm_tracking_ - gps1_lever_arm_tracking_).norm() < 0.10) {
        LOG(ERROR) << "[LOCALIZATION_EKF] dual GPS requires distinct antenna "
                      "lever arms with a baseline of at least 0.10 m";
        return false;
    }

    ekf_.Configure(filter_options);

    if (UsesDualGps() || UsesGpsVelocity()) {
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
              << ", dual_gps_pose=" << UsesDualGps()
              << ", gps_sync_tolerance_sec="
              << dual_gps_sync_tolerance_sec_
              << ", gps_velocity=" << UsesGpsVelocity()
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

void LocalizationSystem::ProcessGps1(
    const sensor_msgs::msg::NavSatFix::SharedPtr& fix) {
    ProcessGps(fix, true);
}

void LocalizationSystem::ProcessGps2(
    const sensor_msgs::msg::NavSatFix::SharedPtr& fix) {
    ProcessGps(fix, false);
}

void LocalizationSystem::ProcessGps(
    const sensor_msgs::msg::NavSatFix::SharedPtr& fix, bool gps1) {
    if (!UsesDualGps() || !fix) return;
    const double stamp = rclcpp::Time(fix->header.stamp).seconds();
    if (!std::isfinite(stamp)) return;

    sensor_msgs::msg::NavSatFix::SharedPtr synchronized_gps1;
    sensor_msgs::msg::NavSatFix::SharedPtr synchronized_gps2;
    {
        std::lock_guard<std::mutex> lock(dual_gps_mutex_);
        if (gps1) {
            pending_gps1_ = fix;
        } else {
            pending_gps2_ = fix;
        }
        if (!pending_gps1_ || !pending_gps2_) return;

        const double gps1_stamp =
            rclcpp::Time(pending_gps1_->header.stamp).seconds();
        const double gps2_stamp =
            rclcpp::Time(pending_gps2_->header.stamp).seconds();
        const double time_difference = gps1_stamp - gps2_stamp;
        if (std::fabs(time_difference) <= dual_gps_sync_tolerance_sec_) {
            synchronized_gps1 = std::move(pending_gps1_);
            synchronized_gps2 = std::move(pending_gps2_);
        } else if (time_difference < 0.0) {
            pending_gps1_.reset();
        } else {
            pending_gps2_.reset();
        }
        if (!synchronized_gps1 || !synchronized_gps2) {
            LOG_EVERY_N(WARNING, 100)
                << "[LOCALIZATION_EKF][DUAL_GPS] unmatched sample discarded"
                << ", dt_sec=" << time_difference
                << ", tolerance_sec=" << dual_gps_sync_tolerance_sec_;
            return;
        }
    }
    HandleDualGpsPair(synchronized_gps1, synchronized_gps2);
}

void LocalizationSystem::HandleDualGpsPair(
    const sensor_msgs::msg::NavSatFix::SharedPtr& gps1,
    const sensor_msgs::msg::NavSatFix::SharedPtr& gps2) {
    if (!gps1 || !gps2) return;

    Eigen::Vector3d gps1_position_map;
    Eigen::Vector3d gps2_position_map;
    Eigen::Matrix3d gps1_covariance_map;
    Eigen::Matrix3d gps2_covariance_map;
    if (!PositionToMap(
            *gps1, &gps1_position_map, &gps1_covariance_map) ||
        !PositionToMap(
            *gps2, &gps2_position_map, &gps2_covariance_map)) {
        LOG_EVERY_N(WARNING, 100)
            << "[LOCALIZATION_EKF][DUAL_GPS] synchronized pair rejected "
               "during coordinate conversion";
        return;
    }

    const Eigen::Vector3d baseline_body =
        gps2_lever_arm_tracking_ - gps1_lever_arm_tracking_;
    const Eigen::Vector3d baseline_map =
        gps2_position_map - gps1_position_map;
    const double expected_baseline_m = baseline_body.norm();
    const double measured_baseline_m = baseline_map.norm();
    const double baseline_error_m =
        std::fabs(measured_baseline_m - expected_baseline_m);
    if (!std::isfinite(measured_baseline_m) ||
        measured_baseline_m < 0.10 ||
        baseline_error_m > dual_gps_baseline_length_tolerance_m_) {
        LOG_EVERY_N(WARNING, 50)
            << "[LOCALIZATION_EKF][DUAL_GPS] baseline rejected"
            << ", measured_m=" << measured_baseline_m
            << ", expected_m=" << expected_baseline_m
            << ", error_m=" << baseline_error_m
            << ", tolerance_m="
            << dual_gps_baseline_length_tolerance_m_;
        return;
    }

    const double gps1_stamp = rclcpp::Time(gps1->header.stamp).seconds();
    const double gps2_stamp = rclcpp::Time(gps2->header.stamp).seconds();
    // Use the newer timestamp so a pair assembled after another observation
    // can never make the EKF step backwards in time.
    const double stamp = std::max(gps1_stamp, gps2_stamp);

    // This minimum-angle transform is used only for independent raw-GPS
    // visualization and initialization when no manual pose exists. The EKF
    // update below consumes both antenna coordinates directly and therefore
    // does not pretend that one baseline observes all three rotation axes.
    Eigen::Quaterniond approximate_rotation =
        Eigen::Quaterniond::FromTwoVectors(baseline_body, baseline_map);
    approximate_rotation.normalize();
    const Eigen::Matrix3d approximate_rotation_matrix =
        approximate_rotation.toRotationMatrix();
    const Eigen::Vector3d gps_body_position_map = 0.5 *
        ((gps1_position_map -
          approximate_rotation_matrix * gps1_lever_arm_tracking_) +
         (gps2_position_map -
          approximate_rotation_matrix * gps2_lever_arm_tracking_));
    const double approximate_yaw =
        loc::EKF::RpyFromRotation(approximate_rotation_matrix).z();

    AppendDebugPath(
        &raw_gps_path_, raw_gps_path_pub_, stamp,
        gps_body_position_map.head<2>(), approximate_yaw);
    if (loc_) {
        loc_->UpdategpsObservationVisualization(
            gps_body_position_map.head<2>());
    }

    LOG_EVERY_N(INFO, 50)
        << "[LOCALIZATION_EKF][DUAL_GPS] synchronized observation"
        << ", stamp=" << stamp
        << ", sync_dt_sec=" << gps1_stamp - gps2_stamp
        << ", gps1_map=" << gps1_position_map.transpose()
        << ", gps2_map=" << gps2_position_map.transpose()
        << ", baseline_map=" << baseline_map.transpose()
        << ", baseline_body=" << baseline_body.transpose()
        << ", raw_body_position_map="
        << gps_body_position_map.transpose();

    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!ekf_.Initialized()) {
        if (manual_initial_guess_pending_) {
            if (!InitializeManualGuess(stamp)) return;
        } else {
            const Eigen::Vector3d initial_rpy =
                loc::EKF::RpyFromRotation(approximate_rotation_matrix);
            loc::EKF::Covariance initial_covariance =
                InitialEkfCovariance(
                    initial_position_std_, kPi, initial_velocity_std_,
                    initial_yaw_rate_std_);
            initial_covariance.block<3, 3>(
                loc::EKF::kPositionX, loc::EKF::kPositionX) =
                0.25 * (gps1_covariance_map + gps2_covariance_map);
            if (!ekf_.Initialize(
                    stamp, gps_body_position_map, initial_rpy,
                    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                    initial_covariance)) {
                return;
            }
            if (UsesLidar() &&
                (!loc_ || !map_ready_ ||
                 !loc_->SetExternalPose(
                     approximate_rotation, gps_body_position_map))) {
                LOG(ERROR) << "[LOCALIZATION_EKF][DUAL_GPS] failed to pass "
                              "initial pose to NDT";
                ekf_.Reset();
                return;
            }
        }
    }

    double distance = 0.0;
    const bool accepted = ekf_.UpdateDualGpsPose(
        stamp, gps1_position_map, gps2_position_map,
        gps1_lever_arm_tracking_, gps2_lever_arm_tracking_,
        gps1_covariance_map, gps2_covariance_map,
        -1.0, &distance);
    if (accepted) {
        PublishResult(BuildEkfResult(
            stamp, "EKF dual-GPS antenna pose update", true));
    } else {
        PublishPredictionIfAdvanced(
            stamp, "EKF prediction; dual-GPS pose rejected");
        LOG_EVERY_N(WARNING, 20)
            << "[LOCALIZATION_EKF][DUAL_GPS] observation rejected"
            << ", stamp=" << stamp
            << ", mahalanobis=" << distance;
    }
}

void LocalizationSystem::ProcessgpsVelocity(
    const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& velocity) {
    if (!UsesGpsVelocity() || !velocity) return;

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
    if (UsesLidar() && !UsesDualGps() && !UsesGpsVelocity()) {
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
    direct_map_from_enu_ = false;
    map_from_enu_ready_ = false;
    map_from_enu_rotation_.setIdentity();
    map_from_enu_translation_.setZero();
    map_from_enu_covariance_.setZero();

    const YAML::Node transform = map_from_enu;

    const bool has_translation = transform && transform["translation"];
    const bool has_quaternion = transform && transform["quaternion"];
    if (has_translation || has_quaternion) {
        if (!has_translation || !has_quaternion) {
            LOG(ERROR) << "[LOCALIZATION_EKF] full 3-D map_from_enu requires "
                          "both translation and quaternion";
            return false;
        }
        const YAML::Node origin = transform["enu_origin"]
            ? transform["enu_origin"]
            : transform;
        if (!origin || !origin["lat"] || !origin["lon"] ||
            !origin["alt"]) {
            LOG(ERROR) << "[LOCALIZATION_EKF] full 3-D map_from_enu requires "
                          "enu_origin.lat/lon/alt";
            return false;
        }
        const double latitude_deg = origin["lat"].as<double>();
        const double longitude_deg = origin["lon"].as<double>();
        const double altitude_m = origin["alt"].as<double>();
        Eigen::Quaterniond quaternion;
        if (!std::isfinite(latitude_deg) || !std::isfinite(longitude_deg) ||
            !std::isfinite(altitude_m) || latitude_deg < -90.0 ||
            latitude_deg > 90.0 || longitude_deg < -180.0 ||
            longitude_deg > 180.0 ||
            !ReadTransformVector3(
                transform["translation"],
                &map_from_enu_translation_) ||
            !ReadTransformQuaternion(
                transform["quaternion"], &quaternion) ||
            !ReadTransformCovariance(
                transform["calibration_covariance"],
                &map_from_enu_covariance_) ||
            !enu_projector_.SetOriginDegrees(
                latitude_deg, longitude_deg, altitude_m)) {
            LOG(ERROR) << "[LOCALIZATION_EKF] invalid full 3-D "
                          "map_from_enu calibration";
            return false;
        }

        map_from_enu_rotation_ = quaternion.toRotationMatrix();
        map_from_true_enu_rotation_ = map_from_enu_rotation_;
        reference_gnss_map_ = map_from_enu_translation_;
        direct_map_from_enu_ = true;
        map_from_enu_ready_ = true;
        LOG(INFO) << "[LOCALIZATION_EKF] calibrated full 3-D map<-ENU "
                     "transform ready"
                  << ", enu_origin_lla=" << latitude_deg << ","
                  << longitude_deg << "," << altitude_m
                  << ", translation="
                  << map_from_enu_translation_.transpose()
                  << ", quaternion_xyzw=" << quaternion.x() << ","
                  << quaternion.y() << "," << quaternion.z() << ","
                  << quaternion.w()
                  << ", calibration_std="
                  << map_from_enu_covariance_.diagonal()
                         .cwiseMax(0.0)
                         .cwiseSqrt()
                         .transpose();
        return true;
    }

    // Backward-compatible manual yaw/UTM path. A calibrated result can instead
    // be copied directly from the finish service as translation+quaternion.
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
    reference_gnss_map_ = gps1_lever_arm_tracking_;
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

    Eigen::Vector3d position_enu = Eigen::Vector3d::Zero();
    if (direct_map_from_enu_) {
        // The calibrated transform and the calibrator use the exact same
        // WGS84 -> ECEF -> local ENU definition.
        position_enu = enu_projector_.ForwardDegrees(
            fix.latitude, fix.longitude, fix.altitude);
        if (!position_enu.allFinite()) return false;
        *position_map =
            map_from_enu_rotation_ * position_enu +
            map_from_enu_translation_;
    } else {
        Eigen::Vector3d position_utm;
        if (!math::JsbsimWgs84Enu::ForwardUtmDegrees(
                fix.latitude, fix.longitude, fix.altitude, utm_zone_,
                &position_utm)) {
            return false;
        }
        // Preserve the legacy fixed-yaw behavior for old configuration files.
        *position_map =
            reference_gnss_map_ +
            map_from_utm_rotation_ * (position_utm - reference_gnss_utm_);
    }

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

    if (direct_map_from_enu_ &&
        map_from_enu_covariance_.cwiseAbs().maxCoeff() > 0.0) {
        // Calibration covariance order is [dtheta, dt], with a right rotation
        // perturbation: R' = R Exp(dtheta). This term prevents millimetre-level
        // receiver covariance from being mistaken for millimetre-level
        // certainty in map coordinates.
        Eigen::Matrix<double, 3, 6> calibration_jacobian =
            Eigen::Matrix<double, 3, 6>::Zero();
        calibration_jacobian.block<3, 3>(0, 0) =
            -map_from_enu_rotation_ *
            math::SKEW_SYM_MATRIX(position_enu);
        calibration_jacobian.block<3, 3>(0, 3) =
            Eigen::Matrix3d::Identity();
        *covariance_map +=
            calibration_jacobian * map_from_enu_covariance_ *
            calibration_jacobian.transpose();
        *covariance_map =
            0.5 * (*covariance_map + covariance_map->transpose());
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
    const Eigen::Vector3d velocity_enu_3d(
        velocity.twist.twist.linear.x,
        velocity.twist.twist.linear.y,
        velocity.twist.twist.linear.z);
    if (!std::isfinite(stamp) || !velocity_enu_3d.allFinite()) return false;

    if (direct_map_from_enu_) {
        *velocity_map =
            (map_from_enu_rotation_ * velocity_enu_3d).head<2>();
        Eigen::Matrix3d covariance_enu_3d;
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                covariance_enu_3d(row, column) =
                    velocity.twist.covariance[row * 6 + column];
            }
        }
        if (IsUsableCovariance(covariance_enu_3d)) {
            const Eigen::Matrix3d covariance_map_3d =
                map_from_enu_rotation_ * covariance_enu_3d *
                map_from_enu_rotation_.transpose();
            *covariance_map = covariance_map_3d.topLeftCorner<2, 2>();
        } else {
            covariance_map->setZero();
        }
    } else {
        // Preserve the old 2-D true-ENU velocity conversion for the legacy
        // yaw/UTM configuration.
        const Eigen::Matrix2d rotation =
            map_from_true_enu_rotation_.topLeftCorner<2, 2>();
        *velocity_map = rotation * velocity_enu_3d.head<2>();
        Eigen::Matrix2d covariance_enu;
        covariance_enu << velocity.twist.covariance[0],
                          velocity.twist.covariance[1],
                          velocity.twist.covariance[6],
                          velocity.twist.covariance[7];
        if (IsUsableCovariance(covariance_enu)) {
            *covariance_map =
                rotation * covariance_enu * rotation.transpose();
        } else {
            covariance_map->setZero();
        }
    }
    if (!IsUsableCovariance(*covariance_map)) {
        covariance_map->setZero();
        (*covariance_map)(0, 0) =
            gps_velocity_std_x_ * gps_velocity_std_x_;
        (*covariance_map)(1, 1) =
            gps_velocity_std_y_ * gps_velocity_std_y_;
    }
    if (direct_map_from_enu_ &&
        map_from_enu_covariance_.topLeftCorner<3, 3>()
                .cwiseAbs().maxCoeff() >
            0.0) {
        const Eigen::Matrix<double, 2, 3> rotation_jacobian =
            (-map_from_enu_rotation_ *
             math::SKEW_SYM_MATRIX(velocity_enu_3d))
                .topRows<2>();
        *covariance_map +=
            rotation_jacobian *
            map_from_enu_covariance_.topLeftCorner<3, 3>() *
            rotation_jacobian.transpose();
        *covariance_map =
            0.5 * (*covariance_map + covariance_map->transpose());
    }
    *covariance_map =
        0.5 * (*covariance_map + covariance_map->transpose());
    return velocity_map->allFinite() && IsUsableCovariance(*covariance_map);
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
    gps1_topic_.clear();
    gps2_topic_.clear();
    velocity_topic_.clear();
    wheel_odometry_topic_.clear();
    with_ui_ = false;
    utm_zone_ = 0;
    map_from_enu_ready_ = false;
    direct_map_from_enu_ = false;
    enu_projector_ = math::JsbsimWgs84Enu();
    map_from_enu_rotation_ = Eigen::Matrix3d::Identity();
    map_from_enu_translation_ = Eigen::Vector3d::Zero();
    map_from_enu_covariance_ =
        Eigen::Matrix<double, 6, 6>::Zero();
    map_from_utm_rotation_ = Eigen::Matrix3d::Identity();
    map_from_utm_translation_ = Eigen::Vector3d::Zero();
    utm_from_true_enu_rotation_ = Eigen::Matrix3d::Identity();
    map_from_true_enu_rotation_ = Eigen::Matrix3d::Identity();
    reference_gnss_utm_ = Eigen::Vector3d::Zero();
    reference_gnss_map_ = Eigen::Vector3d::Zero();
    gps1_lever_arm_tracking_ = Eigen::Vector3d::Zero();
    gps2_lever_arm_tracking_ = Eigen::Vector3d::Zero();
    dual_gps_sync_tolerance_sec_ = 0.05;
    dual_gps_baseline_length_tolerance_m_ = 0.15;
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
    {
        std::lock_guard<std::mutex> lock(dual_gps_mutex_);
        pending_gps1_.reset();
        pending_gps2_.reset();
    }
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
