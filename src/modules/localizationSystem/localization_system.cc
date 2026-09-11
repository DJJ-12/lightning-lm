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
    gps_initialization_sample_count_required_ =
        gps_ins && gps_ins["gps_initialization_sample_count"]
            ? gps_ins["gps_initialization_sample_count"].as<int>()
            : gps_initialization_sample_count_required_;

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
        gps_initialization_sample_count_required_ <= 0 ||
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
              << ", gps_initialization_samples="
              << gps_initialization_sample_count_required_
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
        gps_initial_map_body_ready_ = false;
        enu_from_map_ready_ = false;
        enu_projector_ = math::JsbsimWgs84Enu();
        gps_initialization_sample_count_ = 0;
        gps1_initial_enu_sum_.setZero();
        gps2_initial_enu_sum_.setZero();
        return true;
    }
    if (!loc_ || !map_ready_) return false;
    const bool accepted = loc_->SetExternalPose(init_pose.unit_quaternion(), init_pose.translation());
    if (accepted) {
        std::lock_guard<std::mutex> lock(filter_mutex_);
        ekf_.Reset();
        manual_initial_pose_ = SE3();
        manual_initial_guess_pending_ = false;

        // A new NDT seed starts a new localization run.  Any GPS transform from
        // the previous run must be discarded and rebuilt after relocalization.
        gps_initial_map_body_ready_ = false;
        gps_initial_map_body_pose_ = SE3();
        enu_from_map_ready_ = false;
        enu_from_map_rotation_.setIdentity();
        enu_from_map_translation_.setZero();
        enu_projector_ = math::JsbsimWgs84Enu();
        gps_initialization_sample_count_ = 0;
        gps1_initial_enu_sum_.setZero();
        gps2_initial_enu_sum_.setZero();
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

    const double gps1_stamp = rclcpp::Time(gps1->header.stamp).seconds();
    const double gps2_stamp = rclcpp::Time(gps2->header.stamp).seconds();
    const double stamp = std::max(gps1_stamp, gps2_stamp);
    if (!std::isfinite(stamp)) return;

    // In the normal LiDAR workflow NDT owns initialization.  Keep the existing
    // no-LiDAR/manual-pose path usable by initializing the EKF at the first GPS
    // timestamp before starting the same 10-pair GPS initialization.
    if (!gps_initial_map_body_ready_ && !UsesLidar() &&
        manual_initial_guess_pending_) {
        std::lock_guard<std::mutex> lock(filter_mutex_);
        InitializeManualGuess(stamp);
    }

    // The ENU<-MAP transform is deliberately initialized only after NDT has
    // produced the first reliable MAP<-BODY pose.  Before that, GPS has no
    // absolute relationship to this particular map.
    if (!gps_initial_map_body_ready_) {
        LOG_EVERY_N(INFO, 100)
            << "[LOCALIZATION_EKF][GPS_INIT] waiting for reliable NDT "
               "relocalization before collecting dual-GPS initialization data";
        return;
    }

    // The first accepted GPS1 fix defines a local ENU origin.  This is only a
    // numerical origin; the fixed ENU<-MAP yaw/translation below is what ties
    // the local ENU coordinates to the NDT map.
    if (!enu_projector_.Initialized()) {
        const double gps1_stamp = rclcpp::Time(gps1->header.stamp).seconds();
        if (gps1->status.status ==
                sensor_msgs::msg::NavSatStatus::STATUS_NO_FIX ||
            !std::isfinite(gps1_stamp) ||
            !std::isfinite(gps1->latitude) ||
            !std::isfinite(gps1->longitude) ||
            !std::isfinite(gps1->altitude) ||
            !enu_projector_.SetOriginDegrees(
                gps1->latitude, gps1->longitude, gps1->altitude)) {
            return;
        }
        LOG(INFO) << "[LOCALIZATION_EKF][GPS_INIT] local ENU origin set from "
                     "first accepted GPS1 fix"
                  << ", lat=" << gps1->latitude
                  << ", lon=" << gps1->longitude
                  << ", alt=" << gps1->altitude;
    }

    Eigen::Vector3d gps1_position_enu;
    Eigen::Vector3d gps2_position_enu;
    Eigen::Matrix3d gps1_covariance_enu;
    Eigen::Matrix3d gps2_covariance_enu;
    if (!GnssToEnu(*gps1, &gps1_position_enu, &gps1_covariance_enu) ||
        !GnssToEnu(*gps2, &gps2_position_enu, &gps2_covariance_enu)) {
        LOG_EVERY_N(WARNING, 100)
            << "[LOCALIZATION_EKF][DUAL_GPS] synchronized pair rejected "
               "during WGS84->ENU conversion";
        return;
    }

    const Eigen::Vector3d baseline_body =
        gps2_lever_arm_tracking_ - gps1_lever_arm_tracking_;
    const Eigen::Vector3d baseline_enu =
        gps2_position_enu - gps1_position_enu;
    const double expected_baseline_m = baseline_body.norm();
    const double measured_baseline_m = baseline_enu.norm();
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

    // GPS initialization: while the vehicle is stationary, average the first N
    // valid synchronized pairs.  NDT has already fixed MAP<-BODY at the initial
    // pose, so these N pairs are sufficient to determine one fixed yaw-only
    // ENU<-MAP rotation and one 3-D translation.
    if (!enu_from_map_ready_) {
        if (AddGpsInitializationSample(
                gps1_position_enu, gps2_position_enu) &&
            !FinishGpsInitialization()) {
            LOG(ERROR) << "[LOCALIZATION_EKF][GPS_INIT] failed to initialize "
                          "fixed ENU<-MAP transform";
        }
        return;
    }

    // Convert raw GPS only for visualization.  The EKF itself stays in MAP and
    // compares its antenna prediction directly against the ENU measurements.
    const Eigen::Matrix3d rotation_map_enu =
        enu_from_map_rotation_.transpose();
    const Eigen::Vector3d gps1_position_map =
        rotation_map_enu *
        (gps1_position_enu - enu_from_map_translation_);
    const Eigen::Vector3d gps2_position_map =
        rotation_map_enu *
        (gps2_position_enu - enu_from_map_translation_);
    const Eigen::Vector3d baseline_map =
        gps2_position_map - gps1_position_map;

    // Dual GPS directly gives only the baseline heading.  For the debug path,
    // use that yaw only; do not invent a full 3-D attitude from one vector.
    const double body_baseline_yaw =
        std::atan2(baseline_body.y(), baseline_body.x());
    const double map_baseline_yaw =
        std::atan2(baseline_map.y(), baseline_map.x());
    const double gps_body_yaw_map = loc::EKF::WrapAngle(
        map_baseline_yaw - body_baseline_yaw);
    const Eigen::Matrix3d gps_rotation_map_body =
        Eigen::AngleAxisd(gps_body_yaw_map, Eigen::Vector3d::UnitZ())
            .toRotationMatrix();
    const Eigen::Vector3d gps_body_position_map = 0.5 *
        ((gps1_position_map -
          gps_rotation_map_body * gps1_lever_arm_tracking_) +
         (gps2_position_map -
          gps_rotation_map_body * gps2_lever_arm_tracking_));
    AppendDebugPath(
        &raw_gps_path_, raw_gps_path_pub_, stamp,
        gps_body_position_map.head<2>(), gps_body_yaw_map);
    if (loc_) {
        loc_->UpdategpsObservationVisualization(
            gps_body_position_map.head<2>());
    }

    LOG_EVERY_N(INFO, 50)
        << "[LOCALIZATION_EKF][DUAL_GPS] synchronized ENU observation"
        << ", stamp=" << stamp
        << ", sync_dt_sec=" << gps1_stamp - gps2_stamp
        << ", gps1_enu=" << gps1_position_enu.transpose()
        << ", gps2_enu=" << gps2_position_enu.transpose()
        << ", baseline_enu=" << baseline_enu.transpose();

    std::lock_guard<std::mutex> lock(filter_mutex_);
    // GPS is no longer allowed to create the localization state.  NDT (or the
    // explicit manual-pose path when LiDAR is disabled) must initialize first.
    if (!ekf_.Initialized()) return;

    double distance = 0.0;
    const bool accepted = ekf_.UpdateDualGpsPoseEnu(
        stamp, gps1_position_enu, gps2_position_enu,
        gps1_lever_arm_tracking_, gps2_lever_arm_tracking_,
        gps1_covariance_enu, gps2_covariance_enu,
        enu_from_map_rotation_, enu_from_map_translation_,
        -1.0, &distance);
    if (accepted) {
        PublishResult(BuildEkfResult(
            stamp, "EKF dual-GPS ENU antenna update", true));
    } else {
        PublishPredictionIfAdvanced(
            stamp, "EKF prediction; dual-GPS ENU observation rejected");
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
        // With LiDAR enabled, velocity is never allowed to initialize the EKF
        // from the user's rough NDT seed.  Wait for reliable NDT relocalization.
        if (UsesLidar() || !manual_initial_guess_pending_ ||
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
        BeginGpsInitialization(ndt.pose_);
        LOG(INFO) << "[LOCALIZATION_EKF] NDT relocalization initialized EKF"
                  << ", stamp=" << ndt.timestamp_
                  << ", position=" << position.transpose();
        if (UsesDualGps()) {
            LOG(INFO) << "[LOCALIZATION_EKF][GPS_INIT] NDT pose fixed; "
                         "dual-GPS ENU<-MAP initialization now starts";
        }
    }
}

void LocalizationSystem::BeginGpsInitialization(
    const SE3& initial_map_body_pose) {
    if (!UsesDualGps()) return;

    gps_initial_map_body_pose_ = initial_map_body_pose;
    gps_initial_map_body_ready_ = true;
    gps_initialization_sample_count_ = 0;
    gps1_initial_enu_sum_.setZero();
    gps2_initial_enu_sum_.setZero();
    enu_projector_ = math::JsbsimWgs84Enu();
    enu_from_map_ready_ = false;
    enu_from_map_rotation_.setIdentity();
    enu_from_map_translation_.setZero();

    {
        std::lock_guard<std::mutex> lock(dual_gps_mutex_);
        pending_gps1_.reset();
        pending_gps2_.reset();
    }

    LOG(INFO) << "[LOCALIZATION_EKF][GPS_INIT] NDT initialization complete. "
                 "Keep vehicle stationary; collecting first "
              << gps_initialization_sample_count_required_
              << " synchronized dual-GPS pairs"
              << ", initial_map_body_position="
              << initial_map_body_pose.translation().transpose()
              << ", initial_map_body_yaw_deg="
              << PoseYaw(initial_map_body_pose) / kDegToRad;
}

bool LocalizationSystem::AddGpsInitializationSample(
    const Eigen::Vector3d& gps1_enu,
    const Eigen::Vector3d& gps2_enu) {
    if (!gps_initial_map_body_ready_ || enu_from_map_ready_ ||
        !gps1_enu.allFinite() || !gps2_enu.allFinite()) {
        return false;
    }

    gps1_initial_enu_sum_ += gps1_enu;
    gps2_initial_enu_sum_ += gps2_enu;
    ++gps_initialization_sample_count_;

    LOG(INFO) << "[LOCALIZATION_EKF][GPS_INIT] collected pair "
              << gps_initialization_sample_count_ << "/"
              << gps_initialization_sample_count_required_;
    return gps_initialization_sample_count_ >=
           gps_initialization_sample_count_required_;
}

bool LocalizationSystem::FinishGpsInitialization() {
    if (!gps_initial_map_body_ready_ || enu_from_map_ready_ ||
        gps_initialization_sample_count_ <
            gps_initialization_sample_count_required_) {
        return false;
    }

    const double count =
        static_cast<double>(gps_initialization_sample_count_);
    const Eigen::Vector3d gps1_mean_enu =
        gps1_initial_enu_sum_ / count;
    const Eigen::Vector3d gps2_mean_enu =
        gps2_initial_enu_sum_ / count;

    const Eigen::Vector3d baseline_body =
        gps2_lever_arm_tracking_ - gps1_lever_arm_tracking_;
    const Eigen::Matrix3d rotation_map_body0 =
        gps_initial_map_body_pose_.rotationMatrix();
    const Eigen::Vector3d translation_map_body0 =
        gps_initial_map_body_pose_.translation();
    const Eigen::Vector3d baseline_map =
        rotation_map_body0 * baseline_body;
    const Eigen::Vector3d baseline_enu =
        gps2_mean_enu - gps1_mean_enu;

    // The simple initialization deliberately estimates only a Z-axis rotation
    // between MAP and ENU.  The initial NDT pose already contains the vehicle's
    // roll/pitch, so the horizontal MAP baseline is compared directly with the
    // horizontal ENU baseline.
    const double map_horizontal = baseline_map.head<2>().norm();
    const double enu_horizontal = baseline_enu.head<2>().norm();
    if (map_horizontal < 1e-3 || enu_horizontal < 1e-3) {
        LOG(ERROR) << "[LOCALIZATION_EKF][GPS_INIT] horizontal antenna baseline "
                      "is too small to initialize yaw"
                   << ", baseline_map=" << baseline_map.transpose()
                   << ", baseline_enu=" << baseline_enu.transpose();
        return false;
    }

    const double yaw_map_baseline =
        std::atan2(baseline_map.y(), baseline_map.x());
    const double yaw_enu_baseline =
        std::atan2(baseline_enu.y(), baseline_enu.x());
    const double yaw_enu_map = loc::EKF::WrapAngle(
        yaw_enu_baseline - yaw_map_baseline);
    enu_from_map_rotation_ =
        Eigen::AngleAxisd(yaw_enu_map, Eigen::Vector3d::UnitZ())
            .toRotationMatrix();

    // Each antenna independently gives the same ENU<-MAP translation:
    //   t_EM = p_E - R_EM * (R_MB0 * l_B + t_MB0).
    // Averaging the two candidates is the simplest symmetric estimator.
    const Eigen::Vector3d gps1_initial_map =
        rotation_map_body0 * gps1_lever_arm_tracking_ +
        translation_map_body0;
    const Eigen::Vector3d gps2_initial_map =
        rotation_map_body0 * gps2_lever_arm_tracking_ +
        translation_map_body0;
    const Eigen::Vector3d translation_from_gps1 =
        gps1_mean_enu - enu_from_map_rotation_ * gps1_initial_map;
    const Eigen::Vector3d translation_from_gps2 =
        gps2_mean_enu - enu_from_map_rotation_ * gps2_initial_map;
    enu_from_map_translation_ =
        0.5 * (translation_from_gps1 + translation_from_gps2);
    enu_from_map_ready_ =
        enu_from_map_rotation_.allFinite() &&
        enu_from_map_translation_.allFinite();

    if (!enu_from_map_ready_) return false;

    LOG(INFO) << "[LOCALIZATION_EKF][GPS_INIT] fixed ENU<-MAP transform ready"
              << ", samples=" << gps_initialization_sample_count_
              << ", yaw_enu_map_deg=" << yaw_enu_map / kDegToRad
              << ", translation_enu_map="
              << enu_from_map_translation_.transpose()
              << ", mean_baseline_enu=" << baseline_enu.transpose()
              << ", initial_baseline_map=" << baseline_map.transpose();
    return true;
}

bool LocalizationSystem::GnssToEnu(
    const sensor_msgs::msg::NavSatFix& fix,
    Eigen::Vector3d* position_enu,
    Eigen::Matrix3d* covariance_enu) const {
    if (!position_enu || !covariance_enu || !enu_projector_.Initialized()) {
        return false;
    }
    const double stamp = rclcpp::Time(fix.header.stamp).seconds();
    if (fix.status.status == sensor_msgs::msg::NavSatStatus::STATUS_NO_FIX ||
        !std::isfinite(stamp) || !std::isfinite(fix.latitude) ||
        !std::isfinite(fix.longitude) || !std::isfinite(fix.altitude)) {
        return false;
    }

    *position_enu = enu_projector_.ForwardDegrees(
        fix.latitude, fix.longitude, fix.altitude);
    if (!position_enu->allFinite()) return false;

    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            (*covariance_enu)(row, column) =
                fix.position_covariance[row * 3 + column];
        }
    }
    if (fix.position_covariance_type ==
            sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN ||
        !IsUsableCovariance(*covariance_enu)) {
        covariance_enu->setZero();
        (*covariance_enu)(0, 0) =
            gps_position_std_x_ * gps_position_std_x_;
        (*covariance_enu)(1, 1) =
            gps_position_std_y_ * gps_position_std_y_;
        (*covariance_enu)(2, 2) =
            gps_position_std_z_ * gps_position_std_z_;
    } else {
        *covariance_enu =
            0.5 * (*covariance_enu + covariance_enu->transpose());
    }

    // Keep a configurable lower bound on vertical uncertainty.  Horizontal
    // GNSS accuracy is used as reported; the local ENU origin and the fixed
    // translation handle the altitude datum offset established at startup.
    const double vertical_variance = std::max(
        (*covariance_enu)(2, 2),
        gps_position_std_z_ * gps_position_std_z_);
    covariance_enu->row(2).setZero();
    covariance_enu->col(2).setZero();
    (*covariance_enu)(2, 2) = vertical_variance;
    return IsUsableCovariance(*covariance_enu);
}

bool LocalizationSystem::VelocityToMap(
    const geometry_msgs::msg::TwistWithCovarianceStamped& velocity,
    Eigen::Vector2d* velocity_map,
    Eigen::Matrix2d* covariance_map) const {
    if (!velocity_map || !covariance_map || !enu_from_map_ready_) {
        return false;
    }

    const double stamp = rclcpp::Time(velocity.header.stamp).seconds();
    const Eigen::Vector3d velocity_enu(
        velocity.twist.twist.linear.x,
        velocity.twist.twist.linear.y,
        velocity.twist.twist.linear.z);
    if (!std::isfinite(stamp) || !velocity_enu.allFinite()) return false;

    const Eigen::Matrix3d rotation_map_enu =
        enu_from_map_rotation_.transpose();
    *velocity_map = (rotation_map_enu * velocity_enu).head<2>();

    Eigen::Matrix3d covariance_enu;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            covariance_enu(row, column) =
                velocity.twist.covariance[row * 6 + column];
        }
    }
    if (IsUsableCovariance(covariance_enu)) {
        const Eigen::Matrix3d covariance_map_3d =
            rotation_map_enu * covariance_enu * rotation_map_enu.transpose();
        *covariance_map = covariance_map_3d.topLeftCorner<2, 2>();
    } else {
        covariance_map->setZero();
        (*covariance_map)(0, 0) =
            gps_velocity_std_x_ * gps_velocity_std_x_;
        (*covariance_map)(1, 1) =
            gps_velocity_std_y_ * gps_velocity_std_y_;
    }
    *covariance_map =
        0.5 * (*covariance_map + covariance_map->transpose());
    return velocity_map->allFinite() && IsUsableCovariance(*covariance_map);
}

bool LocalizationSystem::InitializeManualGuess(double stamp) {
    if (ekf_.Initialized()) return true;
    if (UsesLidar() || !manual_initial_guess_pending_ ||
        !std::isfinite(stamp)) {
        return false;
    }
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
    BeginGpsInitialization(manual_initial_pose_);
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
    gps_initial_map_body_ready_ = false;
    gps_initial_map_body_pose_ = SE3();
    gps_initialization_sample_count_ = 0;
    gps1_initial_enu_sum_.setZero();
    gps2_initial_enu_sum_.setZero();
    enu_projector_ = math::JsbsimWgs84Enu();
    enu_from_map_ready_ = false;
    enu_from_map_rotation_.setIdentity();
    enu_from_map_translation_.setZero();
    gps1_lever_arm_tracking_ = Eigen::Vector3d::Zero();
    gps2_lever_arm_tracking_ = Eigen::Vector3d::Zero();
    dual_gps_sync_tolerance_sec_ = 0.05;
    dual_gps_baseline_length_tolerance_m_ = 0.15;
    gps_initialization_sample_count_required_ = 10;
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
