#include "modules/localizationSystem/localization_system.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <vector>

#include <Eigen/Cholesky>
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

Eigen::Matrix3d MapFromTrueEnuRotation(
    double yaw_from_true_north_ccw_degrees) {
    const double yaw_from_true_north_ccw =
        yaw_from_true_north_ccw_degrees * kDegToRad;
    const double sine = std::sin(yaw_from_true_north_ccw);
    const double cosine = std::cos(yaw_from_true_north_ccw);

    // The map +X axis is yaw_from_true_north_ccw counterclockwise from true
    // north, so its standard ENU yaw (counterclockwise from east) is
    // yaw_from_true_north_ccw + 90 deg. The matrix below is the inverse basis
    // matrix: it converts true-ENU coordinate values into map coordinates.
    Eigen::Matrix3d rotation;
    rotation << -sine, cosine, 0.0,
                -cosine, -sine, 0.0,
               0.0, 0.0, 1.0;
    return rotation;
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
    double position_std, double yaw_std, double velocity_std,
    double yaw_rate_std) {
    loc::EKF::Covariance covariance = loc::EKF::Covariance::Zero();
    covariance(loc::EKF::kX, loc::EKF::kX) =
        position_std * position_std;
    covariance(loc::EKF::kY, loc::EKF::kY) =
        position_std * position_std;
    covariance(loc::EKF::kYaw, loc::EKF::kYaw) = yaw_std * yaw_std;
    covariance(loc::EKF::kVelocityX, loc::EKF::kVelocityX) =
        velocity_std * velocity_std;
    covariance(loc::EKF::kVelocityY, loc::EKF::kVelocityY) =
        velocity_std * velocity_std;
    covariance(loc::EKF::kYawRate, loc::EKF::kYawRate) =
        yaw_rate_std * yaw_rate_std;
    return covariance;
}

Eigen::Matrix3d FixedPoseNoise(double position_std_x,
                               double position_std_y,
                               double yaw_std) {
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    covariance(0, 0) = position_std_x * position_std_x;
    covariance(1, 1) = position_std_y * position_std_y;
    covariance(2, 2) = yaw_std * yaw_std;
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

Eigen::Vector2d Rotate2D(double yaw, const Eigen::Vector2d& vector) {
    const double cosine = std::cos(yaw);
    const double sine = std::sin(yaw);
    return Eigen::Vector2d(
        cosine * vector.x() - sine * vector.y(),
        sine * vector.x() + cosine * vector.y());
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
    if (mode == "ekf" || mode == "ndt_rtk_ekf" ||
        mode == "ndt_rtk_ins_ekf" || mode == "ndt_rtk") {
        return Mode::EKF_FUSION;
    }
    if (mode == "rtk_only" || mode == "rtk_ins_only") return Mode::RTK_ONLY;
    if (mode == "auto") return Mode::AUTO;
    return Mode::NDT_ONLY;
}

std::string LocalizationSystem::ModeToString(Mode mode) {
    if (mode == Mode::EKF_FUSION) return "ndt_rtk_ins_ekf";
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

bool LocalizationSystem::Init(const std::string& yaml_path, rclcpp::Node::SharedPtr node) {
    Reset();
    yaml_path_ = yaml_path;
    const YAML::Node yaml = YAML::LoadFile(yaml_path);
    const YAML::Node localization = yaml["localization"];
    const YAML::Node ekf = localization && localization["ekf"]
        ? localization["ekf"]
        : YAML::Node();
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
    rtk_ins_lever_arm_tracking_ = ReadVector3(rtk_ins && rtk_ins["lever_arm_tracking"] ? rtk_ins["lever_arm_tracking"] : YAML::Node(), Eigen::Vector3d::Zero());
    if (rtk_ins && rtk_ins["initial_observation_max_dt"])
        initial_observation_max_dt_ =
            std::max(0.0, rtk_ins["initial_observation_max_dt"].as<double>());

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
    filter_options.max_prediction_step = read_ekf(
        "max_prediction_step", filter_options.max_prediction_step);
    filter_options.rtk_position_gate_chi2 = read_ekf(
        "rtk_position_gate_chi2", filter_options.rtk_position_gate_chi2);
    filter_options.ins_yaw_gate_chi2 = read_ekf(
        "ins_yaw_gate_chi2", filter_options.ins_yaw_gate_chi2);
    filter_options.rtk_velocity_gate_chi2 = read_ekf(
        "rtk_velocity_gate_chi2", filter_options.rtk_velocity_gate_chi2);
    filter_options.ndt_pose_gate_chi2 = read_ekf(
        "ndt_pose_gate_chi2", filter_options.ndt_pose_gate_chi2);
    filter_options.wheel_gate_chi2 = read_ekf(
        "wheel_gate_chi2", filter_options.wheel_gate_chi2);

    initial_position_std_ = read_ekf(
        "initial_position_std", initial_position_std_);
    initial_yaw_std_ = read_ekf(
        "initial_yaw_std_deg", initial_yaw_std_ / kDegToRad) * kDegToRad;
    initial_velocity_std_ = read_ekf(
        "initial_velocity_std", initial_velocity_std_);
    initial_yaw_rate_std_ = read_ekf(
        "initial_yaw_rate_std", initial_yaw_rate_std_);
    rtk_position_std_x_ = read_ekf(
        "rtk_position_std_x", rtk_position_std_x_);
    rtk_position_std_y_ = read_ekf(
        "rtk_position_std_y", rtk_position_std_y_);
    ins_yaw_std_ = read_ekf(
        "ins_yaw_std_deg", ins_yaw_std_ / kDegToRad) * kDegToRad;
    rtk_velocity_std_x_ = read_ekf(
        "rtk_velocity_std_x", rtk_velocity_std_x_);
    rtk_velocity_std_y_ = read_ekf(
        "rtk_velocity_std_y", rtk_velocity_std_y_);
    ndt_position_std_x_ = read_ekf(
        "ndt_position_std_x", ndt_position_std_x_);
    ndt_position_std_y_ = read_ekf(
        "ndt_position_std_y", ndt_position_std_y_);
    ndt_yaw_std_ = read_ekf(
        "ndt_yaw_std_deg", ndt_yaw_std_ / kDegToRad) * kDegToRad;
    wheel_velocity_std_ = read_ekf(
        "wheel_velocity_std", wheel_velocity_std_);

    const std::array<double, 13> standard_deviations = {
        filter_options.process_acceleration_std,
        filter_options.process_yaw_acceleration_std,
        initial_position_std_, initial_yaw_std_, initial_velocity_std_,
        initial_yaw_rate_std_, rtk_position_std_x_, rtk_position_std_y_,
        ins_yaw_std_, rtk_velocity_std_x_, rtk_velocity_std_y_,
        ndt_position_std_x_, ndt_position_std_y_};
    const std::array<double, 5> gates = {
        filter_options.rtk_position_gate_chi2,
        filter_options.ins_yaw_gate_chi2,
        filter_options.rtk_velocity_gate_chi2,
        filter_options.ndt_pose_gate_chi2,
        filter_options.wheel_gate_chi2};
    if (!std::isfinite(filter_options.max_prediction_step) ||
        filter_options.max_prediction_step <= 0.0 ||
        ndt_yaw_std_ <= 0.0 || wheel_velocity_std_ <= 0.0 ||
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
    has_initial_guess_ = !RequiresInitialGuess();
    if (node) SetupPublishers(node);
    LOG(INFO) << "[LOCALIZATION_SYSTEM] mode=" << ModeToString(mode_)
              << ", rtk_position=" << UsesRtk()
              << ", ins_orientation=" << UsesInsOrientation()
              << ", ins_velocity=" << UsesInsVelocity()
              << ", wheel_odometry=" << UsesWheelOdometry()
              << ", imu_input=" << (!imu_topic_.empty())
              << " (not fused into localization EKF)";
    return true;
}

void LocalizationSystem::SetupPublishers(rclcpp::Node::SharedPtr node) {
    if (!node) return;
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node);
    loc_odom_pub_ = node->create_publisher<nav_msgs::msg::Odometry>("/lightning/localization/odom", 10);
    loc_pose_pub_ = node->create_publisher<geometry_msgs::msg::PoseStamped>("/lightning/localization/pose", 10);
    loc_path_pub_ = node->create_publisher<nav_msgs::msg::Path>("/lightning/localization/path", rclcpp::QoS(1).reliable().transient_local());
    raw_rtk_path_pub_ = node->create_publisher<nav_msgs::msg::Path>(
        "/lightning/localization/debug/raw_rtk_path",
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
    has_initial_guess_ = !RequiresInitialGuess();
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
        has_initial_guess_ = true;
        return true;
    }
    if (!loc_ || !map_ready_) return false;
    const bool accepted = loc_->SetExternalPose(init_pose.unit_quaternion(), init_pose.translation());
    has_initial_guess_ = accepted;
    if (accepted) {
        std::lock_guard<std::mutex> lock(filter_mutex_);
        ekf_.Reset();
        manual_initial_guess_pending_ = true;
        has_initial_position_ = false;
        has_initial_yaw_ = false;
    }
    return accepted;
}

loc::LocalizationFrameOutcome LocalizationSystem::ProcessCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud, const loc::LocalizationInputDiagnostic& diagnostic) {
    if (!UsesLidar() || !loc_ || !map_ready_) return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
    if (cloud) {
        const double stamp = rclcpp::Time(cloud->header.stamp).seconds();
        last_lidar_stamp_ = stamp;
        std::lock_guard<std::mutex> lock(filter_mutex_);
        if (ekf_.Initialized() && ekf_.PredictTo(stamp)) {
            loc_->SetPredictionPose(ekf_.Pose());
        }
    }
    return loc_->ProcessLidarMsg(cloud, diagnostic);
}

loc::LocalizationFrameOutcome LocalizationSystem::ProcessCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud, const loc::LocalizationInputDiagnostic& diagnostic) {
    if (!UsesLidar() || !loc_ || !map_ready_) return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
    if (cloud) {
        const double stamp = rclcpp::Time(cloud->header.stamp).seconds();
        last_lidar_stamp_ = stamp;
        std::lock_guard<std::mutex> lock(filter_mutex_);
        if (ekf_.Initialized() && ekf_.PredictTo(stamp)) {
            loc_->SetPredictionPose(ekf_.Pose());
        }
    }
    return loc_->ProcessLivoxLidarMsg(cloud, diagnostic);
}

void LocalizationSystem::ProcessRtkPosition(
    const sensor_msgs::msg::NavSatFix::SharedPtr& fix) {
    if (!UsesRtk() || !fix) return;

    Eigen::Vector2d position_map;
    Eigen::Matrix2d covariance_map;
    if (!PositionToMap(*fix, &position_map, &covariance_map)) {
        LOG_EVERY_N(WARNING, 100)
            << "[LOCALIZATION_EKF] RTK fix rejected before coordinate conversion"
            << ", status=" << static_cast<int>(fix->status.status)
            << ", latitude=" << fix->latitude
            << ", longitude=" << fix->longitude
            << ", altitude=" << fix->altitude;
        return;
    }
    const double stamp = rclcpp::Time(fix->header.stamp).seconds();
    AppendDebugPath(
        &raw_rtk_path_, raw_rtk_path_pub_, stamp, position_map, 0.0);
    // Green UI trajectory: the raw GNSS antenna position after coordinate
    // conversion, before initialization, prediction, gating or EKF update.
    if (loc_) loc_->UpdateRtkObservationVisualization(position_map);

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
    const bool accepted = ekf_.UpdateRtkPosition(
        stamp, position_map, RtkLeverArmForFilter(),
        covariance_map, -1.0, &distance);
    if (accepted) {
        PublishResult(BuildEkfResult(stamp, "EKF RTK position update", true));
    } else {
        PublishPredictionIfAdvanced(
            stamp, "EKF prediction; RTK position rejected");
        LOG_EVERY_N(WARNING, 20)
            << "[LOCALIZATION_EKF] RTK position rejected, stamp=" << stamp
            << ", mahalanobis=" << distance;
    }
}

void LocalizationSystem::ProcessInsOrientation(
    const sensor_msgs::msg::Imu::SharedPtr& orientation) {
    if (!UsesInsOrientation() || !orientation) return;

    double yaw_map = 0.0;
    double variance = 0.0;
    if (!OrientationToMapYaw(*orientation, &yaw_map, &variance)) return;
    const double stamp = rclcpp::Time(orientation->header.stamp).seconds();

    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!ekf_.Initialized()) {
        if (manual_initial_guess_pending_) {
            if (!InitializeManualGuess(stamp)) return;
        } else {
            has_initial_yaw_ = true;
            initial_yaw_stamp_ = stamp;
            initial_yaw_map_ = yaw_map;
            TryInitializeEkf();
            return;
        }
    }

    double distance = 0.0;
    const bool accepted = ekf_.UpdateYaw(
        stamp, yaw_map, variance, -1.0, &distance);
    if (accepted) {
        PublishResult(BuildEkfResult(stamp, "EKF INS yaw update", true));
    } else {
        PublishPredictionIfAdvanced(
            stamp, "EKF prediction; INS yaw rejected");
        LOG_EVERY_N(WARNING, 20)
            << "[LOCALIZATION_EKF] INS yaw rejected, stamp=" << stamp
            << ", mahalanobis=" << distance;
    }
}

void LocalizationSystem::ProcessInsVelocity(
    const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& velocity) {
    if (!UsesInsVelocity() || !velocity) return;

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
            stamp, "EKF INS velocity update", true));
    } else {
        PublishPredictionIfAdvanced(
            stamp, "EKF prediction; INS velocity rejected");
        LOG_EVERY_N(WARNING, 20)
            << "[LOCALIZATION_EKF] INS velocity rejected, stamp=" << stamp
            << ", mahalanobis=" << distance;
    }
}

void LocalizationSystem::ProcessWheelOdometry(
    const nav_msgs::msg::Odometry::SharedPtr& odometry) {
    if (!UsesWheelOdometry() || !odometry) return;
    const double stamp = rclcpp::Time(odometry->header.stamp).seconds();
    const double forward_velocity = odometry->twist.twist.linear.x;
    if (!std::isfinite(stamp) || !std::isfinite(forward_velocity)) {
        return;
    }

    const double message_variance = odometry->twist.covariance[0];
    const double variance =
        std::isfinite(message_variance) && message_variance > 0.0
            ? message_variance
            : wheel_velocity_std_ * wheel_velocity_std_;

    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!ekf_.Initialized()) {
        if (!manual_initial_guess_pending_ ||
            !InitializeManualGuess(stamp)) {
            return;
        }
    }

    double distance = 0.0;
    const bool accepted = ekf_.UpdateWheelOdometry(
        stamp, forward_velocity, variance, -1.0, &distance);
    if (accepted) {
        PublishResult(BuildEkfResult(
            stamp, "EKF wheel odometry update", true));
    } else {
        PublishPredictionIfAdvanced(
            stamp, "EKF prediction; wheel odometry rejected");
        LOG_EVERY_N(WARNING, 20)
            << "[LOCALIZATION_EKF] wheel odometry rejected, stamp=" << stamp
            << ", mahalanobis=" << distance;
    }
}

void LocalizationSystem::ProcessImu(const sensor_msgs::msg::Imu::SharedPtr& imu) {
    // imu_topic remains available to mapping/deskew and future algorithms.
    // It is deliberately not an implicit yaw-rate measurement for this EKF.
    (void)imu;
}

void LocalizationSystem::HandleNdtResult(const loc::LocalizationResult& result) {
    if (result.valid_) {
        AppendDebugPath(
            &raw_ndt_path_, raw_ndt_path_pub_, result.timestamp_,
            result.pose_.translation().head<2>(), PoseYaw(result.pose_));
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
        Eigen::Vector3d measurement;
        measurement << result.pose_.translation().x(),
                       result.pose_.translation().y(),
                       PoseYaw(result.pose_);
        const Eigen::Matrix3d covariance = FixedPoseNoise(
            ndt_position_std_x_, ndt_position_std_y_, ndt_yaw_std_);
        double distance = 0.0;
        pose_accepted = ekf_.UpdateNdtPose(
            result.timestamp_, measurement, covariance, -1.0, &distance);
        if (!pose_accepted) {
            LOG(WARNING) << "[LOCALIZATION_EKF] NDT pose rejected, mahalanobis="
                         << distance;
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
    const Eigen::Vector2d position = ndt.pose_.translation().head<2>();
    const loc::EKF::Covariance covariance = InitialEkfCovariance(
        initial_position_std_, initial_yaw_std_, initial_velocity_std_,
        initial_yaw_rate_std_);
    if (ekf_.Initialize(
            ndt.timestamp_, position.x(), position.y(), PoseYaw(ndt.pose_),
            Eigen::Vector2d::Zero(), 0.0, covariance)) {
        manual_initial_guess_pending_ = false;
    }
}

bool LocalizationSystem::InitializeFixedMapTransform(
    const YAML::Node& map_from_enu) {
    const std::array<const char*, 6> required = {
        "lat", "lon", "alt", "pitch", "roll", "yaw"};
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
    const double pitch_deg = map_from_enu["pitch"].as<double>();
    const double roll_deg = map_from_enu["roll"].as<double>();
    // `yaw` is the counterclockwise angle from true north to the map +X axis.
    const double yaw_from_true_north_ccw_deg =
        map_from_enu["yaw"].as<double>();
    if (!std::isfinite(latitude_deg) || !std::isfinite(longitude_deg) ||
        !std::isfinite(altitude_m) || !std::isfinite(pitch_deg) ||
        !std::isfinite(roll_deg) ||
        !std::isfinite(yaw_from_true_north_ccw_deg) ||
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

    // The map +X axis is (yaw + 90 deg) counterclockwise from ENU east.
    // R_map_true_enu converts coordinate values, so it is the inverse of that
    // axis rotation: R_map_true_enu = Rz(-(yaw + 90 deg)).
    map_from_true_enu_rotation_ =
        MapFromTrueEnuRotation(yaw_from_true_north_ccw_deg);
    const double map_x_yaw_in_true_enu = std::atan2(
        map_from_true_enu_rotation_(0, 1),
        map_from_true_enu_rotation_(0, 0));
    const double map_from_true_enu_yaw = std::atan2(
        map_from_true_enu_rotation_(1, 0),
        map_from_true_enu_rotation_(0, 0));

    // NavSatFix is projected into UTM grid east/north, whereas course,
    // orientation and velocity use true ENU. Remove the fixed grid
    // convergence before applying the map yaw.
    map_from_utm_rotation_ =
        map_from_true_enu_rotation_ *
        utm_from_true_enu_rotation_.transpose();

    // At mapping start the tracking origin is the map origin and its +X axis
    // is the map +X axis. The reference GNSS antenna therefore lies at the
    // configured tracking-frame lever arm in map coordinates.
    reference_gnss_map_ = rtk_ins_lever_arm_tracking_;
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
              << ", reference_roll_deg=" << roll_deg
              << ", reference_pitch_deg=" << pitch_deg
              << ", yaw_from_true_north_ccw_deg="
              << yaw_from_true_north_ccw_deg
              << ", map_x_yaw_in_true_enu_deg="
              << map_x_yaw_in_true_enu / kDegToRad
              << ", map_from_true_enu_yaw_deg="
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
    Eigen::Vector2d* position_map,
    Eigen::Matrix2d* covariance_map) const {
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
    const Eigen::Vector3d position_map_3d =
        reference_gnss_map_ +
        map_from_utm_rotation_ * (position_utm - reference_gnss_utm_);
    *position_map = position_map_3d.head<2>();

    Eigen::Matrix2d covariance_enu;
    covariance_enu << fix.position_covariance[0],
                      fix.position_covariance[1],
                      fix.position_covariance[3],
                      fix.position_covariance[4];
    if (fix.position_covariance_type !=
            sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN &&
        IsUsableCovariance(covariance_enu)) {
        const Eigen::Matrix2d rotation =
            map_from_true_enu_rotation_.topLeftCorner<2, 2>();
        *covariance_map =
            rotation * covariance_enu * rotation.transpose();
        *covariance_map =
            0.5 * (*covariance_map + covariance_map->transpose());
    } else {
        covariance_map->setZero();
        (*covariance_map)(0, 0) =
            rtk_position_std_x_ * rtk_position_std_x_;
        (*covariance_map)(1, 1) =
            rtk_position_std_y_ * rtk_position_std_y_;
    }
    return position_map->allFinite() && IsUsableCovariance(*covariance_map);
}

bool LocalizationSystem::OrientationToMapYaw(
    const sensor_msgs::msg::Imu& orientation,
    double* yaw_map, double* variance) const {
    if (!yaw_map || !variance || !map_from_enu_ready_ ||
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
    const Eigen::Matrix3d rotation_map_tracking =
        map_from_true_enu_rotation_ * quaternion.toRotationMatrix();
    *yaw_map = PoseYaw(SE3(Eigen::Quaterniond(rotation_map_tracking),
                           Eigen::Vector3d::Zero()));
    const double message_variance = orientation.orientation_covariance[8];
    *variance = std::isfinite(message_variance) && message_variance > 0.0
        ? message_variance
        : ins_yaw_std_ * ins_yaw_std_;
    return rotation_map_tracking.allFinite() && std::isfinite(*yaw_map) &&
           std::isfinite(*variance) && *variance > 0.0;
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
            rtk_velocity_std_x_ * rtk_velocity_std_x_;
        (*covariance_map)(1, 1) =
            rtk_velocity_std_y_ * rtk_velocity_std_y_;
    }
    return velocity_map->allFinite() && IsUsableCovariance(*covariance_map);
}

Eigen::Vector2d LocalizationSystem::RtkLeverArmForFilter() const {
    // GNSS measures the antenna position. Converting it to the tracking
    // origin requires the current tracking yaw. Without an enabled absolute
    // yaw observation that correction is unobservable, so keep the measured
    // antenna position unchanged instead of applying a guessed rotation.
    if (!UsesInsOrientation()) return Eigen::Vector2d::Zero();
    return rtk_ins_lever_arm_tracking_.head<2>();
}

void LocalizationSystem::TryInitializeEkf() {
    if (ekf_.Initialized() ||
        manual_initial_guess_pending_ ||
        !has_initial_position_) {
        return;
    }

    const bool yaw_observed = UsesInsOrientation();
    if (yaw_observed) {
        if (!has_initial_yaw_ ||
            std::fabs(initial_position_stamp_ - initial_yaw_stamp_) >
                initial_observation_max_dt_) {
            return;
        }
    }

    const double stamp = yaw_observed
        ? std::max(initial_position_stamp_, initial_yaw_stamp_)
        : initial_position_stamp_;
    const double initial_yaw = yaw_observed ? initial_yaw_map_ : 0.0;
    const Eigen::Vector2d lever_arm = RtkLeverArmForFilter();
    const Eigen::Vector2d tracking_position_map =
        initial_sensor_position_map_ -
        Rotate2D(initial_yaw, lever_arm);
    const loc::EKF::Covariance covariance = InitialEkfCovariance(
        initial_position_std_, yaw_observed ? initial_yaw_std_ : kPi,
        initial_velocity_std_,
        initial_yaw_rate_std_);
    if (ekf_.Initialize(
            stamp, tracking_position_map.x(), tracking_position_map.y(),
            initial_yaw, Eigen::Vector2d::Zero(), 0.0, covariance)) {
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
            has_initial_guess_ = true;
        }
        LOG(INFO) << "[LOCALIZATION_EKF] initialized from "
                  << (yaw_observed ? "RTK position + INS yaw"
                                   : "RTK position only")
                  << ", stamp=" << stamp
                  << ", position_map=" << tracking_position_map.transpose()
                  << ", yaw_map_deg=" << initial_yaw / kDegToRad
                  << ", yaw_observed=" << yaw_observed
                  << ", lever_arm_applied=" << yaw_observed;
        PublishResult(BuildEkfResult(
            stamp,
            yaw_observed
                ? "EKF initialized from fixed-map RTK position + INS yaw"
                : "EKF initialized from fixed-map RTK antenna position; "
                  "yaw unobserved and lever arm disabled",
            yaw_observed));
    }
}

bool LocalizationSystem::InitializeManualGuess(double stamp) {
    if (ekf_.Initialized()) return true;
    if (!manual_initial_guess_pending_ || !std::isfinite(stamp)) return false;
    const Eigen::Vector2d position =
        manual_initial_pose_.translation().head<2>();
    const loc::EKF::Covariance covariance = InitialEkfCovariance(
        initial_position_std_, initial_yaw_std_, initial_velocity_std_,
        initial_yaw_rate_std_);
    if (!ekf_.Initialize(
            stamp, position.x(), position.y(), PoseYaw(manual_initial_pose_),
            Eigen::Vector2d::Zero(), 0.0, covariance)) {
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
    result.velocity_map_ = ekf_.VelocityMap();
    result.angular_velocity_body_ = Eigen::Vector3d(
        0.0, 0.0, ekf_.GetState().yaw_rate);
    result.pose_covariance_.setZero();
    constexpr std::array<int, 3> kPoseIndices = {0, 1, 5};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result.pose_covariance_(kPoseIndices[row], kPoseIndices[column]) =
                ekf_.P()(row, column);
        }
    }
    // z, roll and pitch are not estimated by the planar EKF.
    result.pose_covariance_(2, 2) = 1e8;
    result.pose_covariance_(3, 3) = 1e8;
    result.pose_covariance_(4, 4) = 1e8;
    result.covariance_valid_ = true;
    result.twist_covariance_.setZero();
    // nav_msgs/Odometry expresses twist in child_frame_id (base_link), while
    // the EKF stores velocity in map. Propagate the full state covariance
    // through v_body = R(-yaw) * v_map, including yaw uncertainty and all
    // velocity/yaw-rate cross correlations.
    const double yaw = ekf_.GetState().yaw;
    const double cosine = std::cos(yaw);
    const double sine = std::sin(yaw);
    const double velocity_x = ekf_.GetState().velocity_x_map;
    const double velocity_y = ekf_.GetState().velocity_y_map;
    const double forward_velocity =
        cosine * velocity_x + sine * velocity_y;
    const double lateral_velocity =
        -sine * velocity_x + cosine * velocity_y;
    Eigen::Matrix<double, 3, loc::EKF::kStateDim> twist_jacobian =
        Eigen::Matrix<double, 3, loc::EKF::kStateDim>::Zero();
    twist_jacobian(0, loc::EKF::kYaw) = lateral_velocity;
    twist_jacobian(0, loc::EKF::kVelocityX) = cosine;
    twist_jacobian(0, loc::EKF::kVelocityY) = sine;
    twist_jacobian(1, loc::EKF::kYaw) = -forward_velocity;
    twist_jacobian(1, loc::EKF::kVelocityX) = -sine;
    twist_jacobian(1, loc::EKF::kVelocityY) = cosine;
    twist_jacobian(2, loc::EKF::kYawRate) = 1.0;
    const Eigen::Matrix3d twist_covariance =
        twist_jacobian * ekf_.P() * twist_jacobian.transpose();
    constexpr std::array<int, 3> kTwistIndices = {0, 1, 5};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result.twist_covariance_(
                kTwistIndices[row], kTwistIndices[column]) =
                twist_covariance(row, column);
        }
    }
    // Vertical velocity and roll/pitch rates are outside this planar state,
    // so advertise large uncertainty rather than false certainty.
    result.twist_covariance_(2, 2) = 1e8;
    result.twist_covariance_(3, 3) = 1e8;
    result.twist_covariance_(4, 4) = 1e8;
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
    // Pangolin receives the same final estimator result as ROS publishers.
    // Every localization mode visualizes the EKF state, never raw NDT output.
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
    rtk_fix_topic_.clear();
    rtk_orientation_topic_.clear();
    rtk_velocity_topic_.clear();
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
    rtk_ins_lever_arm_tracking_ = Eigen::Vector3d::Zero();
    initial_position_std_ = 0.5;
    initial_yaw_std_ = 3.0 * kDegToRad;
    initial_velocity_std_ = 2.0;
    initial_yaw_rate_std_ = 0.5;
    rtk_position_std_x_ = 0.05;
    rtk_position_std_y_ = 0.05;
    ins_yaw_std_ = 1.0 * kDegToRad;
    rtk_velocity_std_x_ = 0.10;
    rtk_velocity_std_y_ = 0.10;
    ndt_position_std_x_ = 0.10;
    ndt_position_std_y_ = 0.10;
    ndt_yaw_std_ = 1.0 * kDegToRad;
    wheel_velocity_std_ = 0.10;
    has_initial_position_ = false;
    has_initial_yaw_ = false;
    initial_yaw_map_ = 0.0;
    initial_observation_max_dt_ = 0.05;
    map_ready_ = false;
    has_initial_guess_ = false;
    manual_initial_guess_pending_ = false;
    manual_initial_pose_ = SE3();
    last_lidar_stamp_ = -1.0;
    path_ = nav_msgs::msg::Path();
    {
        std::lock_guard<std::mutex> lock(debug_path_mutex_);
        raw_rtk_path_ = nav_msgs::msg::Path();
        raw_ndt_path_ = nav_msgs::msg::Path();
    }
    { std::lock_guard<std::mutex> lock(result_mutex_); latest_result_ = loc::LocalizationResult(); }
    tf_broadcaster_.reset();
    loc_odom_pub_.reset();
    loc_pose_pub_.reset();
    loc_path_pub_.reset();
    raw_rtk_path_pub_.reset();
    raw_ndt_path_pub_.reset();
    loc_pose_quality_pub_.reset();
}

}  // namespace lightning::modules
