#include "modules/localizationSystem/localization_system.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <map>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <rclcpp/node.hpp>
#include <sensor_msgs/msg/nav_sat_status.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "common/point_def.h"
#include "core/lightning_math.hpp"
#include "core/localization/localization.h"
#include "ui/pangolin_window.h"

namespace lightning::modules {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kCalibrationSampleSpacingM = 1.0;
constexpr double kCalibrationTravelDistanceM = 50.0;
constexpr double kCalibrationSyncToleranceSec = 0.05;
constexpr std::size_t kMinimumCalibrationPairs = 10;

void AppendXYZCloud(const pcl::PointCloud<pcl::PointXYZ>& source,
                    PointCloudType& destination) {
    destination.reserve(destination.size() + source.size());
    for (const auto& source_point : source.points) {
        PointType point;
        point.x = source_point.x;
        point.y = source_point.y;
        point.z = source_point.z;
        point.intensity = 0.0f;
        point.ring = 0;
        point.time = 0.0;
        destination.push_back(point);
    }
}

std::string ReadTopic(const YAML::Node& common, const char* name) {
    if (!common) return std::string();
    const YAML::Node value = common[name];
    return value && value.IsScalar()
        ? value.as<std::string>()
        : std::string();
}

Eigen::Vector3d OrientationMessageToEnuRpy(
    const geometry_msgs::msg::TwistWithCovarianceStamped& orientation) {
    // The temporary DataTransfer bag stores the vendor convention unchanged:
    // angular.x/y are roll/pitch and angular.z is course, all in radians.
    // Course is the clockwise angle from ENU north to the GPS-device +Y axis.
    // When course is zero, GPS +X points east and GPS +Y points north, so the
    // GPS frame is aligned with ENU and its standard right-handed yaw is zero.
    // Consequently ENU yaw is the negated vendor course.
    return Eigen::Vector3d(
        orientation.twist.twist.angular.x,
        orientation.twist.twist.angular.y,
        loc::EKF::WrapAngle(-orientation.twist.twist.angular.z));
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
    gps_topic_ = ReadTopic(common, "gps_topic");
    orientation_topic_ = ReadTopic(common, "orientation_topic");
    velocity_topic_ = ReadTopic(common, "velocity_topic");
    wheel_odometry_topic_ = ReadTopic(common, "wheel_odometry_topic");
    gps_rotation_body_gps_rpy_ = kDegToRad * ReadVector3(
        gps_ins && gps_ins["gps_rotation_body_gps_rpy_deg"]
            ? gps_ins["gps_rotation_body_gps_rpy_deg"]
            : YAML::Node(),
        Eigen::Vector3d::Zero());
    gps_rotation_body_gps_ =
        loc::EKF::RotationFromRpy(gps_rotation_body_gps_rpy_);

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
        "gps_position_gate_chi2",
        filter_options.gps_position_gate_chi2);
    filter_options.gps_orientation_gate_chi2 = read_ekf(
        "gps_orientation_gate_chi2",
        filter_options.gps_orientation_gate_chi2);
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
    gps_orientation_std_roll_ = read_ekf(
        "gps_orientation_std_floor_roll_deg",
        gps_orientation_std_roll_ / kDegToRad) * kDegToRad;
    gps_orientation_std_pitch_ = read_ekf(
        "gps_orientation_std_floor_pitch_deg",
        gps_orientation_std_pitch_ / kDegToRad) * kDegToRad;
    gps_orientation_std_yaw_ = read_ekf(
        "gps_orientation_std_floor_yaw_deg",
        gps_orientation_std_yaw_ / kDegToRad) * kDegToRad;
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
    const std::array<double, 20> standard_deviations = {
        filter_options.process_acceleration_std,
        filter_options.process_yaw_acceleration_std,
        filter_options.process_vertical_position_rate_std,
        filter_options.process_roll_pitch_rate_std,
        initial_position_std_, initial_orientation_std_,
        initial_velocity_std_, initial_yaw_rate_std_,
        gps_position_std_x_, gps_position_std_y_, gps_position_std_z_,
        gps_orientation_std_roll_, gps_orientation_std_pitch_,
        gps_orientation_std_yaw_,
        gps_velocity_std_x_, gps_velocity_std_y_, ndt_position_std_x_,
        ndt_position_std_y_, ndt_position_std_z_, ndt_orientation_std_};
    const std::array<double, 4> gates = {
        filter_options.gps_position_gate_chi2,
        filter_options.gps_orientation_gate_chi2,
        filter_options.gps_velocity_gate_chi2,
        filter_options.ndt_pose_gate_chi2};
    if (!std::isfinite(filter_options.max_prediction_step) ||
        filter_options.max_prediction_step <= 0.0 ||
        !gps_rotation_body_gps_rpy_.allFinite() ||
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
    if (UsesGpsPosition() && !UsesLidar()) {
        LOG(WARNING) << "[MAP_ENU_CALIBRATION] GPS position is configured "
                        "without LiDAR; MAP<-ENU cannot be calibrated from "
                        "GPS/NDT point pairs";
    }
    if (UsesGpsOrientation() && !UsesGpsPosition()) {
        LOG(WARNING) << "[LOCALIZATION_EKF] orientation updates require the "
                        "MAP<-ENU transform produced by GPS/NDT calibration";
    }

    ekf_.Configure(filter_options);


    if (UsesLidar()) {
        ndt_localization_ = std::make_shared<loc::Localization>();
        ndt_localization_->SetResultCallback(
            [this](const loc::LocalizationResult& result) {
                HandleNdtResult(result);
            });
    }
    if (node) SetupPublishers(node);
    LOG(INFO) << "[LOCALIZATION_SYSTEM] mode=" << ModeToString(mode_)
              << ", filter=3d_pose_planar_motion_12_state"
              << ", gps_position=" << UsesGpsPosition()
              << ", gps_orientation="
              << UsesGpsOrientation()
              << ", gps_rotation_body_gps_rpy_deg="
              << (gps_rotation_body_gps_rpy_ / kDegToRad).transpose()
              << ", map_enu_sample_spacing_m="
              << kCalibrationSampleSpacingM
              << ", map_enu_travel_distance_m="
              << kCalibrationTravelDistanceM
              << ", map_enu_sync_tolerance_sec="
              << kCalibrationSyncToleranceSec
              << ", gps_velocity=" << UsesGpsVelocity()
              << ", wheel_input_reserved=" << UsesWheelOdometry()
              << ", imu_input_reserved=" << (!imu_topic_.empty())
              << " (not fused into localization EKF)"
              << ", gps_z_std_floor_m=" << gps_position_std_z_
              << ", gps_orientation_std_floor_deg="
              << (Eigen::Vector3d(
                      gps_orientation_std_roll_,
                      gps_orientation_std_pitch_,
                      gps_orientation_std_yaw_) /
                  kDegToRad).transpose();
    return true;
}

void LocalizationSystem::SetupPublishers(rclcpp::Node::SharedPtr node) {
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
    if (map_path.empty()) return false;
    map_path_ = map_path;
    map_ready_ = false;

    if (UsesLidar() &&
        (!ndt_localization_ ||
         !ndt_localization_->Init(yaml_path_, map_path_))) {
        return false;
    }
    if (with_ui_ && !InitializeVisualization(map_path_)) {
        return false;
    }

    map_ready_ = true;
    LOG(INFO) << "[LOCALIZATION_SYSTEM] map ready"
              << ", path=" << map_path_
              << ", lidar_enabled=" << UsesLidar()
              << ", ui_enabled=" << with_ui_;
    return true;
}

bool LocalizationSystem::InitializeVisualization(
    const std::string& map_path) {
    if (ui_) {
        ui_->Quit();
        ui_.reset();
    }

    ui_ = std::make_shared<ui::PangolinWindow>();
    if (!ui_->Init()) {
        LOG(ERROR) << "[LOCALIZATION_UI] failed to create Pangolin window";
        ui_.reset();
        return false;
    }

    LoadMapForVisualization(map_path);
    return true;
}

void LocalizationSystem::LoadMapForVisualization(
    const std::string& map_path) {
    CloudPtr target_map(new PointCloudType());
    namespace fs = std::filesystem;
    std::string map_source = "BlockMap";
    int block_file_count = 0;
    const fs::path block_map_directory =
        fs::path(map_path) / "BlockMap" / "pointcloud_map";
    if (fs::exists(block_map_directory) &&
        fs::is_directory(block_map_directory)) {
        for (const auto& entry : fs::directory_iterator(block_map_directory)) {
            const std::string extension = entry.path().extension().string();
            if (!entry.is_regular_file() ||
                (extension != ".pcd" && extension != ".PCD")) {
                continue;
            }

            pcl::PointCloud<pcl::PointXYZ> block_cloud;
            if (pcl::io::loadPCDFile(
                    entry.path().string(), block_cloud) != 0) {
                LOG(WARNING)
                    << "[LOCALIZATION_UI] failed to load BlockMap pcd: "
                    << entry.path().string();
                continue;
            }
            AppendXYZCloud(block_cloud, *target_map);
            ++block_file_count;
        }
    }

    if (target_map->empty()) {
        map_source = "global.pcd";
        const std::string global_pcd_path = map_path + "/global.pcd";
        if (pcl::io::loadPCDFile(global_pcd_path, *target_map) != 0 ||
            target_map->empty()) {
            LOG(WARNING)
                << "[LOCALIZATION_UI] failed to load map for UI from "
                   "BlockMap or global.pcd under: "
                << map_path;
            return;
        }
    }

    target_map->height = 1;
    target_map->width = target_map->size();
    target_map->is_dense = false;

    std::map<int, CloudPtr> visualization_map;
    visualization_map.emplace(0, target_map);
    ui_->UpdatePointCloudGlobal(visualization_map);
    LOG(INFO) << "[LOCALIZATION_UI] target map loaded from " << map_source
              << ", block_files=" << block_file_count
              << ", points=" << target_map->size();
}

bool LocalizationSystem::SetInitialGuess(const SE3& init_pose, bool* initialized_now) {
    if (initialized_now) *initialized_now = false;
    if (!UsesLidar() || !map_ready_) return false;
    const bool accepted = ndt_localization_->SetExternalPose(
        init_pose.unit_quaternion(), init_pose.translation());
    if (accepted) {
        {
            std::lock_guard<std::mutex> lock(filter_mutex_);
            ekf_.Reset();
        }
        // A new NDT seed starts a new localization run, so its MAP<-ENU
        // calibration must be rebuilt from the new raw NDT trajectory.
        ResetMapEnuCalibration();
    }
    return accepted;
}

loc::LocalizationFrameOutcome LocalizationSystem::ProcessCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud, const loc::LocalizationInputDiagnostic& diagnostic) {
    if (!UsesLidar() || !map_ready_) return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
    return ndt_localization_->ProcessLidarMsg(cloud, diagnostic);
}

loc::LocalizationFrameOutcome LocalizationSystem::ProcessCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud, const loc::LocalizationInputDiagnostic& diagnostic) {
    if (!UsesLidar() || !map_ready_) return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
    return ndt_localization_->ProcessLivoxLidarMsg(cloud, diagnostic);
}

// ProcessGps/ProcessGpsOrientation/ProcessgpsVelocity are the shared online
// and offline input boundary. Validate message payloads here once; downstream
// conversion and EKF functions operate on that validated data.
void LocalizationSystem::ProcessGps(
    const sensor_msgs::msg::NavSatFix::SharedPtr& fix) {
    if (!UsesGpsPosition() || !fix) return;
    const double stamp = rclcpp::Time(fix->header.stamp).seconds();
    if (fix->status.status == sensor_msgs::msg::NavSatStatus::STATUS_NO_FIX ||
        !std::isfinite(stamp) || !std::isfinite(fix->latitude) ||
        !std::isfinite(fix->longitude) || !std::isfinite(fix->altitude)) {
        return;
    }

    if (!enu_projector_.Initialized()) {
        if (!enu_projector_.SetOriginDegrees(
                fix->latitude, fix->longitude, fix->altitude)) {
            return;
        }
        LOG(INFO) << "[MAP_ENU_CALIBRATION] ENU origin set"
                  << ", latitude=" << fix->latitude
                  << ", longitude=" << fix->longitude
                  << ", altitude=" << fix->altitude;
    }

    Eigen::Vector3d position_enu;
    Eigen::Matrix3d covariance_enu;
    GnssToEnu(*fix, position_enu, covariance_enu);

    if (!map_enu_calibrated_) {
        AddGpsCalibrationSample(stamp, position_enu);
        TryFinishMapEnuCalibration();
        if (!map_enu_calibrated_) return;
    }

    HandleGpsPosition(stamp, position_enu, covariance_enu);
}

void LocalizationSystem::ProcessGpsOrientation(
    const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& orientation) {
    if (!UsesGpsOrientation() || !orientation) return;
    const double stamp = rclcpp::Time(orientation->header.stamp).seconds();
    const auto& angular = orientation->twist.twist.angular;
    if (!std::isfinite(stamp) || !std::isfinite(angular.x) ||
        !std::isfinite(angular.y) || !std::isfinite(angular.z)) {
        return;
    }

    if (!map_enu_calibrated_) return;

    HandleGpsOrientation(*orientation);
}

void LocalizationSystem::AddGpsCalibrationSample(
    double stamp, const Eigen::Vector3d& position_enu) {
    std::lock_guard<std::mutex> lock(map_enu_calibration_mutex_);
    gps_calibration_samples_.push_back({stamp, position_enu});
}

void LocalizationSystem::AddNdtCalibrationSample(
    const loc::LocalizationResult& result) {
    if (!UsesGpsPosition() || !result.reliable_) return;

    bool distance_complete = false;
    std::size_t sample_count = 0;
    double travel_distance = 0.0;
    {
        std::lock_guard<std::mutex> lock(map_enu_calibration_mutex_);
        if (map_enu_calibrated_ || calibration_distance_complete_) return;

        const TimedPosition sample{
            result.timestamp_, result.pose_.translation()};
        if (ndt_calibration_samples_.empty()) {
            ndt_calibration_samples_.push_back(sample);
            return;
        }

        const double distance =
            (sample.position - ndt_calibration_samples_.back().position).norm();
        if (distance < kCalibrationSampleSpacingM) return;

        calibration_travel_distance_m_ += distance;
        ndt_calibration_samples_.push_back(sample);
        calibration_distance_complete_ =
            calibration_travel_distance_m_ >= kCalibrationTravelDistanceM;
        distance_complete = calibration_distance_complete_;
        sample_count = ndt_calibration_samples_.size();
        travel_distance = calibration_travel_distance_m_;
    }

    if (distance_complete) {
        LOG(INFO) << "[MAP_ENU_CALIBRATION] NDT sampling complete"
                  << ", samples=" << sample_count
                  << ", travel_distance_m=" << travel_distance;
        TryFinishMapEnuCalibration();
    }
}

bool LocalizationSystem::TryFinishMapEnuCalibration() {
    std::vector<TimedPosition> gps_samples;
    std::vector<TimedPosition> ndt_samples;
    {
        std::lock_guard<std::mutex> lock(map_enu_calibration_mutex_);
        if (map_enu_calibrated_) return true;
        if (!calibration_distance_complete_) return false;
        gps_samples = gps_calibration_samples_;
        ndt_samples = ndt_calibration_samples_;
    }

    std::vector<Eigen::Vector3d> matched_gps_positions;
    std::vector<Eigen::Vector3d> matched_ndt_positions;
    matched_gps_positions.reserve(ndt_samples.size());
    matched_ndt_positions.reserve(ndt_samples.size());

    if (gps_samples.empty()) return false;
    std::size_t gps_index = 0;
    for (const TimedPosition& ndt_sample : ndt_samples) {
        while (gps_index + 1 < gps_samples.size() &&
               std::abs(gps_samples[gps_index + 1].stamp - ndt_sample.stamp) <
                   std::abs(gps_samples[gps_index].stamp - ndt_sample.stamp)) {
            ++gps_index;
        }

        const TimedPosition& gps_sample = gps_samples[gps_index];
        if (std::abs(gps_sample.stamp - ndt_sample.stamp) <=
            kCalibrationSyncToleranceSec) {
            matched_gps_positions.push_back(gps_sample.position);
            matched_ndt_positions.push_back(ndt_sample.position);
        }
    }

    if (matched_gps_positions.size() < kMinimumCalibrationPairs) {
        LOG_EVERY_N(WARNING, 20)
            << "[MAP_ENU_CALIBRATION] waiting for matched pairs"
            << ", matched=" << matched_gps_positions.size()
            << ", required=" << kMinimumCalibrationPairs
            << ", gps_samples=" << gps_samples.size()
            << ", ndt_samples=" << ndt_samples.size();
        return false;
    }

    const Eigen::Index pair_count =
        static_cast<Eigen::Index>(matched_gps_positions.size());
    Eigen::Matrix3Xd gps_matrix(3, pair_count);
    Eigen::Matrix3Xd ndt_matrix(3, pair_count);
    for (Eigen::Index index = 0; index < pair_count; ++index) {
        gps_matrix.col(index) = matched_gps_positions[index];
        ndt_matrix.col(index) = matched_ndt_positions[index];
    }

    // MAP and ENU are both gravity-aligned fixed frames. Their rotation has
    // only one degree of freedom: yaw about the common +Z axis. Estimate the
    // horizontal rigid transform with fixed scale, embed its 2-D rotation in
    // 3-D, and estimate only a constant vertical translation. A straight
    // trajectory is sufficient because no roll/pitch rotation is estimated.
    const Eigen::Matrix3d transform_map_enu_xy = Eigen::umeyama(
        gps_matrix.topRows<2>(), ndt_matrix.topRows<2>(), false);
    Eigen::Matrix3d rotation_map_enu = Eigen::Matrix3d::Identity();
    rotation_map_enu.topLeftCorner<2, 2>() =
        transform_map_enu_xy.topLeftCorner<2, 2>();
    Eigen::Vector3d translation_map_enu = Eigen::Vector3d::Zero();
    translation_map_enu.head<2>() =
        transform_map_enu_xy.block<2, 1>(0, 2);
    translation_map_enu.z() =
        (ndt_matrix.row(2) - gps_matrix.row(2)).mean();

    double squared_error_sum = 0.0;
    for (Eigen::Index index = 0; index < pair_count; ++index) {
        const Eigen::Vector3d residual =
            rotation_map_enu * gps_matrix.col(index) +
            translation_map_enu - ndt_matrix.col(index);
        squared_error_sum += residual.squaredNorm();
    }
    const double rmse =
        std::sqrt(squared_error_sum / static_cast<double>(pair_count));

    {
        std::lock_guard<std::mutex> lock(map_enu_calibration_mutex_);
        rotation_map_enu_ = rotation_map_enu;
        translation_map_enu_ = translation_map_enu;
        map_enu_calibrated_ = true;
        gps_calibration_samples_.clear();
        ndt_calibration_samples_.clear();
    }

    const double yaw_map_enu =
        std::atan2(rotation_map_enu_(1, 0), rotation_map_enu_(0, 0));
    LOG(INFO) << "[MAP_ENU_CALIBRATION] completed"
              << ", matched_pairs=" << pair_count
              << ", rmse_m=" << rmse
              << ", det_R=" << rotation_map_enu_.determinant()
              << ", roll_map_enu_deg=0"
              << ", pitch_map_enu_deg=0"
              << ", yaw_map_enu_deg=" << yaw_map_enu / kDegToRad
              << ", translation_map_enu="
              << translation_map_enu_.transpose();
    return true;
}

void LocalizationSystem::ResetMapEnuCalibration() {
    std::lock_guard<std::mutex> lock(map_enu_calibration_mutex_);
    enu_projector_ = math::JsbsimWgs84Enu();
    map_enu_calibrated_ = false;
    rotation_map_enu_.setIdentity();
    translation_map_enu_.setZero();
    gps_calibration_samples_.clear();
    ndt_calibration_samples_.clear();
    calibration_travel_distance_m_ = 0.0;
    calibration_distance_complete_ = false;
}

void LocalizationSystem::HandleGpsPosition(
    double stamp, const Eigen::Vector3d& position_enu,
    const Eigen::Matrix3d& covariance_enu) {
    const Eigen::Vector3d position_map =
        rotation_map_enu_ * position_enu + translation_map_enu_;
    Eigen::Matrix3d covariance_map =
        rotation_map_enu_ * covariance_enu * rotation_map_enu_.transpose();
    covariance_map = 0.5 * (covariance_map + covariance_map.transpose());

    std::lock_guard<std::mutex> lock(filter_mutex_);
    double distance = 0.0;
    const bool accepted = ekf_.UpdateGpsPoseMap(
        stamp, position_map, covariance_map,
        -1.0, &distance);
    if (accepted) {
        PublishResult(BuildEkfResult(
            stamp, "EKF GPS MAP position update", true));
    } else {
        PublishPredictionIfAdvanced(
            stamp, "EKF prediction; GPS MAP position observation rejected");
        LOG_EVERY_N(WARNING, 20)
            << "[LOCALIZATION_EKF][GPS] position observation rejected"
            << ", stamp=" << stamp
            << ", mahalanobis=" << distance
            << ", covariance_enu_diag="
            << covariance_enu.diagonal().transpose()
            << ", covariance_map_diag="
            << covariance_map.diagonal().transpose();
    }

    AppendDebugPath(
        raw_gps_path_, raw_gps_path_pub_, stamp,
        position_map.head<2>(), 0.0);

    if (ui_) ui_->UpdategpsPosition(position_map.head<2>());

    LOG_EVERY_N(INFO, 50)
        << "[LOCALIZATION_EKF][GPS] MAP position observation"
        << ", stamp=" << stamp
        << ", gps_enu=" << position_enu.transpose()
        << ", gps_map=" << position_map.transpose()
        << ", covariance_enu_diag="
        << covariance_enu.diagonal().transpose()
        << ", covariance_map_diag="
        << covariance_map.diagonal().transpose();
}

void LocalizationSystem::HandleGpsOrientation(
    const geometry_msgs::msg::TwistWithCovarianceStamped& orientation) {
    const double stamp = rclcpp::Time(orientation.header.stamp).seconds();
    const Eigen::Vector3d measured_rpy_enu_gps =
        OrientationMessageToEnuRpy(orientation);

    // Convert the observation into the exact coordinate convention used by
    // the EKF state before calling the filter:
    //   R_M_B(meas) = R_M_E * R_E_G(meas) * R_B_G^T.
    // R_M_E and R_B_G are fixed, known rotations.  After this conversion the
    // measurement is MAP<-BODY RPY, so the EKF observation Jacobian is I3.
    const Eigen::Matrix3d measured_rotation_enu_gps =
        loc::EKF::RotationFromRpy(measured_rpy_enu_gps);
    const Eigen::Matrix3d measured_rotation_map_body =
        rotation_map_enu_ * measured_rotation_enu_gps *
        gps_rotation_body_gps_.transpose();
    const Eigen::Vector3d measured_rpy_map_body =
        loc::EKF::RpyFromRotation(measured_rotation_map_body);

    Eigen::Matrix3d covariance_map_body;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            covariance_map_body(row, column) =
                orientation.twist.covariance[(row + 3) * 6 + column + 3];
        }
    }

    // yaw_enu = -course has derivative -1 with respect to course.
    // Transform the complete covariance too, so any future roll/course or
    // pitch/course cross-covariance keeps the correct sign. The current bag
    // converter writes only the diagonal, for which the values are unchanged.
    Eigen::Matrix3d course_to_enu_yaw = Eigen::Matrix3d::Identity();
    course_to_enu_yaw(2, 2) = -1.0;
    covariance_map_body =
        course_to_enu_yaw * covariance_map_body * course_to_enu_yaw;

    // The temporary orientation message supplies diagonal roll/pitch/course
    // variances.  Treat those variances as the uncertainty of the converted
    // MAP<-BODY Euler observation; the known frame transforms themselves add
    // no uncertainty.  This keeps the observation model explicit and avoids
    // putting frame-conversion derivatives inside the EKF.

    const Eigen::Vector3d std_floor(
        gps_orientation_std_roll_,
        gps_orientation_std_pitch_,
        gps_orientation_std_yaw_);
    if (!IsUsableCovariance(covariance_map_body)) {
        covariance_map_body =
            std_floor.array().square().matrix().asDiagonal();
    } else {
        covariance_map_body = 0.5 *
            (covariance_map_body + covariance_map_body.transpose());
        for (int axis = 0; axis < 3; ++axis) {
            covariance_map_body(axis, axis) = std::max(
                covariance_map_body(axis, axis),
                std_floor(axis) * std_floor(axis));
        }
    }
    std::lock_guard<std::mutex> lock(filter_mutex_);
    double distance = 0.0;
    const bool accepted = ekf_.UpdateMapOrientation(
        stamp, measured_rpy_map_body, covariance_map_body,
        -1.0, &distance);
    if (accepted) {
        PublishResult(BuildEkfResult(
            stamp, "EKF INS MAP<-BODY orientation update", true));
        LOG_EVERY_N(INFO, 50)
            << "[LOCALIZATION_EKF][INS_ORIENTATION] observation accepted"
            << ", stamp=" << stamp
            << ", vendor_course_deg="
            << orientation.twist.twist.angular.z / kDegToRad
            << ", measured_rpy_enu_gps_deg="
            << (measured_rpy_enu_gps / kDegToRad).transpose()
            << ", measured_rpy_map_body_deg="
            << (measured_rpy_map_body / kDegToRad).transpose()
            << ", mahalanobis=" << distance;
    } else {
        PublishPredictionIfAdvanced(
            stamp,
            "EKF prediction; INS MAP<-BODY orientation observation rejected");
        LOG_EVERY_N(WARNING, 20)
            << "[LOCALIZATION_EKF][INS_ORIENTATION] observation rejected"
            << ", stamp=" << stamp
            << ", vendor_course_deg="
            << orientation.twist.twist.angular.z / kDegToRad
            << ", measured_rpy_enu_gps_deg="
            << (measured_rpy_enu_gps / kDegToRad).transpose()
            << ", measured_rpy_map_body_deg="
            << (measured_rpy_map_body / kDegToRad).transpose()
            << ", mahalanobis=" << distance;
    }
}

void LocalizationSystem::ProcessgpsVelocity(
    const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& velocity) {
    if (!UsesGpsVelocity() || !velocity) return;

    const double stamp = rclcpp::Time(velocity->header.stamp).seconds();
    const auto& linear = velocity->twist.twist.linear;
    if (!std::isfinite(stamp) || !std::isfinite(linear.x) ||
        !std::isfinite(linear.y) || !std::isfinite(linear.z) ||
        !map_enu_calibrated_) {
        return;
    }

    Eigen::Vector2d velocity_map;
    Eigen::Matrix2d covariance_map;
    VelocityToMap(*velocity, velocity_map, covariance_map);

    std::lock_guard<std::mutex> lock(filter_mutex_);
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
    if (!result.valid_) return;

    AddNdtCalibrationSample(result);

    const Eigen::Vector2d ndt_position_map =
        result.pose_.translation().head<2>();
    AppendDebugPath(
        raw_ndt_path_, raw_ndt_path_pub_, result.timestamp_,
        ndt_position_map, PoseYaw(result.pose_));

    if (ui_) ui_->UpdateNdtPosition(ndt_position_map);

    if (!UsesGpsPosition()) {
        PublishResult(result);
        return;
    }

    UpdateEkfWithNdt(result);
}

void LocalizationSystem::UpdateEkfWithNdt(
    const loc::LocalizationResult& result) {
    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (!ekf_.Initialized()) {
        if (!result.reliable_ || !InitializeEkfFromNdt(result)) return;
        PublishResult(BuildEkfResult(
            result.timestamp_, "EKF NDT update", result.reliable_));
        return;
    }

    loc::EKF::Matrix6d covariance = FixedPoseNoise(
        ndt_position_std_x_, ndt_position_std_y_, ndt_position_std_z_,
        ndt_orientation_std_);
    if (result.covariance_valid_ &&
        IsUsableCovariance(result.pose_covariance_)) {
        covariance = 0.5 *
            (result.pose_covariance_ + result.pose_covariance_.transpose());
    }

    double mahalanobis = 0.0;
    if (ekf_.UpdateNdtPose(
            result.timestamp_, result.pose_, covariance, -1.0,
            &mahalanobis)) {
        PublishResult(BuildEkfResult(
            result.timestamp_, "EKF NDT update", result.reliable_));
    } else {
        LOG_EVERY_N(WARNING, 20)
            << "[LOCALIZATION_EKF] NDT rejected, mahalanobis="
            << mahalanobis;
        PublishPredictionIfAdvanced(
            result.timestamp_, "EKF prediction; NDT rejected");
    }
}

bool LocalizationSystem::InitializeEkfFromNdt(
    const loc::LocalizationResult& ndt) {
    const Eigen::Vector3d position = ndt.pose_.translation();
    const Eigen::Vector3d rpy =
        loc::EKF::RpyFromRotation(ndt.pose_.rotationMatrix());
    const loc::EKF::Covariance covariance = InitialEkfCovariance(
        initial_position_std_, initial_orientation_std_, initial_velocity_std_,
        initial_yaw_rate_std_);
    if (!ekf_.Initialize(
            ndt.timestamp_, position, rpy, Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero(), covariance)) {
        return false;
    }

    LOG(INFO) << "[LOCALIZATION_EKF] NDT relocalization initialized EKF"
              << ", stamp=" << ndt.timestamp_
              << ", position=" << position.transpose();
    return true;
}

void LocalizationSystem::GnssToEnu(
    const sensor_msgs::msg::NavSatFix& fix,
    Eigen::Vector3d& position_enu,
    Eigen::Matrix3d& covariance_enu) const {
    position_enu = enu_projector_.ForwardDegrees(
        fix.latitude, fix.longitude, fix.altitude);

    // This project's GNSS publisher uses a non-standard covariance contract:
    // NavSatFix.position_covariance contains the ECEF XYZ covariance in m^2.
    // The WGS84 position and its ECEF covariance must therefore both be
    // transformed into the fixed local ENU frame established by the first fix.
    Eigen::Matrix3d covariance_ecef;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            covariance_ecef(row, column) =
                fix.position_covariance[row * 3 + column];
        }
    }
    if (fix.position_covariance_type ==
            sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN ||
        !IsUsableCovariance(covariance_ecef)) {
        covariance_enu.setZero();
        covariance_enu(0, 0) =
            gps_position_std_x_ * gps_position_std_x_;
        covariance_enu(1, 1) =
            gps_position_std_y_ * gps_position_std_y_;
        covariance_enu(2, 2) =
            gps_position_std_z_ * gps_position_std_z_;
    } else {
        covariance_ecef =
            0.5 * (covariance_ecef + covariance_ecef.transpose());
        covariance_enu = enu_projector_.CovarianceEcefToEnu(
            covariance_ecef);
        covariance_enu =
            0.5 * (covariance_enu + covariance_enu.transpose());
        LOG_EVERY_N(INFO, 50)
            << "[LOCALIZATION_EKF][GPS] ECEF covariance converted to ENU"
            << ", covariance_ecef_diag="
            << covariance_ecef.diagonal().transpose()
            << ", covariance_enu_diag="
            << covariance_enu.diagonal().transpose();
    }

    // Keep the converted ENU covariance intact, including cross terms. Only
    // raise the vertical variance to its configured lower bound; adding
    // variance on one diagonal preserves positive definiteness.
    covariance_enu(2, 2) = std::max(
        covariance_enu(2, 2),
        gps_position_std_z_ * gps_position_std_z_);
}

void LocalizationSystem::VelocityToMap(
    const geometry_msgs::msg::TwistWithCovarianceStamped& velocity,
    Eigen::Vector2d& velocity_map,
    Eigen::Matrix2d& covariance_map) const {
    const Eigen::Vector3d velocity_enu(
        velocity.twist.twist.linear.x,
        velocity.twist.twist.linear.y,
        velocity.twist.twist.linear.z);

    velocity_map = (rotation_map_enu_ * velocity_enu).head<2>();

    Eigen::Matrix3d covariance_enu;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            covariance_enu(row, column) =
                velocity.twist.covariance[row * 6 + column];
        }
    }
    if (IsUsableCovariance(covariance_enu)) {
        const Eigen::Matrix3d covariance_map_3d =
            rotation_map_enu_ * covariance_enu *
            rotation_map_enu_.transpose();
        covariance_map = covariance_map_3d.topLeftCorner<2, 2>();
    } else {
        covariance_map.setZero();
        covariance_map(0, 0) =
            gps_velocity_std_x_ * gps_velocity_std_x_;
        covariance_map(1, 1) =
            gps_velocity_std_y_ * gps_velocity_std_y_;
    }
    covariance_map = 0.5 * (covariance_map + covariance_map.transpose());
}

void LocalizationSystem::PublishPredictionIfAdvanced(
    double stamp, const std::string& message) {
    // Every update function predicts first. If the observation is then gated
    // out, publish that new predicted state; do nothing for stale input whose
    // timestamp could not advance the filter.
    if (std::fabs(ekf_.GetState().stamp - stamp) <= 1e-6) {
        PublishResult(BuildEkfResult(stamp, message, false));
    }
}

loc::LocalizationResult LocalizationSystem::BuildEkfResult(
    double stamp, const std::string& message, bool reliable) const {
    loc::LocalizationResult result;
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
    // The red trajectory receives exactly the same final localization result
    // as the ROS publishers: raw NDT in LiDAR-only mode, EKF in fusion mode.
    // Raw gps and raw NDT observations use independent green/yellow paths.
    if (ui_) ui_->UpdateNavState(result.ToNavState());
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
    nav_msgs::msg::Path& path,
    const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr& publisher,
    double stamp, const Eigen::Vector2d& position, double yaw) {
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
    path.header = pose.header;
    path.poses.push_back(pose);
    if (path.poses.size() > 100000) {
        path.poses.erase(path.poses.begin(), path.poses.begin() + 1000);
    }
    if (publisher) publisher->publish(path);
}

loc::LocalizationResult LocalizationSystem::GetLatestResult() const { std::lock_guard<std::mutex> lock(result_mutex_); return latest_result_; }
double LocalizationSystem::PoseYaw(const SE3& pose) { return std::atan2(pose.so3().matrix()(1, 0), pose.so3().matrix()(0, 0)); }

void LocalizationSystem::Reset() {
    if (ui_) {
        ui_->Quit();
        ui_.reset();
    }
    if (ndt_localization_) ndt_localization_->Finish();
    ndt_localization_.reset();
    { std::lock_guard<std::mutex> lock(filter_mutex_); ekf_.Reset(); }
    lidar_topic_.clear();
    livox_lidar_topic_.clear();
    imu_topic_.clear();
    gps_topic_.clear();
    orientation_topic_.clear();
    velocity_topic_.clear();
    wheel_odometry_topic_.clear();
    with_ui_ = false;
    ResetMapEnuCalibration();
    gps_rotation_body_gps_rpy_.setZero();
    gps_rotation_body_gps_.setIdentity();
    initial_position_std_ = 0.5;
    initial_orientation_std_ = 3.0 * kDegToRad;
    initial_velocity_std_ = 2.0;
    initial_yaw_rate_std_ = 0.5;
    gps_position_std_x_ = 0.05;
    gps_position_std_y_ = 0.05;
    gps_position_std_z_ = 100.0;
    gps_orientation_std_roll_ = 0.2 * kDegToRad;
    gps_orientation_std_pitch_ = 0.2 * kDegToRad;
    gps_orientation_std_yaw_ = 0.5 * kDegToRad;
    gps_velocity_std_x_ = 0.10;
    gps_velocity_std_y_ = 0.10;
    ndt_position_std_x_ = 0.10;
    ndt_position_std_y_ = 0.10;
    ndt_position_std_z_ = 0.20;
    ndt_orientation_std_ = 1.0 * kDegToRad;
    map_ready_ = false;
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
