#include "modules/localizationSystem/localization_system.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <rclcpp/node.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "common/point_def.h"
#include "core/lightning_math.hpp"
#include "core/localization/localization.h"
#include "ui/pangolin_window.h"

namespace lightning::modules {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;

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

std::size_t YamlIndent(const std::string& line) {
    return line.find_first_not_of(' ');
}

std::string TrimYamlLine(const std::string& line) {
    const std::size_t begin = line.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return std::string();
    const std::size_t end = line.find_last_not_of(" \t\r\n");
    return line.substr(begin, end - begin + 1);
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
                std::lock_guard<std::mutex> lock(gps_init_mutex_);
                if (!UsesGpsPosition() || start_EKF_localization_ ||
                    !result.reliable_) {
                    return;
                }

                const Eigen::Vector3d pos = result.pose_.translation();

                if (ndt_positions_.empty()) {
                    ndt_positions_.push_back(pos);
                    ndt_timestamps_.push_back(result.timestamp_);
                    return;
                }

                const double dist_to_last =
                    (pos - ndt_positions_.back()).norm();
                if (dist_to_last <= 1.0) return;

                ndt_positions_.push_back(pos);
                ndt_timestamps_.push_back(result.timestamp_);

                const double dist_to_first =
                    (pos - ndt_positions_.front()).norm();
                if (dist_to_first >= 50.0) {
                    TryHandleGpsInitialization();
                }
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
    if (!std::isfinite(stamp) || !std::isfinite(fix->latitude) ||
        !std::isfinite(fix->longitude) || !std::isfinite(fix->altitude)) {
        return;
    }

    std::lock_guard<std::mutex> lock(gps_init_mutex_);

    if (!start_EKF_localization_) {
        if (!WGS84_Model.Initialized()) {
            if (count < 10) {
                fix_postions_[0] += Eigen::Vector3d(
                    fix->latitude, fix->longitude, fix->altitude);
                fix_timestamps_[0] = stamp;
                ++count;
                return;
            }

            fix_postions_[0] /= count;
            if (!WGS84_Model.SetOriginDegrees(
                    fix_postions_[0].x(), fix_postions_[0].y(), fix_postions_[0].z())) {
                return;
            }

            // 这样做的目的是0点一定要稳：平均前10帧，避免噪声导致零点不稳。
            fix_positions_[0] = WGS84_Model.ForwardDegrees(
                fix_postions_[0].x(), fix_postions_[0].y(),
                fix_postions_[0].z());
            LOG(INFO) << "[MAP_ENU_CALIBRATION] ENU origin set"
                      << ", latitude=" << fix_postions_[0].x()
                      << ", longitude=" << fix_postions_[0].y()
                      << ", altitude=" << fix_postions_[0].z();
        }

        // NDT负责每隔约1m采样；GPS只保存该采样时刻前后50ms内的少量数据。
        Eigen::Vector3d position_enu = WGS84_Model.ForwardDegrees(
            fix->latitude, fix->longitude, fix->altitude);
        if (!ndt_timestamps_.empty() &&
            std::abs(stamp - ndt_timestamps_.back()) < 0.05) {
            fix_positions_.push_back(position_enu);
            fix_timestamps_.push_back(stamp);
        }
        return;
    }

    Eigen::Vector3d position_enu;
    Eigen::Matrix3d covariance_enu;
    GnssToEnu(*fix, position_enu, covariance_enu);
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

    if (!start_EKF_localization_) return;

    HandleGpsOrientation(*orientation);
}

void LocalizationSystem::TryHandleGpsInitialization() {
    if (fix_positions_.size() != fix_timestamps_.size()) return;
    if (ndt_positions_.size() != ndt_timestamps_.size()) return;
    if (ndt_positions_.empty() || fix_positions_.size() <= 1) return;

    std::vector<Eigen::Vector3d> GPS_positions;
    std::vector<Eigen::Vector3d> NDT_positions;
    GPS_positions.reserve(ndt_positions_.size());
    NDT_positions.reserve(ndt_positions_.size());

    int j = 0;
    for (int i = 0; i < static_cast<int>(ndt_positions_.size()); ++i) {
        const double ndt_stamp = ndt_timestamps_[i];

        double min_diff = std::abs(ndt_stamp - fix_timestamps_[j]);
        int best_j = j;

        while (j + 1 < static_cast<int>(fix_timestamps_.size())) {
            const double next_diff =
                std::abs(ndt_stamp - fix_timestamps_[j + 1]);
            if (next_diff < min_diff) {
                min_diff = next_diff;
                best_j = j + 1;
                ++j;
            } else {
                break;
            }
        }

        if (min_diff < 0.05) {
            GPS_positions.push_back(fix_positions_[best_j]);
            NDT_positions.push_back(ndt_positions_[i]);
            j = best_j;
        }
    }

    const int N = static_cast<int>(GPS_positions.size());
    if (N <= 10) return;

    const double matched_distance =
        (NDT_positions.back() - NDT_positions.front()).norm();
    if (matched_distance < 40.0) return;

    Eigen::Matrix2Xd gps_mat(2, N);
    Eigen::Matrix2Xd ndt_mat(2, N);
    for (int i = 0; i < N; ++i) {
        gps_mat.col(i) = GPS_positions[i].head<2>();
        ndt_mat.col(i) = NDT_positions[i].head<2>();
    }

    // MAP和ENU都是重力对齐的固定坐标系，只估计水平面内的yaw和平移。
    const Eigen::Matrix3d T_map_enu =
        Eigen::umeyama(gps_mat, ndt_mat, false);
    R_MAP_ENU.setIdentity();
    R_MAP_ENU.topLeftCorner<2, 2>() =
        T_map_enu.topLeftCorner<2, 2>();
    t_MAP_ENU.head<2>() = T_map_enu.block<2, 1>(0, 2);

    double z_offset_sum = 0.0;
    double squared_error_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        z_offset_sum += NDT_positions[i].z() - GPS_positions[i].z();
        const Eigen::Vector2d residual =
            R_MAP_ENU.topLeftCorner<2, 2>() * GPS_positions[i].head<2>() +
            t_MAP_ENU.head<2>() - NDT_positions[i].head<2>();
        squared_error_sum += residual.squaredNorm();
    }
    t_MAP_ENU.z() = z_offset_sum / static_cast<double>(N);

    const double rmse =
        std::sqrt(squared_error_sum / static_cast<double>(N));
    const double yaw_map_enu =
        std::atan2(R_MAP_ENU(1, 0), R_MAP_ENU(0, 0));
    start_EKF_localization_ = true;

    const Eigen::Vector3d rotation_map_enu_rpy_deg =
        loc::EKF::RpyFromRotation(R_MAP_ENU) / kDegToRad;
    const bool config_saved = SaveMapEnuCalibrationToConfig(
        t_MAP_ENU, rotation_map_enu_rpy_deg);
    LOG(INFO) << "[MAP_ENU_CALIBRATION] completed"
              << ", matched_pairs=" << N
              << ", matched_distance_m=" << matched_distance
              << ", rmse_m=" << rmse
              << ", yaw_map_enu_deg=" << yaw_map_enu / kDegToRad
              << ", rotation_map_enu_rpy_deg="
              << rotation_map_enu_rpy_deg.transpose()
              << ", translation_map_enu="
              << t_MAP_ENU.transpose()
              << ", config_saved=" << config_saved
              << ", config_path=" << yaml_path_;
}

bool LocalizationSystem::SaveMapEnuCalibrationToConfig(
    const Eigen::Vector3d& translation_map_enu,
    const Eigen::Vector3d& rotation_map_enu_rpy_deg) const {
    std::ifstream input(yaml_path_);
    if (!input.is_open()) {
        LOG(ERROR) << "[MAP_ENU_CALIBRATION] cannot open config for reading: "
                   << yaml_path_;
        return false;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) lines.push_back(line);

    const std::size_t not_found = std::string::npos;
    std::size_t localization_begin = not_found;
    std::size_t localization_end = lines.size();
    for (std::size_t index = 0; index < lines.size(); ++index) {
        if (YamlIndent(lines[index]) == 0 &&
            TrimYamlLine(lines[index]) == "localization:") {
            localization_begin = index;
            break;
        }
    }
    if (localization_begin == not_found) {
        LOG(ERROR) << "[MAP_ENU_CALIBRATION] config has no localization "
                      "section: " << yaml_path_;
        return false;
    }

    for (std::size_t index = localization_begin + 1;
         index < lines.size(); ++index) {
        const std::string trimmed = TrimYamlLine(lines[index]);
        if (!trimmed.empty() && trimmed.front() != '#' &&
            YamlIndent(lines[index]) == 0) {
            localization_end = index;
            break;
        }
    }

    std::ostringstream translation;
    translation << std::setprecision(15)
                << "    translation_map_enu: ["
                << translation_map_enu.x() << ", "
                << translation_map_enu.y() << ", "
                << translation_map_enu.z() << "]";
    std::ostringstream rotation;
    rotation << std::setprecision(15)
             << "    rotation_map_enu_rpy_deg: ["
             << rotation_map_enu_rpy_deg.x() << ", "
             << rotation_map_enu_rpy_deg.y() << ", "
             << rotation_map_enu_rpy_deg.z() << "]";
    const std::vector<std::string> result_block{
        "  map_enu_calibration:",
        "    # p_map = R_map_enu * p_enu + translation_map_enu",
        translation.str(),
        rotation.str(),
        ""};

    std::size_t result_begin = not_found;
    std::size_t result_end = localization_end;
    for (std::size_t index = localization_begin + 1;
         index < localization_end; ++index) {
        if (YamlIndent(lines[index]) == 2 &&
            TrimYamlLine(lines[index]) == "map_enu_calibration:") {
            result_begin = index;
            result_end = index + 1;
            while (result_end < localization_end) {
                const std::string trimmed = TrimYamlLine(lines[result_end]);
                if (!trimmed.empty() && trimmed.front() != '#' &&
                    YamlIndent(lines[result_end]) <= 2) {
                    break;
                }
                ++result_end;
            }
            break;
        }
    }

    if (result_begin == not_found) {
        lines.insert(lines.begin() + localization_end,
                     result_block.begin(), result_block.end());
    } else {
        lines.erase(lines.begin() + result_begin, lines.begin() + result_end);
        lines.insert(lines.begin() + result_begin,
                     result_block.begin(), result_block.end());
    }

    const std::filesystem::path config_path(yaml_path_);
    const std::filesystem::path temporary_path =
        config_path.string() + ".map_enu.tmp";
    std::ofstream output(temporary_path, std::ios::trunc);
    if (!output.is_open()) {
        LOG(ERROR) << "[MAP_ENU_CALIBRATION] cannot open temporary config: "
                   << temporary_path.string();
        return false;
    }
    for (const std::string& output_line : lines) {
        output << output_line << '\n';
    }
    output.close();
    if (!output) {
        LOG(ERROR) << "[MAP_ENU_CALIBRATION] failed writing temporary config: "
                   << temporary_path.string();
        std::error_code remove_error;
        std::filesystem::remove(temporary_path, remove_error);
        return false;
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary_path, config_path, rename_error);
    if (rename_error) {
        LOG(ERROR) << "[MAP_ENU_CALIBRATION] failed replacing config: "
                   << rename_error.message();
        std::error_code remove_error;
        std::filesystem::remove(temporary_path, remove_error);
        return false;
    }

    LOG(INFO) << "[MAP_ENU_CALIBRATION] result saved to " << yaml_path_;
    return true;
}

void LocalizationSystem::ResetMapEnuCalibration() {
    std::lock_guard<std::mutex> lock(gps_init_mutex_);
    WGS84_Model = math::JsbsimWgs84Enu();
    start_EKF_localization_ = false;
    R_MAP_ENU.setIdentity();
    t_MAP_ENU.setZero();
    fix_postions_.assign(1, Eigen::Vector3d::Zero());
    fix_positions_.assign(1, Eigen::Vector3d::Zero());
    fix_timestamps_.assign(1, 0.0);
    ndt_positions_.clear();
    ndt_timestamps_.clear();
    count = 0;
}

void LocalizationSystem::HandleGpsPosition(
    double stamp, const Eigen::Vector3d& position_enu,
    const Eigen::Matrix3d& covariance_enu) {
    const Eigen::Vector3d position_map =
        R_MAP_ENU * position_enu + t_MAP_ENU;
    Eigen::Matrix3d covariance_map =
        R_MAP_ENU * covariance_enu * R_MAP_ENU.transpose();
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
        R_MAP_ENU * measured_rotation_enu_gps *
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
        !start_EKF_localization_) {
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
    position_enu = WGS84_Model.ForwardDegrees(
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
        covariance_enu = WGS84_Model.CovarianceEcefToEnu(
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

    velocity_map = (R_MAP_ENU * velocity_enu).head<2>();

    Eigen::Matrix3d covariance_enu;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            covariance_enu(row, column) =
                velocity.twist.covariance[row * 6 + column];
        }
    }
    if (IsUsableCovariance(covariance_enu)) {
        const Eigen::Matrix3d covariance_map_3d =
            R_MAP_ENU * covariance_enu *
            R_MAP_ENU.transpose();
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
