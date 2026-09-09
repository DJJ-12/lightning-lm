#include "modules/mapEnuCalibration/map_enu_calibrator.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <glog/logging.h>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/nav_sat_status.hpp>
#include <yaml-cpp/yaml.h>

namespace lightning::modules {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kTimeEpsilon = 1e-6;

double Clamp(double value, double lower, double upper) {
    return std::max(lower, std::min(upper, value));
}

Eigen::Vector3d ReadVector3(const YAML::Node& node,
                           const Eigen::Vector3d& fallback) {
    if (!node || !node.IsSequence() || node.size() != 3) return fallback;
    return Eigen::Vector3d(
        node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
}

template <typename T>
T ReadValue(const YAML::Node& node, const char* key, const T& fallback) {
    return node && node[key] ? node[key].as<T>() : fallback;
}

bool IsFiniteRotation(const Eigen::Matrix3d& rotation) {
    return rotation.allFinite() &&
           std::fabs(rotation.determinant() - 1.0) < 1e-3 &&
           (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
                   .norm() <
               1e-3;
}

template <int Dimension>
Eigen::Matrix<double, Dimension, Dimension> SymmetricPositiveDefinite(
    const Eigen::Matrix<double, Dimension, Dimension>& input,
    double minimum_eigenvalue = 1e-10,
    double maximum_eigenvalue = 1e8) {
    using Matrix = Eigen::Matrix<double, Dimension, Dimension>;
    const Matrix symmetric = 0.5 * (input + input.transpose());
    Eigen::SelfAdjointEigenSolver<Matrix> solver(symmetric);
    if (solver.info() != Eigen::Success ||
        !solver.eigenvalues().allFinite()) {
        return Matrix::Identity();
    }
    Eigen::Matrix<double, Dimension, 1> eigenvalues = solver.eigenvalues();
    for (int index = 0; index < Dimension; ++index) {
        eigenvalues(index) = Clamp(
            eigenvalues(index), minimum_eigenvalue, maximum_eigenvalue);
    }
    return solver.eigenvectors() * eigenvalues.asDiagonal() *
           solver.eigenvectors().transpose();
}

template <int Dimension>
Eigen::Matrix<double, Dimension, Dimension> InverseSpd(
    const Eigen::Matrix<double, Dimension, Dimension>& covariance) {
    using Matrix = Eigen::Matrix<double, Dimension, Dimension>;
    const Matrix positive =
        SymmetricPositiveDefinite<Dimension>(covariance);
    Eigen::LDLT<Matrix> decomposition(positive);
    if (decomposition.info() != Eigen::Success ||
        !decomposition.isPositive()) {
        return Matrix::Identity();
    }
    return decomposition.solve(Matrix::Identity());
}

double Percentile95(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t index = std::min(
        values.size() - 1,
        static_cast<std::size_t>(std::ceil(0.95 * values.size()) - 1.0));
    return values[index];
}

}  // namespace

bool MapEnuCalibrator::Configure(const std::string& yaml_path,
                                 std::string* message) {
    std::lock_guard<std::mutex> lock(mutex_);
    yaml_path_ = yaml_path;
    localization_system_.reset();
    last_localization_stamp_ = -std::numeric_limits<double>::infinity();
    configured_ = false;
    active_ = false;
    solved_ = false;
    phase_ = "DISABLED";
    configuration_error_.clear();
    last_message_.clear();
    main_gnss_buffer_.clear();
    slave_gnss_buffer_.clear();
    pending_ndt_.clear();
    samples_.clear();
    result_ = MapEnuCalibrationResult();
    main_gnss_received_ = 0;
    main_gnss_valid_ = 0;
    slave_gnss_received_ = 0;
    slave_gnss_valid_ = 0;
    ndt_received_ = 0;
    ndt_valid_ = 0;
    rejected_time_sync_ = 0;
    rejected_baseline_ = 0;
    rejected_sampling_ = 0;
    baseline_squared_error_sum_ = 0.0;
    attitude_coverage_deg_ = 0.0;

    try {
        const YAML::Node yaml = YAML::LoadFile(yaml_path);
        const YAML::Node common = yaml["common"];
        const YAML::Node localization = yaml["localization"];
        const YAML::Node calibration =
            localization ? localization["map_enu_calibration"] : YAML::Node();
        if (!calibration) {
            configuration_error_ =
                "Map-ENU calibration is disabled: configuration section is absent";
            last_message_ = configuration_error_;
            if (message) *message = last_message_;
            LOG(INFO) << "[MAP_ENU_CALIBRATION] " << last_message_;
            return true;
        }

        options_ = Options();
        const std::string gps1_topic = ReadValue<std::string>(
            common, "gps1_topic", std::string());
        const std::string gps2_topic = ReadValue<std::string>(
            common, "gps2_topic", std::string());

        // Topic emptiness follows the same convention as the rest of
        // Lightning: it disables this optional sensor path. This is a normal
        // startup state, not a configuration failure for mapping/localization.
        if (gps1_topic.empty() || gps2_topic.empty()) {
            configuration_error_ =
                "Map-ENU calibration is disabled: both common.gps1_topic "
                "and common.gps2_topic are required";
            last_message_ = configuration_error_;
            if (message) *message = last_message_;
            LOG(INFO) << "[MAP_ENU_CALIBRATION] " << last_message_;
            return true;
        }

        const YAML::Node dual_gnss = calibration["dual_gnss"]
            ? calibration["dual_gnss"]
            : calibration;
        const YAML::Node gps_ins =
            localization && localization["gps_ins"]
                ? localization["gps_ins"]
                : YAML::Node();
        const Eigen::Vector3d legacy_main = ReadVector3(
            dual_gnss["main_antenna_in_body"], Eigen::Vector3d::Zero());
        const Eigen::Vector3d legacy_slave = ReadVector3(
            dual_gnss["slave_antenna_in_body"], Eigen::Vector3d::Zero());
        options_.main_antenna_in_body = ReadVector3(
            gps_ins ? gps_ins["lever_arm_tracking"] : YAML::Node(),
            legacy_main);
        options_.slave_antenna_in_body = ReadVector3(
            gps_ins ? gps_ins["gps2_lever_arm_tracking"] : YAML::Node(),
            legacy_slave);

        YAML::Node origin = calibration["enu_origin"];
        if (!origin && localization && localization["map_from_enu"]) {
            const YAML::Node old_map_from_enu = localization["map_from_enu"];
            origin = old_map_from_enu["enu_origin"]
                ? old_map_from_enu["enu_origin"]
                : old_map_from_enu;
        }
        if (!origin || !origin["lat"] || !origin["lon"] ||
            !origin["alt"]) {
            configuration_error_ =
                "map_enu_calibration.enu_origin requires lat/lon/alt";
            if (message) *message = configuration_error_;
            return false;
        }
        options_.enu_origin_lla = Eigen::Vector3d(
            origin["lat"].as<double>(), origin["lon"].as<double>(),
            origin["alt"].as<double>());

        options_.gnss_time_offset_sec = ReadValue<double>(
            calibration, "gnss_time_offset_sec",
            options_.gnss_time_offset_sec);
        options_.gnss_buffer_duration_sec = ReadValue<double>(
            calibration, "gnss_buffer_duration_sec",
            options_.gnss_buffer_duration_sec);
        options_.interpolation_max_gap_sec = ReadValue<double>(
            calibration, "interpolation_max_gap_sec",
            options_.interpolation_max_gap_sec);
        options_.baseline_length_tolerance_m = ReadValue<double>(
            calibration, "baseline_length_tolerance_m",
            options_.baseline_length_tolerance_m);
        options_.min_sample_interval_sec = ReadValue<double>(
            calibration, "min_sample_interval_sec",
            options_.min_sample_interval_sec);
        options_.min_sample_distance_m = ReadValue<double>(
            calibration, "min_sample_distance_m",
            options_.min_sample_distance_m);
        options_.min_baseline_direction_change_deg = ReadValue<double>(
            calibration, "min_baseline_direction_change_deg",
            options_.min_baseline_direction_change_deg);
        options_.min_samples = ReadValue<std::size_t>(
            calibration, "min_samples", options_.min_samples);
        options_.max_samples = ReadValue<std::size_t>(
            calibration, "max_samples", options_.max_samples);
        options_.max_gnss_samples = ReadValue<std::size_t>(
            calibration, "max_gnss_samples", options_.max_gnss_samples);
        options_.max_pending_ndt_samples = ReadValue<std::size_t>(
            calibration, "max_pending_ndt_samples",
            options_.max_pending_ndt_samples);
        options_.require_gnss_covariance = ReadValue<bool>(
            calibration, "require_gnss_covariance",
            options_.require_gnss_covariance);
        options_.require_ndt_covariance = ReadValue<bool>(
            calibration, "require_ndt_covariance",
            options_.require_ndt_covariance);
        options_.gnss_std_floor_m = ReadVector3(
            calibration["gnss_std_floor_m"], options_.gnss_std_floor_m);
        options_.fallback_ndt_position_std_m = ReadVector3(
            calibration["fallback_ndt_position_std_m"],
            options_.fallback_ndt_position_std_m);
        options_.fallback_ndt_rotation_std_deg = ReadVector3(
            calibration["fallback_ndt_rotation_std_deg"],
            options_.fallback_ndt_rotation_std_deg);
        options_.antenna_position_std_m = ReadValue<double>(
            calibration, "antenna_position_std_m",
            options_.antenna_position_std_m);
        options_.minimum_ndt_tp = ReadValue<double>(
            calibration, "minimum_ndt_tp", options_.minimum_ndt_tp);
        options_.minimum_ndt_nvtl = ReadValue<double>(
            calibration, "minimum_ndt_nvtl", options_.minimum_ndt_nvtl);
        options_.maximum_ndt_iterations = ReadValue<int>(
            calibration, "maximum_ndt_iterations",
            options_.maximum_ndt_iterations);
        options_.minimum_attitude_coverage_deg = ReadValue<double>(
            calibration, "minimum_attitude_coverage_deg",
            options_.minimum_attitude_coverage_deg);
        options_.minimum_rotation_singular_ratio = ReadValue<double>(
            calibration, "minimum_rotation_singular_ratio",
            options_.minimum_rotation_singular_ratio);
        options_.maximum_rotation_condition_number = ReadValue<double>(
            calibration, "maximum_rotation_condition_number",
            options_.maximum_rotation_condition_number);
        options_.maximum_rotation_rms_deg = ReadValue<double>(
            calibration, "maximum_rotation_rms_deg",
            options_.maximum_rotation_rms_deg);
        options_.maximum_translation_rms_m = ReadValue<double>(
            calibration, "maximum_translation_rms_m",
            options_.maximum_translation_rms_m);
        options_.maximum_residual_m = ReadValue<double>(
            calibration, "maximum_residual_m",
            options_.maximum_residual_m);
        options_.huber_delta_sigma = ReadValue<double>(
            calibration, "huber_delta_sigma", options_.huber_delta_sigma);
        options_.maximum_refinement_iterations = ReadValue<int>(
            calibration, "maximum_refinement_iterations",
            options_.maximum_refinement_iterations);

        const Eigen::Vector3d baseline_body =
            options_.slave_antenna_in_body -
            options_.main_antenna_in_body;
        const bool vectors_valid = options_.enu_origin_lla.allFinite() &&
            options_.main_antenna_in_body.allFinite() &&
            options_.slave_antenna_in_body.allFinite() &&
            options_.gnss_std_floor_m.allFinite() &&
            (options_.gnss_std_floor_m.array() > 0.0).all() &&
            options_.fallback_ndt_position_std_m.allFinite() &&
            (options_.fallback_ndt_position_std_m.array() > 0.0).all() &&
            options_.fallback_ndt_rotation_std_deg.allFinite() &&
            (options_.fallback_ndt_rotation_std_deg.array() > 0.0).all();
        const bool scalar_values_valid =
            std::isfinite(options_.gnss_time_offset_sec) &&
            options_.gnss_buffer_duration_sec > 0.0 &&
            options_.interpolation_max_gap_sec > 0.0 &&
            options_.baseline_length_tolerance_m > 0.0 &&
            options_.min_sample_interval_sec >= 0.0 &&
            options_.min_sample_distance_m >= 0.0 &&
            options_.min_baseline_direction_change_deg >= 0.0 &&
            options_.antenna_position_std_m >= 0.0 &&
            options_.minimum_attitude_coverage_deg > 0.0 &&
            options_.minimum_rotation_singular_ratio > 0.0 &&
            options_.maximum_rotation_condition_number > 1.0 &&
            options_.maximum_rotation_rms_deg > 0.0 &&
            options_.maximum_translation_rms_m > 0.0 &&
            options_.maximum_residual_m > 0.0 &&
            options_.huber_delta_sigma > 0.0 &&
            options_.maximum_refinement_iterations > 0;
        const bool counts_valid = options_.min_samples >= 3 &&
            options_.max_samples >= options_.min_samples &&
            options_.max_gnss_samples >= 2 &&
            options_.max_pending_ndt_samples >= 1;
        if (gps1_topic == gps2_topic) {
            configuration_error_ =
                "common.gps1_topic and common.gps2_topic must be different";
        } else if (!vectors_valid || !scalar_values_valid || !counts_valid) {
            configuration_error_ =
                "map_enu_calibration contains an invalid option";
        } else if (baseline_body.norm() < 0.10) {
            configuration_error_ =
                "dual-GNSS antenna baseline must be at least 0.10 m";
        } else if (!enu_projector_.SetOriginDegrees(
                       options_.enu_origin_lla.x(),
                       options_.enu_origin_lla.y(),
                       options_.enu_origin_lla.z())) {
            configuration_error_ = "invalid WGS84 ENU origin";
        }
        if (!configuration_error_.empty()) {
            if (message) *message = configuration_error_;
            return false;
        }

        configured_ = true;
        phase_ = "IDLE";
        last_message_ = "Map-ENU calibration is configured";
        if (message) *message = last_message_;
        LOG(INFO) << "[MAP_ENU_CALIBRATION] configured"
                  << ", gps1_topic=" << gps1_topic
                  << ", gps2_topic=" << gps2_topic
                  << ", origin_lla=" << options_.enu_origin_lla.transpose()
                  << ", main_antenna_body="
                  << options_.main_antenna_in_body.transpose()
                  << ", slave_antenna_body="
                  << options_.slave_antenna_in_body.transpose()
                  << ", baseline_m=" << baseline_body.norm();
        return true;
    } catch (const std::exception& exception) {
        configuration_error_ =
            std::string("failed to read map_enu_calibration: ") +
            exception.what();
        if (message) *message = configuration_error_;
        return false;
    }
}

bool MapEnuCalibrator::Start(const std::string& map_path,
                             rclcpp::Node::SharedPtr node,
                             std::string* message) {
    std::string yaml_path;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!configured_) {
            const std::string error = configuration_error_.empty()
                ? "Map-ENU calibration is not configured"
                : configuration_error_;
            if (message) *message = error;
            return false;
        }
        if (active_) {
            if (message) {
                *message = "Map-ENU calibration is already collecting";
            }
            return false;
        }
        yaml_path = yaml_path_;
    }
    if (!node || map_path.empty()) {
        if (message) *message = "calibration requires a node and map_path";
        return false;
    }

    // LocalizationSystem is deliberately treated as a black box. Calibration
    // gives it a map and LiDAR frames, then reads the latest public result.
    auto localization = std::make_shared<LocalizationSystem>();
    if (!localization->Init(yaml_path, node)) {
        if (message) {
            *message = "failed to initialize calibration LocalizationSystem";
        }
        return false;
    }
    if (!localization->SetMapPath(map_path)) {
        if (message) *message = "failed to load calibration map: " + map_path;
        return false;
    }
    const SE3 map_origin(
        Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    if (!localization->SetInitialGuess(map_origin)) {
        if (message) {
            *message = "failed to set map-origin calibration initial pose";
        }
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    main_gnss_buffer_.clear();
    slave_gnss_buffer_.clear();
    pending_ndt_.clear();
    samples_.clear();
    result_ = MapEnuCalibrationResult();
    main_gnss_received_ = 0;
    main_gnss_valid_ = 0;
    slave_gnss_received_ = 0;
    slave_gnss_valid_ = 0;
    ndt_received_ = 0;
    ndt_valid_ = 0;
    rejected_time_sync_ = 0;
    rejected_baseline_ = 0;
    rejected_sampling_ = 0;
    baseline_squared_error_sum_ = 0.0;
    attitude_coverage_deg_ = 0.0;
    localization_system_ = std::move(localization);
    last_localization_stamp_ = -std::numeric_limits<double>::infinity();
    solved_ = false;
    active_ = true;
    phase_ = "STATIC_COLLECTION";
    last_message_ =
        "collecting dual-GNSS and reliable NDT observations";
    if (message) *message = last_message_;
    LOG(INFO) << "[MAP_ENU_CALIBRATION] collection started";
    return true;
}

void MapEnuCalibrator::ResetSession() {
    std::lock_guard<std::mutex> lock(mutex_);
    localization_system_.reset();
    last_localization_stamp_ = -std::numeric_limits<double>::infinity();
    active_ = false;
    solved_ = false;
    phase_ = configured_ ? "IDLE" : "DISABLED";
    main_gnss_buffer_.clear();
    slave_gnss_buffer_.clear();
    pending_ndt_.clear();
    samples_.clear();
    result_ = MapEnuCalibrationResult();
    main_gnss_received_ = 0;
    main_gnss_valid_ = 0;
    slave_gnss_received_ = 0;
    slave_gnss_valid_ = 0;
    ndt_received_ = 0;
    ndt_valid_ = 0;
    rejected_time_sync_ = 0;
    rejected_baseline_ = 0;
    rejected_sampling_ = 0;
    baseline_squared_error_sum_ = 0.0;
    attitude_coverage_deg_ = 0.0;
    last_message_ = configured_
        ? "Map-ENU calibration session reset"
        : configuration_error_;
}

template <typename CloudMessage>
loc::LocalizationFrameOutcome MapEnuCalibrator::ProcessCloudImpl(
    const std::shared_ptr<CloudMessage>& cloud,
    const loc::LocalizationInputDiagnostic& diagnostic) {
    std::shared_ptr<LocalizationSystem> localization;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!active_ || !localization_system_) {
            return loc::LocalizationFrameOutcome::SYSTEM_NOT_READY;
        }
        localization = localization_system_;
    }

    const loc::LocalizationFrameOutcome outcome =
        localization->ProcessCloud(cloud, diagnostic);
    const loc::LocalizationResult result = localization->GetLatestResult();

    bool is_new_result = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_ && result.valid_ &&
            result.timestamp_ > last_localization_stamp_ + kTimeEpsilon) {
            last_localization_stamp_ = result.timestamp_;
            is_new_result = true;
        }
    }
    if (is_new_result) AddNdtObservation(result);
    return outcome;
}

loc::LocalizationFrameOutcome MapEnuCalibrator::ProcessCloud(
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud,
    const loc::LocalizationInputDiagnostic& diagnostic) {
    return ProcessCloudImpl(cloud, diagnostic);
}

loc::LocalizationFrameOutcome MapEnuCalibrator::ProcessCloud(
    const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud,
    const loc::LocalizationInputDiagnostic& diagnostic) {
    return ProcessCloudImpl(cloud, diagnostic);
}

bool MapEnuCalibrator::Configured() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return configured_;
}

bool MapEnuCalibrator::Active() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_;
}

void MapEnuCalibrator::AddMainGnss(
    const sensor_msgs::msg::NavSatFix& fix) {
    AddGnss(fix, true);
}

void MapEnuCalibrator::AddSlaveGnss(
    const sensor_msgs::msg::NavSatFix& fix) {
    AddGnss(fix, false);
}

void MapEnuCalibrator::AddGnss(
    const sensor_msgs::msg::NavSatFix& fix, bool main) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) return;
    if (main) {
        ++main_gnss_received_;
    } else {
        ++slave_gnss_received_;
    }

    GnssSample sample;
    if (!ConvertGnss(fix, &sample)) return;
    if (main) {
        ++main_gnss_valid_;
        InsertGnss(&main_gnss_buffer_, std::move(sample));
    } else {
        ++slave_gnss_valid_;
        InsertGnss(&slave_gnss_buffer_, std::move(sample));
    }
    TryMatchPendingNdt();
}

bool MapEnuCalibrator::ConvertGnss(
    const sensor_msgs::msg::NavSatFix& fix, GnssSample* sample) const {
    if (!sample ||
        fix.status.status == sensor_msgs::msg::NavSatStatus::STATUS_NO_FIX ||
        !std::isfinite(fix.latitude) || !std::isfinite(fix.longitude) ||
        !std::isfinite(fix.altitude) || fix.latitude < -90.0 ||
        fix.latitude > 90.0 || fix.longitude < -180.0 ||
        fix.longitude > 180.0) {
        return false;
    }
    const double raw_stamp = rclcpp::Time(fix.header.stamp).seconds();
    sample->stamp = raw_stamp + options_.gnss_time_offset_sec;
    if (!std::isfinite(sample->stamp)) return false;

    sample->position_enu = enu_projector_.ForwardDegrees(
        fix.latitude, fix.longitude, fix.altitude);
    if (!sample->position_enu.allFinite()) return false;

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            covariance(row, column) =
                fix.position_covariance[row * 3 + column];
        }
    }
    bool covariance_valid =
        fix.position_covariance_type !=
            sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN &&
        covariance.allFinite();
    if (covariance_valid) {
        covariance = 0.5 * (covariance + covariance.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        covariance_valid = solver.info() == Eigen::Success &&
            solver.eigenvalues().minCoeff() > 0.0;
    }
    if (!covariance_valid && options_.require_gnss_covariance) return false;
    if (!covariance_valid) covariance.setZero();

    for (int axis = 0; axis < 3; ++axis) {
        covariance(axis, axis) = std::max(
            covariance(axis, axis),
            options_.gnss_std_floor_m(axis) *
                options_.gnss_std_floor_m(axis));
    }
    sample->covariance_enu =
        SymmetricPositiveDefinite<3>(covariance);
    return true;
}

void MapEnuCalibrator::InsertGnss(
    std::deque<GnssSample>* buffer, GnssSample sample) {
    if (!buffer) return;
    const auto position = std::lower_bound(
        buffer->begin(), buffer->end(), sample.stamp,
        [](const GnssSample& existing, double stamp) {
            return existing.stamp < stamp;
        });
    if (position != buffer->end() &&
        std::fabs(position->stamp - sample.stamp) <= kTimeEpsilon) {
        *position = std::move(sample);
    } else {
        buffer->insert(position, std::move(sample));
    }
    while (buffer->size() > options_.max_gnss_samples) {
        buffer->pop_front();
    }
    if (!buffer->empty()) {
        const double newest_stamp = buffer->back().stamp;
        while (buffer->size() > 2 &&
               newest_stamp - buffer->front().stamp >
                   options_.gnss_buffer_duration_sec) {
            buffer->pop_front();
        }
    }
}

bool MapEnuCalibrator::ValidateNdt(
    const loc::LocalizationResult& ndt, NdtSample* sample) const {
    if (!sample || !ndt.valid_ || !ndt.reliable_ ||
        !std::isfinite(ndt.timestamp_) ||
        !ndt.pose_.translation().allFinite() ||
        !IsFiniteRotation(ndt.pose_.rotationMatrix()) ||
        !std::isfinite(ndt.tp_) || ndt.tp_ < options_.minimum_ndt_tp ||
        !std::isfinite(ndt.nvtl_) || ndt.nvtl_ < options_.minimum_ndt_nvtl ||
        ndt.iterations_ < 0 ||
        ndt.iterations_ > options_.maximum_ndt_iterations) {
        return false;
    }

    sample->stamp = ndt.timestamp_;
    sample->rotation_map_body = ndt.pose_.rotationMatrix();
    sample->translation_map_body = ndt.pose_.translation();
    bool covariance_valid = ndt.covariance_valid_ &&
        ndt.pose_covariance_.allFinite();
    if (covariance_valid) {
        const NdtPoseCovariance symmetric = 0.5 *
            (ndt.pose_covariance_ + ndt.pose_covariance_.transpose());
        Eigen::SelfAdjointEigenSolver<NdtPoseCovariance> solver(symmetric);
        covariance_valid = solver.info() == Eigen::Success &&
            solver.eigenvalues().minCoeff() > 0.0;
        if (covariance_valid) sample->covariance = symmetric;
    }
    if (!covariance_valid && options_.require_ndt_covariance) return false;
    if (!covariance_valid) {
        sample->covariance.setZero();
        for (int axis = 0; axis < 3; ++axis) {
            sample->covariance(axis, axis) =
                options_.fallback_ndt_position_std_m(axis) *
                options_.fallback_ndt_position_std_m(axis);
            const double rotation_std =
                options_.fallback_ndt_rotation_std_deg(axis) * kDegToRad;
            sample->covariance(3 + axis, 3 + axis) =
                rotation_std * rotation_std;
        }
    }
    sample->covariance =
        SymmetricPositiveDefinite<6>(sample->covariance);
    return true;
}

void MapEnuCalibrator::AddNdtObservation(
    const loc::LocalizationResult& ndt) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) return;
    ++ndt_received_;

    NdtSample sample;
    if (!ValidateNdt(ndt, &sample)) return;
    ++ndt_valid_;
    const auto position = std::lower_bound(
        pending_ndt_.begin(), pending_ndt_.end(), sample.stamp,
        [](const NdtSample& existing, double stamp) {
            return existing.stamp < stamp;
        });
    pending_ndt_.insert(position, std::move(sample));
    while (pending_ndt_.size() > options_.max_pending_ndt_samples) {
        pending_ndt_.pop_front();
        ++rejected_time_sync_;
    }
    TryMatchPendingNdt();
}

MapEnuCalibrator::InterpolationState MapEnuCalibrator::InterpolateGnss(
    const std::deque<GnssSample>& buffer, double stamp,
    GnssSample* interpolated) const {
    if (!interpolated || buffer.empty()) {
        return InterpolationState::WAIT_FOR_FUTURE;
    }
    if (stamp < buffer.front().stamp - kTimeEpsilon) {
        return InterpolationState::INVALID;
    }
    if (stamp > buffer.back().stamp + kTimeEpsilon) {
        return InterpolationState::WAIT_FOR_FUTURE;
    }

    const auto right = std::lower_bound(
        buffer.begin(), buffer.end(), stamp,
        [](const GnssSample& existing, double requested_stamp) {
            return existing.stamp < requested_stamp;
        });
    if (right != buffer.end() &&
        std::fabs(right->stamp - stamp) <= kTimeEpsilon) {
        *interpolated = *right;
        interpolated->stamp = stamp;
        return InterpolationState::READY;
    }
    if (right == buffer.begin()) return InterpolationState::INVALID;
    if (right == buffer.end()) return InterpolationState::WAIT_FOR_FUTURE;

    const GnssSample& after = *right;
    const GnssSample& before = *(right - 1);
    const double interval = after.stamp - before.stamp;
    if (!std::isfinite(interval) || interval <= 0.0 ||
        interval > options_.interpolation_max_gap_sec) {
        return InterpolationState::INVALID;
    }
    const double alpha = (stamp - before.stamp) / interval;
    if (alpha < 0.0 || alpha > 1.0) {
        return InterpolationState::INVALID;
    }
    interpolated->stamp = stamp;
    interpolated->position_enu =
        (1.0 - alpha) * before.position_enu + alpha * after.position_enu;
    interpolated->covariance_enu =
        (1.0 - alpha) * (1.0 - alpha) * before.covariance_enu +
        alpha * alpha * after.covariance_enu;
    interpolated->covariance_enu =
        SymmetricPositiveDefinite<3>(interpolated->covariance_enu);
    return interpolated->position_enu.allFinite()
        ? InterpolationState::READY
        : InterpolationState::INVALID;
}

void MapEnuCalibrator::TryMatchPendingNdt() {
    while (!pending_ndt_.empty()) {
        const NdtSample& ndt = pending_ndt_.front();
        GnssSample main;
        GnssSample slave;
        const InterpolationState main_state =
            InterpolateGnss(main_gnss_buffer_, ndt.stamp, &main);
        const InterpolationState slave_state =
            InterpolateGnss(slave_gnss_buffer_, ndt.stamp, &slave);
        if (main_state == InterpolationState::WAIT_FOR_FUTURE ||
            slave_state == InterpolationState::WAIT_FOR_FUTURE) {
            break;
        }
        if (main_state == InterpolationState::INVALID ||
            slave_state == InterpolationState::INVALID) {
            pending_ndt_.pop_front();
            ++rejected_time_sync_;
            continue;
        }
        BuildAndStoreSample(ndt, main, slave);
        pending_ndt_.pop_front();
    }
}

bool MapEnuCalibrator::BuildAndStoreSample(
    const NdtSample& ndt, const GnssSample& main,
    const GnssSample& slave) {
    const Eigen::Vector3d baseline_body =
        options_.slave_antenna_in_body - options_.main_antenna_in_body;
    const Eigen::Vector3d baseline_enu =
        slave.position_enu - main.position_enu;
    const double body_length = baseline_body.norm();
    const double measured_length = baseline_enu.norm();
    if (!std::isfinite(measured_length) || measured_length < 0.10 ||
        std::fabs(measured_length - body_length) >
            options_.baseline_length_tolerance_m) {
        ++rejected_baseline_;
        return false;
    }

    CalibrationSample sample;
    sample.stamp = ndt.stamp;
    sample.rotation_map_body = ndt.rotation_map_body;
    sample.translation_map_body = ndt.translation_map_body;
    sample.main_enu = main.position_enu;
    sample.slave_enu = slave.position_enu;
    sample.main_gnss_covariance = main.covariance_enu;
    sample.slave_gnss_covariance = slave.covariance_enu;
    sample.ndt_pose_covariance = ndt.covariance;
    sample.baseline_map = ndt.rotation_map_body * baseline_body;
    sample.baseline_enu = baseline_enu;
    if (!sample.baseline_map.allFinite()) {
        ++rejected_baseline_;
        return false;
    }

    if (!samples_.empty()) {
        const CalibrationSample& previous = samples_.back();
        const double elapsed = sample.stamp - previous.stamp;
        const double distance =
            (sample.translation_map_body -
             previous.translation_map_body).norm();
        const double cosine = Clamp(
            sample.baseline_map.normalized().dot(
                previous.baseline_map.normalized()),
            -1.0, 1.0);
        const double angle_deg = std::acos(cosine) / kDegToRad;
        if (elapsed < options_.min_sample_interval_sec &&
            distance < options_.min_sample_distance_m &&
            angle_deg < options_.min_baseline_direction_change_deg) {
            ++rejected_sampling_;
            return false;
        }
    }
    if (samples_.size() >= options_.max_samples) {
        ++rejected_sampling_;
        return false;
    }

    const double length_error = measured_length - body_length;
    baseline_squared_error_sum_ += length_error * length_error;
    const Eigen::Vector3d new_direction = sample.baseline_map.normalized();
    for (const CalibrationSample& existing : samples_) {
        const double cosine = Clamp(
            new_direction.dot(existing.baseline_map.normalized()),
            -1.0, 1.0);
        attitude_coverage_deg_ = std::max(
            attitude_coverage_deg_, std::acos(cosine) / kDegToRad);
    }
    samples_.push_back(std::move(sample));
    if (attitude_coverage_deg_ >=
        options_.minimum_attitude_coverage_deg) {
        phase_ = "DYNAMIC_COLLECTION";
    }
    return true;
}

bool MapEnuCalibrator::EstimateRotation(
    const std::vector<CalibrationSample>& samples,
    Eigen::Matrix3d* rotation_enu_map,
    double* condition_number,
    double* singular_ratio) {
    if (!rotation_enu_map || samples.size() < 2) return false;
    Eigen::Matrix3d correlation = Eigen::Matrix3d::Zero();
    double total_weight = 0.0;
    for (const CalibrationSample& sample : samples) {
        const double baseline_length = sample.baseline_enu.norm();
        if (sample.baseline_map.norm() < 1e-9 || baseline_length < 1e-9) {
            continue;
        }
        const Eigen::Matrix3d baseline_covariance =
            sample.main_gnss_covariance + sample.slave_gnss_covariance;
        const double rotation_variance =
            sample.ndt_pose_covariance.block<3, 3>(3, 3).trace() / 3.0;
        const double variance = std::max(
            1e-10,
            baseline_covariance.trace() / 3.0 +
                baseline_length * baseline_length * rotation_variance);
        const double weight = 1.0 / variance;
        const Eigen::Vector3d direction_map =
            sample.baseline_map.normalized();
        const Eigen::Vector3d direction_enu =
            sample.baseline_enu.normalized();
        correlation +=
            weight * direction_enu * direction_map.transpose();
        total_weight += weight;
    }
    if (total_weight <= 0.0 || !correlation.allFinite()) return false;
    correlation /= total_weight;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        correlation, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.info() != Eigen::Success) return false;
    const Eigen::Vector3d singular_values = svd.singularValues();
    if (!singular_values.allFinite() || singular_values(0) <= 1e-12) {
        return false;
    }
    const double second_ratio =
        singular_values(1) / singular_values(0);
    if (singular_ratio) *singular_ratio = second_ratio;
    if (condition_number) {
        *condition_number = singular_values(0) /
            std::max(1e-12, singular_values(1));
    }

    Eigen::Matrix3d correction = Eigen::Matrix3d::Identity();
    correction(2, 2) =
        (svd.matrixU() * svd.matrixV().transpose()).determinant();
    *rotation_enu_map =
        svd.matrixU() * correction * svd.matrixV().transpose();
    return IsFiniteRotation(*rotation_enu_map);
}

Eigen::Matrix3d MapEnuCalibrator::Skew(
    const Eigen::Vector3d& vector) {
    Eigen::Matrix3d matrix;
    matrix << 0.0, -vector.z(), vector.y(),
              vector.z(), 0.0, -vector.x(),
              -vector.y(), vector.x(), 0.0;
    return matrix;
}

Eigen::Matrix3d MapEnuCalibrator::ExpSo3(
    const Eigen::Vector3d& angle) {
    const double norm = angle.norm();
    if (norm < 1e-12) {
        return Eigen::Matrix3d::Identity() + Skew(angle);
    }
    return Eigen::AngleAxisd(norm, angle / norm).toRotationMatrix();
}

Eigen::Matrix3d MapEnuCalibrator::PointCovarianceInMap(
    const CalibrationSample& sample,
    const Eigen::Vector3d& antenna_in_body,
    double antenna_position_std_m) {
    Eigen::Matrix<double, 3, 6> jacobian =
        Eigen::Matrix<double, 3, 6>::Zero();
    jacobian.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    jacobian.block<3, 3>(0, 3) =
        -sample.rotation_map_body * Skew(antenna_in_body);
    Eigen::Matrix3d covariance =
        jacobian * sample.ndt_pose_covariance * jacobian.transpose();
    covariance += Eigen::Matrix3d::Identity() *
        antenna_position_std_m * antenna_position_std_m;
    return SymmetricPositiveDefinite<3>(covariance);
}

Eigen::Matrix3d MapEnuCalibrator::PointInformationInEnu(
    const CalibrationSample& sample,
    const Eigen::Vector3d& antenna_in_body,
    const Eigen::Matrix3d& gnss_covariance,
    const Eigen::Matrix3d& rotation_enu_map,
    double antenna_position_std_m) {
    const Eigen::Matrix3d covariance_map = PointCovarianceInMap(
        sample, antenna_in_body, antenna_position_std_m);
    const Eigen::Matrix3d covariance_enu =
        gnss_covariance + rotation_enu_map * covariance_map *
                              rotation_enu_map.transpose();
    return InverseSpd<3>(covariance_enu);
}

bool MapEnuCalibrator::EstimateTranslation(
    const std::vector<CalibrationSample>& samples,
    const Options& options,
    const Eigen::Matrix3d& rotation_enu_map,
    Eigen::Vector3d* translation_enu_map) {
    if (!translation_enu_map || samples.empty()) return false;
    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d right_hand_side = Eigen::Vector3d::Zero();
    const auto accumulate_antenna = [&](
        const CalibrationSample& sample,
        const Eigen::Vector3d& antenna_body,
        const Eigen::Vector3d& position_enu,
        const Eigen::Matrix3d& gnss_covariance) {
        const Eigen::Vector3d position_map =
            sample.rotation_map_body * antenna_body +
            sample.translation_map_body;
        const Eigen::Vector3d translation_candidate =
            position_enu - rotation_enu_map * position_map;
        const Eigen::Matrix3d information = PointInformationInEnu(
            sample, antenna_body, gnss_covariance, rotation_enu_map,
            options.antenna_position_std_m);
        normal += information;
        right_hand_side += information * translation_candidate;
    };
    for (const CalibrationSample& sample : samples) {
        accumulate_antenna(
            sample, options.main_antenna_in_body, sample.main_enu,
            sample.main_gnss_covariance);
        accumulate_antenna(
            sample, options.slave_antenna_in_body, sample.slave_enu,
            sample.slave_gnss_covariance);
    }
    Eigen::LDLT<Eigen::Matrix3d> decomposition(
        SymmetricPositiveDefinite<3>(normal));
    if (decomposition.info() != Eigen::Success ||
        !decomposition.isPositive()) {
        return false;
    }
    *translation_enu_map = decomposition.solve(right_hand_side);
    return translation_enu_map->allFinite();
}

bool MapEnuCalibrator::RefineTransform(
    const std::vector<CalibrationSample>& samples,
    const Options& options,
    Eigen::Matrix3d* rotation_enu_map,
    Eigen::Vector3d* translation_enu_map,
    MapEnuCovariance* covariance_enu_map) {
    if (!rotation_enu_map || !translation_enu_map ||
        !covariance_enu_map || samples.empty()) {
        return false;
    }

    const auto build_system = [&](const Eigen::Matrix3d& rotation,
                                  const Eigen::Vector3d& translation,
                                  MapEnuCovariance* normal,
                                  Eigen::Matrix<double, 6, 1>* rhs) {
        normal->setZero();
        rhs->setZero();
        const auto accumulate_antenna = [&](
            const CalibrationSample& sample,
            const Eigen::Vector3d& antenna_body,
            const Eigen::Vector3d& observed_enu,
            const Eigen::Matrix3d& gnss_covariance) {
            const Eigen::Vector3d point_map =
                sample.rotation_map_body * antenna_body +
                sample.translation_map_body;
            const Eigen::Vector3d residual =
                observed_enu - (rotation * point_map + translation);
            const Eigen::Matrix3d information = PointInformationInEnu(
                sample, antenna_body, gnss_covariance, rotation,
                options.antenna_position_std_m);
            const double normalized_residual = std::sqrt(std::max(
                0.0, residual.dot(information * residual)));
            const double huber_weight =
                normalized_residual <= options.huber_delta_sigma ||
                    normalized_residual < 1e-12
                ? 1.0
                : options.huber_delta_sigma / normalized_residual;

            Eigen::Matrix<double, 3, 6> jacobian =
                Eigen::Matrix<double, 3, 6>::Zero();
            // Prediction Jacobian for R <- R Exp(dtheta), t <- t + dt.
            jacobian.block<3, 3>(0, 0) = -rotation * Skew(point_map);
            jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
            *normal += huber_weight *
                jacobian.transpose() * information * jacobian;
            *rhs += huber_weight *
                jacobian.transpose() * information * residual;
        };

        for (const CalibrationSample& sample : samples) {
            accumulate_antenna(
                sample, options.main_antenna_in_body, sample.main_enu,
                sample.main_gnss_covariance);
            accumulate_antenna(
                sample, options.slave_antenna_in_body, sample.slave_enu,
                sample.slave_gnss_covariance);
        }
        *normal = 0.5 * (*normal + normal->transpose());
    };

    MapEnuCovariance final_normal = MapEnuCovariance::Zero();
    for (int iteration = 0;
         iteration < options.maximum_refinement_iterations; ++iteration) {
        Eigen::Matrix<double, 6, 1> right_hand_side =
            Eigen::Matrix<double, 6, 1>::Zero();
        build_system(
            *rotation_enu_map, *translation_enu_map, &final_normal,
            &right_hand_side);
        final_normal += MapEnuCovariance::Identity() * 1e-9;
        Eigen::LDLT<MapEnuCovariance> decomposition(final_normal);
        if (decomposition.info() != Eigen::Success ||
            !decomposition.isPositive()) {
            return false;
        }
        Eigen::Matrix<double, 6, 1> increment =
            decomposition.solve(right_hand_side);
        if (!increment.allFinite()) return false;

        const double rotation_norm = increment.head<3>().norm();
        if (rotation_norm > 0.20) {
            increment.head<3>() *= 0.20 / rotation_norm;
        }
        const double translation_norm = increment.tail<3>().norm();
        if (translation_norm > 2.0) {
            increment.tail<3>() *= 2.0 / translation_norm;
        }
        *rotation_enu_map =
            *rotation_enu_map * ExpSo3(increment.head<3>());
        *translation_enu_map += increment.tail<3>();
        Eigen::Quaterniond normalized_rotation(*rotation_enu_map);
        normalized_rotation.normalize();
        *rotation_enu_map = normalized_rotation.toRotationMatrix();

        if (increment.head<3>().norm() < 1e-9 &&
            increment.tail<3>().norm() < 1e-7) {
            break;
        }
    }

    Eigen::Matrix<double, 6, 1> unused_rhs;
    build_system(
        *rotation_enu_map, *translation_enu_map, &final_normal,
        &unused_rhs);
    Eigen::SelfAdjointEigenSolver<MapEnuCovariance> solver(final_normal);
    if (solver.info() != Eigen::Success ||
        !solver.eigenvalues().allFinite() ||
        solver.eigenvalues().minCoeff() <= 1e-12) {
        return false;
    }
    *covariance_enu_map = solver.eigenvectors() *
        solver.eigenvalues().cwiseInverse().asDiagonal() *
        solver.eigenvectors().transpose();
    *covariance_enu_map = 0.5 *
        (*covariance_enu_map + covariance_enu_map->transpose());
    return IsFiniteRotation(*rotation_enu_map) &&
        translation_enu_map->allFinite() &&
        covariance_enu_map->allFinite();
}

void MapEnuCalibrator::ComputeQuality(
    const std::vector<CalibrationSample>& samples,
    const Eigen::Matrix3d& rotation_enu_map,
    MapEnuCalibrationResult* result) {
    if (!result || samples.empty()) return;
    std::vector<double> rotation_errors_deg;
    rotation_errors_deg.reserve(samples.size());
    double rotation_squared_sum = 0.0;
    double rotation_max = 0.0;
    double baseline_squared_sum = 0.0;

    for (const CalibrationSample& sample : samples) {
        const Eigen::Vector3d predicted_direction =
            (rotation_enu_map * sample.baseline_map).normalized();
        const Eigen::Vector3d observed_direction =
            sample.baseline_enu.normalized();
        const double angle_deg = std::acos(Clamp(
            predicted_direction.dot(observed_direction), -1.0, 1.0)) /
            kDegToRad;
        rotation_errors_deg.push_back(angle_deg);
        rotation_squared_sum += angle_deg * angle_deg;
        rotation_max = std::max(rotation_max, angle_deg);

        const Eigen::Vector3d baseline_body =
            sample.rotation_map_body.transpose() * sample.baseline_map;
        const double length_error =
            sample.baseline_enu.norm() - baseline_body.norm();
        baseline_squared_sum += length_error * length_error;

    }
    result->baseline_rms_m =
        std::sqrt(baseline_squared_sum / samples.size());
    result->rotation_rms_deg =
        std::sqrt(rotation_squared_sum / samples.size());
    result->rotation_p95_deg = Percentile95(rotation_errors_deg);
    result->rotation_max_deg = rotation_max;
}

bool MapEnuCalibrator::Finish(
    MapEnuCalibrationResult* output_result,
    std::string* message) {
    Options options;
    std::vector<CalibrationSample> samples;
    double attitude_coverage_deg = 0.0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!configured_) {
            const std::string error = configuration_error_.empty()
                ? "Map-ENU calibration is not configured"
                : configuration_error_;
            if (message) *message = error;
            return false;
        }
        if (!active_) {
            const std::string error = solved_
                ? "Map-ENU calibration has already been solved"
                : "Map-ENU calibration is not collecting";
            if (message) *message = error;
            return false;
        }
        active_ = false;
        phase_ = "SOLVING";
        localization_system_.reset();
        last_localization_stamp_ =
            -std::numeric_limits<double>::infinity();
        rejected_time_sync_ += pending_ndt_.size();
        pending_ndt_.clear();
        options = options_;
        samples = samples_;
        attitude_coverage_deg = attitude_coverage_deg_;
    }

    MapEnuCalibrationResult candidate;
    candidate.sample_count = samples.size();
    candidate.enu_origin_lla = options.enu_origin_lla;

    const auto fail = [&](const std::string& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        phase_ = "FAILED";
        solved_ = false;
        last_message_ = error;
        result_ = candidate;
        if (output_result) *output_result = candidate;
        if (message) *message = error;
        LOG(ERROR) << "[MAP_ENU_CALIBRATION] " << error;
        return false;
    };

    if (samples.size() < options.min_samples) {
        return fail(
            "insufficient calibration samples: " +
            std::to_string(samples.size()) + "/" +
            std::to_string(options.min_samples));
    }
    candidate.attitude_coverage_deg = attitude_coverage_deg;
    if (candidate.attitude_coverage_deg <
        options.minimum_attitude_coverage_deg) {
        return fail(
            "insufficient attitude excitation: coverage=" +
            std::to_string(candidate.attitude_coverage_deg) + " deg");
    }

    double singular_ratio = 0.0;
    if (!EstimateRotation(
            samples, &candidate.enu_from_map_rotation,
            &candidate.rotation_condition_number, &singular_ratio)) {
        return fail("Wahba/SVD rotation estimation failed");
    }
    if (singular_ratio < options.minimum_rotation_singular_ratio ||
        candidate.rotation_condition_number >
            options.maximum_rotation_condition_number) {
        return fail(
            "rotation is poorly observable: condition=" +
            std::to_string(candidate.rotation_condition_number) +
            ", second_singular_ratio=" + std::to_string(singular_ratio));
    }
    if (!EstimateTranslation(
            samples, options, candidate.enu_from_map_rotation,
            &candidate.enu_from_map_translation)) {
        return fail("translation estimation failed");
    }
    if (!RefineTransform(
            samples, options, &candidate.enu_from_map_rotation,
            &candidate.enu_from_map_translation,
            &candidate.enu_from_map_covariance)) {
        return fail("robust SE(3) refinement failed");
    }

    candidate.map_from_enu_rotation =
        candidate.enu_from_map_rotation.transpose();
    candidate.map_from_enu_translation =
        -candidate.map_from_enu_rotation *
        candidate.enu_from_map_translation;
    MapEnuCovariance inverse_jacobian = MapEnuCovariance::Zero();
    inverse_jacobian.block<3, 3>(0, 0) =
        -candidate.enu_from_map_rotation;
    inverse_jacobian.block<3, 3>(3, 0) =
        Skew(candidate.map_from_enu_translation);
    inverse_jacobian.block<3, 3>(3, 3) =
        -candidate.map_from_enu_rotation;
    candidate.map_from_enu_covariance =
        inverse_jacobian * candidate.enu_from_map_covariance *
        inverse_jacobian.transpose();
    candidate.map_from_enu_covariance = 0.5 *
        (candidate.map_from_enu_covariance +
         candidate.map_from_enu_covariance.transpose());

    ComputeQuality(
        samples, candidate.enu_from_map_rotation, &candidate);
    // Compute position residuals with the configured antenna coordinates.
    double squared_residual_sum = 0.0;
    double maximum_residual = 0.0;
    std::size_t residual_count = 0;
    for (const CalibrationSample& sample : samples) {
        const auto accumulate = [&](const Eigen::Vector3d& antenna_body,
                                    const Eigen::Vector3d& observed_enu) {
            const Eigen::Vector3d point_map =
                sample.rotation_map_body * antenna_body +
                sample.translation_map_body;
            const double residual =
                (observed_enu -
                 (candidate.enu_from_map_rotation * point_map +
                  candidate.enu_from_map_translation)).norm();
            squared_residual_sum += residual * residual;
            maximum_residual = std::max(maximum_residual, residual);
            ++residual_count;
        };
        accumulate(options.main_antenna_in_body, sample.main_enu);
        accumulate(options.slave_antenna_in_body, sample.slave_enu);
    }
    candidate.translation_rms_m = residual_count > 0
        ? std::sqrt(squared_residual_sum / residual_count)
        : std::numeric_limits<double>::infinity();
    candidate.max_residual_m = maximum_residual;

    if (candidate.rotation_rms_deg > options.maximum_rotation_rms_deg) {
        return fail(
            "rotation residual is too large: rms=" +
            std::to_string(candidate.rotation_rms_deg) + " deg");
    }
    if (candidate.translation_rms_m > options.maximum_translation_rms_m ||
        candidate.max_residual_m > options.maximum_residual_m) {
        return fail(
            "position residual is too large: rms=" +
            std::to_string(candidate.translation_rms_m) +
            " m, max=" + std::to_string(candidate.max_residual_m) + " m");
    }
    candidate.valid = true;
    const std::string success_message =
        "Map-ENU calibration solved; result returned by finish service";

    {
        std::lock_guard<std::mutex> lock(mutex_);
        result_ = candidate;
        solved_ = true;
        phase_ = "SUCCESS";
        last_message_ = success_message;
    }
    if (output_result) *output_result = candidate;
    if (message) *message = success_message;
    LOG(INFO) << "[MAP_ENU_CALIBRATION] success"
              << ", samples=" << candidate.sample_count
              << ", coverage_deg=" << candidate.attitude_coverage_deg
              << ", rotation_rms_deg=" << candidate.rotation_rms_deg
              << ", translation_rms_m=" << candidate.translation_rms_m
              << ", max_residual_m=" << candidate.max_residual_m
              << ", condition=" << candidate.rotation_condition_number;
    return true;
}

MapEnuCalibrationStatus MapEnuCalibrator::GetStatus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    MapEnuCalibrationStatus status;
    status.configured = configured_;
    status.active = active_;
    status.solved = solved_;
    status.phase = phase_;
    status.gps1_received = main_gnss_received_;
    status.gps1_valid = main_gnss_valid_;
    status.gps2_received = slave_gnss_received_;
    status.gps2_valid = slave_gnss_valid_;
    status.ndt_received = ndt_received_;
    status.ndt_valid = ndt_valid_;
    status.accepted_samples = samples_.size();
    status.pending_ndt_samples = pending_ndt_.size();
    status.rejected_time_sync = rejected_time_sync_;
    status.rejected_baseline = rejected_baseline_;
    status.rejected_sampling = rejected_sampling_;
    status.attitude_coverage_deg = attitude_coverage_deg_;
    status.baseline_rms_m = samples_.empty()
        ? 0.0
        : std::sqrt(baseline_squared_error_sum_ / samples_.size());
    Eigen::Matrix3d unused_rotation;
    double singular_ratio = 0.0;
    if (!EstimateRotation(
            samples_, &unused_rotation,
            &status.rotation_condition_number, &singular_ratio)) {
        status.rotation_condition_number = 0.0;
    }
    if (result_.sample_count > 0) {
        status.rotation_rms_deg = result_.rotation_rms_deg;
        status.translation_rms_m = result_.translation_rms_m;
        status.max_residual_m = result_.max_residual_m;
        if (result_.rotation_condition_number > 0.0) {
            status.rotation_condition_number =
                result_.rotation_condition_number;
        }
    }
    status.message = last_message_.empty()
        ? configuration_error_
        : last_message_;
    return status;
}

}  // namespace lightning::modules
