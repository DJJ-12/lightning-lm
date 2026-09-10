#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "core/lightning_math.hpp"
#include "core/localization/localization_result.h"
#include "modules/localizationSystem/localization_system.h"

namespace lightning::modules {

// The covariance order used by both the solver and the finish service is:
// [rotation_x, rotation_y, rotation_z, translation_x, translation_y,
//  translation_z]. Rotations are small right perturbations of the stored
// transform.
using MapEnuCovariance = Eigen::Matrix<double, 6, 6>;

// Raw NDT pose covariance keeps Lightning's existing pose order:
// [translation_x, translation_y, translation_z, rotation_x, rotation_y,
//  rotation_z]. It is deliberately named separately from MapEnuCovariance,
// whose parameter order is rotation first.
using NdtPoseCovariance = Eigen::Matrix<double, 6, 6>;

struct MapEnuCalibrationResult {
    bool valid = false;
    std::size_t sample_count = 0;

    Eigen::Vector3d enu_origin_lla = Eigen::Vector3d::Zero();
    Eigen::Matrix3d enu_from_map_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d enu_from_map_translation = Eigen::Vector3d::Zero();
    MapEnuCovariance enu_from_map_covariance =
        MapEnuCovariance::Zero();

    Eigen::Matrix3d map_from_enu_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d map_from_enu_translation = Eigen::Vector3d::Zero();
    MapEnuCovariance map_from_enu_covariance =
        MapEnuCovariance::Zero();

    double attitude_coverage_deg = 0.0;
    double baseline_rms_m = 0.0;
    double rotation_condition_number = 0.0;
    double rotation_rms_deg = 0.0;
    double rotation_p95_deg = 0.0;
    double rotation_max_deg = 0.0;
    double translation_rms_m = 0.0;
    double max_residual_m = 0.0;
};

struct MapEnuCalibrationStatus {
    bool configured = false;
    bool active = false;
    bool solved = false;
    std::string phase = "DISABLED";

    std::uint64_t gps1_received = 0;
    std::uint64_t gps1_valid = 0;
    std::uint64_t gps2_received = 0;
    std::uint64_t gps2_valid = 0;
    std::uint64_t ndt_received = 0;
    std::uint64_t ndt_valid = 0;
    std::uint64_t accepted_samples = 0;
    std::uint64_t pending_ndt_samples = 0;
    std::uint64_t rejected_time_sync = 0;
    std::uint64_t rejected_baseline = 0;
    std::uint64_t rejected_sampling = 0;

    double attitude_coverage_deg = 0.0;
    double baseline_rms_m = 0.0;
    double rotation_condition_number = 0.0;
    double rotation_rms_deg = 0.0;
    double translation_rms_m = 0.0;
    double max_residual_m = 0.0;
    std::string message;
};

// Complete Map-ENU calibration module. It owns a LocalizationSystem and uses
// that object only through its public interface. Lightning supplies LiDAR and
// dual-GNSS messages; this class owns all localization/calibration details.
class MapEnuCalibrator {
   public:
    bool Configure(const std::string& yaml_path, std::string* message = nullptr);
    bool Start(const std::string& map_path,
               rclcpp::Node::SharedPtr node,
               std::string* message = nullptr);
    bool Finish(MapEnuCalibrationResult* result,
                std::string* message = nullptr);
    void ResetSession();

    loc::LocalizationFrameOutcome ProcessCloud(
        const sensor_msgs::msg::PointCloud2::SharedPtr& cloud,
        const loc::LocalizationInputDiagnostic& diagnostic = {});
    loc::LocalizationFrameOutcome ProcessCloud(
        const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud,
        const loc::LocalizationInputDiagnostic& diagnostic = {});
    void AddMainGnss(const sensor_msgs::msg::NavSatFix& fix);
    void AddSlaveGnss(const sensor_msgs::msg::NavSatFix& fix);

    MapEnuCalibrationStatus GetStatus() const;
    bool Configured() const;
    bool Active() const;

   private:
    struct Options {
        Eigen::Vector3d enu_origin_lla = Eigen::Vector3d::Zero();
        Eigen::Vector3d main_antenna_in_body = Eigen::Vector3d::Zero();
        Eigen::Vector3d slave_antenna_in_body = Eigen::Vector3d::Zero();

        double gnss_time_offset_sec = 0.0;
        double gnss_buffer_duration_sec = 30.0;
        double interpolation_max_gap_sec = 0.25;
        double baseline_length_tolerance_m = 0.15;
        double min_sample_interval_sec = 0.20;
        double min_sample_distance_m = 0.30;
        double min_baseline_direction_change_deg = 2.0;
        std::size_t min_samples = 30;
        std::size_t max_samples = 5000;
        std::size_t max_gnss_samples = 20000;
        std::size_t max_pending_ndt_samples = 1000;

        bool require_gnss_covariance = true;
        bool require_ndt_covariance = false;
        Eigen::Vector3d gnss_std_floor_m =
            Eigen::Vector3d(0.01, 0.01, 0.02);
        Eigen::Vector3d fallback_ndt_position_std_m =
            Eigen::Vector3d(0.10, 0.10, 0.20);
        Eigen::Vector3d fallback_ndt_rotation_std_deg =
            Eigen::Vector3d(1.0, 1.0, 1.0);
        double antenna_position_std_m = 0.01;

        double minimum_ndt_tp = 0.0;
        double minimum_ndt_nvtl = 0.0;
        int maximum_ndt_iterations = 1000;
        double minimum_attitude_coverage_deg = 15.0;
        double minimum_rotation_singular_ratio = 0.03;
        double maximum_rotation_condition_number = 100.0;
        double maximum_rotation_rms_deg = 5.0;
        double maximum_translation_rms_m = 1.0;
        double maximum_residual_m = 3.0;
        double huber_delta_sigma = 2.5;
        int maximum_refinement_iterations = 20;
    };

    struct GnssSample {
        double stamp = 0.0;
        Eigen::Vector3d position_enu = Eigen::Vector3d::Zero();
        Eigen::Matrix3d covariance_enu = Eigen::Matrix3d::Identity();
    };

    struct NdtSample {
        double stamp = 0.0;
        Eigen::Matrix3d rotation_map_body = Eigen::Matrix3d::Identity();
        Eigen::Vector3d translation_map_body = Eigen::Vector3d::Zero();
        NdtPoseCovariance covariance = NdtPoseCovariance::Identity();
    };

    struct CalibrationSample {
        double stamp = 0.0;
        Eigen::Matrix3d rotation_map_body = Eigen::Matrix3d::Identity();
        Eigen::Vector3d translation_map_body = Eigen::Vector3d::Zero();
        Eigen::Vector3d main_enu = Eigen::Vector3d::Zero();
        Eigen::Vector3d slave_enu = Eigen::Vector3d::Zero();
        Eigen::Matrix3d main_gnss_covariance = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d slave_gnss_covariance = Eigen::Matrix3d::Identity();
        NdtPoseCovariance ndt_pose_covariance =
            NdtPoseCovariance::Identity();
        Eigen::Vector3d baseline_map = Eigen::Vector3d::Zero();
        Eigen::Vector3d baseline_enu = Eigen::Vector3d::Zero();
    };

    enum class InterpolationState { READY, WAIT_FOR_FUTURE, INVALID };

    void ClearSessionDataLocked();
    void AddNdtObservation(const loc::LocalizationResult& ndt);
    void AddGnss(const sensor_msgs::msg::NavSatFix& fix, bool main);
    bool ValidateNdt(const loc::LocalizationResult& ndt,
                     NdtSample* sample) const;
    bool ConvertGnss(const sensor_msgs::msg::NavSatFix& fix,
                     GnssSample* sample) const;
    void InsertGnss(std::deque<GnssSample>* buffer, GnssSample sample);
    void TryMatchPendingNdt();
    InterpolationState InterpolateGnss(
        const std::deque<GnssSample>& buffer, double stamp,
        GnssSample* interpolated) const;
    bool BuildAndStoreSample(const NdtSample& ndt,
                             const GnssSample& main,
                             const GnssSample& slave);

    static bool EstimateRotation(
        const std::vector<CalibrationSample>& samples,
        Eigen::Matrix3d* rotation_enu_map,
        double* condition_number,
        double* singular_ratio);
    static bool EstimateTranslation(
        const std::vector<CalibrationSample>& samples,
        const Options& options,
        const Eigen::Matrix3d& rotation_enu_map,
        Eigen::Vector3d* translation_enu_map);
    static bool RefineTransform(
        const std::vector<CalibrationSample>& samples,
        const Options& options,
        Eigen::Matrix3d* rotation_enu_map,
        Eigen::Vector3d* translation_enu_map,
        MapEnuCovariance* covariance_enu_map);
    static void ComputeQuality(
        const std::vector<CalibrationSample>& samples,
        const Eigen::Matrix3d& rotation_enu_map,
        MapEnuCalibrationResult* result);
    static Eigen::Matrix3d PointCovarianceInMap(
        const CalibrationSample& sample,
        const Eigen::Vector3d& antenna_in_body,
        double antenna_position_std_m);
    static Eigen::Matrix3d PointInformationInEnu(
        const CalibrationSample& sample,
        const Eigen::Vector3d& antenna_in_body,
        const Eigen::Matrix3d& gnss_covariance,
        const Eigen::Matrix3d& rotation_enu_map,
        double antenna_position_std_m);
    static Eigen::Matrix3d ExpSo3(const Eigen::Vector3d& angle);
    static Eigen::Matrix3d Skew(const Eigen::Vector3d& vector);

    template <typename CloudMessage>
    loc::LocalizationFrameOutcome ProcessCloudImpl(
        const std::shared_ptr<CloudMessage>& cloud,
        const loc::LocalizationInputDiagnostic& diagnostic);

    mutable std::mutex mutex_;
    std::string yaml_path_;
    Options options_;
    math::JsbsimWgs84Enu enu_projector_;
    std::shared_ptr<LocalizationSystem> localization_system_;
    double last_localization_stamp_ =
        -std::numeric_limits<double>::infinity();
    bool configured_ = false;
    bool active_ = false;
    bool solved_ = false;
    std::string phase_ = "DISABLED";
    std::string configuration_error_;
    std::string last_message_;

    std::deque<GnssSample> main_gnss_buffer_;
    std::deque<GnssSample> slave_gnss_buffer_;
    std::deque<NdtSample> pending_ndt_;
    std::vector<CalibrationSample> samples_;
    MapEnuCalibrationResult result_;

    std::uint64_t main_gnss_received_ = 0;
    std::uint64_t main_gnss_valid_ = 0;
    std::uint64_t slave_gnss_received_ = 0;
    std::uint64_t slave_gnss_valid_ = 0;
    std::uint64_t ndt_received_ = 0;
    std::uint64_t ndt_valid_ = 0;
    std::uint64_t rejected_time_sync_ = 0;
    std::uint64_t rejected_baseline_ = 0;
    std::uint64_t rejected_sampling_ = 0;
    double baseline_squared_error_sum_ = 0.0;
    double attitude_coverage_deg_ = 0.0;
};

}  // namespace lightning::modules
