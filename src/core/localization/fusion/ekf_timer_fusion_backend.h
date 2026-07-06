#pragma once

#include <atomic>
#include <cstddef>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include "common/eigen_types.h"
#include "core/localization/fusion/fusion_measurements.h"

namespace lightning::loc {


// Autoware-style fixed-length extended delay-state Kalman filter.
// State layout is [x(k), x(k-1), x(k-2), ...], each block has dim_x variables.
// predictWithDelay() shifts the latest state into the first delay slot and propagates the
// current state. updateWithDelay() applies a measurement to one delayed block while updating the
// whole extended state through the cross-covariance, matching Autoware time_delay_kalman_filter's
// core mechanism.
class TimeDelayEkf {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TimeDelayEkf() = default;

    void Init(const Eigen::VectorXd& x, const Eigen::MatrixXd& p, int extend_state_step);
    bool IsInitialized() const { return initialized_; }
    int DimX() const { return dim_x_; }
    int ExtendStateStep() const { return extend_state_step_; }

    // Match Autoware TimeDelayKalmanFilter semantics: latest means the current 6D block.
    Eigen::VectorXd GetLatestX() const;
    Eigen::MatrixXd GetLatestP() const;
    Eigen::VectorXd GetCurrentState() const;
    Eigen::MatrixXd GetCurrentCovariance() const;
    // Full extended state/covariance accessors for delayed-measurement gating/logging.
    const Eigen::VectorXd& GetExtendedX() const { return x_ext_; }
    const Eigen::MatrixXd& GetExtendedP() const { return p_ext_; }
    double GetXElement(int index) const;

    void PredictWithDelay(const Eigen::VectorXd& x_next,
                          const Eigen::MatrixXd& a,
                          const Eigen::MatrixXd& q);
    bool UpdateWithDelay(const Eigen::VectorXd& y,
                         const Eigen::MatrixXd& c,
                         const Eigen::MatrixXd& r,
                         int delay_step,
                         Eigen::VectorXd* innovation = nullptr,
                         double* mahalanobis = nullptr);
    void NormalizeYawStates(int yaw_index);

   private:
    bool initialized_ = false;
    int dim_x_ = 0;
    int extend_state_step_ = 0;
    Eigen::VectorXd x_ext_;
    Eigen::MatrixXd p_ext_;
};

class EkfTimerFusionBackend {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EkfTimerFusionBackend();
    ~EkfTimerFusionBackend();

    bool Init(const std::string& yaml_path);

    void SetInitialMapOdom(const SE3& map_odom);
    void Start();
    void Stop();

    void FeedImu(const FusionImuMeasurement& imu);
    void FeedLio(const FusionLioMeasurement& lio);
    void FeedNdt(const FusionNdtMeasurement& ndt);
    void FeedWheel(const FusionWheelMeasurement& wheel);
    bool RunTimerTick(double timestamp, SE3* map_base, SE3* map_odom, std::uint64_t* seq = nullptr);

    bool GetLatestMapOdom(SE3* map_odom, std::uint64_t* seq = nullptr) const;
    bool GetLatestOptimizedMapBase(SE3* map_base) const;
    bool IsInitialized() const;

   private:
    enum class FusionEventType { IMU, LIO, NDT, WHEEL };

    struct FusionEvent {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        double timestamp = 0.0;
        FusionEventType type = FusionEventType::IMU;
        FusionImuMeasurement imu;
        FusionLioMeasurement lio;
        FusionNdtMeasurement ndt;
        FusionWheelMeasurement wheel;
    };

    struct Options {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        bool enable = false;

        // EKF state: [x, y, yaw, yaw_bias, vx, wz]
        bool ekf_enable = true;
        double initial_sigma_xy = 1.0;
        double initial_sigma_yaw_deg = 10.0;
        double initial_sigma_yaw_bias_deg = 5.0;
        double initial_sigma_vx = 1.0;
        double initial_sigma_wz_deg = 20.0;

        double process_sigma_xy = 0.05;
        double process_sigma_yaw_deg = 1.0;
        double process_sigma_yaw_bias_deg = 0.01;
        double process_sigma_vx = 0.50;
        double process_sigma_wz_deg = 5.0;

        // NDT covariance first version: fixed covariance + quality scaling.
        double ndt_sigma_x = 0.30;
        double ndt_sigma_y = 0.30;
        double ndt_sigma_yaw_deg = 3.0;
        double ndt_bad_confidence_threshold = 0.20;
        double ndt_bad_cov_scale = 10.0;
        double ndt_mahalanobis_gate = 9.0;
        bool ndt_force_accept = true;
        // Autoware-style extended delay state. 10Hz timer with 50 steps keeps ~5 seconds.
        int extend_state_step = 50;
        double delay_time_tolerance = 0.06;
        // Kept for backward-compatible YAML keys/logs; not used as variable-size pose clone count.
        int max_pose_clones = 60;
        double clone_time_tolerance = 0.06;
        double clone_remove_before_sec = 0.0;


        // LIO-SAM Hessian covariance is used to derive twist covariance.
        double lio_twist_sigma_vx_fallback = 0.10;
        double lio_twist_sigma_wz_deg_fallback = 2.0;
        double lio_twist_max_dt = 5.0;
        double lio_twist_min_dt = 1e-3;
        double lio_twist_max_vx = 5.0;
        double lio_twist_max_wz_deg = 90.0;
        // LIO-SAM odom difference is algorithm-derived twist, not a hardware velocity sensor.
        // Low-pass it before EKF update to prevent small pose jitter becoming vx/wz jitter.
        bool lio_twist_enable_low_pass = true;
        double lio_twist_low_pass_tau = 0.80;      // seconds; larger = smoother
        double lio_twist_filter_reset_gap = 2.0;   // reset after long LIO gap
        double lio_twist_max_vx_step = 0.50;       // m/s per LIO update, <=0 disables step limit
        double lio_twist_max_wz_step_deg = 10.0;   // deg/s per LIO update, <=0 disables step limit
        double lio_cov_min_variance = 1e-6;
        double lio_cov_max_variance = 1e4;

        double wheel_sigma_vx = 0.20;
        double wheel_sigma_wz_deg = 5.0;
        double wheel_max_vx = 10.0;
        double wheel_max_wz_deg = 180.0;

        // IMU gyro covariance / preintegration covariance source.
        double imu_gravity = 9.80511;
        double imu_acc_noise = 9.0e-4;
        double imu_gyr_noise = 1.5636343949698187e-03;
        double imu_acc_bias_noise = 5.0e-4;
        double imu_gyr_bias_noise = 5.0e-5;
        double imu_integration_sigma = 1.0e-4;
        // First debugging/tuning version: do not let IMU gyro.z drive yaw-rate unless explicitly enabled.
        // LIO-SAM relative pose is the primary wz source; wheel can weakly supplement it.
        bool imu_use_wz_update = false;
        double imu_gyro_z_sigma_deg = 10.0;
        bool imu_use_preintegration_covariance = false;
        double imu_preintegration_min_update_dt = 0.02;
        double imu_preintegration_max_update_dt = 0.20;
        double imu_preintegration_max_dt = 0.10;
        Mat3d R_base_imu = Mat3d::Identity();

        // map->odom output smoothing. Set alpha=1.0 to directly use EKF output.
        bool enable_map_odom_filter = true;
        double map_odom_alpha = 0.20;

        int max_imu_queue_size = 4000;
        int max_lio_queue_size = 200;
        int max_ndt_queue_size = 100;
        int max_wheel_queue_size = 200;
        int max_lio_buffer_size = 2000;
        int max_imu_buffer_size = 8000;
    };

    void BackendLoop();
    void ProcessEvent(const FusionEvent& event);
    void ProcessEventUnlocked(const FusionEvent& event);

    void BufferImuMeasurement(const FusionImuMeasurement& imu);
    void BufferLioMeasurement(const FusionLioMeasurement& lio);
    void BufferNdtMeasurement(const FusionNdtMeasurement& ndt);
    void BufferWheelMeasurement(const FusionWheelMeasurement& wheel);

    void ProcessImuMeasurement(const FusionImuMeasurement& imu);
    void ProcessLioMeasurement(const FusionLioMeasurement& lio);
    void ProcessNdtMeasurement(const FusionNdtMeasurement& ndt);
    void ProcessWheelMeasurement(const FusionWheelMeasurement& wheel);

    void PredictTo(double timestamp);
    void InitializeEkfFromPose(double timestamp, const SE3& map_base);
    bool UpdatePoseClone2D(double timestamp, const SE3& map_base, const Eigen::Matrix3d& covariance_xy_yaw,
                           const std::string& source);
    void UpdateTwist(double timestamp, double vx, double wz, const Eigen::Matrix2d& covariance,
                     const std::string& source);
    void AddPoseClone(double timestamp);
    int FindPoseCloneIndex(double timestamp) const;
    int FindDelayStep(double measurement_timestamp) const;
    void AccumulateDelayTime(double dt);
    void RemovePoseClone(std::size_t clone_index);
    void RemovePoseClonesUpTo(double timestamp);
    void ApplyAugmentedDelta(const Eigen::VectorXd& dx);
    void SyncStateFromFilter();

    Eigen::Matrix3d BuildNdtCovarianceXYYaw(const FusionNdtMeasurement& ndt) const;
    bool ComputeLioTwist(const FusionLioMeasurement& prev,
                         const FusionLioMeasurement& curr,
                         double* vx,
                         double* wz,
                         Eigen::Matrix2d* covariance) const;
    bool FilterLioTwist(double timestamp, double raw_vx, double raw_wz,
                        const Eigen::Matrix2d& raw_covariance,
                        double* filtered_vx, double* filtered_wz, Eigen::Matrix2d* filtered_covariance);
    bool ComputeImuYawRate(const FusionImuMeasurement& imu, double* wz, double* variance) const;
    bool PropagateImuPreintegration(const FusionImuMeasurement& imu, double gyro_z_base, double gyro_z_variance,
                                    double* wz, double* variance);
    void ResetImuPreintegration(double timestamp);

    void UpdateMapOdomFromEkf(const FusionLioMeasurement* lio_for_output, const std::string& reason);
    SE3 StateToMapBase() const;
    void PublishOutputLocked(const SE3& map_base, const SE3& map_odom, const std::string& reason);
    void TrimBuffers();
    void LogStatsIfNeeded();

    static double NormalizeAngle(double angle);
    static double DegToRad(double deg);
    static double RadToDeg(double rad);
    static double YawOf(const SE3& pose);

    Options options_;

    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<FusionImuMeasurement> imu_queue_;
    std::deque<FusionLioMeasurement> lio_queue_;
    std::deque<FusionNdtMeasurement> ndt_queue_;
    std::deque<FusionWheelMeasurement> wheel_queue_;
    bool stop_requested_ = false;
    std::atomic<bool> running_{false};
    std::thread backend_thread_;

    std::deque<FusionImuMeasurement> imu_buffer_;
    std::deque<FusionLioMeasurement> lio_buffer_;
    std::deque<FusionWheelMeasurement> wheel_buffer_;
    bool has_prev_lio_for_twist_ = false;
    FusionLioMeasurement prev_lio_for_twist_;

    bool lio_twist_filter_initialized_ = false;
    double lio_twist_filter_timestamp_ = -1.0;
    double lio_twist_filtered_vx_ = 0.0;
    double lio_twist_filtered_wz_ = 0.0;

    bool imu_preint_initialized_ = false;
    double imu_preint_last_time_ = -1.0;
    double imu_preint_delta_t_ = 0.0;
    double imu_preint_delta_yaw_ = 0.0;
    double imu_preint_yaw_variance_ = 0.0;

    mutable std::mutex output_mutex_;
    mutable std::mutex state_mutex_;
    SE3 initial_map_odom_;
    SE3 latest_map_odom_smooth_;
    SE3 latest_map_base_opt_;
    bool has_map_odom_ = false;
    std::atomic<bool> ekf_initialized_{false};
    std::uint64_t map_odom_update_seq_ = 0;

    struct PoseClone {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        double timestamp = 0.0;
        Eigen::Vector3d xyyaw = Eigen::Vector3d::Zero();
    };

    Eigen::Matrix<double, 6, 1> x_ = Eigen::Matrix<double, 6, 1>::Zero();
    TimeDelayEkf delay_filter_;
    std::vector<double> accumulated_delay_times_;
    // Deprecated compatibility/logging container; the active delay state is delay_filter_.
    std::vector<PoseClone, Eigen::aligned_allocator<PoseClone>> pose_clones_;
    double last_predict_time_ = -1.0;

    std::chrono::steady_clock::time_point stat_last_log_time_;
    std::size_t stat_imu_buffered_ = 0;
    std::size_t stat_lio_buffered_ = 0;
    std::size_t stat_ndt_buffered_ = 0;
    std::size_t stat_wheel_buffered_ = 0;
    std::size_t stat_pose_updates_ = 0;
    std::size_t stat_clone_updates_ = 0;
    std::size_t stat_twist_updates_ = 0;
    std::size_t stat_imu_updates_ = 0;
    std::size_t stat_output_updates_ = 0;

    std::size_t stat_window_imu_ = 0;
    std::size_t stat_window_lio_ = 0;
    std::size_t stat_window_ndt_ = 0;
    std::size_t stat_window_wheel_ = 0;
    std::size_t stat_window_pose_updates_ = 0;
    std::size_t stat_window_clone_updates_ = 0;
    std::size_t stat_window_twist_updates_ = 0;
    std::size_t stat_window_imu_updates_ = 0;
    std::size_t stat_window_output_updates_ = 0;
};

}  // namespace lightning::loc
