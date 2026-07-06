#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>

#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

#include "common/eigen_types.h"
#include "core/localization/fusion/fusion_measurements.h"
#include "core/localization/fusion/se3_low_pass_filter.h"

namespace lightning::loc {

class PoseChainFusionBackend {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PoseChainFusionBackend() = default;
    ~PoseChainFusionBackend() = default;

    bool Init(const std::string& yaml_path);
    void SetInitialMapOdom(const SE3& map_odom);

    void Start() {}
    void Stop() {}

    void FeedImu(const FusionImuMeasurement& imu);
    void FeedLio(const FusionLioMeasurement& lio);
    void FeedNdt(const FusionNdtMeasurement& ndt);
    void FeedWheel(const FusionWheelMeasurement& wheel);

    // The ROS wall timer is only a trigger.  This function publishes only when a 0.1s IMU
    // delta segment has been prepared.  The output timestamp is always sensor time.
    bool RunTimerTick(double timestamp, SE3* map_base, SE3* map_odom, std::uint64_t* seq,
                      double* output_timestamp = nullptr);

    bool GetLatestMapOdom(SE3* map_odom, std::uint64_t* seq) const;
    bool GetLatestOptimizedMapBase(SE3* map_base) const;
    bool IsInitialized() const;

   private:
    struct Options {
        // NDT -> map_odom second-order low-pass.
        double map_odom_low_pass_alpha = 0.35;
        double map_odom_ignore_trans_threshold = 5.0;
        double map_odom_ignore_yaw_threshold_deg = 45.0;
        bool ndt_first_fix_force_init = true;
        double ndt_lio_interp_max_gap = 1.5;

        // IMU delta segment generation.  IMU samples are grouped into fixed sensor-time
        // windows [t0+n*T, t0+(n+1)*T], averaged once, and converted to one SE2 delta.
        bool imu_prediction_enable = true;
        double publish_period = 0.10;
        double imu_acc_low_pass_tau = 0.20;
        double imu_gyro_low_pass_tau = 0.20;
        double imu_max_velocity = 1.0;
        double imu_max_yaw_rate_deg = 60.0;
        double imu_max_acc = 3.0;
        bool imu_use_accel_xy = true;
        bool imu_use_gyro_z = true;
        Eigen::Matrix3d R_base_imu = Eigen::Matrix3d::Identity();

        // Delayed LIO correction.  The current state is corrected by replacing the predicted
        // pose at t1 with the strict LIO pose at t1 while preserving the already-integrated
        // delta from t1 to current_time.
        bool lio_delay_correction_enable = true;
        bool reset_velocity_on_lio_correction = false;
        int prediction_history_size = 500;
        double prediction_history_age = 30.0;

        int max_imu_buffer_size = 10000;
        int max_lio_buffer_size = 200;
        int max_ndt_queue_size = 50;
        double max_imu_buffer_age = 30.0;
        double max_lio_buffer_age = 30.0;
    };

    struct PredictedPose {
        double timestamp = 0.0;
        SE3 map_base = SE3();
        SE3 map_odom = SE3();
    };

    struct ImuDeltaSegment {
        double start_time = 0.0;
        double end_time = 0.0;
        SE3 delta = SE3();
        Eigen::Vector2d vel_body_after = Eigen::Vector2d::Zero();
        Vec3d avg_acc_base = Vec3d::Zero();
        Vec3d avg_gyro_base = Vec3d::Zero();
        int used_samples = 0;
    };

    class SecondOrderVectorLowPass {
       public:
        void Reset() { initialized_ = false; }
        Vec3d Update(const Vec3d& x, double dt, double tau);

       private:
        bool initialized_ = false;
        Vec3d y1_ = Vec3d::Zero();
        Vec3d y2_ = Vec3d::Zero();
    };

    bool ParseOptions(const std::string& yaml_path);
    void TrimBuffersUnlocked(double now);
    void TrimPredictionHistoryUnlocked(double now);
    bool InterpolatePredictedStateUnlocked(double timestamp, SE3* map_base, SE3* map_odom,
                                           double* nearest_gap) const;
    bool InterpolateLioPoseUnlocked(double timestamp, SE3* odom_base, double* nearest_gap) const;

    void InitializeFromLioUnlocked(const FusionLioMeasurement& lio, const SE3& map_base_at_lio);
    void ApplyDelayedLioCorrectionUnlocked(const FusionLioMeasurement& lio);
    void ProcessNdtQueueUnlocked();
    bool ApplyNdtMapOdomObservationUnlocked(const FusionNdtMeasurement& ndt);

    void BuildPendingImuDeltasUnlocked();
    bool BuildOneImuDeltaSegmentUnlocked(double start_time, double end_time, ImuDeltaSegment* seg);
    bool PopAndApplyPendingDeltasUnlocked(ImuDeltaSegment* last_seg, int* applied_count);
    void PushPredictionHistoryUnlocked(double timestamp, const SE3& map_base, const SE3& map_odom);
    void PublishOutputLocked(double timestamp, const SE3& map_base, const SE3& map_odom);
    void ResetImuDeltaStateUnlocked(double start_time, bool reset_velocity);

    static double NormalizeAngle(double angle);
    static double DegToRad(double deg);
    static double RadToDeg(double rad);
    static double YawOf(const SE3& pose);
    static double Clamp(double v, double lo, double hi);

    mutable std::mutex mutex_;
    Options options_;

    bool initialized_ = false;
    SE3 map_odom_filtered_ = SE3();
    SE3 latest_map_base_ = SE3();
    std::uint64_t output_seq_ = 0;
    double latest_output_time_ = -1.0;

    SE3LowPassFilter map_odom_lp_stage1_;
    SE3LowPassFilter map_odom_lp_stage2_;

    std::deque<FusionImuMeasurement> imu_buffer_;
    std::deque<FusionLioMeasurement> lio_buffer_;
    std::deque<FusionNdtMeasurement> ndt_queue_;
    std::deque<PredictedPose> prediction_history_;
    std::deque<ImuDeltaSegment> delta_queue_;

    bool has_latest_lio_ = false;
    FusionLioMeasurement latest_lio_;

    bool has_current_ = false;
    SE3 current_map_base_ = SE3();
    double current_time_ = -1.0;
    bool pending_anchor_publish_ = false;

    // Fixed sensor-time IMU segment scheduler.
    bool imu_segment_initialized_ = false;
    double next_segment_start_time_ = -1.0;
    double next_segment_end_time_ = -1.0;
    double imu_filter_last_time_ = -1.0;
    Eigen::Vector2d imu_velocity_body_ = Eigen::Vector2d::Zero();
    SecondOrderVectorLowPass imu_acc_filter_;
    SecondOrderVectorLowPass imu_gyro_filter_;

    mutable std::uint64_t stat_timer_ticks_ = 0;
    mutable std::uint64_t stat_ndt_updates_ = 0;
    mutable std::uint64_t stat_ndt_skips_ = 0;
    mutable std::uint64_t stat_delta_segments_ = 0;
};

}  // namespace lightning::loc
