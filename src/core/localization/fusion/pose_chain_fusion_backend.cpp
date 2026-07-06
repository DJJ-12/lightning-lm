#include "core/localization/fusion/pose_chain_fusion_backend.h"

#include <algorithm>
#include <Eigen/Geometry>
#include <cmath>
#include <iomanip>

#include <glog/logging.h>

namespace lightning::loc {
namespace {

Eigen::Matrix3d LoadRotationMatrix(const YAML::Node& node, const Eigen::Matrix3d& fallback) {
    if (!node || !node.IsSequence() || node.size() != 9) {
        return fallback;
    }
    Eigen::Matrix3d R;
    R << node[0].as<double>(), node[1].as<double>(), node[2].as<double>(),
         node[3].as<double>(), node[4].as<double>(), node[5].as<double>(),
         node[6].as<double>(), node[7].as<double>(), node[8].as<double>();
    return R;
}

}  // namespace

Vec3d PoseChainFusionBackend::SecondOrderVectorLowPass::Update(const Vec3d& x, double dt, double tau) {
    if (!initialized_) {
        y1_ = x;
        y2_ = x;
        initialized_ = true;
        return y2_;
    }
    const double a = tau <= 1e-6 ? 1.0 : std::clamp(dt / (tau + dt), 0.0, 1.0);
    y1_ = y1_ + a * (x - y1_);
    y2_ = y2_ + a * (y1_ - y2_);
    return y2_;
}

bool PoseChainFusionBackend::Init(const std::string& yaml_path) {
    if (!ParseOptions(yaml_path)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    initialized_ = false;
    map_odom_filtered_ = SE3();
    latest_map_base_ = SE3();
    output_seq_ = 0;
    latest_output_time_ = -1.0;
    imu_buffer_.clear();
    lio_buffer_.clear();
    ndt_queue_.clear();
    prediction_history_.clear();
    delta_queue_.clear();
    has_latest_lio_ = false;
    has_current_ = false;
    current_map_base_ = SE3();
    current_time_ = -1.0;
    pending_anchor_publish_ = false;
    imu_segment_initialized_ = false;
    next_segment_start_time_ = -1.0;
    next_segment_end_time_ = -1.0;
    imu_filter_last_time_ = -1.0;
    imu_velocity_body_.setZero();
    imu_acc_filter_.Reset();
    imu_gyro_filter_.Reset();
    stat_timer_ticks_ = 0;
    stat_ndt_updates_ = 0;
    stat_ndt_skips_ = 0;
    stat_delta_segments_ = 0;
    LOG(INFO) << "[POSE_CHAIN_INIT] backend=scheduled_imu_delta, publish_period=" << options_.publish_period
              << ", imu_use_accel_xy=" << options_.imu_use_accel_xy
              << ", imu_use_gyro_z=" << options_.imu_use_gyro_z
              << ", reset_velocity_on_lio_correction=" << options_.reset_velocity_on_lio_correction;
    return true;
}

bool PoseChainFusionBackend::ParseOptions(const std::string& yaml_path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    } catch (const std::exception& e) {
        LOG(ERROR) << "[POSE_CHAIN_CONFIG] failed to load yaml: " << yaml_path << ", err=" << e.what();
        return false;
    }
    const YAML::Node fusion = root["fusion"];
    if (!fusion) {
        return true;
    }

    if (fusion["map_odom_low_pass_alpha"]) options_.map_odom_low_pass_alpha = fusion["map_odom_low_pass_alpha"].as<double>();
    if (fusion["pose_chain_map_odom_low_pass_alpha"]) options_.map_odom_low_pass_alpha = fusion["pose_chain_map_odom_low_pass_alpha"].as<double>();
    if (fusion["map_odom_ignore_trans_threshold"]) options_.map_odom_ignore_trans_threshold = fusion["map_odom_ignore_trans_threshold"].as<double>();
    if (fusion["pose_chain_map_odom_ignore_trans_threshold"]) options_.map_odom_ignore_trans_threshold = fusion["pose_chain_map_odom_ignore_trans_threshold"].as<double>();
    if (fusion["map_odom_ignore_yaw_threshold_deg"]) options_.map_odom_ignore_yaw_threshold_deg = fusion["map_odom_ignore_yaw_threshold_deg"].as<double>();
    if (fusion["pose_chain_map_odom_ignore_yaw_threshold_deg"]) options_.map_odom_ignore_yaw_threshold_deg = fusion["pose_chain_map_odom_ignore_yaw_threshold_deg"].as<double>();
    if (fusion["ndt_first_fix_force_init"]) options_.ndt_first_fix_force_init = fusion["ndt_first_fix_force_init"].as<bool>();
    if (fusion["pose_chain_ndt_first_fix_force_init"]) options_.ndt_first_fix_force_init = fusion["pose_chain_ndt_first_fix_force_init"].as<bool>();
    if (fusion["ndt_lio_interp_max_gap"]) options_.ndt_lio_interp_max_gap = fusion["ndt_lio_interp_max_gap"].as<double>();
    if (fusion["pose_chain_ndt_lio_interp_max_gap"]) options_.ndt_lio_interp_max_gap = fusion["pose_chain_ndt_lio_interp_max_gap"].as<double>();

    if (fusion["pose_chain_publish_period"]) options_.publish_period = fusion["pose_chain_publish_period"].as<double>();
    if (fusion["ekf_timer_period_ms"] && !fusion["pose_chain_publish_period"]) {
        options_.publish_period = fusion["ekf_timer_period_ms"].as<double>() * 1e-3;
    }
    if (fusion["imu_prediction_enable"]) options_.imu_prediction_enable = fusion["imu_prediction_enable"].as<bool>();
    if (fusion["pose_chain_imu_prediction_enable"]) options_.imu_prediction_enable = fusion["pose_chain_imu_prediction_enable"].as<bool>();
    if (fusion["imu_acc_low_pass_tau"]) options_.imu_acc_low_pass_tau = fusion["imu_acc_low_pass_tau"].as<double>();
    if (fusion["imu_gyro_low_pass_tau"]) options_.imu_gyro_low_pass_tau = fusion["imu_gyro_low_pass_tau"].as<double>();
    if (fusion["imu_max_velocity"]) options_.imu_max_velocity = fusion["imu_max_velocity"].as<double>();
    if (fusion["imu_max_yaw_rate_deg"]) options_.imu_max_yaw_rate_deg = fusion["imu_max_yaw_rate_deg"].as<double>();
    if (fusion["imu_max_acc"]) options_.imu_max_acc = fusion["imu_max_acc"].as<double>();
    if (fusion["imu_use_accel_xy"]) options_.imu_use_accel_xy = fusion["imu_use_accel_xy"].as<bool>();
    if (fusion["imu_use_gyro_z"]) options_.imu_use_gyro_z = fusion["imu_use_gyro_z"].as<bool>();
    if (fusion["R_base_imu"]) options_.R_base_imu = LoadRotationMatrix(fusion["R_base_imu"], options_.R_base_imu);

    if (fusion["pose_chain_lio_delay_correction_enable"]) options_.lio_delay_correction_enable = fusion["pose_chain_lio_delay_correction_enable"].as<bool>();
    if (fusion["pose_chain_reset_velocity_on_lio_correction"]) {
        options_.reset_velocity_on_lio_correction = fusion["pose_chain_reset_velocity_on_lio_correction"].as<bool>();
    }
    if (fusion["pose_chain_prediction_history_size"]) options_.prediction_history_size = fusion["pose_chain_prediction_history_size"].as<int>();
    if (fusion["pose_chain_prediction_history_age"]) options_.prediction_history_age = fusion["pose_chain_prediction_history_age"].as<double>();

    if (fusion["max_imu_buffer_size"]) options_.max_imu_buffer_size = fusion["max_imu_buffer_size"].as<int>();
    if (fusion["max_lio_buffer_size"]) options_.max_lio_buffer_size = fusion["max_lio_buffer_size"].as<int>();
    if (fusion["max_ndt_queue_size"]) options_.max_ndt_queue_size = fusion["max_ndt_queue_size"].as<int>();
    if (fusion["max_imu_buffer_age"]) options_.max_imu_buffer_age = fusion["max_imu_buffer_age"].as<double>();
    if (fusion["max_lio_buffer_age"]) options_.max_lio_buffer_age = fusion["max_lio_buffer_age"].as<double>();

    options_.map_odom_low_pass_alpha = std::clamp(options_.map_odom_low_pass_alpha, 0.0, 1.0);
    options_.publish_period = std::clamp(options_.publish_period, 0.02, 1.0);
    options_.imu_acc_low_pass_tau = std::max(0.0, options_.imu_acc_low_pass_tau);
    options_.imu_gyro_low_pass_tau = std::max(0.0, options_.imu_gyro_low_pass_tau);
    options_.imu_max_velocity = std::max(0.0, options_.imu_max_velocity);
    options_.imu_max_yaw_rate_deg = std::max(0.0, options_.imu_max_yaw_rate_deg);
    options_.imu_max_acc = std::max(0.0, options_.imu_max_acc);
    options_.prediction_history_size = std::max(20, options_.prediction_history_size);
    options_.prediction_history_age = std::max(2.0, options_.prediction_history_age);
    options_.max_imu_buffer_size = std::max(500, options_.max_imu_buffer_size);
    options_.max_lio_buffer_size = std::max(10, options_.max_lio_buffer_size);
    options_.max_ndt_queue_size = std::max(1, options_.max_ndt_queue_size);
    return true;
}

void PoseChainFusionBackend::SetInitialMapOdom(const SE3& map_odom) {
    std::lock_guard<std::mutex> lock(mutex_);
    map_odom_filtered_ = map_odom;
    map_odom_lp_stage1_.Reset(map_odom);
    map_odom_lp_stage2_.Reset(map_odom);
    initialized_ = true;
    ++output_seq_;
    LOG(INFO) << "[POSE_CHAIN_SET_INITIAL_MAP_ODOM] trans=" << map_odom.translation().transpose()
              << ", yaw_deg=" << RadToDeg(YawOf(map_odom));
}

void PoseChainFusionBackend::FeedImu(const FusionImuMeasurement& imu) {
    if (!std::isfinite(imu.timestamp)) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (imu_buffer_.empty() || imu.timestamp >= imu_buffer_.back().timestamp) {
        imu_buffer_.push_back(imu);
    } else {
        auto it = std::upper_bound(imu_buffer_.begin(), imu_buffer_.end(), imu.timestamp,
            [](double t, const FusionImuMeasurement& m) { return t < m.timestamp; });
        imu_buffer_.insert(it, imu);
    }
    while (static_cast<int>(imu_buffer_.size()) > options_.max_imu_buffer_size) {
        imu_buffer_.pop_front();
    }
    BuildPendingImuDeltasUnlocked();
}

void PoseChainFusionBackend::FeedLio(const FusionLioMeasurement& lio) {
    if (!lio.pose_valid || !std::isfinite(lio.timestamp)) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (lio_buffer_.empty() || lio.timestamp >= lio_buffer_.back().timestamp) {
        lio_buffer_.push_back(lio);
    } else {
        auto it = std::upper_bound(lio_buffer_.begin(), lio_buffer_.end(), lio.timestamp,
            [](double t, const FusionLioMeasurement& m) { return t < m.timestamp; });
        lio_buffer_.insert(it, lio);
    }
    while (static_cast<int>(lio_buffer_.size()) > options_.max_lio_buffer_size) {
        lio_buffer_.pop_front();
    }

    latest_lio_ = lio;
    has_latest_lio_ = true;
    if (!initialized_) {
        map_odom_filtered_ = SE3();
        map_odom_lp_stage1_.Reset(map_odom_filtered_);
        map_odom_lp_stage2_.Reset(map_odom_filtered_);
        initialized_ = true;
    }
    ApplyDelayedLioCorrectionUnlocked(lio);
    BuildPendingImuDeltasUnlocked();
    TrimBuffersUnlocked(current_time_ > 0.0 ? current_time_ : lio.timestamp);
    LOG(INFO) << "[POSE_CHAIN_LIO_FEED] t=" << std::setprecision(14) << lio.timestamp
              << ", odom_base_trans=" << lio.odom_base.translation().transpose()
              << ", odom_base_yaw_deg=" << RadToDeg(YawOf(lio.odom_base));
}

void PoseChainFusionBackend::FeedNdt(const FusionNdtMeasurement& ndt) {
    if (!ndt.converged || !std::isfinite(ndt.timestamp)) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    ndt_queue_.push_back(ndt);
    while (static_cast<int>(ndt_queue_.size()) > options_.max_ndt_queue_size) {
        ndt_queue_.pop_front();
    }
}

void PoseChainFusionBackend::FeedWheel(const FusionWheelMeasurement& wheel) {
    (void)wheel;
}

bool PoseChainFusionBackend::RunTimerTick(double timestamp, SE3* map_base, SE3* map_odom,
                                          std::uint64_t* seq, double* output_timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    ProcessNdtQueueUnlocked();
    BuildPendingImuDeltasUnlocked();

    if (!has_current_) {
        LOG(WARNING) << "[POSE_CHAIN_TIMER_SKIP] reason=no_lio_anchor, timer_t=" << std::setprecision(14)
                     << timestamp << ", imu_buffer_size=" << imu_buffer_.size();
        return false;
    }

    if (pending_anchor_publish_) {
        PublishOutputLocked(current_time_, current_map_base_, map_odom_filtered_);
        pending_anchor_publish_ = false;
    } else {
        ImuDeltaSegment last_seg;
        int applied = 0;
        if (!PopAndApplyPendingDeltasUnlocked(&last_seg, &applied)) {
            LOG(WARNING) << "[POSE_CHAIN_TIMER_SKIP] reason=no_delta_segment, timer_t=" << std::setprecision(14)
                         << timestamp << ", current_t=" << current_time_
                         << ", next_seg_start=" << next_segment_start_time_
                         << ", next_seg_end=" << next_segment_end_time_
                         << ", newest_imu_t=" << (imu_buffer_.empty() ? -1.0 : imu_buffer_.back().timestamp)
                         << ", delta_queue_size=" << delta_queue_.size();
            return false;
        }
        PublishOutputLocked(current_time_, current_map_base_, map_odom_filtered_);
        LOG(INFO) << "[POSE_CHAIN_PUBLISH_FROM_DELTA] applied_segments=" << applied
                  << ", last_seg=[" << std::setprecision(14) << last_seg.start_time << ","
                  << last_seg.end_time << "]"
                  << ", used_last=" << last_seg.used_samples
                  << ", current_t=" << current_time_;
    }

    if (map_base) *map_base = latest_map_base_;
    if (map_odom) *map_odom = map_odom_filtered_;
    if (seq) *seq = output_seq_;
    if (output_timestamp) *output_timestamp = latest_output_time_;
    ++stat_timer_ticks_;
    return true;
}

bool PoseChainFusionBackend::GetLatestMapOdom(SE3* map_odom, std::uint64_t* seq) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) return false;
    if (map_odom) *map_odom = map_odom_filtered_;
    if (seq) *seq = output_seq_;
    return true;
}

bool PoseChainFusionBackend::GetLatestOptimizedMapBase(SE3* map_base) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_current_ && !initialized_) return false;
    if (map_base) *map_base = latest_map_base_;
    return true;
}

bool PoseChainFusionBackend::IsInitialized() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initialized_;
}

void PoseChainFusionBackend::InitializeFromLioUnlocked(const FusionLioMeasurement& lio,
                                                       const SE3& map_base_at_lio) {
    has_current_ = true;
    current_map_base_ = map_base_at_lio;
    current_time_ = lio.timestamp;
    latest_map_base_ = current_map_base_;
    latest_output_time_ = -1.0;
    prediction_history_.clear();
    PushPredictionHistoryUnlocked(current_time_, current_map_base_, map_odom_filtered_);
    pending_anchor_publish_ = true;
    delta_queue_.clear();
    ResetImuDeltaStateUnlocked(lio.timestamp, true);
    TrimBuffersUnlocked(lio.timestamp);
    LOG(INFO) << "[POSE_CHAIN_FIRST_LIO_ANCHOR] t0=" << std::setprecision(14) << lio.timestamp
              << ", map_base_trans=" << map_base_at_lio.translation().transpose()
              << ", map_base_yaw_deg=" << RadToDeg(YawOf(map_base_at_lio))
              << ", next_delta=[" << next_segment_start_time_ << "," << next_segment_end_time_ << "]";
}

void PoseChainFusionBackend::ApplyDelayedLioCorrectionUnlocked(const FusionLioMeasurement& lio) {
    SE3 map_odom_at_lio = map_odom_filtered_;
    SE3 predicted_at_lio;
    double history_gap = 0.0;
    const bool has_pred_t1 = InterpolatePredictedStateUnlocked(lio.timestamp, &predicted_at_lio,
                                                               &map_odom_at_lio, &history_gap);
    const SE3 lio_map_base_t1 = map_odom_at_lio * lio.odom_base;

    if (!has_current_) {
        InitializeFromLioUnlocked(lio, lio_map_base_t1);
        return;
    }

    if (!options_.lio_delay_correction_enable) {
        if (lio.timestamp >= current_time_ - 1e-4) {
            current_map_base_ = lio_map_base_t1;
            current_time_ = lio.timestamp;
            latest_map_base_ = current_map_base_;
            PushPredictionHistoryUnlocked(current_time_, current_map_base_, map_odom_filtered_);
            ResetImuDeltaStateUnlocked(current_time_, true);
            pending_anchor_publish_ = true;
        }
        return;
    }

    const bool delayed = current_time_ > lio.timestamp + 1e-4;
    if (delayed && has_pred_t1) {
        // Strict delayed alignment requested by the user:
        //   delta_pred(t1->t2) = inv(X_predict(t1)) * X_current(t2)
        //   X_corrected(t2)   = X_lio(t1) * delta_pred(t1->t2)
        // The same correction is also applied to the saved history after t1.
        const SE3 delta_pred_t1_current = predicted_at_lio.inverse() * current_map_base_;
        const SE3 corrected_current = lio_map_base_t1 * delta_pred_t1_current;
        const SE3 history_correction = lio_map_base_t1 * predicted_at_lio.inverse();
        int corrected_history_count = 0;
        for (auto& p : prediction_history_) {
            if (p.timestamp + 1e-6 >= lio.timestamp) {
                p.map_base = history_correction * p.map_base;
                ++corrected_history_count;
            }
        }
        current_map_base_ = corrected_current;
        latest_map_base_ = corrected_current;
        if (options_.reset_velocity_on_lio_correction) {
            imu_velocity_body_.setZero();
        }
        LOG(INFO) << "[POSE_CHAIN_LIO_DELAY_CORRECT] lio_t=" << std::setprecision(14) << lio.timestamp
                  << ", current_t=" << current_time_
                  << ", history_gap=" << history_gap
                  << ", pred_t1_trans=" << predicted_at_lio.translation().transpose()
                  << ", lio_t1_trans=" << lio_map_base_t1.translation().transpose()
                  << ", delta_t1_current_trans=" << delta_pred_t1_current.translation().transpose()
                  << ", corrected_current_trans=" << corrected_current.translation().transpose()
                  << ", history_corrected=" << corrected_history_count
                  << ", vel_body=" << imu_velocity_body_.transpose();
        return;
    }

    // If the LIO stamp is not older than the current predicted state, it is a new anchor.
    // Restart the 0.1s IMU segment scheduler from the LIO timestamp.
    if (!delayed) {
        current_map_base_ = lio_map_base_t1;
        current_time_ = lio.timestamp;
        latest_map_base_ = current_map_base_;
        PushPredictionHistoryUnlocked(current_time_, current_map_base_, map_odom_filtered_);
        pending_anchor_publish_ = true;
        ResetImuDeltaStateUnlocked(current_time_, true);
        LOG(INFO) << "[POSE_CHAIN_LIO_ANCHOR_RESET] reason=not_delayed, lio_t=" << std::setprecision(14)
                  << lio.timestamp << ", current_trans=" << current_map_base_.translation().transpose()
                  << ", current_yaw_deg=" << RadToDeg(YawOf(current_map_base_));
    } else {
        LOG(WARNING) << "[POSE_CHAIN_LIO_DELAY_SKIP] reason=no_prediction_history, lio_t=" << std::setprecision(14)
                     << lio.timestamp << ", current_t=" << current_time_
                     << ", history_size=" << prediction_history_.size();
    }
}

void PoseChainFusionBackend::ResetImuDeltaStateUnlocked(double start_time, bool reset_velocity) {
    imu_segment_initialized_ = true;
    next_segment_start_time_ = start_time;
    next_segment_end_time_ = start_time + options_.publish_period;
    imu_filter_last_time_ = -1.0;
    imu_acc_filter_.Reset();
    imu_gyro_filter_.Reset();
    if (reset_velocity) {
        imu_velocity_body_.setZero();
    }
    delta_queue_.clear();
    while (!imu_buffer_.empty() && imu_buffer_.front().timestamp < start_time - 1.0) {
        imu_buffer_.pop_front();
    }
    LOG(INFO) << "[POSE_CHAIN_IMU_DELTA_SCHED_RESET] start_t=" << std::setprecision(14) << start_time
              << ", next_end=" << next_segment_end_time_
              << ", reset_velocity=" << reset_velocity
              << ", vel_body=" << imu_velocity_body_.transpose();
}

void PoseChainFusionBackend::BuildPendingImuDeltasUnlocked() {
    if (!options_.imu_prediction_enable || !has_current_ || !imu_segment_initialized_) {
        return;
    }
    while (!imu_buffer_.empty() && imu_buffer_.back().timestamp + 1e-6 >= next_segment_end_time_) {
        ImuDeltaSegment seg;
        if (!BuildOneImuDeltaSegmentUnlocked(next_segment_start_time_, next_segment_end_time_, &seg)) {
            break;
        }
        delta_queue_.push_back(seg);
        ++stat_delta_segments_;
        LOG(INFO) << "[POSE_CHAIN_IMU_DELTA_READY] start_t=" << std::setprecision(14) << seg.start_time
                  << ", end_t=" << seg.end_time
                  << ", dt=" << (seg.end_time - seg.start_time)
                  << ", used=" << seg.used_samples
                  << ", delta_trans=" << seg.delta.translation().transpose()
                  << ", delta_yaw_deg=" << RadToDeg(YawOf(seg.delta))
                  << ", avg_acc=" << seg.avg_acc_base.transpose()
                  << ", avg_gyro=" << seg.avg_gyro_base.transpose()
                  << ", vel_after=" << seg.vel_body_after.transpose()
                  << ", queue_size=" << delta_queue_.size();
        next_segment_start_time_ = next_segment_end_time_;
        next_segment_end_time_ += options_.publish_period;

        // Keep a little overlap for low-pass continuity and late interpolation, but do not let
        // the IMU buffer grow without bound.
        const double keep_after = next_segment_start_time_ - 1.0;
        while (!imu_buffer_.empty() && imu_buffer_.front().timestamp < keep_after) {
            imu_buffer_.pop_front();
        }
    }
}

bool PoseChainFusionBackend::BuildOneImuDeltaSegmentUnlocked(double start_time, double end_time,
                                                             ImuDeltaSegment* seg) {
    if (!seg || end_time <= start_time + 1e-6) {
        return false;
    }
    Vec3d acc_sum = Vec3d::Zero();
    Vec3d gyro_sum = Vec3d::Zero();
    int count = 0;

    double last_filter_time = imu_filter_last_time_;
    for (const auto& imu : imu_buffer_) {
        if (imu.timestamp <= start_time + 1e-9) {
            continue;
        }
        if (imu.timestamp > end_time + 1e-9) {
            break;
        }
        double dt = options_.publish_period / 10.0;
        if (last_filter_time > 0.0 && imu.timestamp > last_filter_time) {
            dt = std::min(0.05, imu.timestamp - last_filter_time);
        }
        last_filter_time = imu.timestamp;

        Vec3d acc_base = options_.R_base_imu * imu.acc;
        Vec3d gyro_base = options_.R_base_imu * imu.gyro;
        acc_base = imu_acc_filter_.Update(acc_base, dt, options_.imu_acc_low_pass_tau);
        gyro_base = imu_gyro_filter_.Update(gyro_base, dt, options_.imu_gyro_low_pass_tau);

        if (options_.imu_max_acc > 0.0) {
            acc_base.x() = Clamp(acc_base.x(), -options_.imu_max_acc, options_.imu_max_acc);
            acc_base.y() = Clamp(acc_base.y(), -options_.imu_max_acc, options_.imu_max_acc);
        }
        const double max_wz = DegToRad(options_.imu_max_yaw_rate_deg);
        if (max_wz > 0.0) {
            gyro_base.z() = Clamp(gyro_base.z(), -max_wz, max_wz);
        }
        acc_sum += acc_base;
        gyro_sum += gyro_base;
        ++count;
    }

    if (count <= 0) {
        return false;
    }
    imu_filter_last_time_ = last_filter_time;

    const double dt = end_time - start_time;
    const Vec3d avg_acc = acc_sum / static_cast<double>(count);
    const Vec3d avg_gyro = gyro_sum / static_cast<double>(count);

    Eigen::Vector2d acc_xy = Eigen::Vector2d::Zero();
    if (options_.imu_use_accel_xy) {
        acc_xy = Eigen::Vector2d(avg_acc.x(), avg_acc.y());
    }
    double wz = options_.imu_use_gyro_z ? avg_gyro.z() : 0.0;

    const Eigen::Vector2d dp_body = imu_velocity_body_ * dt + 0.5 * acc_xy * dt * dt;
    imu_velocity_body_ += acc_xy * dt;
    const double speed = imu_velocity_body_.norm();
    if (options_.imu_max_velocity > 0.0 && speed > options_.imu_max_velocity) {
        imu_velocity_body_ *= options_.imu_max_velocity / speed;
    }
    const double dyaw = NormalizeAngle(wz * dt);

    seg->start_time = start_time;
    seg->end_time = end_time;
    seg->delta = SE3(Quatd(Eigen::AngleAxisd(dyaw, Vec3d::UnitZ())),
                     Vec3d(dp_body.x(), dp_body.y(), 0.0));
    seg->vel_body_after = imu_velocity_body_;
    seg->avg_acc_base = avg_acc;
    seg->avg_gyro_base = avg_gyro;
    seg->used_samples = count;
    return true;
}

bool PoseChainFusionBackend::PopAndApplyPendingDeltasUnlocked(ImuDeltaSegment* last_seg, int* applied_count) {
    if (applied_count) *applied_count = 0;
    if (delta_queue_.empty() || !has_current_) {
        return false;
    }
    int applied = 0;
    ImuDeltaSegment last;
    while (!delta_queue_.empty()) {
        const ImuDeltaSegment seg = delta_queue_.front();
        delta_queue_.pop_front();
        current_map_base_ = current_map_base_ * seg.delta;
        current_time_ = seg.end_time;
        latest_map_base_ = current_map_base_;
        PushPredictionHistoryUnlocked(current_time_, current_map_base_, map_odom_filtered_);
        last = seg;
        ++applied;
    }
    if (last_seg) *last_seg = last;
    if (applied_count) *applied_count = applied;
    return applied > 0;
}

bool PoseChainFusionBackend::InterpolatePredictedStateUnlocked(double timestamp, SE3* map_base, SE3* map_odom,
                                                               double* nearest_gap) const {
    if (prediction_history_.empty()) {
        return false;
    }
    if (prediction_history_.size() == 1) {
        const double gap = std::abs(prediction_history_.front().timestamp - timestamp);
        if (map_base) *map_base = prediction_history_.front().map_base;
        if (map_odom) *map_odom = prediction_history_.front().map_odom;
        if (nearest_gap) *nearest_gap = gap;
        return true;
    }
    auto upper = std::lower_bound(prediction_history_.begin(), prediction_history_.end(), timestamp,
        [](const PredictedPose& p, double t) { return p.timestamp < t; });
    if (upper == prediction_history_.begin()) {
        const double gap = std::abs(upper->timestamp - timestamp);
        if (map_base) *map_base = upper->map_base;
        if (map_odom) *map_odom = upper->map_odom;
        if (nearest_gap) *nearest_gap = gap;
        return true;
    }
    if (upper == prediction_history_.end()) {
        const auto& last = prediction_history_.back();
        const double gap = std::abs(last.timestamp - timestamp);
        if (map_base) *map_base = last.map_base;
        if (map_odom) *map_odom = last.map_odom;
        if (nearest_gap) *nearest_gap = gap;
        return true;
    }
    const auto& b = *upper;
    const auto& a = *(upper - 1);
    const double dt = b.timestamp - a.timestamp;
    if (dt <= 1e-6) {
        if (map_base) *map_base = a.map_base;
        if (map_odom) *map_odom = a.map_odom;
        if (nearest_gap) *nearest_gap = std::abs(a.timestamp - timestamp);
        return true;
    }
    const double s = std::clamp((timestamp - a.timestamp) / dt, 0.0, 1.0);
    if (map_base) {
        const SE3 delta = a.map_base.inverse() * b.map_base;
        *map_base = a.map_base * SE3::exp(s * delta.log());
    }
    if (map_odom) {
        const SE3 delta = a.map_odom.inverse() * b.map_odom;
        *map_odom = a.map_odom * SE3::exp(s * delta.log());
    }
    if (nearest_gap) *nearest_gap = std::min(std::abs(timestamp - a.timestamp), std::abs(b.timestamp - timestamp));
    return true;
}

bool PoseChainFusionBackend::InterpolateLioPoseUnlocked(double timestamp, SE3* odom_base, double* nearest_gap) const {
    if (lio_buffer_.empty()) return false;
    if (lio_buffer_.size() == 1) {
        const double gap = std::abs(lio_buffer_.front().timestamp - timestamp);
        if (gap > options_.ndt_lio_interp_max_gap) return false;
        if (odom_base) *odom_base = lio_buffer_.front().odom_base;
        if (nearest_gap) *nearest_gap = gap;
        return true;
    }
    auto upper = std::lower_bound(lio_buffer_.begin(), lio_buffer_.end(), timestamp,
        [](const FusionLioMeasurement& m, double t) { return m.timestamp < t; });
    if (upper == lio_buffer_.begin()) {
        const double gap = std::abs(upper->timestamp - timestamp);
        if (gap > options_.ndt_lio_interp_max_gap) return false;
        if (odom_base) *odom_base = upper->odom_base;
        if (nearest_gap) *nearest_gap = gap;
        return true;
    }
    if (upper == lio_buffer_.end()) {
        const auto& last = lio_buffer_.back();
        const double gap = std::abs(last.timestamp - timestamp);
        if (gap > options_.ndt_lio_interp_max_gap) return false;
        if (odom_base) *odom_base = last.odom_base;
        if (nearest_gap) *nearest_gap = gap;
        return true;
    }
    const auto& b = *upper;
    const auto& a = *(upper - 1);
    const double dt = b.timestamp - a.timestamp;
    if (dt <= 1e-6) return false;
    const double s = std::clamp((timestamp - a.timestamp) / dt, 0.0, 1.0);
    const SE3 delta = a.odom_base.inverse() * b.odom_base;
    if (odom_base) *odom_base = a.odom_base * SE3::exp(s * delta.log());
    if (nearest_gap) *nearest_gap = std::min(std::abs(timestamp - a.timestamp), std::abs(b.timestamp - timestamp));
    return true;
}

void PoseChainFusionBackend::ProcessNdtQueueUnlocked() {
    while (!ndt_queue_.empty()) {
        const FusionNdtMeasurement ndt = ndt_queue_.front();
        ndt_queue_.pop_front();
        if (!ApplyNdtMapOdomObservationUnlocked(ndt)) {
            ++stat_ndt_skips_;
        }
    }
}

bool PoseChainFusionBackend::ApplyNdtMapOdomObservationUnlocked(const FusionNdtMeasurement& ndt) {
    SE3 odom_base_at_ndt;
    double lio_gap = 0.0;
    if (!InterpolateLioPoseUnlocked(ndt.timestamp, &odom_base_at_ndt, &lio_gap)) {
        LOG(WARNING) << "[POSE_CHAIN_NDT_SKIP] reason=no_lio_for_ndt, ndt_t=" << std::setprecision(14)
                     << ndt.timestamp << ", lio_buffer_size=" << lio_buffer_.size();
        return false;
    }

    const SE3 old_map_odom = map_odom_filtered_;
    const SE3 raw_map_odom = ndt.map_base * odom_base_at_ndt.inverse();
    bool accepted1 = false;
    bool accepted2 = false;
    std::string reason1;
    std::string reason2;

    if (options_.ndt_first_fix_force_init && stat_ndt_updates_ == 0) {
        map_odom_lp_stage1_.Reset(raw_map_odom);
        map_odom_lp_stage2_.Reset(raw_map_odom);
        map_odom_filtered_ = raw_map_odom;
        accepted1 = accepted2 = true;
        reason1 = reason2 = "first_ndt_force_init";
    } else {
        const double yaw_threshold = DegToRad(options_.map_odom_ignore_yaw_threshold_deg);
        const SE3 stage1 = map_odom_lp_stage1_.Update(raw_map_odom,
                                                       options_.map_odom_low_pass_alpha,
                                                       options_.map_odom_ignore_trans_threshold,
                                                       yaw_threshold,
                                                       &accepted1,
                                                       &reason1);
        map_odom_filtered_ = map_odom_lp_stage2_.Update(stage1,
                                                         options_.map_odom_low_pass_alpha,
                                                         options_.map_odom_ignore_trans_threshold,
                                                         yaw_threshold,
                                                         &accepted2,
                                                         &reason2);
    }

    // NDT changes the map->odom transform.  Apply the corresponding map-frame correction to
    // already predicted states, instead of resetting the IMU delta schedule.
    const SE3 map_correction = map_odom_filtered_ * old_map_odom.inverse();
    if (has_current_) {
        current_map_base_ = map_correction * current_map_base_;
        latest_map_base_ = current_map_base_;
        for (auto& p : prediction_history_) {
            p.map_base = map_correction * p.map_base;
            p.map_odom = map_odom_filtered_;
        }
    }

    ++output_seq_;
    ++stat_ndt_updates_;
    LOG(INFO) << "[POSE_CHAIN_NDT_UPDATE] t=" << std::setprecision(14) << ndt.timestamp
              << ", lio_gap=" << lio_gap
              << ", raw_map_odom_trans=" << raw_map_odom.translation().transpose()
              << ", raw_map_odom_yaw_deg=" << RadToDeg(YawOf(raw_map_odom))
              << ", filtered_trans=" << map_odom_filtered_.translation().transpose()
              << ", filtered_yaw_deg=" << RadToDeg(YawOf(map_odom_filtered_))
              << ", accepted1=" << accepted1 << ", reason1=" << reason1
              << ", accepted2=" << accepted2 << ", reason2=" << reason2
              << ", current_corrected=" << has_current_
              << ", seq=" << output_seq_;
    return true;
}

void PoseChainFusionBackend::PushPredictionHistoryUnlocked(double timestamp, const SE3& map_base, const SE3& map_odom) {
    if (prediction_history_.empty() || timestamp >= prediction_history_.back().timestamp) {
        prediction_history_.push_back(PredictedPose{timestamp, map_base, map_odom});
    } else {
        auto it = std::upper_bound(prediction_history_.begin(), prediction_history_.end(), timestamp,
            [](double t, const PredictedPose& p) { return t < p.timestamp; });
        prediction_history_.insert(it, PredictedPose{timestamp, map_base, map_odom});
    }
    TrimPredictionHistoryUnlocked(timestamp);
}

void PoseChainFusionBackend::PublishOutputLocked(double timestamp, const SE3& map_base, const SE3& map_odom) {
    latest_output_time_ = timestamp;
    latest_map_base_ = map_base;
    ++output_seq_;
    LOG(INFO) << "[POSE_CHAIN_OUTPUT] t=" << std::setprecision(14) << timestamp
              << ", map_base_trans=" << map_base.translation().transpose()
              << ", map_base_yaw_deg=" << RadToDeg(YawOf(map_base))
              << ", map_odom_trans=" << map_odom.translation().transpose()
              << ", map_odom_yaw_deg=" << RadToDeg(YawOf(map_odom))
              << ", current_t=" << current_time_
              << ", history_size=" << prediction_history_.size()
              << ", delta_queue_size=" << delta_queue_.size();
}

void PoseChainFusionBackend::TrimBuffersUnlocked(double now) {
    while (static_cast<int>(lio_buffer_.size()) > options_.max_lio_buffer_size) lio_buffer_.pop_front();
    while (!lio_buffer_.empty() && now - lio_buffer_.front().timestamp > options_.max_lio_buffer_age) lio_buffer_.pop_front();
    while (static_cast<int>(imu_buffer_.size()) > options_.max_imu_buffer_size) imu_buffer_.pop_front();
    while (!imu_buffer_.empty() && now - imu_buffer_.front().timestamp > options_.max_imu_buffer_age) imu_buffer_.pop_front();
}

void PoseChainFusionBackend::TrimPredictionHistoryUnlocked(double now) {
    while (static_cast<int>(prediction_history_.size()) > options_.prediction_history_size) prediction_history_.pop_front();
    while (!prediction_history_.empty() && now - prediction_history_.front().timestamp > options_.prediction_history_age) {
        prediction_history_.pop_front();
    }
}

// static helpers
double PoseChainFusionBackend::NormalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

double PoseChainFusionBackend::DegToRad(double deg) { return deg * M_PI / 180.0; }
double PoseChainFusionBackend::RadToDeg(double rad) { return rad * 180.0 / M_PI; }

double PoseChainFusionBackend::YawOf(const SE3& pose) {
    const Mat3d R = pose.so3().matrix();
    return std::atan2(R(1, 0), R(0, 0));
}

double PoseChainFusionBackend::Clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

}  // namespace lightning::loc
