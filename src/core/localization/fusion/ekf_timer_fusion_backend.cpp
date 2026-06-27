#include "core/localization/fusion/ekf_timer_fusion_backend.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>

#include <Eigen/Cholesky>

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

namespace lightning::loc {

namespace {

template <typename T>
T ReadYaml(const YAML::Node& node, const std::string& key, const T& default_value) {
    if (!node || !node[key]) {
        return default_value;
    }
    return node[key].as<T>();
}


std::vector<double> ReadYamlVector(const YAML::Node& root,
                                   const std::string& section,
                                   const std::string& key,
                                   const std::vector<double>& default_value) {
    if (root && root[section] && root[section][key]) {
        return root[section][key].as<std::vector<double>>();
    }
    if (root && root[key]) {
        return root[key].as<std::vector<double>>();
    }
    return default_value;
}

Mat3d Mat3FromVector(const std::vector<double>& values, const Mat3d& default_value) {
    if (values.size() != 9) {
        return default_value;
    }
    Mat3d mat;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            mat(r, c) = values[static_cast<std::size_t>(r * 3 + c)];
        }
    }
    return mat;
}

SE3 PoseFromXYYaw(double x, double y, double yaw) {
    const Eigen::AngleAxisd aa(yaw, Vec3d::UnitZ());
    return SE3(Quatd(aa), Vec3d(x, y, 0.0));
}

}  // namespace


void TimeDelayEkf::Init(const Eigen::VectorXd& x, const Eigen::MatrixXd& p, int extend_state_step) {
    // This mirrors autoware::kalman_filter::TimeDelayKalmanFilter::init(): the extended
    // delayed states are initialized as repeated state values, but the initial covariance is
    // block-diagonal only. Cross-covariances are then created by predictWithDelay().
    dim_x_ = static_cast<int>(x.rows());
    extend_state_step_ = std::max(1, extend_state_step);
    x_ext_ = Eigen::VectorXd::Zero(dim_x_ * extend_state_step_);
    p_ext_ = Eigen::MatrixXd::Zero(dim_x_ * extend_state_step_, dim_x_ * extend_state_step_);
    for (int i = 0; i < extend_state_step_; ++i) {
        x_ext_.segment(i * dim_x_, dim_x_) = x;
        p_ext_.block(i * dim_x_, i * dim_x_, dim_x_, dim_x_) = p;
    }
    initialized_ = true;
}

Eigen::VectorXd TimeDelayEkf::GetLatestX() const {
    if (!initialized_ || dim_x_ <= 0) {
        return Eigen::VectorXd();
    }
    return x_ext_.segment(0, dim_x_);
}

Eigen::MatrixXd TimeDelayEkf::GetLatestP() const {
    if (!initialized_ || dim_x_ <= 0) {
        return Eigen::MatrixXd();
    }
    return p_ext_.block(0, 0, dim_x_, dim_x_);
}

Eigen::VectorXd TimeDelayEkf::GetCurrentState() const {
    return GetLatestX();
}

Eigen::MatrixXd TimeDelayEkf::GetCurrentCovariance() const {
    return GetLatestP();
}

double TimeDelayEkf::GetXElement(int index) const {
    if (!initialized_ || index < 0 || index >= x_ext_.size()) {
        return 0.0;
    }
    return x_ext_(index);
}

void TimeDelayEkf::PredictWithDelay(const Eigen::VectorXd& x_next,
                                             const Eigen::MatrixXd& a,
                                             const Eigen::MatrixXd& q) {
    if (!initialized_ || dim_x_ <= 0 || extend_state_step_ <= 0) {
        return;
    }
    const int n = dim_x_ * extend_state_step_;
    Eigen::MatrixXd a_ext = Eigen::MatrixXd::Zero(n, n);
    a_ext.block(0, 0, dim_x_, dim_x_) = a;
    for (int i = 1; i < extend_state_step_; ++i) {
        a_ext.block(i * dim_x_, (i - 1) * dim_x_, dim_x_, dim_x_) =
            Eigen::MatrixXd::Identity(dim_x_, dim_x_);
    }

    Eigen::VectorXd x_old = x_ext_;
    Eigen::VectorXd x_new = Eigen::VectorXd::Zero(n);
    x_new.segment(0, dim_x_) = x_next;
    for (int i = 1; i < extend_state_step_; ++i) {
        x_new.segment(i * dim_x_, dim_x_) = x_old.segment((i - 1) * dim_x_, dim_x_);
    }

    Eigen::MatrixXd q_ext = Eigen::MatrixXd::Zero(n, n);
    q_ext.block(0, 0, dim_x_, dim_x_) = q;

    x_ext_ = x_new;
    p_ext_ = a_ext * p_ext_ * a_ext.transpose() + q_ext;
    p_ext_ = 0.5 * (p_ext_ + p_ext_.transpose());
}

bool TimeDelayEkf::UpdateWithDelay(const Eigen::VectorXd& y,
                                            const Eigen::MatrixXd& c,
                                            const Eigen::MatrixXd& r,
                                            int delay_step,
                                            Eigen::VectorXd* innovation,
                                            double* mahalanobis) {
    if (!initialized_ || delay_step < 0 || delay_step >= extend_state_step_) {
        return false;
    }
    const int dim_y = static_cast<int>(y.rows());
    const int n = x_ext_.size();
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(dim_y, n);
    h.block(0, delay_step * dim_x_, dim_y, dim_x_) = c;

    const Eigen::VectorXd y_pred = h * x_ext_;
    Eigen::VectorXd v = y - y_pred;
    if (innovation) {
        *innovation = v;
    }

    Eigen::MatrixXd s = h * p_ext_ * h.transpose() + r;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(s);
    if (ldlt.info() != Eigen::Success) {
        return false;
    }
    const Eigen::MatrixXd s_inv = ldlt.solve(Eigen::MatrixXd::Identity(dim_y, dim_y));
    if (mahalanobis) {
        *mahalanobis = (v.transpose() * s_inv * v)(0, 0);
    }
    const Eigen::MatrixXd p_ht = p_ext_ * h.transpose();
    const Eigen::MatrixXd k = p_ht * s_inv;
    if (k.array().isNaN().any() || k.array().isInf().any()) {
        return false;
    }

    // Same extended-delay-state measurement model as Autoware, but use the
    // Joseph covariance update for better numerical symmetry/positive-semidefiniteness.
    // This changes only the covariance update form; the delayed-state architecture,
    // block selection, and Kalman gain are the same as Autoware time_delay_kalman_filter.
    const Eigen::MatrixXd i = Eigen::MatrixXd::Identity(n, n);
    x_ext_ = x_ext_ + k * v;
    const Eigen::MatrixXd i_kh = i - k * h;
    p_ext_ = i_kh * p_ext_ * i_kh.transpose() + k * r * k.transpose();
    p_ext_ = 0.5 * (p_ext_ + p_ext_.transpose());
    return true;
}

void TimeDelayEkf::NormalizeYawStates(int yaw_index) {
    if (!initialized_ || yaw_index < 0 || yaw_index >= dim_x_) {
        return;
    }
    for (int i = 0; i < extend_state_step_; ++i) {
        const int idx = i * dim_x_ + yaw_index;
        x_ext_(idx) = std::atan2(std::sin(x_ext_(idx)), std::cos(x_ext_(idx)));
    }
}

EkfTimerFusionBackend::EkfTimerFusionBackend() {
    initial_map_odom_ = SE3();
    latest_map_odom_smooth_ = SE3();
    latest_map_base_opt_ = SE3();
    stat_last_log_time_ = std::chrono::steady_clock::now();
}

EkfTimerFusionBackend::~EkfTimerFusionBackend() { Stop(); }

bool EkfTimerFusionBackend::Init(const std::string& yaml_path) {
    const YAML::Node yaml = YAML::LoadFile(yaml_path);
    const YAML::Node fusion = yaml["fusion"];

    options_.enable = ReadYaml<bool>(fusion, "enable", false);
    options_.ekf_enable = ReadYaml<bool>(fusion, "ekf_enable", true);

    options_.initial_sigma_xy = ReadYaml<double>(fusion, "ekf_initial_sigma_xy", 1.0);
    options_.initial_sigma_yaw_deg = ReadYaml<double>(fusion, "ekf_initial_sigma_yaw_deg", 10.0);
    options_.initial_sigma_yaw_bias_deg = ReadYaml<double>(fusion, "ekf_initial_sigma_yaw_bias_deg", 5.0);
    options_.initial_sigma_vx = ReadYaml<double>(fusion, "ekf_initial_sigma_vx", 1.0);
    options_.initial_sigma_wz_deg = ReadYaml<double>(fusion, "ekf_initial_sigma_wz_deg", 20.0);

    options_.process_sigma_xy = ReadYaml<double>(fusion, "ekf_process_sigma_xy", 0.05);
    options_.process_sigma_yaw_deg = ReadYaml<double>(fusion, "ekf_process_sigma_yaw_deg", 1.0);
    options_.process_sigma_yaw_bias_deg = ReadYaml<double>(fusion, "ekf_process_sigma_yaw_bias_deg", 0.01);
    options_.process_sigma_vx = ReadYaml<double>(fusion, "ekf_process_sigma_vx", 0.50);
    options_.process_sigma_wz_deg = ReadYaml<double>(fusion, "ekf_process_sigma_wz_deg", 5.0);

    options_.ndt_sigma_x = ReadYaml<double>(fusion, "ndt_sigma_x", 0.30);
    options_.ndt_sigma_y = ReadYaml<double>(fusion, "ndt_sigma_y", 0.30);
    options_.ndt_sigma_yaw_deg = ReadYaml<double>(fusion, "ndt_sigma_yaw_deg", 3.0);
    options_.ndt_bad_confidence_threshold = ReadYaml<double>(fusion, "ndt_bad_confidence_threshold", 0.20);
    options_.ndt_bad_cov_scale = ReadYaml<double>(fusion, "ndt_bad_cov_scale", 10.0);
    options_.ndt_mahalanobis_gate = ReadYaml<double>(fusion, "ndt_mahalanobis_gate", 9.0);
    options_.ndt_force_accept = ReadYaml<bool>(fusion, "ndt_force_accept", true);
    options_.extend_state_step = ReadYaml<int>(fusion, "extend_state_step", 50);
    options_.delay_time_tolerance = ReadYaml<double>(fusion, "delay_time_tolerance", 0.06);
    options_.max_pose_clones = ReadYaml<int>(fusion, "max_pose_clones", options_.extend_state_step);
    options_.clone_time_tolerance = ReadYaml<double>(fusion, "clone_time_tolerance", options_.delay_time_tolerance);
    options_.clone_remove_before_sec = ReadYaml<double>(fusion, "clone_remove_before_sec", 0.0);

    options_.lio_twist_sigma_vx_fallback = ReadYaml<double>(fusion, "lio_twist_sigma_vx_fallback", 0.10);
    options_.lio_twist_sigma_wz_deg_fallback = ReadYaml<double>(fusion, "lio_twist_sigma_wz_deg_fallback", 2.0);
    options_.lio_twist_max_dt = ReadYaml<double>(fusion, "lio_twist_max_dt", 5.0);
    options_.lio_twist_min_dt = ReadYaml<double>(fusion, "lio_twist_min_dt", 1e-3);
    options_.lio_twist_max_vx = ReadYaml<double>(fusion, "lio_twist_max_vx", 5.0);
    options_.lio_twist_max_wz_deg = ReadYaml<double>(fusion, "lio_twist_max_wz_deg", 90.0);
    options_.lio_twist_enable_low_pass = ReadYaml<bool>(fusion, "lio_twist_enable_low_pass", true);
    options_.lio_twist_low_pass_tau = ReadYaml<double>(fusion, "lio_twist_low_pass_tau", 0.80);
    options_.lio_twist_filter_reset_gap = ReadYaml<double>(fusion, "lio_twist_filter_reset_gap", 2.0);
    options_.lio_twist_max_vx_step = ReadYaml<double>(fusion, "lio_twist_max_vx_step", 0.50);
    options_.lio_twist_max_wz_step_deg = ReadYaml<double>(fusion, "lio_twist_max_wz_step_deg", 10.0);
    options_.lio_cov_min_variance = ReadYaml<double>(fusion, "lio_cov_min_variance", 1e-6);
    options_.lio_cov_max_variance = ReadYaml<double>(fusion, "lio_cov_max_variance", 1e4);

    options_.wheel_sigma_vx = ReadYaml<double>(fusion, "wheel_sigma_vx", 0.20);
    options_.wheel_sigma_wz_deg = ReadYaml<double>(fusion, "wheel_sigma_wz_deg", 5.0);
    options_.wheel_max_vx = ReadYaml<double>(fusion, "wheel_max_vx", 10.0);
    options_.wheel_max_wz_deg = ReadYaml<double>(fusion, "wheel_max_wz_deg", 180.0);

    const YAML::Node lio_sam = yaml["lio_sam"];
    options_.imu_gravity = ReadYaml<double>(fusion, "imu_gravity",
                                            ReadYaml<double>(lio_sam, "imuGravity", 9.80511));
    options_.imu_acc_noise = ReadYaml<double>(fusion, "imu_acc_noise",
                                             ReadYaml<double>(lio_sam, "imuAccNoise", 9.0e-4));
    options_.imu_gyr_noise = ReadYaml<double>(fusion, "imu_gyr_noise",
                                             ReadYaml<double>(lio_sam, "imuGyrNoise", 1.5636343949698187e-03));
    options_.imu_acc_bias_noise = ReadYaml<double>(fusion, "imu_acc_bias_noise",
                                                  ReadYaml<double>(lio_sam, "imuAccBiasN", 5.0e-4));
    options_.imu_gyr_bias_noise = ReadYaml<double>(fusion, "imu_gyr_bias_noise",
                                                  ReadYaml<double>(lio_sam, "imuGyrBiasN", 5.0e-5));
    options_.imu_integration_sigma = ReadYaml<double>(fusion, "imu_integration_sigma", 1.0e-4);
    options_.imu_use_wz_update = ReadYaml<bool>(fusion, "imu_use_wz_update", false);
    options_.imu_gyro_z_sigma_deg = ReadYaml<double>(fusion, "imu_gyro_z_sigma_deg", 10.0);
    options_.imu_use_preintegration_covariance =
        ReadYaml<bool>(fusion, "imu_use_preintegration_covariance", false);
    options_.imu_preintegration_min_update_dt =
        ReadYaml<double>(fusion, "imu_preintegration_min_update_dt", 0.02);
    options_.imu_preintegration_max_update_dt =
        ReadYaml<double>(fusion, "imu_preintegration_max_update_dt", 0.20);
    options_.imu_preintegration_max_dt =
        ReadYaml<double>(fusion, "imu_preintegration_max_dt", 0.10);

    const std::vector<double> identity9{1.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0,
                                        0.0, 0.0, 1.0};
    // Preferred semantic key: R_base_imu maps angular velocity from IMU frame to base_link frame.
    // Fallback to extrinsicBaseImuRot, then to extrinsicBaseLidarRot only if no explicit IMU extrinsic
    // exists. The fallback preserves old deployments where the IMU axes are configured as lidar axes.
    std::vector<double> R_base_imu_vec = ReadYamlVector(yaml, "fusion", "R_base_imu", {});
    if (R_base_imu_vec.size() != 9) {
        R_base_imu_vec = ReadYamlVector(yaml, "common", "extrinsicBaseImuRot", {});
    }
    if (R_base_imu_vec.size() != 9) {
        R_base_imu_vec = ReadYamlVector(yaml, "common", "extrinsicBaseLidarRot", identity9);
    }
    options_.R_base_imu = Mat3FromVector(R_base_imu_vec, Mat3d::Identity());

    options_.enable_map_odom_filter = ReadYaml<bool>(fusion, "enable_map_odom_filter", true);
    options_.map_odom_alpha = ReadYaml<double>(fusion, "map_odom_alpha", 0.20);
    options_.max_imu_queue_size = ReadYaml<int>(fusion, "max_imu_queue_size", 4000);
    options_.max_lio_queue_size = ReadYaml<int>(fusion, "max_lio_queue_size", 200);
    options_.max_ndt_queue_size = ReadYaml<int>(fusion, "max_ndt_queue_size", 100);
    options_.max_wheel_queue_size = ReadYaml<int>(fusion, "max_wheel_queue_size", 200);
    options_.max_lio_buffer_size = ReadYaml<int>(fusion, "max_lio_buffer_size", 2000);
    options_.max_imu_buffer_size = ReadYaml<int>(fusion, "max_imu_buffer_size", 8000);

    LOG(INFO) << "[FUSION_EKF_PARAM] enable=" << options_.enable
              << ", ekf_enable=" << options_.ekf_enable
              << ", ndt_sigma_xy=" << options_.ndt_sigma_x << "/" << options_.ndt_sigma_y
              << ", ndt_sigma_yaw_deg=" << options_.ndt_sigma_yaw_deg
              << ", extend_state_step=" << options_.extend_state_step
              << ", delay_time_tolerance=" << options_.delay_time_tolerance
              << ", ndt_mahalanobis_gate=" << options_.ndt_mahalanobis_gate
              << ", lio_twist_sigma_vx_fallback=" << options_.lio_twist_sigma_vx_fallback
              << ", lio_twist_sigma_wz_deg_fallback=" << options_.lio_twist_sigma_wz_deg_fallback
              << ", lio_twist_enable_low_pass=" << options_.lio_twist_enable_low_pass
              << ", lio_twist_low_pass_tau=" << options_.lio_twist_low_pass_tau
              << ", lio_twist_max_vx_step=" << options_.lio_twist_max_vx_step
              << ", lio_twist_max_wz_step_deg=" << options_.lio_twist_max_wz_step_deg
              << ", wheel_sigma_vx=" << options_.wheel_sigma_vx
              << ", wheel_sigma_wz_deg=" << options_.wheel_sigma_wz_deg
              << ", imu_use_preintegration_covariance=" << options_.imu_use_preintegration_covariance
              << ", imu_acc_noise=" << options_.imu_acc_noise
              << ", imu_gyr_noise=" << options_.imu_gyr_noise
              << ", imu_gyro_z_sigma_deg=" << options_.imu_gyro_z_sigma_deg
              << ", map_odom_alpha=" << options_.map_odom_alpha
              << ", R_base_imu=" << options_.R_base_imu;
    LOG(INFO) << "[FUSION_IMU_EXTRINSIC] semantic: R_base_imu maps gyro/acc vector from IMU frame to base_link;"
              << " translation is intentionally ignored for vector conversion";
    return true;
}

void EkfTimerFusionBackend::SetInitialMapOdom(const SE3& map_odom) {
    std::lock_guard<std::mutex> lock(output_mutex_);
    initial_map_odom_ = map_odom;
    latest_map_odom_smooth_ = map_odom;
    has_map_odom_ = true;
}

void EkfTimerFusionBackend::Start() {
    // The EKF timer is the master pipeline.  Do not start a second consumer thread here;
    // RunTimerTick() drains the queues at the timer timestamp, predicts to that timestamp,
    // then applies velocity/NDT updates in a deterministic order.
    stop_requested_ = false;
    running_ = true;
}

void EkfTimerFusionBackend::Stop() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_requested_ = true;
    }
    queue_cv_.notify_all();
    if (backend_thread_.joinable()) {
        backend_thread_.join();
    }
    running_ = false;
}

void EkfTimerFusionBackend::FeedImu(const FusionImuMeasurement& imu) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        imu_queue_.push_back(imu);
        while (imu_queue_.size() > static_cast<std::size_t>(options_.max_imu_queue_size)) {
            imu_queue_.pop_front();
        }
    }
    queue_cv_.notify_one();
}

void EkfTimerFusionBackend::FeedLio(const FusionLioMeasurement& lio) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        lio_queue_.push_back(lio);
        while (lio_queue_.size() > static_cast<std::size_t>(options_.max_lio_queue_size)) {
            lio_queue_.pop_front();
        }
    }
    queue_cv_.notify_one();
}

void EkfTimerFusionBackend::FeedNdt(const FusionNdtMeasurement& ndt) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        ndt_queue_.push_back(ndt);
        while (ndt_queue_.size() > static_cast<std::size_t>(options_.max_ndt_queue_size)) {
            ndt_queue_.pop_front();
        }
    }
    queue_cv_.notify_one();
}

void EkfTimerFusionBackend::FeedWheel(const FusionWheelMeasurement& wheel) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        wheel_queue_.push_back(wheel);
        while (wheel_queue_.size() > static_cast<std::size_t>(options_.max_wheel_queue_size)) {
            wheel_queue_.pop_front();
        }
    }
    queue_cv_.notify_one();
}

bool EkfTimerFusionBackend::GetLatestMapOdom(SE3* map_odom, std::uint64_t* seq) const {
    if (!map_odom) {
        return false;
    }
    std::lock_guard<std::mutex> lock(output_mutex_);
    if (!has_map_odom_) {
        return false;
    }
    *map_odom = latest_map_odom_smooth_;
    if (seq) {
        *seq = map_odom_update_seq_;
    }
    return true;
}

bool EkfTimerFusionBackend::GetLatestOptimizedMapBase(SE3* map_base) const {
    if (!map_base) {
        return false;
    }
    std::lock_guard<std::mutex> lock(output_mutex_);
    if (!ekf_initialized_.load()) {
        return false;
    }
    *map_base = latest_map_base_opt_;
    return true;
}

bool EkfTimerFusionBackend::IsInitialized() const {
    std::lock_guard<std::mutex> lock(output_mutex_);
    return ekf_initialized_.load();
}

bool EkfTimerFusionBackend::RunTimerTick(double timestamp,
                                           SE3* map_base,
                                           SE3* map_odom,
                                           std::uint64_t* seq) {
    if (!map_base || !map_odom) {
        return false;
    }
    if (timestamp <= 0.0 || !std::isfinite(timestamp)) {
        return false;
    }

    std::vector<FusionEvent> events;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        events.reserve(imu_queue_.size() + lio_queue_.size() + ndt_queue_.size() + wheel_queue_.size());
        for (const auto& imu : imu_queue_) {
            FusionEvent e;
            e.timestamp = imu.timestamp;
            e.type = FusionEventType::IMU;
            e.imu = imu;
            events.push_back(e);
        }
        imu_queue_.clear();
        for (const auto& lio : lio_queue_) {
            FusionEvent e;
            e.timestamp = lio.timestamp;
            e.type = FusionEventType::LIO;
            e.lio = lio;
            events.push_back(e);
        }
        lio_queue_.clear();
        for (const auto& wheel : wheel_queue_) {
            FusionEvent e;
            e.timestamp = wheel.timestamp;
            e.type = FusionEventType::WHEEL;
            e.wheel = wheel;
            events.push_back(e);
        }
        wheel_queue_.clear();
        for (const auto& ndt : ndt_queue_) {
            FusionEvent e;
            e.timestamp = ndt.timestamp;
            e.type = FusionEventType::NDT;
            e.ndt = ndt;
            events.push_back(e);
        }
        ndt_queue_.clear();
    }

    std::vector<FusionEvent> lio_events;
    std::vector<FusionEvent> ndt_events;
    FusionEvent latest_imu;
    FusionEvent latest_wheel;
    bool has_latest_imu = false;
    bool has_latest_wheel = false;
    for (const auto& e : events) {
        switch (e.type) {
            case FusionEventType::IMU:
                if (!has_latest_imu || e.timestamp > latest_imu.timestamp) {
                    latest_imu = e;
                    has_latest_imu = true;
                }
                break;
            case FusionEventType::WHEEL:
                if (!has_latest_wheel || e.timestamp > latest_wheel.timestamp) {
                    latest_wheel = e;
                    has_latest_wheel = true;
                }
                break;
            case FusionEventType::LIO:
                lio_events.push_back(e);
                break;
            case FusionEventType::NDT:
                ndt_events.push_back(e);
                break;
        }
    }
    auto by_stamp = [](const FusionEvent& a, const FusionEvent& b) { return a.timestamp < b.timestamp; };
    std::sort(lio_events.begin(), lio_events.end(), by_stamp);
    std::sort(ndt_events.begin(), ndt_events.end(), by_stamp);

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!ekf_initialized_.load()) {
            // EKF timer is the main timeline.  Start from an all-zero map->base_link state at
            // the first timer tick.  This mirrors Autoware's timer-driven EKF lifecycle: predict
            // on the timer, apply queued observations, then publish the latest state.
            InitializeEkfFromPose(timestamp, SE3());
        }

        // Timer-master order for this robot:
        // 1) Apply the newest velocity-like measurements to the current block.  LIO-SAM relative
        //    pose is the primary vx/wz source until a newer LIO frame arrives; wheel odom can fill
        //    the vx gap.  IMU gyro.z is disabled by default because the current sensor yaw-rate is
        //    not reliable enough for this factory robot.
        // 2) Predict the time-delay EKF to the timer timestamp with that latest velocity.
        // 3) Apply delayed absolute NDT pose measurements to the proper delay block.
        // This keeps the extended-delay architecture from Autoware while matching the actual
        // LIO-SAM + NDT timing pipeline in lightning.
        if (has_latest_imu) {
            ProcessImuMeasurement(latest_imu.imu);
        }
        if (has_latest_wheel) {
            ProcessWheelMeasurement(latest_wheel.wheel);
        }
        for (const auto& e : lio_events) {
            ProcessLioMeasurement(e.lio);
        }

        PredictTo(timestamp);

        // NDT is the only delayed absolute pose source.  Its residual is computed against the
        // delay-state block whose accumulated delay is closest to the NDT scan timestamp, then
        // applied to the whole extended state through the current/delayed cross-covariance.
        for (const auto& e : ndt_events) {
            ProcessNdtMeasurement(e.ndt);
        }

        if (lio_buffer_.empty()) {
            std::lock_guard<std::mutex> lock_output(output_mutex_);
            latest_map_base_opt_ = StateToMapBase();
        } else {
            UpdateMapOdomFromEkf(&lio_buffer_.back(), "timer_10hz_master");
        }
        LogStatsIfNeeded();
    }
    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        if (!has_map_odom_) {
            return false;
        }
        *map_base = latest_map_base_opt_;
        *map_odom = latest_map_odom_smooth_;
        if (seq) {
            *seq = map_odom_update_seq_;
        }
    }
    return true;
}

void EkfTimerFusionBackend::BackendLoop() {
    while (true) {
        std::vector<FusionEvent> events;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(20), [this]() {
                return stop_requested_ || !imu_queue_.empty() || !lio_queue_.empty() ||
                       !ndt_queue_.empty() || !wheel_queue_.empty();
            });
            if (stop_requested_) {
                break;
            }
            events.reserve(imu_queue_.size() + lio_queue_.size() + ndt_queue_.size() + wheel_queue_.size());
            for (const auto& imu : imu_queue_) {
                FusionEvent e;
                e.timestamp = imu.timestamp;
                e.type = FusionEventType::IMU;
                e.imu = imu;
                events.push_back(e);
            }
            imu_queue_.clear();
            for (const auto& lio : lio_queue_) {
                FusionEvent e;
                e.timestamp = lio.timestamp;
                e.type = FusionEventType::LIO;
                e.lio = lio;
                events.push_back(e);
            }
            lio_queue_.clear();
            for (const auto& ndt : ndt_queue_) {
                FusionEvent e;
                e.timestamp = ndt.timestamp;
                e.type = FusionEventType::NDT;
                e.ndt = ndt;
                events.push_back(e);
            }
            ndt_queue_.clear();
            for (const auto& wheel : wheel_queue_) {
                FusionEvent e;
                e.timestamp = wheel.timestamp;
                e.type = FusionEventType::WHEEL;
                e.wheel = wheel;
                events.push_back(e);
            }
            wheel_queue_.clear();
        }

        auto priority = [](FusionEventType type) {
            switch (type) {
                case FusionEventType::IMU:
                    return 0;
                case FusionEventType::LIO:
                    return 1;
                case FusionEventType::WHEEL:
                    return 2;
                case FusionEventType::NDT:
                    return 3;
            }
            return 4;
        };
        std::sort(events.begin(), events.end(), [priority](const FusionEvent& a, const FusionEvent& b) {
            if (a.timestamp == b.timestamp) {
                return priority(a.type) < priority(b.type);
            }
            return a.timestamp < b.timestamp;
        });
        for (const auto& event : events) {
            ProcessEvent(event);
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            LogStatsIfNeeded();
        }
    }
}

void EkfTimerFusionBackend::ProcessEvent(const FusionEvent& event) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    ProcessEventUnlocked(event);
}

void EkfTimerFusionBackend::ProcessEventUnlocked(const FusionEvent& event) {
    switch (event.type) {
        case FusionEventType::IMU:
            ProcessImuMeasurement(event.imu);
            break;
        case FusionEventType::LIO:
            ProcessLioMeasurement(event.lio);
            break;
        case FusionEventType::NDT:
            ProcessNdtMeasurement(event.ndt);
            break;
        case FusionEventType::WHEEL:
            ProcessWheelMeasurement(event.wheel);
            break;
    }
}

void EkfTimerFusionBackend::BufferImuMeasurement(const FusionImuMeasurement& imu) {
    imu_buffer_.push_back(imu);
    while (imu_buffer_.size() > static_cast<std::size_t>(options_.max_imu_buffer_size)) {
        imu_buffer_.pop_front();
    }
    ++stat_imu_buffered_;
    ++stat_window_imu_;
}

void EkfTimerFusionBackend::BufferLioMeasurement(const FusionLioMeasurement& lio) {
    lio_buffer_.push_back(lio);
    while (lio_buffer_.size() > static_cast<std::size_t>(options_.max_lio_buffer_size)) {
        lio_buffer_.pop_front();
    }
    ++stat_lio_buffered_;
    ++stat_window_lio_;
}

void EkfTimerFusionBackend::BufferNdtMeasurement(const FusionNdtMeasurement& ndt) {
    ++stat_ndt_buffered_;
    ++stat_window_ndt_;
}

void EkfTimerFusionBackend::BufferWheelMeasurement(const FusionWheelMeasurement& wheel) {
    wheel_buffer_.push_back(wheel);
    while (wheel_buffer_.size() > static_cast<std::size_t>(options_.max_wheel_queue_size)) {
        wheel_buffer_.pop_front();
    }
    ++stat_wheel_buffered_;
    ++stat_window_wheel_;
}

void EkfTimerFusionBackend::ProcessImuMeasurement(const FusionImuMeasurement& imu) {
    BufferImuMeasurement(imu);
    if (!options_.imu_use_wz_update) {
        static std::size_t skipped_imu_log_count = 0;
        if ((skipped_imu_log_count++ % 200) == 0) {
            LOG(INFO) << "[FUSION_IMU_SKIP] reason=imu_wz_update_disabled, t="
                      << std::setprecision(14) << imu.timestamp;
        }
        return;
    }
    double gyro_z_base = 0.0;
    double gyro_z_variance = 0.0;
    if (!ComputeImuYawRate(imu, &gyro_z_base, &gyro_z_variance)) {
        return;
    }
    if (!ekf_initialized_.load()) {
        ResetImuPreintegration(imu.timestamp);
        return;
    }

    double wz = gyro_z_base;
    double var = gyro_z_variance;
    if (options_.imu_use_preintegration_covariance) {
        if (!PropagateImuPreintegration(imu, gyro_z_base, gyro_z_variance, &wz, &var)) {
            return;
        }
    }

    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    // Use only wz. vx gets a very large variance to make this a yaw-rate update only.
    R(0, 0) = 1e6;
    R(1, 1) = std::max(var, 1e-8);
    UpdateTwist(imu.timestamp, x_(4), wz, R,
                options_.imu_use_preintegration_covariance ? "imu_preintegration_wz" : "imu_gyro");
    ++stat_imu_updates_;
    ++stat_window_imu_updates_;
}

void EkfTimerFusionBackend::ProcessLioMeasurement(const FusionLioMeasurement& lio) {
    if (!lio.pose_valid) {
        return;
    }
    BufferLioMeasurement(lio);

    if (ekf_initialized_.load() && has_prev_lio_for_twist_) {
        double vx = 0.0;
        double wz = 0.0;
        Eigen::Matrix2d cov = Eigen::Matrix2d::Identity();
        if (ComputeLioTwist(prev_lio_for_twist_, lio, &vx, &wz, &cov)) {
            double vx_update = vx;
            double wz_update = wz;
            Eigen::Matrix2d cov_update = cov;
            if (FilterLioTwist(lio.timestamp, vx, wz, cov, &vx_update, &wz_update, &cov_update)) {
                UpdateTwist(lio.timestamp, vx_update, wz_update, cov_update,
                            options_.lio_twist_enable_low_pass ? "lio_filtered_twist" : "lio_hessian_twist");
            }
        }
    }
    prev_lio_for_twist_ = lio;
    has_prev_lio_for_twist_ = true;

    if (ekf_initialized_.load()) {
        UpdateMapOdomFromEkf(&lio, "lio_event_velocity_update");
    }
}

void EkfTimerFusionBackend::ProcessNdtMeasurement(const FusionNdtMeasurement& ndt) {
    BufferNdtMeasurement(ndt);
    if (!options_.ndt_force_accept && !ndt.converged) {
        LOG(WARNING) << "[FUSION_NDT_SKIP] reason=not_converged, t=" << std::setprecision(14) << ndt.timestamp;
        return;
    }

    if (!ekf_initialized_.load()) {
        InitializeEkfFromPose(ndt.timestamp, ndt.map_base);
        UpdateMapOdomFromEkf(lio_buffer_.empty() ? nullptr : &lio_buffer_.back(), "init_from_ndt");
        return;
    }

    const Eigen::Matrix3d R = BuildNdtCovarianceXYYaw(ndt);
    if (UpdatePoseClone2D(ndt.timestamp, ndt.map_base, R,
                          ndt.has_pose_covariance ? "ndt_delay_hessian_cov" : "ndt_delay_fixed_cov")) {
        RemovePoseClonesUpTo(ndt.timestamp + options_.delay_time_tolerance - options_.clone_remove_before_sec);
        UpdateMapOdomFromEkf(lio_buffer_.empty() ? nullptr : &lio_buffer_.back(), "ndt_delay_update");
    }
}

void EkfTimerFusionBackend::ProcessWheelMeasurement(const FusionWheelMeasurement& wheel) {
    BufferWheelMeasurement(wheel);
    if (!ekf_initialized_.load() || !wheel.valid) {
        return;
    }
    if (!std::isfinite(wheel.vx) || !std::isfinite(wheel.wz) ||
        std::abs(wheel.vx) > options_.wheel_max_vx ||
        std::abs(wheel.wz) > DegToRad(options_.wheel_max_wz_deg)) {
        LOG(WARNING) << "[FUSION_WHEEL_SKIP] reason=bad_value, t=" << std::setprecision(14)
                     << wheel.timestamp << ", vx=" << wheel.vx << ", wz=" << wheel.wz;
        return;
    }

    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    if (wheel.has_twist_covariance) {
        R = wheel.twist_covariance;
    } else {
        R(0, 0) = std::pow(options_.wheel_sigma_vx, 2);
        R(1, 1) = std::pow(DegToRad(options_.wheel_sigma_wz_deg), 2);
    }
    UpdateTwist(wheel.timestamp, wheel.vx, wheel.wz, R, "wheel_twist");
}

void EkfTimerFusionBackend::PredictTo(double timestamp) {
    if (!ekf_initialized_.load()) {
        return;
    }
    if (last_predict_time_ < 0.0) {
        last_predict_time_ = timestamp;
        return;
    }
    double dt = timestamp - last_predict_time_;
    if (dt <= 0.0) {
        return;
    }
    dt = std::min(dt, 1.0);

    const double yaw = x_(2);
    const double yaw_bias = x_(3);
    const double vx = x_(4);
    const double wz = x_(5);
    const double theta = yaw + yaw_bias;

    Eigen::Matrix<double, 6, 1> x_next = x_;
    x_next(0) += vx * std::cos(theta) * dt;
    x_next(1) += vx * std::sin(theta) * dt;
    x_next(2) = NormalizeAngle(x_next(2) + wz * dt);

    // Same kinematic Jacobian as Autoware EKF localizer's state_transition.cpp.
    Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Identity();
    A(0, 2) = -vx * std::sin(theta) * dt;
    A(0, 3) = -vx * std::sin(theta) * dt;
    A(0, 4) = std::cos(theta) * dt;
    A(1, 2) = vx * std::cos(theta) * dt;
    A(1, 3) = vx * std::cos(theta) * dt;
    A(1, 4) = std::sin(theta) * dt;
    A(2, 5) = dt;

    // Match Autoware's process-noise semantics: only yaw, vx and wz are driven by process noise.
    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
    Q(2, 2) = std::pow(DegToRad(options_.process_sigma_yaw_deg) * dt, 2);
    Q(4, 4) = std::pow(options_.process_sigma_vx * dt, 2);
    Q(5, 5) = std::pow(DegToRad(options_.process_sigma_wz_deg) * dt, 2);

    delay_filter_.PredictWithDelay(x_next, A, Q);
    delay_filter_.NormalizeYawStates(2);
    AccumulateDelayTime(dt);
    SyncStateFromFilter();
    last_predict_time_ = timestamp;

    LOG(INFO) << "[FUSION_EKF_DELAY_PREDICT] t=" << std::setprecision(14) << timestamp
              << ", dt=" << dt
              << ", delay0=" << (accumulated_delay_times_.empty() ? -1.0 : accumulated_delay_times_.front())
              << ", delay_back=" << (accumulated_delay_times_.empty() ? -1.0 : accumulated_delay_times_.back())
              << ", state_xy_yaw_vx_wz=" << x_(0) << " " << x_(1) << " "
              << RadToDeg(x_(2)) << " " << x_(4) << " " << x_(5);
}

void EkfTimerFusionBackend::InitializeEkfFromPose(double timestamp, const SE3& map_base) {
    const Vec3d t = map_base.translation();
    x_.setZero();
    x_(0) = t.x();
    x_(1) = t.y();
    x_(2) = YawOf(map_base);
    x_(3) = 0.0;
    x_(4) = 0.0;
    x_(5) = 0.0;

    Eigen::Matrix<double, 6, 6> P0 = Eigen::Matrix<double, 6, 6>::Zero();
    P0(0, 0) = std::pow(options_.initial_sigma_xy, 2);
    P0(1, 1) = std::pow(options_.initial_sigma_xy, 2);
    P0(2, 2) = std::pow(DegToRad(options_.initial_sigma_yaw_deg), 2);
    P0(3, 3) = std::pow(DegToRad(options_.initial_sigma_yaw_bias_deg), 2);
    P0(4, 4) = std::pow(options_.initial_sigma_vx, 2);
    P0(5, 5) = std::pow(DegToRad(options_.initial_sigma_wz_deg), 2);

    delay_filter_.Init(x_, P0, options_.extend_state_step);
    accumulated_delay_times_.assign(static_cast<std::size_t>(options_.extend_state_step), 1.0e15);
    if (!accumulated_delay_times_.empty()) {
        accumulated_delay_times_[0] = 0.0;
    }
    pose_clones_.clear();

    ekf_initialized_ = true;
    last_predict_time_ = timestamp;
    ResetImuPreintegration(timestamp);
    const SE3 init_map_base = StateToMapBase();
    {
        std::lock_guard<std::mutex> lock_output(output_mutex_);
        latest_map_base_opt_ = init_map_base;
    }
    LOG(INFO) << "[FUSION_EKF_INIT_AUTOWARE_DELAY] t=" << std::setprecision(14) << timestamp
              << ", extend_state_step=" << options_.extend_state_step
              << ", xyz=" << init_map_base.translation().transpose()
              << ", yaw_deg=" << RadToDeg(x_(2));
}

bool EkfTimerFusionBackend::UpdatePoseClone2D(double timestamp,
                                               const SE3& map_base,
                                               const Eigen::Matrix3d& covariance_xy_yaw,
                                               const std::string& source) {
    const int delay_step = FindDelayStep(timestamp);
    if (delay_step < 0) {
        LOG(WARNING) << "[FUSION_EKF_DELAY_UPDATE_SKIP] reason=no_delay_step, source=" << source
                     << ", meas_t=" << std::setprecision(14) << timestamp
                     << ", current_t=" << last_predict_time_
                     << ", delay=" << (last_predict_time_ - timestamp)
                     << ", delay_back=" << (accumulated_delay_times_.empty() ? -1.0 : accumulated_delay_times_.back());
        return false;
    }

    Eigen::VectorXd z(3);
    z << map_base.translation().x(), map_base.translation().y(), YawOf(map_base);
    const double ekf_yaw_delay = delay_filter_.GetXElement(delay_step * 6 + 2);
    z(2) = ekf_yaw_delay + NormalizeAngle(z(2) - ekf_yaw_delay);

    Eigen::Matrix<double, 3, 6> C = Eigen::Matrix<double, 3, 6>::Zero();
    C(0, 0) = 1.0;
    C(1, 1) = 1.0;
    C(2, 2) = 1.0;

    Eigen::Matrix3d R = covariance_xy_yaw;
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(R(i, i)) || R(i, i) < 1e-8) {
            R(i, i) = 1e-8;
        }
    }

    Eigen::VectorXd innovation;
    double maha = std::numeric_limits<double>::quiet_NaN();
    // First compute innovation/maha on a temporary copy by calling UpdateWithDelay on the real filter only
    // after the gate is checked. Rebuild the same H here so the gate uses the delayed block covariance.
    const Eigen::MatrixXd& P = delay_filter_.GetExtendedP();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, P.cols());
    H.block<3, 6>(0, delay_step * 6) = C;
    const Eigen::VectorXd y_pred = H * delay_filter_.GetExtendedX();
    innovation = z - y_pred;
    innovation(2) = NormalizeAngle(innovation(2));
    const Eigen::Matrix3d S = H * P * H.transpose() + R;
    Eigen::LDLT<Eigen::Matrix3d> ldlt(S);
    if (ldlt.info() != Eigen::Success) {
        LOG(WARNING) << "[FUSION_EKF_DELAY_UPDATE_SKIP] reason=bad_S, source=" << source
                     << ", meas_t=" << std::setprecision(14) << timestamp
                     << ", delay_step=" << delay_step;
        return false;
    }
    const Eigen::Matrix3d S_inv = ldlt.solve(Eigen::Matrix3d::Identity());
    maha = (innovation.transpose() * S_inv * innovation)(0, 0);
    if (std::isfinite(options_.ndt_mahalanobis_gate) && options_.ndt_mahalanobis_gate > 0.0 &&
        std::isfinite(maha) && maha > options_.ndt_mahalanobis_gate) {
        LOG(WARNING) << "[FUSION_EKF_DELAY_UPDATE_REJECT] source=" << source
                     << ", meas_t=" << std::setprecision(14) << timestamp
                     << ", current_t=" << last_predict_time_
                     << ", delay_step=" << delay_step
                     << ", delay_time=" << (last_predict_time_ - timestamp)
                     << ", stored_delay=" << accumulated_delay_times_[static_cast<std::size_t>(delay_step)]
                     << ", maha=" << maha
                     << ", gate=" << options_.ndt_mahalanobis_gate
                     << ", residual=" << innovation.transpose()
                     << ", Rdiag=" << R.diagonal().transpose();
        return false;
    }

    // UpdateWithDelay computes its own residual internally.  Feed a yaw-wrapped measurement
    // equivalent to the normalized innovation so wrap-around near +/-pi is handled correctly.
    Eigen::Vector3d z_update = z;
    z_update(2) = y_pred(2) + innovation(2);

    Eigen::VectorXd update_innovation;
    double update_maha = 0.0;
    if (!delay_filter_.UpdateWithDelay(z_update, C, R, delay_step, &update_innovation, &update_maha)) {
        LOG(WARNING) << "[FUSION_EKF_DELAY_UPDATE_SKIP] reason=update_failed, source=" << source
                     << ", meas_t=" << std::setprecision(14) << timestamp
                     << ", delay_step=" << delay_step;
        return false;
    }
    delay_filter_.NormalizeYawStates(2);
    SyncStateFromFilter();

    ++stat_pose_updates_;
    ++stat_window_pose_updates_;
    ++stat_clone_updates_;
    ++stat_window_clone_updates_;
    LOG(INFO) << "[FUSION_EKF_DELAY_UPDATE] source=" << source
              << ", meas_t=" << std::setprecision(14) << timestamp
              << ", current_t=" << last_predict_time_
              << ", delay_step=" << delay_step
              << ", delay_time=" << (last_predict_time_ - timestamp)
              << ", stored_delay=" << accumulated_delay_times_[static_cast<std::size_t>(delay_step)]
              << ", residual=" << update_innovation.transpose()
              << ", maha=" << update_maha
              << ", Rdiag=" << R.diagonal().transpose()
              << ", current_xy_yaw=" << x_(0) << " " << x_(1) << " " << RadToDeg(x_(2));
    return true;
}

void EkfTimerFusionBackend::UpdateTwist(double timestamp, double vx, double wz,
                                         const Eigen::Matrix2d& covariance,
                                         const std::string& source) {
    if (!ekf_initialized_.load()) {
        return;
    }
    Eigen::VectorXd z(2);
    z << vx, wz;

    Eigen::Matrix<double, 2, 6> C = Eigen::Matrix<double, 2, 6>::Zero();
    C(0, 4) = 1.0;
    C(1, 5) = 1.0;

    Eigen::Matrix2d R = covariance;
    for (int i = 0; i < 2; ++i) {
        if (!std::isfinite(R(i, i)) || R(i, i) < 1e-8) {
            R(i, i) = 1e-8;
        }
    }

    Eigen::VectorXd innovation;
    double maha = 0.0;
    // Velocity observations are treated as current-state updates by design: LIO/wheel provides the
    // latest available speed estimate, and the EKF timer holds it until the next speed observation.
    if (!delay_filter_.UpdateWithDelay(z, C, R, 0, &innovation, &maha)) {
        LOG(WARNING) << "[FUSION_EKF_TWIST_UPDATE_SKIP] source=" << source
                     << ", t=" << std::setprecision(14) << timestamp;
        return;
    }
    delay_filter_.NormalizeYawStates(2);
    SyncStateFromFilter();

    ++stat_twist_updates_;
    ++stat_window_twist_updates_;
    LOG(INFO) << "[FUSION_EKF_TWIST_UPDATE] source=" << source
              << ", t=" << std::setprecision(14) << timestamp
              << ", z_vx_wz=" << vx << " " << wz
              << ", residual=" << innovation.transpose()
              << ", maha=" << maha
              << ", Rdiag=" << R.diagonal().transpose()
              << ", state_vx_wz=" << x_(4) << " " << x_(5);
}

void EkfTimerFusionBackend::AddPoseClone(double timestamp) {
    // Deprecated. Autoware-style delay states are shifted inside PredictTo()/PredictWithDelay().
    LOG(INFO) << "[FUSION_EKF_DELAY_STATE] t=" << std::setprecision(14) << timestamp
              << ", steps=" << options_.extend_state_step
              << ", current_xy_yaw=" << x_(0) << " " << x_(1) << " " << RadToDeg(x_(2));
}

int EkfTimerFusionBackend::FindPoseCloneIndex(double timestamp) const {
    return FindDelayStep(timestamp);
}

int EkfTimerFusionBackend::FindDelayStep(double measurement_timestamp) const {
    if (!ekf_initialized_.load() || last_predict_time_ < 0.0 || accumulated_delay_times_.empty()) {
        return -1;
    }
    const double delay_time = std::max(0.0, last_predict_time_ - measurement_timestamp);
    int best = -1;
    double best_dt = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < accumulated_delay_times_.size(); ++i) {
        const double dt = std::abs(accumulated_delay_times_[i] - delay_time);
        if (dt < best_dt) {
            best_dt = dt;
            best = static_cast<int>(i);
        }
    }
    if (best >= 0 && best_dt <= options_.delay_time_tolerance) {
        return best;
    }
    return -1;
}

void EkfTimerFusionBackend::AccumulateDelayTime(double dt) {
    if (accumulated_delay_times_.empty()) {
        accumulated_delay_times_.assign(static_cast<std::size_t>(options_.extend_state_step), 1.0e15);
    }
    // Keep the timestamps aligned with TimeDelayEkf::PredictWithDelay():
    // block 0 is the newly predicted current state, block 1 is the previous current state, etc.
    for (std::size_t i = accumulated_delay_times_.size() - 1; i > 0; --i) {
        accumulated_delay_times_[i] = accumulated_delay_times_[i - 1] + dt;
    }
    accumulated_delay_times_[0] = 0.0;
}

void EkfTimerFusionBackend::RemovePoseClone(std::size_t clone_index) {
    (void)clone_index;
}

void EkfTimerFusionBackend::RemovePoseClonesUpTo(double timestamp) {
    // Autoware-style delay states are fixed-length; nothing is explicitly removed after an NDT update.
    LOG(INFO) << "[FUSION_EKF_DELAY_KEEP] up_to=" << std::setprecision(14) << timestamp
              << ", delay_back=" << (accumulated_delay_times_.empty() ? -1.0 : accumulated_delay_times_.back());
}

void EkfTimerFusionBackend::ApplyAugmentedDelta(const Eigen::VectorXd& dx) {
    (void)dx;
    SyncStateFromFilter();
}

void EkfTimerFusionBackend::SyncStateFromFilter() {
    if (!delay_filter_.IsInitialized()) {
        return;
    }
    const Eigen::VectorXd x_cur = delay_filter_.GetCurrentState();
    if (x_cur.size() >= 6) {
        x_ = x_cur.head<6>();
        x_(2) = NormalizeAngle(x_(2));
    }
}

Eigen::Matrix3d EkfTimerFusionBackend::BuildNdtCovarianceXYYaw(const FusionNdtMeasurement& ndt) const {
    double scale = 1.0;
    if (std::isfinite(ndt.confidence) && ndt.confidence < options_.ndt_bad_confidence_threshold) {
        scale = options_.ndt_bad_cov_scale;
    }

    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    R(0, 0) = std::pow(options_.ndt_sigma_x, 2);
    R(1, 1) = std::pow(options_.ndt_sigma_y, 2);
    R(2, 2) = std::pow(DegToRad(options_.ndt_sigma_yaw_deg), 2);

    if (ndt.has_pose_covariance) {
        R(0, 0) += std::clamp(ndt.pose_covariance(3, 3), 1e-8, 1e4);
        R(1, 1) += std::clamp(ndt.pose_covariance(4, 4), 1e-8, 1e4);
        R(2, 2) += std::clamp(ndt.pose_covariance(2, 2), 1e-8, 1e4);
    }

    R *= scale;
    LOG(INFO) << "[FUSION_NDT_COV] t=" << std::setprecision(14) << ndt.timestamp
              << ", has_hessian_cov=" << ndt.has_pose_covariance
              << ", confidence=" << ndt.confidence
              << ", scale=" << scale
              << ", Rdiag=" << R.diagonal().transpose();
    return R;
}

bool EkfTimerFusionBackend::ComputeLioTwist(const FusionLioMeasurement& prev,
                                             const FusionLioMeasurement& curr,
                                             double* vx,
                                             double* wz,
                                             Eigen::Matrix2d* covariance) const {
    if (!vx || !wz || !covariance) {
        return false;
    }
    const double dt = curr.timestamp - prev.timestamp;
    if (dt < options_.lio_twist_min_dt || dt > options_.lio_twist_max_dt) {
        LOG(WARNING) << "[FUSION_LIO_TWIST_SKIP] reason=bad_dt, dt=" << dt;
        return false;
    }

    const SE3 rel = prev.odom_base.inverse() * curr.odom_base;
    const double yaw_rel = YawOf(rel);
    const double vx_meas = rel.translation().x() / dt;
    const double wz_meas = NormalizeAngle(yaw_rel) / dt;

    if (!std::isfinite(vx_meas) || !std::isfinite(wz_meas) ||
        std::abs(vx_meas) > options_.lio_twist_max_vx ||
        std::abs(wz_meas) > DegToRad(options_.lio_twist_max_wz_deg)) {
        LOG(WARNING) << "[FUSION_LIO_TWIST_SKIP] reason=bad_value, dt=" << dt
                     << ", vx=" << vx_meas << ", wz=" << wz_meas;
        return false;
    }

    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    const double floor_vx_var = std::pow(options_.lio_twist_sigma_vx_fallback, 2);
    const double floor_wz_var = std::pow(DegToRad(options_.lio_twist_sigma_wz_deg_fallback), 2);
    if (curr.has_pose_covariance) {
        const double var_x = std::clamp(curr.pose_covariance(3, 3),
                                        options_.lio_cov_min_variance, options_.lio_cov_max_variance);
        const double var_yaw = std::clamp(curr.pose_covariance(2, 2),
                                          options_.lio_cov_min_variance, options_.lio_cov_max_variance);
        R(0, 0) = std::max(floor_vx_var, var_x / (dt * dt));
        R(1, 1) = std::max(floor_wz_var, var_yaw / (dt * dt));
    } else {
        R(0, 0) = floor_vx_var;
        R(1, 1) = floor_wz_var;
    }

    *vx = vx_meas;
    *wz = wz_meas;
    *covariance = R;
    LOG(INFO) << "[FUSION_LIO_TWIST] dt=" << dt
              << ", vx=" << vx_meas << ", wz=" << wz_meas
              << ", cov=" << R.diagonal().transpose()
              << ", has_hessian_cov=" << curr.has_pose_covariance;
    return true;
}

bool EkfTimerFusionBackend::FilterLioTwist(double timestamp,
                                           double raw_vx,
                                           double raw_wz,
                                           const Eigen::Matrix2d& raw_covariance,
                                           double* filtered_vx,
                                           double* filtered_wz,
                                           Eigen::Matrix2d* filtered_covariance) {
    if (!filtered_vx || !filtered_wz || !filtered_covariance) {
        return false;
    }
    if (!std::isfinite(timestamp) || !std::isfinite(raw_vx) || !std::isfinite(raw_wz)) {
        LOG(WARNING) << "[FUSION_LIO_TWIST_FILTER_SKIP] reason=bad_input, t=" << timestamp
                     << ", raw_vx=" << raw_vx << ", raw_wz=" << raw_wz;
        return false;
    }

    *filtered_covariance = raw_covariance;
    if (!options_.lio_twist_enable_low_pass || options_.lio_twist_low_pass_tau <= 1e-6) {
        *filtered_vx = raw_vx;
        *filtered_wz = raw_wz;
        return true;
    }

    const bool need_reset = !lio_twist_filter_initialized_ ||
                            lio_twist_filter_timestamp_ <= 0.0 ||
                            timestamp <= lio_twist_filter_timestamp_ ||
                            (timestamp - lio_twist_filter_timestamp_) > options_.lio_twist_filter_reset_gap;
    if (need_reset) {
        lio_twist_filtered_vx_ = raw_vx;
        lio_twist_filtered_wz_ = raw_wz;
        lio_twist_filter_timestamp_ = timestamp;
        lio_twist_filter_initialized_ = true;
        *filtered_vx = lio_twist_filtered_vx_;
        *filtered_wz = lio_twist_filtered_wz_;
        LOG(INFO) << "[FUSION_LIO_TWIST_FILTER_RESET] t=" << std::setprecision(14) << timestamp
                  << ", vx=" << raw_vx << ", wz=" << raw_wz;
        return true;
    }

    const double dt = std::max(1e-3, timestamp - lio_twist_filter_timestamp_);
    const double alpha = std::clamp(dt / (options_.lio_twist_low_pass_tau + dt), 0.0, 1.0);

    double target_vx = raw_vx;
    double target_wz = raw_wz;
    if (options_.lio_twist_max_vx_step > 0.0) {
        const double delta_vx = std::clamp(raw_vx - lio_twist_filtered_vx_,
                                          -options_.lio_twist_max_vx_step,
                                          options_.lio_twist_max_vx_step);
        target_vx = lio_twist_filtered_vx_ + delta_vx;
    }
    if (options_.lio_twist_max_wz_step_deg > 0.0) {
        const double max_wz_step = DegToRad(options_.lio_twist_max_wz_step_deg);
        const double delta_wz = std::clamp(raw_wz - lio_twist_filtered_wz_, -max_wz_step, max_wz_step);
        target_wz = lio_twist_filtered_wz_ + delta_wz;
    }

    lio_twist_filtered_vx_ = (1.0 - alpha) * lio_twist_filtered_vx_ + alpha * target_vx;
    lio_twist_filtered_wz_ = (1.0 - alpha) * lio_twist_filtered_wz_ + alpha * target_wz;
    lio_twist_filter_timestamp_ = timestamp;

    *filtered_vx = lio_twist_filtered_vx_;
    *filtered_wz = lio_twist_filtered_wz_;

    // Do not over-trust a filtered algorithmic velocity. Keep the Hessian/floor covariance, and
    // slightly inflate it when the raw value differs a lot from the filtered value so EKF updates
    // stay conservative during pose jitter or sudden relocalization.
    const double vx_jump_ratio = std::abs(raw_vx - *filtered_vx) / std::max(options_.lio_twist_sigma_vx_fallback, 1e-3);
    const double wz_jump_ratio = std::abs(raw_wz - *filtered_wz) / std::max(DegToRad(options_.lio_twist_sigma_wz_deg_fallback), 1e-4);
    const double inflate = std::clamp(1.0 + 0.25 * std::max(vx_jump_ratio, wz_jump_ratio), 1.0, 10.0);
    (*filtered_covariance) *= inflate;

    LOG(INFO) << "[FUSION_LIO_TWIST_FILTER] t=" << std::setprecision(14) << timestamp
              << ", dt=" << dt
              << ", alpha=" << alpha
              << ", raw_vx=" << raw_vx
              << ", raw_wz=" << raw_wz
              << ", filtered_vx=" << *filtered_vx
              << ", filtered_wz=" << *filtered_wz
              << ", cov_inflate=" << inflate
              << ", Rdiag=" << filtered_covariance->diagonal().transpose();
    return true;
}

bool EkfTimerFusionBackend::ComputeImuYawRate(const FusionImuMeasurement& imu, double* wz, double* variance) const {
    if (!wz || !variance) {
        return false;
    }
    const Vec3d gyro_base = options_.R_base_imu * imu.gyro;
    // In this project gyro.z is used only as a weak current yaw-rate observation.  Do not let
    // unrealistically small IMU noise parameters dominate LIO-SAM yaw-rate.  The configured
    // imu_gyro_z_sigma_deg is a minimum variance floor.
    double var = std::pow(DegToRad(options_.imu_gyro_z_sigma_deg), 2);
    if (imu.has_gyro_covariance) {
        const Eigen::Matrix3d R_base = options_.R_base_imu * imu.gyro_covariance * options_.R_base_imu.transpose();
        if (std::isfinite(R_base(2, 2)) && R_base(2, 2) > 1e-10) {
            var = std::max(var, R_base(2, 2));
        }
    }
    *wz = gyro_base.z();
    *variance = var;
    return std::isfinite(*wz) && std::isfinite(*variance);
}

bool EkfTimerFusionBackend::PropagateImuPreintegration(const FusionImuMeasurement& imu,
                                                         double gyro_z_base,
                                                         double gyro_z_variance,
                                                         double* wz,
                                                         double* variance) {
    if (!wz || !variance) {
        return false;
    }
    if (!imu_preint_initialized_) {
        ResetImuPreintegration(imu.timestamp);
        return false;
    }

    double dt = imu.timestamp - imu_preint_last_time_;
    if (dt <= 0.0 || dt > options_.imu_preintegration_max_dt) {
        LOG(WARNING) << "[FUSION_IMU_PREINT_RESET] reason=bad_dt, dt=" << dt
                     << ", t=" << std::setprecision(14) << imu.timestamp;
        ResetImuPreintegration(imu.timestamp);
        return false;
    }

    const double gyro_bias_var = std::pow(std::max(options_.imu_gyr_bias_noise, 0.0), 2);
    const double integration_var = std::pow(std::max(options_.imu_integration_sigma, 0.0), 2);

    imu_preint_delta_yaw_ += gyro_z_base * dt;
    imu_preint_yaw_variance_ += std::max(gyro_z_variance + gyro_bias_var, 1e-12) * dt * dt + integration_var * dt;
    imu_preint_delta_t_ += dt;
    imu_preint_last_time_ = imu.timestamp;

    if (imu_preint_delta_t_ < options_.imu_preintegration_min_update_dt &&
        imu_preint_delta_t_ < options_.imu_preintegration_max_update_dt) {
        return false;
    }

    const double used_dt = std::max(imu_preint_delta_t_, 1e-6);
    *wz = imu_preint_delta_yaw_ / used_dt;
    *variance = std::clamp(imu_preint_yaw_variance_ / (used_dt * used_dt), 1e-12, 1e4);

    LOG(INFO) << "[FUSION_IMU_PREINT] dt=" << used_dt
              << ", delta_yaw=" << imu_preint_delta_yaw_
              << ", wz=" << *wz
              << ", variance=" << *variance;

    imu_preint_delta_t_ = 0.0;
    imu_preint_delta_yaw_ = 0.0;
    imu_preint_yaw_variance_ = 0.0;
    return std::isfinite(*wz) && std::isfinite(*variance);
}

void EkfTimerFusionBackend::ResetImuPreintegration(double timestamp) {
    imu_preint_initialized_ = true;
    imu_preint_last_time_ = timestamp;
    imu_preint_delta_t_ = 0.0;
    imu_preint_delta_yaw_ = 0.0;
    imu_preint_yaw_variance_ = 0.0;
}

void EkfTimerFusionBackend::UpdateMapOdomFromEkf(const FusionLioMeasurement* lio_for_output,
                                                   const std::string& reason) {
    if (!ekf_initialized_.load() || lio_buffer_.empty()) {
        return;
    }
    const FusionLioMeasurement& lio = lio_for_output ? *lio_for_output : lio_buffer_.back();
    const SE3 map_base = StateToMapBase();
    const SE3 raw_map_odom = map_base * lio.odom_base.inverse();

    SE3 smooth = raw_map_odom;
    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        if (has_map_odom_ && options_.enable_map_odom_filter) {
            const double alpha = std::clamp(options_.map_odom_alpha, 0.0, 1.0);
            const SE3 delta = latest_map_odom_smooth_.inverse() * raw_map_odom;
            smooth = latest_map_odom_smooth_ * SE3::exp(alpha * delta.log());
        }
        PublishOutputLocked(map_base, smooth, reason);
    }
}

SE3 EkfTimerFusionBackend::StateToMapBase() const {
    return PoseFromXYYaw(x_(0), x_(1), x_(2));
}

void EkfTimerFusionBackend::PublishOutputLocked(const SE3& map_base, const SE3& map_odom,
                                                  const std::string& reason) {
    latest_map_base_opt_ = map_base;
    latest_map_odom_smooth_ = map_odom;
    has_map_odom_ = true;
    ++map_odom_update_seq_;
    ++stat_output_updates_;
    ++stat_window_output_updates_;
    LOG(INFO) << "[FUSION_EKF_MAP_ODOM_UPDATE] seq=" << map_odom_update_seq_
              << ", reason=" << reason
              << ", map_base=" << map_base.translation().transpose()
              << ", yaw_deg=" << RadToDeg(YawOf(map_base))
              << ", map_odom=" << map_odom.translation().transpose()
              << ", Pdiag=" << delay_filter_.GetCurrentCovariance().diagonal().transpose();
}

void EkfTimerFusionBackend::TrimBuffers() {}

void EkfTimerFusionBackend::LogStatsIfNeeded() {
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - stat_last_log_time_).count();
    if (dt < 5.0) {
        return;
    }
    const double inv = dt > 1e-6 ? 1.0 / dt : 0.0;
    LOG(INFO) << "[FUSION_EKF_STAT] imu_freq=" << stat_window_imu_ * inv
              << ", lio_freq=" << stat_window_lio_ * inv
              << ", ndt_freq=" << stat_window_ndt_ * inv
              << ", wheel_freq=" << stat_window_wheel_ * inv
              << ", pose_update_freq=" << stat_window_pose_updates_ * inv
              << ", clone_update_freq=" << stat_window_clone_updates_ * inv
              << ", twist_update_freq=" << stat_window_twist_updates_ * inv
              << ", imu_update_freq=" << stat_window_imu_updates_ * inv
              << ", output_freq=" << stat_window_output_updates_ * inv
              << ", total_pose_updates=" << stat_pose_updates_
              << ", total_clone_updates=" << stat_clone_updates_
              << ", total_twist_updates=" << stat_twist_updates_
              << ", total_output=" << stat_output_updates_
              << ", delay_steps=" << options_.extend_state_step
              << ", delay_back=" << (accumulated_delay_times_.empty() ? -1.0 : accumulated_delay_times_.back())
              << ", initialized=" << ekf_initialized_.load();

    stat_window_imu_ = 0;
    stat_window_lio_ = 0;
    stat_window_ndt_ = 0;
    stat_window_wheel_ = 0;
    stat_window_pose_updates_ = 0;
    stat_window_clone_updates_ = 0;
    stat_window_twist_updates_ = 0;
    stat_window_imu_updates_ = 0;
    stat_window_output_updates_ = 0;
    stat_last_log_time_ = now;
}

double EkfTimerFusionBackend::NormalizeAngle(double angle) {
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

double EkfTimerFusionBackend::DegToRad(double deg) { return deg * M_PI / 180.0; }

double EkfTimerFusionBackend::RadToDeg(double rad) { return rad * 180.0 / M_PI; }

double EkfTimerFusionBackend::YawOf(const SE3& pose) {
    const Mat3d R = pose.so3().matrix();
    return std::atan2(R(1, 0), R(0, 0));
}

}  // namespace lightning::loc
