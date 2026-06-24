#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include "core/localization/lidar_loc/lidar_loc.h"
#include "core/localization/localization.h"

#include <opencv2/highgui.hpp>

#include <chrono>
#include "core/lightning_math.hpp"
#include "core/localization/pose_graph/pgo.h"
#include "io/yaml_io.h"
#include "ui/pangolin_window.h"
#include <iomanip>
namespace lightning::loc {

// ！ 构造函数
Localization::Localization(Options options) { options_ = options; }

// ！初始化函数
bool Localization::Init(const std::string& yaml_path, const std::string& global_map_path) {
    UL lock(global_mutex_);
    if (lidar_loc_ != nullptr) {
        // 若已经启动，则变为初始化
        Finish();
    }

    YAML_IO yaml(yaml_path);
    options_.with_ui_ = yaml.GetValue<bool>("system", "with_ui");

    std::string frontend = yaml.GetValue<std::string>("system", "frontend");
    
    use_lio_sam_ = frontend == "lio_sam" || frontend == "liosam";

    YAML::Node yaml_node = YAML::LoadFile(yaml_path);
    if (yaml_node["common"] && yaml_node["common"]["base_link_frame"]) {
        base_link_frame_ = yaml_node["common"]["base_link_frame"].as<std::string>();
    }
    std::vector<double> base_lidar_t{0.0, 0.0, 0.0};
    std::vector<double> base_lidar_R{1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0};
    if (yaml_node["extrinsicBaseLidarTrans"]) {
        base_lidar_t = yaml_node["extrinsicBaseLidarTrans"].as<std::vector<double>>();
    } else if (yaml_node["common"] && yaml_node["common"]["extrinsicBaseLidarTrans"]) {
        base_lidar_t = yaml_node["common"]["extrinsicBaseLidarTrans"].as<std::vector<double>>();
    }
    if (yaml_node["extrinsicBaseLidarRot"]) {
        base_lidar_R = yaml_node["extrinsicBaseLidarRot"].as<std::vector<double>>();
    } else if (yaml_node["common"] && yaml_node["common"]["extrinsicBaseLidarRot"]) {
        base_lidar_R = yaml_node["common"]["extrinsicBaseLidarRot"].as<std::vector<double>>();
    }
    CHECK_EQ(base_lidar_t.size(), 3);
    CHECK_EQ(base_lidar_R.size(), 9);
    Mat3d R_base_lidar;
    R_base_lidar << base_lidar_R[0], base_lidar_R[1], base_lidar_R[2],
        base_lidar_R[3], base_lidar_R[4], base_lidar_R[5],
        base_lidar_R[6], base_lidar_R[7], base_lidar_R[8];
    Quatd q_base_lidar(R_base_lidar);
    q_base_lidar.normalize();
    options_.T_base_lidar_ = SE3(q_base_lidar, Vec3d(base_lidar_t[0], base_lidar_t[1], base_lidar_t[2]));
    LOG(INFO) << "[BASE_LIDAR] T_base_lidar trans=" << options_.T_base_lidar_.translation().transpose();

    std::vector<double> initial_map_odom_t{0.0, 0.0, 0.0};
    std::vector<double> initial_map_odom_R{1.0, 0.0, 0.0,
                                           0.0, 1.0, 0.0,
                                           0.0, 0.0, 1.0};
    if (yaml_node["relocalization"]) {
        if (yaml_node["relocalization"]["initialMapOdomTrans"]) {
            initial_map_odom_t = yaml_node["relocalization"]["initialMapOdomTrans"].as<std::vector<double>>();
        }
        if (yaml_node["relocalization"]["initialMapOdomRot"]) {
            initial_map_odom_R = yaml_node["relocalization"]["initialMapOdomRot"].as<std::vector<double>>();
        }
    }
    CHECK_EQ(initial_map_odom_t.size(), 3);
    CHECK_EQ(initial_map_odom_R.size(), 9);
    Mat3d R_map_odom;
    R_map_odom << initial_map_odom_R[0], initial_map_odom_R[1], initial_map_odom_R[2],
        initial_map_odom_R[3], initial_map_odom_R[4], initial_map_odom_R[5],
        initial_map_odom_R[6], initial_map_odom_R[7], initial_map_odom_R[8];
    Quatd q_map_odom(R_map_odom);
    q_map_odom.normalize();
    {
        UL lock_map_odom(map_odom_mutex_);
        map_odom_pose_ = SE3(q_map_odom,
                             Vec3d(initial_map_odom_t[0], initial_map_odom_t[1], initial_map_odom_t[2]));
    }
    {
        UL lock_lo(lo_pose_mutex_);
        lo_pose_queue_.clear();
    }
    lidar_loc_frame_count_ = 0;
    stat_ros_cloud_in_ = 0;
    stat_preprocess_ok_ = 0;
    stat_preprocess_fail_ = 0;
    stat_base_transform_ok_ = 0;
    stat_odom_queue_push_ = 0;
    stat_preprocess_total_ms_ = 0.0;
    stat_base_transform_total_ms_ = 0.0;
    stat_add_odom_queue_total_ms_ = 0.0;
    stat_process_lidar_total_ms_ = 0.0;
    stat_window_ros_cloud_in_ = 0;
    stat_window_preprocess_ok_ = 0;
    stat_window_preprocess_fail_ = 0;
    stat_window_base_transform_ok_ = 0;
    stat_window_odom_queue_push_ = 0;
    stat_window_preprocess_ms_ = 0.0;
    stat_window_base_transform_ms_ = 0.0;
    stat_window_add_odom_queue_ms_ = 0.0;
    stat_window_process_lidar_ms_ = 0.0;
    stat_input_last_log_time_ = std::chrono::steady_clock::now();
    stat_odom_proc_called_ = 0;
    stat_lio_process_cloud_called_ = 0;
    stat_lio_run_called_ = 0;
    stat_lio_run_success_ = 0;
    stat_lio_run_fail_ = 0;
    stat_lo_pose_published_ = 0;
    stat_lo_pose_not_ok_ = 0;
    stat_window_lo_run_ = 0;
    stat_window_lo_publish_ = 0;
    stat_latest_lo_timestamp_ = -1.0;
    stat_latest_publish_timestamp_ = -1.0;
    stat_odom_last_log_time_ = std::chrono::steady_clock::now();
    stat_need_loc_scan_true_ = 0;
    stat_need_loc_scan_false_ = 0;
    stat_ndt_scan_sent_ = 0;
    {
        UL lock_result(loc_result_mutex_);
        loc_result_ = LocalizationResult();
    }
    LOG(INFO) << "[MAP_ODOM_INIT] map_odom_pose trans=" << map_odom_pose_.translation().transpose();

    preprocess_ = std::make_shared<PointCloudPreprocess>();
    if (!preprocess_->Init(yaml_path)) {
        LOG(ERROR) << "failed to init input preprocess";
        return false;
    }

    /// lidar odom前端
    if (use_lio_sam_) {
        LioSamMapping::Options opt_lio_sam;
        opt_lio_sam.is_in_slam_mode_ = false;
        opt_lio_sam.online_mode_ = options_.online_mode_;
        lio_sam_ = std::make_shared<LioSamMapping>(opt_lio_sam);
        if (!lio_sam_->Init(yaml_path)) {
            LOG(ERROR) << "failed to init LIO-SAM odometry frontend";
            return false;
        }
    } else {
        LaserMapping::Options opt_lio;
        opt_lio.is_in_slam_mode_ = false;
        lio_ = std::make_shared<LaserMapping>(opt_lio);
        if (!lio_->Init(yaml_path)) {
            LOG(ERROR) << "failed to init laser odometry frontend";
            return false;
        }
    }

    /// 激光定位
    LidarLoc::Options lidar_loc_options;
    lidar_loc_options.update_dynamic_cloud_ = yaml.GetValue<bool>("lidar_loc", "update_dynamic_cloud");
    lidar_loc_options.force_2d_ = yaml.GetValue<bool>("lidar_loc", "force_2d");
    lidar_loc_options.map_option_.enable_dynamic_polygon_ = false;
    lidar_loc_options.map_option_.map_path_ = global_map_path;
    lidar_loc_ = std::make_shared<LidarLoc>(lidar_loc_options);

    if (options_.with_ui_) {
        ui_ = std::make_shared<ui::PangolinWindow>();
        ui_->SetCurrentScanSize(1);
        ui_->Init();

        lidar_loc_->SetUI(ui_);

        // lio_->SetUI(ui_);
    }

    if (!lidar_loc_->Init(yaml_path)) {
        LOG(ERROR) << "failed to initialize localization map";
        return false;
    }
    {
        UL lock_map_odom(map_odom_mutex_);
        lidar_loc_->SetMapOdomPose(map_odom_pose_);
    }

    /// pose graph
    pgo_ = std::make_shared<PGO>();
    pgo_->SetDebug(false);
    ///  各模块的异步调用
    options_.enable_lidar_loc_skip_ = yaml.GetValue<bool>("system", "enable_lidar_loc_skip");
    options_.enable_lidar_loc_rviz_ = yaml.GetValue<bool>("system", "enable_lidar_loc_rviz");
    options_.lidar_loc_skip_num_ = yaml.GetValue<int>("system", "lidar_loc_skip_num");
    options_.enable_lidar_odom_skip_ = yaml.GetValue<bool>("system", "enable_lidar_odom_skip");
    options_.lidar_odom_skip_num_ = yaml.GetValue<int>("system", "lidar_odom_skip_num");
    options_.loc_on_kf_ = yaml.GetValue<bool>("lidar_loc", "loc_on_kf");

    lidar_odom_proc_cloud_.SetMaxSize(1);
    lidar_loc_proc_cloud_.SetMaxSize(1);

    lidar_odom_proc_cloud_.SetName("激光里程计");
    lidar_loc_proc_cloud_.SetName("激光定位");
    lidar_odom_proc_cloud_.SetStatsEnabled(true);

    // 允许跳帧
    // NDT skip is handled before LIO-SAM generates recent_cloud_.
    // Do not skip again inside lidar_loc_proc_cloud_, otherwise skip is applied twice.
    lidar_loc_proc_cloud_.SetSkipParam(false, 1);
    lidar_odom_proc_cloud_.SetSkipParam(options_.enable_lidar_odom_skip_, options_.lidar_odom_skip_num_);

    lidar_odom_proc_cloud_.SetProcFunc([this](CloudPtr cloud) { LidarOdomProcCloud(cloud); });
    lidar_loc_proc_cloud_.SetProcFunc([this](CloudPtr cloud) { LidarLocProcCloud(cloud); });

    if (options_.online_mode_) {
        lidar_odom_proc_cloud_.Start();
        lidar_loc_proc_cloud_.Start();
    }

    /// TODO: 发布
    return true;
}

void Localization::PublishLatestResult() {
    LocalizationResult latest;
    {
        UL lock_result(loc_result_mutex_);
        latest = loc_result_;
    }

    if (tf_callback_ && latest.valid_) {
        tf_callback_(latest.ToGeoMsg());
    }
}

bool Localization::ShouldRunLidarLocThisFrame() {
    if (!options_.enable_lidar_loc_skip_ || options_.lidar_loc_skip_num_ <= 1) {
        return true;
    }

    const int idx = lidar_loc_frame_count_++;
    return (idx % options_.lidar_loc_skip_num_) == 0;
}

void Localization::UpdateMapOdomByPGOResult(const LocalizationResult& pgo_result) {
    if (!pgo_result.valid_ || !pgo_result.lidar_loc_valid_) {
        return;
    }

    SE3 odom_base_pose;
    NavState best_match;
    {
        UL lock_lo(lo_pose_mutex_);
        bool interp_success = math::PoseInterp<NavState>(
            pgo_result.timestamp_, lo_pose_queue_, [](const NavState& s) { return s.timestamp_; },
            [](const NavState& s) { return s.GetPose(); }, odom_base_pose, best_match, 5.0);
        if (!interp_success) {
            return;
        }
    }

    SE3 new_map_odom = pgo_result.pose_ * odom_base_pose.inverse();
    {
        UL lock_map_odom(map_odom_mutex_);
        map_odom_pose_ = new_map_odom;
    }

    if (lidar_loc_) {
        lidar_loc_->SetMapOdomPose(new_map_odom);
    }

}

void Localization::UpdateMapOdomByLidarLocResult(const LocalizationResult& loc_result) {
    if (!loc_result.lidar_loc_valid_ || loc_result.status_ != LocalizationStatus::GOOD) {
        return;
    }

    SE3 odom_base_pose;
    NavState best_match;
    {
        UL lock_lo(lo_pose_mutex_);
        bool interp_success = math::PoseInterp<NavState>(
            loc_result.timestamp_, lo_pose_queue_, [](const NavState& s) { return s.timestamp_; },
            [](const NavState& s) { return s.GetPose(); }, odom_base_pose, best_match, 1.0);
        if (!interp_success) {
            LOG(WARNING) << "[MAP_ODOM_UPDATE_FROM_NDT] failed to get odom->base pose at t="
                         << std::setprecision(14) << loc_result.timestamp_;
            return;
        }
    }

    SE3 new_map_odom = loc_result.pose_ * odom_base_pose.inverse();
    {
        UL lock_map_odom(map_odom_mutex_);
        map_odom_pose_ = new_map_odom;
    }

    if (lidar_loc_) {
        lidar_loc_->SetMapOdomPose(new_map_odom);
    }
}

void Localization::PublishHighFrequencyResultByLO(const NavState& lo_state) {
    if (!lo_state.pose_is_ok_) {
        return;
    }

    LocalizationResult result;
    result.timestamp_ = lo_state.timestamp_;
    result.valid_ = true;
    result.lidar_loc_valid_ = true;
    result.status_ = LocalizationStatus::GOOD;
    result.confidence_ = 1.0;

    {
        UL lock_map_odom(map_odom_mutex_);
        result.pose_ = map_odom_pose_ * lo_state.GetPose();
    }

    {
        UL lock_result(loc_result_mutex_);
        loc_result_ = result;
    }

    if (tf_callback_ && result.valid_) {
        tf_callback_(result.ToGeoMsg());
        ++stat_lo_pose_published_;
        ++stat_window_lo_publish_;
        stat_latest_publish_timestamp_ = lo_state.timestamp_;
    }

    if (ui_) {
        ui_->UpdateNavState(result.ToNavState());
        ui_->UpdateRecentPose(result.pose_);
    }
}

void Localization::ProcessLidarMsg(const sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
    const auto t0 = std::chrono::steady_clock::now();
    std::shared_ptr<PointCloudPreprocess> preprocess;
    SE3 T_base_lidar;
    bool online_mode = false;
    std::string base_link_frame;
    {
        UL lock(global_mutex_);
        if (preprocess_ == nullptr || lidar_loc_ == nullptr ||
            (!use_lio_sam_ && lio_ == nullptr) ||
            (use_lio_sam_ && lio_sam_ == nullptr)) {
            return;
        }
        preprocess = preprocess_;
        T_base_lidar = options_.T_base_lidar_;
        online_mode = options_.online_mode_;
        base_link_frame = base_link_frame_;
    }

    ++stat_ros_cloud_in_;
    ++stat_window_ros_cloud_in_;

    // 串行模式
    CloudPtr laser_cloud(new PointCloudType);
    preprocess->Process(cloud, laser_cloud);
    const auto t_preprocess = std::chrono::steady_clock::now();
    const double preprocess_ms =
        std::chrono::duration<double, std::milli>(t_preprocess - t0).count();
    stat_preprocess_total_ms_ += preprocess_ms;
    stat_window_preprocess_ms_ += preprocess_ms;
    if (laser_cloud->empty()) {
        ++stat_preprocess_fail_;
        ++stat_window_preprocess_fail_;
    } else {
        ++stat_preprocess_ok_;
        ++stat_window_preprocess_ok_;
    }
    // TODO(perf): Fuse preprocessing and T_base_lidar transform into one pass.
    CloudPtr cloud_base(new PointCloudType);
    pcl::transformPointCloud(*laser_cloud, *cloud_base, T_base_lidar.matrix().cast<float>());
    cloud_base->header = laser_cloud->header;
    cloud_base->header.frame_id = base_link_frame;
    laser_cloud = cloud_base;
    const auto t_transform = std::chrono::steady_clock::now();
    ++stat_base_transform_ok_;
    ++stat_window_base_transform_ok_;
    const double transform_ms =
        std::chrono::duration<double, std::milli>(t_transform - t_preprocess).count();
    stat_base_transform_total_ms_ += transform_ms;
    stat_window_base_transform_ms_ += transform_ms;

    if (online_mode) {
        lidar_odom_proc_cloud_.AddMessage(laser_cloud);
        ++stat_odom_queue_push_;
        ++stat_window_odom_queue_push_;
    } else {
        UL lock(global_mutex_);
        LidarOdomProcCloud(laser_cloud);
    }
    const auto t_end = std::chrono::steady_clock::now();
    const double add_queue_ms =
        std::chrono::duration<double, std::milli>(t_end - t_transform).count();
    const double process_lidar_ms =
        std::chrono::duration<double, std::milli>(t_end - t0).count();
    stat_add_odom_queue_total_ms_ += add_queue_ms;
    stat_window_add_odom_queue_ms_ += add_queue_ms;
    stat_process_lidar_total_ms_ += process_lidar_ms;
    stat_window_process_lidar_ms_ += process_lidar_ms;

    const double report_interval =
        std::chrono::duration<double>(t_end - stat_input_last_log_time_).count();
    if (report_interval >= 5.0) {
        const double window_count = static_cast<double>(stat_window_ros_cloud_in_);
        const double total_count = static_cast<double>(stat_ros_cloud_in_);
        LOG(INFO) << "[LIO_PIPELINE_INPUT] window_s=" << report_interval
                  << ", window_ros_cloud_in=" << stat_window_ros_cloud_in_
                  << ", window_preprocess_ok=" << stat_window_preprocess_ok_
                  << ", window_preprocess_fail=" << stat_window_preprocess_fail_
                  << ", window_base_transform_ok=" << stat_window_base_transform_ok_
                  << ", window_odom_queue_push=" << stat_window_odom_queue_push_
                  << ", window_preprocess_ms_avg=" << stat_window_preprocess_ms_ / window_count
                  << ", window_transform_ms_avg=" << stat_window_base_transform_ms_ / window_count
                  << ", window_add_queue_ms_avg=" << stat_window_add_odom_queue_ms_ / window_count
                  << ", window_total_ms_avg=" << stat_window_process_lidar_ms_ / window_count
                  << ", total_ros_cloud_in=" << stat_ros_cloud_in_
                  << ", total_preprocess_ok=" << stat_preprocess_ok_
                  << ", total_preprocess_fail=" << stat_preprocess_fail_
                  << ", total_base_transform_ok=" << stat_base_transform_ok_
                  << ", total_odom_queue_push=" << stat_odom_queue_push_
                  << ", total_preprocess_ms_avg=" << stat_preprocess_total_ms_ / total_count
                  << ", total_transform_ms_avg=" << stat_base_transform_total_ms_ / total_count
                  << ", total_add_queue_ms_avg=" << stat_add_odom_queue_total_ms_ / total_count
                  << ", total_input_ms_avg=" << stat_process_lidar_total_ms_ / total_count;
        stat_window_ros_cloud_in_ = 0;
        stat_window_preprocess_ok_ = 0;
        stat_window_preprocess_fail_ = 0;
        stat_window_base_transform_ok_ = 0;
        stat_window_odom_queue_push_ = 0;
        stat_window_preprocess_ms_ = 0.0;
        stat_window_base_transform_ms_ = 0.0;
        stat_window_add_odom_queue_ms_ = 0.0;
        stat_window_process_lidar_ms_ = 0.0;
        stat_input_last_log_time_ = t_end;
    }
}

void Localization::ProcessLivoxLidarMsg(const livox_ros_driver2::msg::CustomMsg::SharedPtr cloud) {
    UL lock(global_mutex_);
    if (lidar_loc_ == nullptr || (!use_lio_sam_ && lio_ == nullptr) ||
        (use_lio_sam_ && lio_sam_ == nullptr)) {
        return;
    }

    // 串行模式
    CloudPtr laser_cloud(new PointCloudType);
    preprocess_->Process(cloud, laser_cloud);
    // TODO(perf): Fuse preprocessing and T_base_lidar transform into one pass.
    CloudPtr cloud_base(new PointCloudType);
    pcl::transformPointCloud(*laser_cloud, *cloud_base, options_.T_base_lidar_.matrix().cast<float>());
    cloud_base->header = laser_cloud->header;
    cloud_base->header.frame_id = base_link_frame_;
    laser_cloud = cloud_base;

    if (options_.online_mode_) {
        lidar_odom_proc_cloud_.AddMessage(laser_cloud);
    } else {
        LidarOdomProcCloud(laser_cloud);
    }
}

void Localization::LidarOdomProcCloud(CloudPtr cloud) {
    ++stat_odom_proc_called_;
    auto log_odom_stats = [this]() {
        const auto now = std::chrono::steady_clock::now();
        const double report_interval =
            std::chrono::duration<double>(now - stat_odom_last_log_time_).count();
        if (report_interval < 5.0) {
            return;
        }
        LOG(INFO) << "[LO_RUN_FREQ] freq=" << stat_window_lo_run_ / report_interval
                  << " Hz, window=" << stat_window_lo_run_
                  << ", total=" << stat_lio_run_success_
                  << ", latest_lo_stamp=" << std::setprecision(14)
                  << stat_latest_lo_timestamp_;
        LOG(INFO) << "[LO_PUBLISH_FREQ] freq=" << stat_window_lo_publish_ / report_interval
                  << " Hz, window=" << stat_window_lo_publish_
                  << ", total=" << stat_lo_pose_published_
                  << ", latest_publish_stamp=" << std::setprecision(14)
                  << stat_latest_publish_timestamp_;
        LOG(INFO) << "[LIO_ODOM_PROC_STAT] called=" << stat_odom_proc_called_
                  << ", process_cloud_called=" << stat_lio_process_cloud_called_
                  << ", run_called=" << stat_lio_run_called_
                  << ", run_success=" << stat_lio_run_success_
                  << ", run_fail=" << stat_lio_run_fail_
                  << ", pose_published=" << stat_lo_pose_published_
                  << ", pose_not_ok=" << stat_lo_pose_not_ok_
                  << ", need_loc_scan_true=" << stat_need_loc_scan_true_
                  << ", need_loc_scan_false=" << stat_need_loc_scan_false_
                  << ", ndt_scan_sent=" << stat_ndt_scan_sent_;
        stat_window_lo_run_ = 0;
        stat_window_lo_publish_ = 0;
        stat_odom_last_log_time_ = now;
    };

    if ((!use_lio_sam_ && lio_ == nullptr) || (use_lio_sam_ && lio_sam_ == nullptr)) {
        log_odom_stats();
        return;
    }
    const bool need_loc_scan = options_.online_mode_ ? ShouldRunLidarLocThisFrame() : true;
    if (need_loc_scan) {
        ++stat_need_loc_scan_true_;
    } else {
        ++stat_need_loc_scan_false_;
    }
    NavState lo_state;
    CloudPtr scan = nullptr;
    Keyframe::Ptr kf = nullptr;
    /// NOTE: 在NCLT这种数据集中，lio内部是有缓存的，它拿到的点云不一定是最新时刻的点云
    if (use_lio_sam_) {
        ++stat_lio_process_cloud_called_;
        lio_sam_->ProcessPointCloud2(cloud);
        ++stat_lio_run_called_;
        if (!lio_sam_->Run(need_loc_scan)) {
            ++stat_lio_run_fail_;
            log_odom_stats();
            return;
        }
        ++stat_lio_run_success_;
        lo_state = lio_sam_->GetState();
        if (need_loc_scan) {
            scan = lio_sam_->GetProjCloudShared();
        }
    } else {
        ++stat_lio_process_cloud_called_;
        lio_->ProcessPointCloud2(cloud);
        ++stat_lio_run_called_;
        if (!lio_->Run()) {
            ++stat_lio_run_fail_;
            log_odom_stats();
            return;
        }
        ++stat_lio_run_success_;
        lo_state = lio_->GetState();
        if (need_loc_scan) {
            scan = lio_->GetProjCloud();
        }
    }

    ++stat_window_lo_run_;
    stat_latest_lo_timestamp_ = lo_state.timestamp_;
    if (!lo_state.pose_is_ok_) {
        ++stat_lo_pose_not_ok_;
    }

    {
        UL lock_lo(lo_pose_mutex_);
        if (lo_pose_queue_.empty() || lo_state.timestamp_ >= lo_pose_queue_.back().timestamp_) {
            lo_pose_queue_.emplace_back(lo_state);
            while (lo_pose_queue_.size() > 1000) {
                lo_pose_queue_.pop_front();
            }
        } else {
            LOG(WARNING) << "[MAP_ODOM_LO_QUEUE] LO timestamp went backward: "
                         << lo_state.timestamp_ - lo_pose_queue_.back().timestamp_;
        }
    }

    if (lidar_loc_) {
        lidar_loc_->ProcessLO(lo_state);
    }
    // pgo_->ProcessLidarOdom(lo_state);
    PublishHighFrequencyResultByLO(lo_state);

    if (!need_loc_scan) {
        log_odom_stats();
        return;
    }

    if (scan == nullptr || scan->empty()) {
        LOG(WARNING) << "[LIDAR_LOC_INPUT] empty scan from frontend, skip NDT";
        log_odom_stats();
        return;
    }

    // LOG(INFO) << "LO pose: " << std::setprecision(12) << lo_state.timestamp_ << " "
    //           << lo_state.GetPose().translation().transpose();

    /// 获得lio的关键帧

    if (options_.loc_on_kf_) {
        if (use_lio_sam_) {
           kf = lio_sam_->GetKeyframe();
        } else {
           kf = lio_->GetKeyframe();
        }
      
        if (!kf || kf == lio_kf_) {
            /// 关键帧未更新，那就只更新IMU状态

            // auto dr_state = lio_->GetState();
            // lidar_loc_->ProcessDR(dr_state);
            // pgo_->ProcessDR(dr_state);
            log_odom_stats();
            return;
        }

        // if (ui_) {
        //     ui_->UpdateKF(kf);
        // }

        lio_kf_ = kf;

        // auto scan = lio_->GetScanUndist();

        if (options_.online_mode_) {
            lidar_loc_proc_cloud_.AddMessage(scan);
        } else {
            LidarLocProcCloud(scan);
        }
        ++stat_ndt_scan_sent_;
    } else {
        // auto scan = cloud;   // 这个cloud应该差一个外参

        if (options_.online_mode_) {
            lidar_loc_proc_cloud_.AddMessage(scan);
        } else {
            LidarLocProcCloud(scan);
        }
        ++stat_ndt_scan_sent_;
    }
    log_odom_stats();
}

void Localization::LidarLocProcCloud(CloudPtr scan_undist) {
    lidar_loc_->ProcessCloud(scan_undist);

    auto res = lidar_loc_->GetLocalizationResult();
    UpdateMapOdomByLidarLocResult(res);
    // pgo_->ProcessLidarLoc(res);
    // UI 显示什么定位结果，RViz TF 就发布什么定位结果。
    /* 注释掉TF 直发
    if (tf_callback_ && res.lidar_loc_valid_) {
        tf_callback_(res.ToGeoMsg());
    }
    */
    if (ui_) {
        // Twi with Til, here pose means Twl, thus Til=I
        ui_->UpdateScan(scan_undist, res.pose_);
    }

    if (loc_state_callback_) {
        auto loc_state = std::make_shared<std_msgs::msg::Int32>();
        loc_state->data = static_cast<int>(res.status_);
        // LOG(INFO) << "loc_state: " << loc_state->data;
        loc_state_callback_(*loc_state);
    }

    // cv::Mat img(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
    // cv::imshow("img", img);
    // cv::waitKey(0);
}

void Localization::ProcessIMUMsg(IMUPtr imu) {
    bool use_lio_sam = false;
    std::shared_ptr<LaserMapping> lio;
    std::shared_ptr<LioSamMapping> lio_sam;
    {
        UL lock(global_mutex_);
        if (lidar_loc_ == nullptr || (!use_lio_sam_ && lio_ == nullptr) ||
            (use_lio_sam_ && lio_sam_ == nullptr)) {
            return;
        }
        use_lio_sam = use_lio_sam_;
        lio = lio_;
        lio_sam = lio_sam_;
    }

    double this_imu_time = imu->timestamp;
    if (last_imu_time_ > 0 && this_imu_time < last_imu_time_) {
        LOG(WARNING) << "IMU 时间异常：" << this_imu_time << ", last: " << last_imu_time_;
    }
    last_imu_time_ = this_imu_time;

    /// 里程计处理IMU
    if (use_lio_sam) {
        lio_sam->ProcessIMU(imu);
    } else {
        lio->ProcessIMU(imu);
    }

    return;

    /// 这里需要 IMU predict，否则没法process DR了
    #if 0
    auto dr_state = use_lio_sam_ ? lio_sam_->GetIMUState() : lio_->GetIMUState();

    if (!dr_state.pose_is_ok_) {
        return;
    }

    if (!dr_state.pos_.allFinite() ||
        !dr_state.vel_.allFinite() ||
        !dr_state.bg_.allFinite() ||
        !dr_state.grav_.allFinite() ||
        !dr_state.rot_.unit_quaternion().coeffs().allFinite()) {
        LOG(WARNING) << "[DR_STATE] invalid DR state, skip. t="
                    << std::setprecision(14) << dr_state.timestamp_;
        return;
    }

    // /// 停车判定
    // constexpr auto kThVbrbStill = 0.05;  // 0.08;
    // constexpr auto kThOmegaStill = 0.05;

    // if (dr_state.GetVel().norm() < kThVbrbStill && imu->angular_velocity.norm() < kThOmegaStill) {
    //     dr_state.is_parking_ = true;
    //     dr_state.SetVel(Vec3d::Zero());
    // }

    /// 如果没有odm, 用lio替代DR

    // LOG(INFO) << "dr state: " << std::setprecision(12) << dr_state.timestamp_ << " "
    //           << dr_state.GetPose().translation().transpose()
    //           << ", q=" << dr_state.GetPose().unit_quaternion().coeffs().transpose();

    lidar_loc_->ProcessDR(dr_state);
    #endif
}

// void Localization::ProcessOdomMsg(const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
//     UL lock(global_mutex_);
//
//     if (lidar_loc_ == nullptr || lio_ == nullptr || pgo_ == nullptr) {
//         return;
//     }
//     double this_odom_time = ToSec(odom_msg->header.stamp);
//     if (last_odom_time_ > 0 && this_odom_time < last_odom_time_) {
//         LOG(WARNING) << "Odom Time Abnormal:" << this_odom_time << ", last: " << last_odom_time_;
//     }
//     last_odom_time_ = this_odom_time;
//
//     lio_->ProcessOdometry(odom_msg);
//
//     if (!lio_->GetbOdomHF()) {
//         return;
//     }
//
//     auto dr_state = lio_->GetStateHF(mapping::FasterLioMapping::kHFStateOdomFiltered);
//
//     constexpr auto kThVbrbStill = 0.03;  // 0.08;
//     constexpr auto kThOmegaStill = 0.03;
//     if (dr_state.Getvwi().norm() < kThVbrbStill && dr_state.Getwii().norm() < kThOmegaStill) {
//         dr_state.is_parking_ = true;
//         dr_state.Setvwi(Vec3d::Zero());
//         dr_state.Setwii(Vec3d::Zero());
//     }
//
//     lidar_loc_->ProcessDR(dr_state);
//     pgo_->ProcessDR(dr_state);
// }

void Localization::Finish() {
    lidar_loc_->Finish();
    if (ui_) {
        ui_->Quit();
    }

    lidar_loc_proc_cloud_.Quit();
    lidar_odom_proc_cloud_.Quit();
}

void Localization::SetExternalPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
    UL lock(global_mutex_);
    /// 设置外部重定位的pose
    if (lidar_loc_) {
        lidar_loc_->SetInitialPose(SE3(q, t));
    }
}

void Localization::SetTFCallback(Localization::TFCallback&& callback) { tf_callback_ = callback; }

}  // namespace lightning::loc

