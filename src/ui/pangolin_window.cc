#include "ui/pangolin_window_impl.h"

#include <glog/logging.h>

#include <thread>

namespace lightning::ui {

PangolinWindow::PangolinWindow() = default;

PangolinWindow::~PangolinWindow() {
    LOG(INFO) << "[UI析构修复][PangolinWindow][01] 进入析构"
              << ", this=" << this
              << ", thread_id=" << std::this_thread::get_id();
    Quit();
    LOG(INFO) << "[UI析构修复][PangolinWindow][02] 析构完成";
}

bool PangolinWindow::Init() {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (render_thread_.joinable()) {
        LOG(ERROR) << "[UI初始化] 渲染线程已经存在，拒绝重复初始化";
        return false;
    }

    auto impl = std::make_shared<PangolinWindowImpl>();
    impl->cloud_global_need_update_.store(false);
    impl->kf_result_need_update_.store(false);
    impl->rtk_position_need_update_.store(false);
    impl->lidarloc_need_update_.store(false);
    impl->current_scan_need_update_.store(false);

    if (!impl->Init()) {
        LOG(ERROR) << "[UI初始化] PangolinWindowImpl::Init 失败";
        return false;
    }

    closed_.store(false);
    impl_ = impl;
    render_thread_ = std::thread([this, render_impl = std::move(impl)]() mutable {
        LOG(INFO) << "[UI析构修复][渲染线程][01] 渲染线程启动"
                  << ", thread_id=" << std::this_thread::get_id()
                  << ", impl=" << render_impl.get();
        const std::string window_name = render_impl->GetWindowName();
        render_impl->Render();

        // Render 返回时仍然持有当前 OpenGL context。
        // 必须先在本线程销毁 Plotter、GlText、GlBuffer 等资源，之后才能销毁 Pangolin context。
        LOG(INFO) << "[UI析构修复][渲染线程][02] 渲染循环结束，准备在当前OpenGL上下文中析构全部UI资源"
                  << ", impl=" << render_impl.get();
        {
            // 与所有 Update 接口使用同一把生命周期锁，等待已经开始的UI更新结束，
            // 并阻止新的更新在资源析构过程中取得 Impl 强引用。
            std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
            impl_.reset();
            render_impl.reset();
        }
        LOG(INFO) << "[UI析构修复][渲染线程][03] 全部UI资源已经在渲染线程析构";

        // RemoveCurrent 只会解除当前线程与context的绑定，不会从 Pangolin 的全局context表注销窗口名。
        // 必须调用 DestroyWindow，才能允许后续任务再次使用相同的窗口名创建UI。
        LOG(INFO) << "[UI析构修复][渲染线程][04] 准备销毁并注销Pangolin上下文"
                  << ", window_name=" << window_name;
        pangolin::DestroyWindow(window_name);
        LOG(INFO) << "[UI析构修复][渲染线程][05] Pangolin上下文已经销毁并注销"
                  << ", window_name=" << window_name;
    });
    return true;
}

void PangolinWindow::Reset(const std::vector<Keyframe::Ptr>& keyframes) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    auto impl = impl_.lock();
    if (impl) {
        impl->Reset(keyframes);
    }
}

void PangolinWindow::Quit() {
    LOG(INFO) << "[UI析构修复][PangolinWindow::Quit][01] 进入"
              << ", this=" << this
              << ", thread_id=" << std::this_thread::get_id();
    if (closed_.exchange(true)) {
        LOG(INFO) << "[UI析构修复][PangolinWindow::Quit][02] 已经关闭，直接返回";
        return;
    }

    {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        auto impl = impl_.lock();
        if (impl) {
            impl->exit_flag_.store(true);
            LOG(INFO) << "[UI析构修复][PangolinWindow::Quit][03] 已设置渲染线程退出标志"
                      << ", impl=" << impl.get();
        }

        // 在 join 之前清除外层弱引用。离开该作用域后，渲染线程持有唯一强引用。
        // 这样 Render 返回后的 render_impl.reset() 会在渲染线程真正触发 Impl 析构。
        impl_.reset();
    }

    if (render_thread_.joinable()) {
        LOG(INFO) << "[UI析构修复][PangolinWindow::Quit][04] 准备等待渲染线程"
                  << ", render_thread_id=" << render_thread_.get_id();
        if (render_thread_.get_id() == std::this_thread::get_id()) {
            LOG(ERROR) << "[UI析构修复][PangolinWindow::Quit][05] 禁止渲染线程等待自身，执行detach";
            render_thread_.detach();
            return;
        }
        render_thread_.join();
        LOG(INFO) << "[UI析构修复][PangolinWindow::Quit][05] 渲染线程已经join，UI资源与上下文均已安全释放";
    }
}

void PangolinWindow::UpdatePointCloudGlobal(const std::map<int, CloudPtr>& cloud) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    auto impl = impl_.lock();
    if (!impl) {
        return;
    }
    std::lock_guard<std::mutex> lock(impl->mtx_map_cloud_);
    impl->cloud_global_map_ = cloud;
    impl->cloud_global_need_update_.store(true);
}

void PangolinWindow::UpdatePointCloudDynamic(const std::map<int, CloudPtr>& cloud) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    auto impl = impl_.lock();
    if (!impl) {
        return;
    }
    std::unique_lock<std::mutex> lock(impl->mtx_map_cloud_);
    impl->cloud_dynamic_map_.clear();  // need deep copy

    for (auto& cp : cloud) {
        CloudPtr c(new PointCloudType());
        *c = *cp.second;
        impl->cloud_dynamic_map_.emplace(cp.first, c);
    }

    for (auto iter = impl->cloud_dynamic_map_.begin(); iter != impl->cloud_dynamic_map_.end();) {
        if (cloud.find(iter->first) == cloud.end()) {
            iter = impl->cloud_dynamic_map_.erase(iter);
        } else {
            iter++;
        }
    }

    impl->cloud_dynamic_need_update_.store(true);
}

void PangolinWindow::UpdateNavState(const NavState& state) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    auto impl = impl_.lock();
    if (!impl) {
        return;
    }
    std::unique_lock<std::mutex> lock_lio_res(impl->mtx_nav_state_);
    // Do not overwrite an unrendered estimator state. Offline bags can feed
    // observations much faster than the screen refresh rate, so retain every
    // final EKF state until the render thread appends it to the trajectory.
    impl->pending_nav_states_.push_back(state);
    impl->kf_result_need_update_.store(true);
}

void PangolinWindow::UpdateRtkPosition(
    const Eigen::Vector2d& position_map) {
    if (!position_map.allFinite()) return;
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    auto impl = impl_.lock();
    if (!impl) return;

    std::lock_guard<std::mutex> lock(impl->mtx_rtk_position_);
    impl->pending_rtk_positions_.emplace_back(
        position_map.x(), position_map.y(), 0.0);
    impl->rtk_position_need_update_.store(true);
}

void PangolinWindow::UpdateRecentPose(const SE3& pose) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    auto impl = impl_.lock();
    if (!impl) {
        return;
    }
    std::lock_guard<std::mutex> lock(impl->mtx_nav_state_);
    impl->newest_frontend_pose_ = pose;
}

void PangolinWindow::UpdatePredictPose(const SE3& pose) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    auto impl = impl_.lock();
    if (!impl) {
        return;
    }
    UL lock(impl->mtx_nav_state_);
    impl->predicted_pose_ = pose;
}

void PangolinWindow::UpdateScan(CloudPtr cloud, const SE3& pose) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    auto impl = impl_.lock();
    if (!impl) {
        return;
    }
    std::lock_guard<std::mutex> lock(impl->mtx_current_scan_);
    std::lock_guard<std::mutex> lock2(impl->mtx_nav_state_);

    *impl->current_scan_ = *cloud;  // need deep copy
    impl->current_scan_pose_ = pose;
    impl->current_scan_need_update_.store(true);
}

void PangolinWindow::UpdateKF(std::shared_ptr<Keyframe> kf) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    auto impl = impl_.lock();
    if (!impl) {
        return;
    }
    UL lock(impl->mtx_current_scan_);
    impl->all_keyframes_.emplace_back(kf);
}

void PangolinWindow::SetCurrentScanSize(int current_scan_size) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    auto impl = impl_.lock();
    if (impl) {
        impl->max_size_of_current_scan_ = current_scan_size;
    }
}

void PangolinWindow::SetTImuLidar(const SE3& T_imu_lidar) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    auto impl = impl_.lock();
    if (impl) {
        impl->T_imu_lidar_ = T_imu_lidar;
    }
}

bool PangolinWindow::ShouldQuit() {
    if (closed_.load()) {
        return true;
    }
    return pangolin::ShouldQuit();
}

}  // namespace lightning::ui
