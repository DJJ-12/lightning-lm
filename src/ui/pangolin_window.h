#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "common/eigen_types.h"
#include "common/keyframe.h"
#include "common/nav_state.h"
#include "common/point_def.h"

namespace lightning::ui {

class PangolinWindowImpl;

/**
 * @note 此类本身不直接涉及任何opengl和pangolin操作，应当放到PangolinWindowImpl中
 */
class PangolinWindow {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PangolinWindow();
    ~PangolinWindow();

    /// @brief 初始化窗口，后台启动render线程。
    /// @note 与opengl/pangolin无关的初始化，尽量放到此函数体中;
    ///       opengl/pangolin相关的内容，尽量放到PangolinWindowImpl::Init中。
    bool Init();

    void Reset(const std::vector<Keyframe::Ptr>& keyframes);

    /// 更新激光地图点云，在激光定位中的地图发生改变时，由fusion调用
    void UpdatePointCloudGlobal(const std::map<int, CloudPtr>& cloud);

    /// 更新动态图层点云
    void UpdatePointCloudDynamic(const std::map<int, CloudPtr>& cloud);

    /// 更新kalman滤波器状态
    void UpdateNavState(const NavState& state);

    /// 增加一帧已经转换到map坐标系的原始RTK位置观测
    void UpdateRtkPosition(const Eigen::Vector2d& position_map);

    /// 增加一帧原始NDT位置观测
    void UpdateNdtPosition(const Eigen::Vector2d& position_map);

    /// 更新最新的pose
    void UpdateRecentPose(const SE3& pose);

    void UpdatePredictPose(const SE3& pose);

    /// 更新一次scan和它对应的Pose
    void UpdateScan(CloudPtr cloud, const SE3& pose);

    void UpdateKF(std::shared_ptr<Keyframe> kf);

    /// 等待显示线程结束，并在显示线程持有OpenGL上下文时释放全部UI资源
    void Quit();

    /// 用户是否已经退出UI
    bool ShouldQuit();

    /// 设置IMU到雷达的外参
    void SetTImuLidar(const SE3& T_imu_lidar);

    /// 设置需要保留多少个扫描数据
    void SetCurrentScanSize(int current_scan_size);

   private:
    // 渲染线程是 PangolinWindowImpl 的唯一强所有者。
    // 外部更新接口只保存弱引用，保证 Impl 最终一定在渲染线程中析构。
    std::weak_ptr<PangolinWindowImpl> impl_;
    std::thread render_thread_;
    std::mutex lifecycle_mutex_;
    std::atomic_bool closed_{false};
};
}  // namespace lightning::ui
