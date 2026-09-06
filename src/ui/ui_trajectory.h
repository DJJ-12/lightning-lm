#pragma once

#include "common/eigen_types.h"

#include <pangolin/gl/glvbo.h>

namespace lightning::ui {

/// UI中的轨迹绘制
class UiTrajectory {
   public:
    UiTrajectory(const Vec3f& color, float line_width = 5.0f)
        : color_(color), line_width_(line_width) {
        pos_.reserve(max_size_);
    }

    /// 增加一个轨迹点到opengl缓冲区
    void AddPt(const SE3& pose);

    /// 渲染此轨迹
    void Render();

    void Clear() {
        pos_.clear();
        pos_.reserve(max_size_);
        vbo_.Free();
    }

    Vec3f At(const uint64_t idx) const { return pos_.at(idx); }

   private:
    int max_size_ = 1e6;                               // 记录的最大点数
    std::vector<Eigen::Vector3f> pos_;                 // 轨迹记录数据
    Eigen::Vector3f color_ = Eigen::Vector3f::Zero();  // 轨迹颜色显示
    float line_width_ = 5.0f;
    pangolin::GlBuffer vbo_;                           // 显存顶点信息
};

}  // namespace lightning::ui
