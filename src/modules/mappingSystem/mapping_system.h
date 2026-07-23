#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/keyframe.h"
#include "common/point_def.h"

namespace lightning {
class LaserMapping;
class LioSamMapping;
class PointCloudPreprocess;
namespace ui { class PangolinWindow; }
namespace modules {

struct MappingSystemOptions {
    bool online_input = false;
    bool with_ui = false;
};

struct MappingSystemResult {
    bool valid = false;
    std::vector<Keyframe::Ptr> keyframes;
    CloudPtr global_map;
    SE3 T_base_lidar = SE3();
    bool global_map_is_lidar_frame = false;
};

class MappingSystem {
   public:
    MappingSystem() = default;
    ~MappingSystem();

    bool Init(const std::string& yaml_path, const MappingSystemOptions& options);
    bool Start();
    void Stop();
    void Reset();

    void ProcessIMU(const sensor_msgs::msg::Imu::SharedPtr& imu);
    void ProcessIMU(const IMUPtr& imu);
    void ProcessCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud);
    void ProcessCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud);

    MappingSystemResult GetResult();
    CloudPtr BuildCurrentMapInBaseFrame();
    nav_msgs::msg::Path BuildCurrentPath();
    bool ConsumeMappingUpdate();

   private:
    bool BuildInputCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud, CloudPtr& input);
    bool BuildInputCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud, CloudPtr& input);
    void HandleProcessedKeyframe(const Keyframe::Ptr& kf);

    MappingSystemOptions options_;
    bool running_ = false;
    bool use_lio_sam_ = false;
    std::string yaml_path_;
    std::string base_link_frame_ = "base_link";
    SE3 T_base_lidar_ = SE3();

    std::shared_ptr<PointCloudPreprocess> preprocess_;
    std::shared_ptr<LioSamMapping> lio_sam_;
    std::shared_ptr<LaserMapping> lio_;
    std::shared_ptr<ui::PangolinWindow> ui_;
    Keyframe::Ptr cur_kf_;
    bool mapping_update_pending_ = false;
    std::size_t keyframe_count_ = 0;
    std::size_t keyframe_cloud_bytes_ = 0;
};

}  // namespace modules
}  // namespace lightning
