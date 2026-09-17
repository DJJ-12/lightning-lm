#pragma once

#include <atomic>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <yaml-cpp/yaml.h>

#include "common/eigen_types.h"
#include "common/keyframe.h"
#include "common/point_def.h"
#include "core/lio_sam/utility.hpp"

namespace lightning {

class LioSamMapping;
namespace ui { class PangolinWindow; }

namespace modules {

enum class SensorType {
    VELODYNE,
    OUSTER,
    LIVOX,
    HESAI,
};

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
    void Start();
    void Stop();
    void Reset();

    void ProcessIMU(const sensor_msgs::msg::Imu::SharedPtr& imu);
    void ProcessCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud);
    void ProcessCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud);

    MappingSystemResult GetResult();
    CloudPtr BuildCurrentMapInBaseFrame();
    nav_msgs::msg::Path BuildCurrentPath();
    bool ConsumeMappingUpdate();
    std::size_t GetKeyframeCount() const { return keyframe_count_.load(); }
    std::size_t GetKeyframeCloudBytes() const { return keyframe_cloud_bytes_.load(); }

   private:
    void LoadMappingParams(const YAML::Node& yaml);
    void ResetProjectionState();
    bool cachePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& laserCloudMsg);
    bool cachePointCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& laserCloudMsg);
    bool deskewInfo();
    void imuConverter(const sensor_msgs::msg::Imu& imu_in, sensor_msgs::msg::Imu& imu_out) const;
    void imuDeskewInfo();
    void findRotation(double pointTime, float* rotXCur, float* rotYCur, float* rotZCur);
    void findPosition(double relTime, float* posXCur, float* posYCur, float* posZCur);
    PointType deskewPoint(PointType* point, double relTime);
    void projectPointCloud();
    void cloudExtraction();
    void RunLioSamFrame(
        const std::shared_ptr<LioSamCloudInfo>& cloud_info);
    void HandleProcessedKeyframe(const Keyframe::Ptr& kf);
    rclcpp::Logger get_logger() const {
        return rclcpp::get_logger("mapping_system");
    }

    MappingSystemOptions options_;
    bool running_ = false;
    std::string base_link_frame_ = "base_link";
    SE3 T_base_lidar_ = SE3();

    SensorType sensor_ = SensorType::VELODYNE;
    int N_SCAN_ = 16;
    int Horizon_SCAN_ = 1800;
    int downsampleRate_ = 1;
    int point_filter_num_ = 1;
    double lidarMinRange_ = 1.0;
    double lidarMaxRange_ = 1000.0;
    Mat3d extRot = Mat3d::Identity();
    Quatd extQRPY = Quatd::Identity();
    bool useImuAccelRollPitchInitialization = false;
    double imuRollInit = 0.0;
    double imuPitchInit = 0.0;
    double imuYawInit = 0.0;

    CloudPtr laserCloudIn_{new PointCloudType()};
    CloudPtr fullCloud_{new PointCloudType()};
    CloudPtr extractedCloud_{new PointCloudType()};
    std::vector<float> rangeMat_;
    std::vector<int> columnIdnCountVec_;

    std::deque<sensor_msgs::msg::PointCloud2> cloudQueue_;
    std::deque<livox_ros_driver2::msg::CustomMsg> livoxCloudQueue_;
    std::deque<sensor_msgs::msg::Imu> imuQueue_;
    double last_timestamp_imu_ = -1.0;
    double last_timestamp_lidar_ = -1.0;

    std::vector<double> imuTime_;
    std::vector<double> imuRotX_;
    std::vector<double> imuRotY_;
    std::vector<double> imuRotZ_;
    int imuPointerCur_ = 0;
    bool firstPointFlag_ = true;
    Eigen::Affine3f transStartInverse_ = Eigen::Affine3f::Identity();
    int ringFlag_ = 0;
    int deskewFlag_ = 0;

    std::shared_ptr<LioSamCloudInfo> cloudInfo_{
        std::make_shared<LioSamCloudInfo>()};
    double timeScanCur_ = 0.0;
    double timeScanEnd_ = 0.0;
    double timeScanHeader_ = 0.0;
    std_msgs::msg::Header cloudHeader_;

    std::shared_ptr<LioSamMapping> lio_sam_;
    std::shared_ptr<ui::PangolinWindow> ui_;

    mutable std::mutex mtx_;
    Keyframe::Ptr cur_kf_;
    bool mapping_update_pending_ = false;
    std::atomic<std::size_t> keyframe_count_{0};
    std::atomic<std::size_t> keyframe_cloud_bytes_{0};
};

}  // namespace modules
}  // namespace lightning
