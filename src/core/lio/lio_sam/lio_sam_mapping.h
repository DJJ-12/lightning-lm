#ifndef LIGHTNING_LIO_SAM_MAPPING_H
#define LIGHTNING_LIO_SAM_MAPPING_H

#include <deque>
#include <cstdint>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <rclcpp/rclcpp.hpp>

#include "common/keyframe.h"
#include "common/imu.h"
#include "common/nav_state.h"
#include "common/options.h"

struct LioSamCloudInfo;

class DeskewFeatureExtractor;
class mapOptimization;

namespace lightning {

namespace ui {
class PangolinWindow;
}

class LioSamMapping {
   public:
    enum class MappingRuntimeMode {
        OFFLINE_MAPPING = 0,
        ONLINE_MAPPING = 1
    };

    struct Options {
        MappingRuntimeMode mapping_mode_ = MappingRuntimeMode::OFFLINE_MAPPING;
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LioSamMapping();
    explicit LioSamMapping(Options options);
    ~LioSamMapping();

    bool Init(const std::string& config_yaml);
    bool Run();

    void ProcessIMU(const IMUPtr& imu);
    void ProcessPointCloud2(CloudPtr cloud);

    void SetUI(std::shared_ptr<ui::PangolinWindow> ui) { ui_ = std::move(ui); }

    Keyframe::Ptr GetKeyframe() const { return last_kf_; }
    std::vector<Keyframe::Ptr> GetAllKeyframes() const { return all_keyframes_; }
    //������λģ�����Ҫ��
     NavState GetState() const { return state_; }

    CloudPtr GetScanUndist() const {
        if (!recent_cloud_) {
            return nullptr;
        }
        return CloudPtr(new PointCloudType(*recent_cloud_));
    }
    CloudPtr GetGlobalMap(bool use_lio_pose, bool use_voxel = true, float res = 0.1);
    void SyncOptimizedKeyframePoses();

   private:
    struct SyncedPackage {
        CloudPtr cloud;
        std::vector<sensor_msgs::msg::Imu> imus;
        double lidar_begin_time = 0.0;
        double lidar_end_time = 0.0;
        std::string frame_id;
    };

    bool LoadParamsFromYAML(const std::string& yaml_path);
    bool SyncPackages();
    bool MakeLightningKeyframeIfNeeded();
    void SyncLightningKeyframePoses();
    bool IsOnlineMapping() const {
        return options_.mapping_mode_ == MappingRuntimeMode::ONLINE_MAPPING;
    }

    Options options_;
    rclcpp::NodeOptions node_options_;
    bool owns_rclcpp_context_ = false;

    std::unique_ptr<::DeskewFeatureExtractor> deskew_feature_extractor_;
    std::unique_ptr<::mapOptimization> map_optimization_;
    std::unique_ptr<::LioSamCloudInfo> frontend_cloud_info_;

    mutable std::mutex mtx_buffer_;
    std::deque<CloudPtr> lidar_buffer_;
    std::deque<double> time_buffer_;
    std::deque<double> scan_duration_buffer_;
    std::deque<std::string> frame_id_buffer_;
    std::deque<sensor_msgs::msg::Imu> imu_buffer_;

    SyncedPackage measures_;
    CloudPtr scan_undistort_{new PointCloudType()};
    CloudPtr recent_cloud_{new PointCloudType()};
    NavState state_;

    double last_timestamp_imu_ = -1.0;
    double last_timestamp_lidar_ = -1.0;
    double lidar_begin_time_ = 0.0;
    double lidar_end_time_ = 0.0;
    bool lidar_pushed_ = false;

    std::vector<Keyframe::Ptr> all_keyframes_;
    Keyframe::Ptr last_kf_ = nullptr;
    int kf_id_ = 0;
    size_t map_keyframe_count_ = 0;

    std::uint64_t diagnostic_cloud_inputs_ = 0;
    std::uint64_t diagnostic_synced_packages_ = 0;
    std::uint64_t diagnostic_map_optimization_executed_ = 0;
    std::uint64_t diagnostic_map_optimization_skipped_ = 0;
    std::uint64_t diagnostic_time_log_count_ = 0;
    double diagnostic_previous_cloud_stamp_ = 0.0;
    double last_scan_duration_ = 0.1;

    std::shared_ptr<ui::PangolinWindow> ui_ = nullptr;
};

}  // namespace lightning

#endif  // LIGHTNING_LIO_SAM_MAPPING_H
