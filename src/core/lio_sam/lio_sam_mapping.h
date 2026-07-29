#ifndef LIGHTNING_LIO_SAM_MAPPING_H
#define LIGHTNING_LIO_SAM_MAPPING_H

#include <cstdint>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "common/keyframe.h"
#include "common/nav_state.h"
#include "common/options.h"

struct LioSamCloudInfo;

class FeatureExtractor;
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
    bool Run(::LioSamCloudInfo& cloud_info);

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

    bool LoadParamsFromYAML(const std::string& yaml_path);
    bool MakeLightningKeyframeIfNeeded();
    void SyncLightningKeyframePoses();
    bool IsOnlineMapping() const {
        return options_.mapping_mode_ == MappingRuntimeMode::ONLINE_MAPPING;
    }

    Options options_;
    rclcpp::NodeOptions node_options_;

    std::unique_ptr<::FeatureExtractor> feature_extraction_;
    std::unique_ptr<::mapOptimization> map_optimization_;

    std::string current_frame_id_;
    
    CloudPtr scan_undistort_{new PointCloudType()};
    CloudPtr recent_cloud_{new PointCloudType()};
    NavState state_;

    std::vector<Keyframe::Ptr> all_keyframes_;
    Keyframe::Ptr last_kf_ = nullptr;
    int kf_id_ = 0;
    size_t map_keyframe_count_ = 0;

    std::uint64_t diagnostic_map_optimization_executed_ = 0;
    std::uint64_t diagnostic_map_optimization_skipped_ = 0;

    std::shared_ptr<ui::PangolinWindow> ui_ = nullptr;
};

}  // namespace lightning

#endif  // LIGHTNING_LIO_SAM_MAPPING_H
