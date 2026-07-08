#ifndef __GLOBAL_LOCALIZER_H__
#define __GLOBAL_LOCALIZER_H__ 

#include <string>

#include <Eigen/Core>
#include "GlobalLocalizer/ndt_omp/multigrid_ndt_omp.h"
#include "MapLoader/MapLoader.hpp"

namespace robot_localizer
{

// 定位质量评估阈值配置
struct QualityThresholds {
    double excellent_threshold = 4.0;           // 优秀质量阈值
    double good_threshold = 2.5;                // 良好质量阈值
    double fair_threshold = 1.5;                // 一般质量阈值
    int excellent_iteration_threshold = 20;     // 优秀质量的迭代次数阈值
};

// 定位结果质量结构体
struct LocalizationQuality {
    double transform_probability = 0.0;           // 变换概率
    double nearest_voxel_likelihood = 0.0;        // 最近体素似然度
    int iteration_num = 0;                        // 迭代次数
    bool is_reliable = false;                     // 是否可靠
    std::string quality_level = "unknown";       // 质量等级：excellent/good/fair/poor

    // 评估定位质量（使用默认阈值）
    void evaluate() {
        QualityThresholds thresholds;
        evaluate(thresholds);
    }

    // 评估定位质量（使用自定义阈值）
    void evaluate(const QualityThresholds& thresholds) {
        if (transform_probability > thresholds.excellent_threshold && 
            iteration_num < thresholds.excellent_iteration_threshold) {
            quality_level = "excellent";
            is_reliable = true;
        } else if (transform_probability > thresholds.good_threshold) {
            quality_level = "good";
            is_reliable = true;
        } else if (transform_probability > thresholds.fair_threshold) {
            quality_level = "fair";
            is_reliable = true;
        } else {
            quality_level = "poor";
            is_reliable = false;
        }
    }
};

class Localizer
{
private:
    using PointSource = pcl::PointXYZ;
    using PointTarget = pcl::PointXYZ;
    using NormalDistributionsTransform = pclomp::MultiGridNormalDistributionsTransform<PointSource, PointTarget>;
    std::shared_ptr<NormalDistributionsTransform> ndt_ptr_;
    std::shared_ptr<NormalDistributionsTransform> ndt_rough_ptr_;
    
    
    std::string pcd_metadata_path_; 
    std::string pcd_directory_;
    MapManager::MapLoader map_loader_;
    QualityThresholds quality_thresholds_;  // 定位质量评估阈值

    pclomp::NdtResult AlignPose(const Eigen::Matrix4d &initial_pose_with_cov);

    Eigen::Matrix4d last_last_pose_ = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d last_pose_ = Eigen::Matrix4d::Identity();
    
public:
    Localizer();
    ~Localizer(){};

    // 设置定位质量评估阈值
    void SetQualityThresholds(const QualityThresholds& thresholds) {
        quality_thresholds_ = thresholds;
    }

    void GetInitPose(const Eigen::Matrix4d& init_guess, Eigen::Matrix4d& align_pose, const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud);

    bool GetInitPose(const Eigen::Matrix4d& init_guess, Eigen::Matrix4d& align_pose, const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud, LocalizationQuality& quality);

    void ResetLocalizationState();

    // 原有接口（保持兼容性）
    bool RegisterFrame(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud, Eigen::Matrix4d& align_pose);
    
    // 新增接口（带定位质量输出）
    bool RegisterFrame(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud, Eigen::Matrix4d& align_pose, LocalizationQuality& quality);

    void SetStaticMap(std::string pcd_metadata_path, std::string pcd_directory)
    {
        pcd_metadata_path_ = pcd_metadata_path;
        pcd_directory_ = pcd_directory;
        map_loader_.Initialize(pcd_metadata_path_, pcd_directory_);
    }

    void UpdateMap(const Eigen::Vector2d &pose);
    pcl::PointCloud<pcl::PointXYZ>::Ptr GetLocateMap(){ return map_loader_.GetMapCloud();}
    void MapReset() {map_loader_.reset();}
};
}

#endif
