#include "GlobalLocalizer/GlobalLocalizer.h"
#include "GlobalLocalizer/tree_structured_parzen_estimator.hpp"
#include <glog/logging.h>
#include <pcl/filters/voxel_grid.h>
#include <algorithm>
#include <chrono>
#include "GlobalLocalizer.h"

namespace robot_localizer
{
Eigen::Vector3d getRPYFromEigenMatrix(const Eigen::Matrix3d& R) {
    Eigen::Vector3d rpy;
    double& roll = rpy.x();   // roll对应x分量
    double& pitch = rpy.y();  // pitch对应y分量
    double& yaw = rpy.z();    // yaw对应z分量

    // 计算Pitch角（绕Y轴）
    pitch = std::asin(-R(2, 0));

    // 处理万向锁（cos(pitch)接近0时）
    const double cos_pitch = std::cos(pitch);
    if (std::fabs(cos_pitch) > 1e-6) {
        // 非万向锁：计算roll和yaw
        roll = std::atan2(R(2, 1), R(2, 2));
        yaw = std::atan2(R(1, 0), R(0, 0));
    } else {
        // 万向锁情况（pitch≈±90°）：固定yaw=0，计算roll
        yaw = 0.0;
        roll = std::atan2(-R(0, 1), R(1, 1));
    }

    return rpy;
}

pclomp::NdtResult Localizer::AlignPose(const Eigen::Matrix4d &initial_pose_with_cov)
{
    // 获取初始位姿的RPY角度。
    // 2.5D 重定位约束：粗搜索阶段只允许 x / y / yaw 撒种子；
    // z / roll / pitch 必须固定为输入初值，避免地面机器人产生几十米高度或大姿态角的无效候选。
    const auto base_rpy = getRPYFromEigenMatrix(initial_pose_with_cov.block<3, 3>(0, 0));
    const double fixed_z = initial_pose_with_cov(2, 3);
    const double fixed_roll = base_rpy.x();
    const double fixed_pitch = base_rpy.y();
    const double fixed_yaw = base_rpy.z();

    constexpr double kSearchStddevX = 3.0;
    constexpr double kSearchStddevY = 3.0;
    constexpr double kSearchStddevYaw = 0.17453;  // 约 10 deg

    // 定义采样均值和标准差：只对 x / y / yaw 采样；z / roll / pitch 固定。
    const std::vector<double> sample_mean{
        initial_pose_with_cov(0,3), // trans_x
        initial_pose_with_cov(1,3), // trans_y
        fixed_z,                    // trans_z，2.5D 固定
        fixed_roll,                 // roll，2.5D 固定
        fixed_pitch,                // pitch，2.5D 固定
        fixed_yaw,                  // yaw
    };
    const std::vector<double> sample_stddev{
        kSearchStddevX,
        kSearchStddevY,
        0.0,                        // trans_z 不撒种子
        0.0,                        // roll 不撒种子
        0.0,                        // pitch 不撒种子
        kSearchStddevYaw,
    };

    // 2.5D 粗搜索：优化 x / y / yaw；z / roll / pitch 由下面的固定约束强制锁定。
    TreeStructuredParzenEstimator tpe(TreeStructuredParzenEstimator::Direction::MAXIMIZE, 100, sample_mean, sample_stddev);

    auto output_cloud = std::make_shared<pcl::PointCloud<PointSource> >();
    std::vector<pclomp::NdtResult> result_array;

    for (int64_t i = 0; i < 200; i++)
    {
        // 第 0 个候选直接使用用户输入初值；后续候选再围绕 x / y / yaw 做 2.5D 搜索。
        TreeStructuredParzenEstimator::Input input = (i == 0) ? sample_mean : tpe.get_next_input();

        // 双保险：即使 TPE 内部以后被改动，这里仍强制 2.5D 候选种子。
        input[TreeStructuredParzenEstimator::TRANS_Z] = fixed_z;
        input[TreeStructuredParzenEstimator::ANGLE_X] = fixed_roll;
        input[TreeStructuredParzenEstimator::ANGLE_Y] = fixed_pitch;

        Eigen::Matrix3d rot = Eigen::AngleAxisd(input[TreeStructuredParzenEstimator::ANGLE_Z], Eigen::Vector3d::UnitZ())
                            * Eigen::AngleAxisd(input[TreeStructuredParzenEstimator::ANGLE_Y], Eigen::Vector3d::UnitY())
                            * Eigen::AngleAxisd(input[TreeStructuredParzenEstimator::ANGLE_X], Eigen::Vector3d::UnitX()).matrix();
        Eigen::Matrix4d init_pose_matrix = Eigen::Matrix4d::Identity();
        init_pose_matrix.block<3, 3>(0, 0) = rot;
        init_pose_matrix.block<3, 1>(0, 3) = Eigen::Vector3d(input[TreeStructuredParzenEstimator::TRANS_X],
                                                            input[TreeStructuredParzenEstimator::TRANS_Y],
                                                            input[TreeStructuredParzenEstimator::TRANS_Z]);

        LOG(INFO) << "Initial Pose: " <<  "x: " << input[0] << ", y: " << input[1] << ", z: " << input[2] << ", roll: " << input[3] << ", pitch: " << input[4] << ", yaw: " << input[5] << std::endl;
        
        // 执行NDT配准
        const Eigen::Matrix4f initial_pose_matrix = init_pose_matrix.cast<float>();
        ndt_rough_ptr_->align(*output_cloud, initial_pose_matrix);
        const pclomp::NdtResult ndt_result = ndt_rough_ptr_->getResult();

        const auto align_pose_rpy = getRPYFromEigenMatrix(ndt_result.pose.block<3, 3>(0, 0).cast<double>());
        std::cout << "aligned pose: " << "x: " << ndt_result.pose(0,3) << ", y: " << ndt_result.pose(1,3) << ", z: " << ndt_result.pose(2,3) << ", roll: " << align_pose_rpy.x() << ", pitch: " << align_pose_rpy.y() << ", yaw: " << align_pose_rpy.z() << std::endl;
        // 构建TPE结果
        TreeStructuredParzenEstimator::Input result(6);
        result[0] = ndt_result.pose(0,3);
        result[1] = ndt_result.pose(1,3);
        result[2] = ndt_result.pose(2,3);
        result[3] = align_pose_rpy.x();
        result[4] = align_pose_rpy.y();
        result[5] = align_pose_rpy.z();

        // 添加试验结果到TPE
        tpe.add_trial(TreeStructuredParzenEstimator::Trial{result, ndt_result.transform_probability});
        result_array.push_back(ndt_result);
    }

    // 找到最佳粒子
    auto best_particle_ptr = std::max_element(std::begin(result_array), std::end(result_array),
                                              [](const pclomp::NdtResult &lhs, const pclomp::NdtResult &rhs)
                                              { return lhs.nearest_voxel_transformation_likelihood < rhs.nearest_voxel_transformation_likelihood; });

    LOG(INFO) << "rough NDT alignment completed - best particle score: " << best_particle_ptr->nearest_voxel_transformation_likelihood;
    
    const auto best_rpy = getRPYFromEigenMatrix(best_particle_ptr->pose.block<3, 3>(0, 0).cast<double>());
    LOG(INFO) << "[INIT_2_5D] rough best pose: x: "
            << best_particle_ptr->pose(0, 3)
            << ", y: " << best_particle_ptr->pose(1, 3)
            << ", z: " << best_particle_ptr->pose(2, 3)
            << ", roll: " << best_rpy.x()
            << ", pitch: " << best_rpy.y()
            << ", yaw: " << best_rpy.z()
            << ", NVTL: " << best_particle_ptr->nearest_voxel_transformation_likelihood
            << ", TP: " << best_particle_ptr->transform_probability;

    return *best_particle_ptr;
}

Localizer::Localizer()
{
    ndt_ptr_ = std::make_shared<NormalDistributionsTransform>();
    pclomp::NdtParams ndt{};
    ndt.num_threads = 20;
    ndt.step_size = 0.1;
    ndt.trans_epsilon = 0.005;
    ndt.max_iterations = 30;
    ndt.resolution = 1.0;
    ndt_ptr_->setParams(ndt);

    ndt_rough_ptr_ = std::make_shared<NormalDistributionsTransform>();
    pclomp::NdtParams ndt_rough{};
    ndt_rough.num_threads = 20;
    ndt_rough.step_size = 0.1;
    ndt_rough.trans_epsilon = 0.01;
    ndt_rough.max_iterations = 30;
    ndt_rough.resolution = 5.0;
    ndt_rough_ptr_->setParams(ndt_rough);
    LOG(INFO) << "Localizer initialized";
}

void Localizer::GetInitPose(const Eigen::Matrix4d &init_guess, Eigen::Matrix4d &align_pose, const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud)
{
    LocalizationQuality quality;
    (void)GetInitPose(init_guess, align_pose, pc, output_cloud, quality);
}

bool Localizer::GetInitPose(const Eigen::Matrix4d &init_guess, Eigen::Matrix4d &align_pose, const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud, LocalizationQuality& quality)
{
    quality = LocalizationQuality();

    if(!ndt_ptr_ || !ndt_rough_ptr_)
    {
        ndt_ptr_ = std::make_shared<NormalDistributionsTransform>();
        pclomp::NdtParams ndt{};
        ndt.num_threads = 4;
        ndt.step_size = 0.1;
        ndt.trans_epsilon = 0.01;
        ndt.max_iterations = 30;
        ndt.resolution = 2.0;
        ndt_ptr_->setParams(ndt);

        ndt_rough_ptr_ = std::make_shared<NormalDistributionsTransform>();
        pclomp::NdtParams ndt_rough{};
        ndt_rough.num_threads = 4;
        ndt_rough.step_size = 0.1;
        ndt_rough.trans_epsilon = 0.01;
        ndt_rough.max_iterations = 30;
        ndt_rough.resolution = 5.0;
        ndt_rough_ptr_->setParams(ndt_rough);
        LOG(INFO) << "Localizer initialized";
    }

    if (!pc || pc->empty()) {
        LOG(WARNING) << "GetInitPose failed: input cloud is empty";
        quality.quality_level = "poor";
        quality.is_reliable = false;
        return false;
    }

    MapManager::DiffMapReqInfo req;
    req.center_x = init_guess(0,3);
    req.center_y = init_guess(1,3);
    req.radius = 2000.0;
    req.cache_ids = ndt_ptr_->getCurrentMapIDs();

    MapManager::DiffMapResult res;
    map_loader_.GetDiffMap(req, res);

    auto& maps_to_add = res.new_pointcloud_with_ids;
    auto& map_ids_to_remove = res.ids_to_remove;

    if (maps_to_add.empty() && map_ids_to_remove.empty())
    {
        LOG(INFO) << "No map to add or remove";
    }

    // Add pcd
    for (auto& map : maps_to_add)
    {
        ndt_ptr_->addTarget(map.pointcloud.makeShared(), map.cell_id);
        ndt_rough_ptr_->addTarget(map.pointcloud.makeShared(), map.cell_id);
    }

    // Remove pcd
    for (const std::string& map_id_to_remove : map_ids_to_remove)
    {
        ndt_ptr_->removeTarget(map_id_to_remove);
        ndt_rough_ptr_->removeTarget(map_id_to_remove);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(2.0, 2.0, 2.0);
    voxel_filter.setInputCloud(pc);
    voxel_filter.filter(*input_cloud);

    if (!input_cloud || input_cloud->empty()) {
        LOG(WARNING) << "GetInitPose failed: filtered input cloud is empty";
        quality.quality_level = "poor";
        quality.is_reliable = false;
        return false;
    }

    ndt_ptr_->createVoxelKdtree();
    ndt_ptr_->setInputSource(pc);

    ndt_rough_ptr_->createVoxelKdtree();
    ndt_rough_ptr_->setInputSource(input_cloud);

    pclomp::NdtResult ndt_res = AlignPose(init_guess);
    align_pose = ndt_res.pose.cast<double>();

    ndt_ptr_->align(*output_cloud, align_pose.cast<float>());

    ndt_res = ndt_ptr_->getResult();
    align_pose = ndt_res.pose.cast<double>();

    quality.transform_probability = ndt_res.transform_probability;
    quality.nearest_voxel_likelihood = ndt_res.nearest_voxel_transformation_likelihood;
    quality.iteration_num = ndt_res.iteration_num;
    quality.evaluate(quality_thresholds_);

    LOG(INFO) << "precise NDT alignment completed - score: " << ndt_res.transform_probability
              << ", NVTL: " << ndt_res.nearest_voxel_transformation_likelihood
              << ", quality: " << quality.quality_level
              << ", reliable: " << (quality.is_reliable ? "yes" : "no");

    if (!quality.is_reliable) {
        LOG(WARNING) << "initial localization rejected by quality gate; last pose is not updated";
        return false;
    }

    last_last_pose_ = align_pose;
    last_pose_ = align_pose;
    return true;
}

void Localizer::ResetLocalizationState()
{
    last_last_pose_ = Eigen::Matrix4d::Identity();
    last_pose_ = Eigen::Matrix4d::Identity();
}

bool Localizer::RegisterFrame(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pc, pcl::PointCloud<pcl::PointXYZ>::Ptr &output_cloud, Eigen::Matrix4d &align_pose)
{
    LocalizationQuality quality;
    return RegisterFrame(pc, output_cloud, align_pose, quality);
}

bool Localizer::RegisterFrame(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pc, pcl::PointCloud<pcl::PointXYZ>::Ptr &output_cloud, Eigen::Matrix4d &align_pose, LocalizationQuality& quality) 
{
    Eigen::Matrix4d init_guess = last_pose_ * last_last_pose_.inverse() * last_pose_;

    ndt_ptr_->setInputSource(pc);
    ndt_ptr_->align(*output_cloud, init_guess.cast<float>());
    pclomp::NdtResult ndt_res = ndt_ptr_->getResult();
    align_pose = ndt_res.pose.cast<double>();

    // 填充定位质量信息
    quality.transform_probability = ndt_res.transform_probability;
    quality.nearest_voxel_likelihood = ndt_res.nearest_voxel_transformation_likelihood;
    quality.iteration_num = ndt_res.iteration_num;
    
    // 评估定位质量（使用配置的阈值）
    quality.evaluate(quality_thresholds_);

    // Tracking and publishing use different gates:
    // - return value still means "reliable enough to publish/use as a trusted localization result";
    // - last_pose_ may be updated for tracking when the score is not completely unusable,
    //   otherwise a single poor frame can freeze the constant-velocity predictor and cause
    //   a domino failure in the following frames.
    const double tracking_min_tp = 0.5;  // conservative fallback gate; keep publish gate stricter.
    const bool acceptable_for_tracking =
        quality.is_reliable || quality.transform_probability > tracking_min_tp;

    if (acceptable_for_tracking) {
        last_last_pose_ = last_pose_;
        last_pose_ = align_pose;
    } else {
        LOG(WARNING) << "RegisterFrame rejected by tracking gate; last pose is not updated";
    }

    LOG(INFO) << "Localization quality: " << quality.quality_level
              << ", TP: " << quality.transform_probability
              << ", NVTL: " << quality.nearest_voxel_likelihood
              << ", iterations: " << quality.iteration_num
              << ", reliable: " << (quality.is_reliable ? "yes" : "no");

    return quality.is_reliable;
}

void Localizer::UpdateMap(const Eigen::Vector2d &pose) 
{ 
    map_loader_.LoadMapOnPose(pose); 
}

}
