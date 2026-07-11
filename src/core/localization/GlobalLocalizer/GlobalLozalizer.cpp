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
    double& roll = rpy.x();   // roll��Ӧx����
    double& pitch = rpy.y();  // pitch��Ӧy����
    double& yaw = rpy.z();    // yaw��Ӧz����

    // ����Pitch�ǣ���Y�ᣩ
    pitch = std::asin(-R(2, 0));

    // ������������cos(pitch)�ӽ�0ʱ��
    const double cos_pitch = std::cos(pitch);
    if (std::fabs(cos_pitch) > 1e-6) {
        // ��������������roll��yaw
        roll = std::atan2(R(2, 1), R(2, 2));
        yaw = std::atan2(R(1, 0), R(0, 0));
    } else {
        // �����������pitch�֡�90�㣩���̶�yaw=0������roll
        yaw = 0.0;
        roll = std::atan2(-R(0, 1), R(1, 1));
    }

    return rpy;
}

pclomp::NdtResult Localizer::AlignPose(const Eigen::Matrix4d &initial_pose_with_cov)
{
    // ��ȡ��ʼλ�˵�RPY�Ƕȡ�
    // 2.5D �ض�λԼ�����������׶�ֻ���� x / y / yaw �����ӣ�
    // z / roll / pitch ����̶�Ϊ�����ֵ�������������˲�����ʮ�׸߶Ȼ����̬�ǵ���Ч��ѡ��
    const auto base_rpy = getRPYFromEigenMatrix(initial_pose_with_cov.block<3, 3>(0, 0));
    const double stddev_z = 10.0;
    const double stddev_roll = 0.01;
    const double stddev_pitch = 0.01;

    const std::vector<double> sample_mean{
        initial_pose_with_cov(0,3),
        initial_pose_with_cov(1,3),
        initial_pose_with_cov(2,3),
        base_rpy.x(),
        base_rpy.y(),
        base_rpy.z(),
    };
    const std::vector<double> sample_stddev{
        3.0, 3.0, stddev_z, stddev_roll, stddev_pitch, 0.17453
    };

    TreeStructuredParzenEstimator tpe(
        TreeStructuredParzenEstimator::Direction::MAXIMIZE,
        100,
        sample_mean,
        sample_stddev);

    auto output_cloud = std::make_shared<pcl::PointCloud<PointSource>>();
    std::vector<pclomp::NdtResult> result_array;

    for (int64_t i = 0; i < 200; i++)
    {
        const TreeStructuredParzenEstimator::Input input = tpe.get_next_input();

        Eigen::Matrix3d rot = Eigen::AngleAxisd(input[5], Eigen::Vector3d::UnitZ())
                            * Eigen::AngleAxisd(input[4], Eigen::Vector3d::UnitY())
                            * Eigen::AngleAxisd(input[3], Eigen::Vector3d::UnitX()).matrix();
        Eigen::Matrix4d init_pose_matrix = Eigen::Matrix4d::Identity();
        init_pose_matrix.block<3, 3>(0, 0) = rot;
        init_pose_matrix.block<3, 1>(0, 3) = Eigen::Vector3d(input[0], input[1], input[2]);

        LOG(INFO) << "Initial Pose: "
                  << "x: " << input[0]
                  << ", y: " << input[1]
                  << ", z: " << input[2]
                  << ", roll: " << input[3]
                  << ", pitch: " << input[4]
                  << ", yaw: " << input[5] << std::endl;

        const Eigen::Matrix4f initial_pose_matrix = init_pose_matrix.cast<float>();
        ndt_rough_ptr_->align(*output_cloud, initial_pose_matrix);
        const pclomp::NdtResult ndt_result = ndt_rough_ptr_->getResult();

        const auto align_pose_rpy = getRPYFromEigenMatrix(
            ndt_result.pose.block<3, 3>(0, 0).cast<double>());
        std::cout << "aligned pose: "
                  << "x: " << ndt_result.pose(0,3)
                  << ", y: " << ndt_result.pose(1,3)
                  << ", z: " << ndt_result.pose(2,3)
                  << ", roll: " << align_pose_rpy.x()
                  << ", pitch: " << align_pose_rpy.y()
                  << ", yaw: " << align_pose_rpy.z() << std::endl;

        TreeStructuredParzenEstimator::Input result(6);
        result[0] = ndt_result.pose(0,3);
        result[1] = ndt_result.pose(1,3);
        result[2] = ndt_result.pose(2,3);
        result[3] = align_pose_rpy.x();
        result[4] = align_pose_rpy.y();
        result[5] = align_pose_rpy.z();

        tpe.add_trial(TreeStructuredParzenEstimator::Trial{
            result,
            ndt_result.transform_probability});
        result_array.push_back(ndt_result);
    }

    auto best_particle_ptr = std::max_element(
        std::begin(result_array),
        std::end(result_array),
        [](const pclomp::NdtResult &lhs, const pclomp::NdtResult &rhs)
        {
            return lhs.nearest_voxel_transformation_likelihood <
                   rhs.nearest_voxel_transformation_likelihood;
        });

    LOG(INFO) << "rough NDT alignment completed - best particle score: "
              << best_particle_ptr->nearest_voxel_transformation_likelihood;

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

bool Localizer::GetInitPose(
    const Eigen::Matrix4d &init_guess,
    Eigen::Matrix4d &align_pose,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud,
    LocalizationQuality& quality)
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

    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(2.0, 2.0, 2.0);
    voxel_filter.setInputCloud(pc);
    voxel_filter.filter(*input_cloud);

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
    quality.nearest_voxel_likelihood =
        ndt_res.nearest_voxel_transformation_likelihood;
    quality.iteration_num = ndt_res.iteration_num;
    quality.evaluate(quality_thresholds_);

    LOG(INFO) << "precise NDT alignment completed - score: " << ndt_res.transform_probability
              << ", NVTL: " << ndt_res.nearest_voxel_transformation_likelihood
              << ", quality: " << quality.quality_level
              << ", reliable: " << (quality.is_reliable ? "yes" : "no");

    const Eigen::Matrix4d precise_pose = ndt_res.pose.cast<double>();
    const Eigen::Vector3d precise_translation = precise_pose.block<3, 1>(0, 3);
    const Eigen::Vector3d precise_rpy = precise_pose.block<3, 3>(0, 0).eulerAngles(0, 1, 2);
    LOG(INFO) << "[INIT_2_5D] precise final pose: x: "
              << precise_translation.x()
              << ", y: " << precise_translation.y()
              << ", z: " << precise_translation.z()
              << ", roll: " << precise_rpy.x()
              << ", pitch: " << precise_rpy.y()
              << ", yaw: " << precise_rpy.z()
              << ", score: " << ndt_res.transform_probability
              << ", NVTL: " << ndt_res.nearest_voxel_transformation_likelihood;

    last_last_pose_ = align_pose;
    last_pose_ = align_pose;
    return true;
}

void Localizer::ResetLocalizationState()
{
    last_last_pose_ = Eigen::Matrix4d::Identity();
    last_pose_ = Eigen::Matrix4d::Identity();
}

bool Localizer::RegisterFrame(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &pc,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &output_cloud,
    Eigen::Matrix4d &align_pose,
    LocalizationQuality& quality)
{
    Eigen::Matrix4d init_guess = last_pose_ * last_last_pose_.inverse() * last_pose_;

    ndt_ptr_->setInputSource(pc);
    ndt_ptr_->align(*output_cloud, init_guess.cast<float>());
    pclomp::NdtResult ndt_res = ndt_ptr_->getResult();
    align_pose = ndt_res.pose.cast<double>();

    // ��䶨λ������Ϣ
    quality.transform_probability = ndt_res.transform_probability;
    quality.nearest_voxel_likelihood = ndt_res.nearest_voxel_transformation_likelihood;
    quality.iteration_num = ndt_res.iteration_num;
    
    // ������λ������ʹ�����õ���ֵ��
    quality.evaluate(quality_thresholds_);

    last_last_pose_ = last_pose_;
    last_pose_ = align_pose;

    LOG(INFO) << "Localization quality: " << quality.quality_level
              << ", TP: " << quality.transform_probability
              << ", NVTL: " << quality.nearest_voxel_likelihood
              << ", iterations: " << quality.iteration_num
              << ", reliable: " << (quality.is_reliable ? "yes" : "no");

    return quality.is_reliable;
}

}
