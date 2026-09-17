#include "modules/mappingSystem/save_map.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <pcl/common/transforms.h>
#include <pcl/common/point_tests.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <sstream>

#include <glog/logging.h>

#include "common/nav_state.h"

namespace lightning::modules {

bool SaveMap::Save(const std::string& save_path, const MappingSystemResult& result,
                   const SaveMapOptions& options) const {
    if (save_path.empty()) {
        LOG(ERROR) << "SaveMap failed: save_path is empty";
        return false;
    }
    if (!result.valid || !result.global_map || result.global_map->empty() || result.keyframes.empty()) {
        LOG(ERROR) << "SaveMap failed: mapping result is empty";
        return false;
    }

    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists(save_path) && options.overwrite) {
        fs::remove_all(save_path, ec);
        if (ec) {
            LOG(ERROR) << "failed to remove existing map path: " << save_path
                       << ", error: " << ec.message();
            return false;
        }
    }
    fs::create_directories(save_path, ec);
    if (ec) {
        LOG(ERROR) << "failed to create save path: " << save_path
                   << ", error: " << ec.message();
        return false;
    }

    CloudPtr global_map_to_save = BuildMapForSave(result);
    if (global_map_to_save->empty()) {
        LOG(ERROR) << "SaveMap failed: converted global map is empty";
        return false;
    }

    pcl::io::savePCDFileBinaryCompressed(save_path + "/global.pcd", *global_map_to_save);

    if (!SaveBlockMap(save_path, global_map_to_save, options)) {
        return false;
    }
    if (!SavePoseFile(save_path, result.keyframes, result.global_map_is_lidar_frame)) {
        return false;
    }

    LOG(INFO) << "map saved to: " << save_path;
    return true;
}

CloudPtr SaveMap::BuildMapForSave(const MappingSystemResult& result) const {
    if (!result.global_map_is_lidar_frame) {
        return result.global_map;
    }

    CloudPtr global_map_base(new PointCloudType());
    pcl::transformPointCloud(
        *result.global_map,
        *global_map_base,
        result.T_base_lidar.matrix().cast<float>());
    global_map_base->header = result.global_map->header;
    global_map_base->height = 1;
    global_map_base->width = global_map_base->size();
    global_map_base->is_dense = result.global_map->is_dense;
    LOG(INFO) << "LIO-SAM global map transformed from lidar0 to base0 before saving";
    return global_map_base;
}

bool SaveMap::SaveBlockMap(const std::string& save_path, const CloudPtr& global_map,
                           const SaveMapOptions& options) const {
    namespace fs = std::filesystem;
    const std::string block_map_dir = save_path + "/BlockMap";
    const std::string pcd_dir = block_map_dir + "/pointcloud_map";
    const std::string metadata_path = block_map_dir + "/pointcloud_map_metadata.yaml";
    fs::create_directories(pcd_dir);

    using SegmentIndex = std::pair<int, int>;
    std::map<SegmentIndex, pcl::PointCloud<pcl::PointXYZ>::Ptr> segment_clouds;
    const int block_resolution = options.block_resolution;

    for (const auto& pt : global_map->points) {
        if (!pcl::isFinite(pt)) {
            continue;
        }
        const int seg_x = static_cast<int>(std::floor(pt.x / static_cast<double>(block_resolution))) * block_resolution;
        const int seg_y = static_cast<int>(std::floor(pt.y / static_cast<double>(block_resolution))) * block_resolution;
        SegmentIndex seg{seg_x, seg_y};
        auto& segment_cloud = segment_clouds[seg];
        if (!segment_cloud) {
            segment_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
        }
        pcl::PointXYZ p;
        p.x = pt.x;
        p.y = pt.y;
        p.z = pt.z;
        segment_cloud->push_back(p);
    }

    std::ofstream metadata_file(metadata_path);
    if (!metadata_file.is_open()) {
        LOG(ERROR) << "failed to open BlockMap metadata file: " << metadata_path;
        return false;
    }
    metadata_file << "x_resolution: " << block_resolution << "\n";
    metadata_file << "y_resolution: " << block_resolution << "\n";

    int segment_num = 0;
    for (auto& item : segment_clouds) {
        const SegmentIndex& seg = item.first;
        auto& segment_cloud = item.second;
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (options.block_voxel_size > 0.0) {
            pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
            voxel_filter.setLeafSize(options.block_voxel_size, options.block_voxel_size, options.block_voxel_size);
            voxel_filter.setInputCloud(segment_cloud);
            voxel_filter.filter(*filtered_cloud);
        } else {
            filtered_cloud = segment_cloud;
        }
        if (filtered_cloud->empty()) {
            continue;
        }
        filtered_cloud->width = filtered_cloud->size();
        filtered_cloud->height = 1;
        filtered_cloud->is_dense = false;

        std::ostringstream filename_stream;
        filename_stream << "seg_" << seg.first << "_" << seg.second << ".pcd";
        const std::string filename = filename_stream.str();
        pcl::io::savePCDFileBinary(pcd_dir + "/" + filename, *filtered_cloud);
        metadata_file << filename << ": [" << seg.first << ", " << seg.second << "]\n";
        ++segment_num;
    }

    LOG(INFO) << "BlockMap saved to " << block_map_dir << ", segment num: " << segment_num;
    return segment_num > 0;
}

bool SaveMap::SavePoseFile(const std::string& save_path,
                           const std::vector<Keyframe::Ptr>& keyframes,
                           bool poses_are_lidar_frame) const {
    std::ofstream pose_file(save_path + "/pose.txt");
    if (!pose_file.is_open()) {
        LOG(ERROR) << "failed to open pose.txt";
        return false;
    }
    if (poses_are_lidar_frame) {
        pose_file << "# parent_frame: lidar0\n";
        pose_file << "# child_frame: lidar\n";
        pose_file << "# pose: T_L0_Lk\n";
    } else {
        pose_file << "# parent_frame: base0\n";
        pose_file << "# child_frame: base_link\n";
        pose_file << "# pose: T_B0_Bk\n";
    }
    pose_file << "# id timestamp tx ty tz qx qy qz qw\n";
    for (const auto& kf : keyframes) {
        if (!kf) {
            continue;
        }
        SE3 pose = kf->GetLIOPose();
        NavState state = kf->GetState();
        Vec3d t = pose.translation();
        Quatd q = pose.unit_quaternion();
        pose_file << std::setprecision(18) << kf->GetID() << " " << state.timestamp_ << " "
                  << t.x() << " " << t.y() << " " << t.z() << " "
                  << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
    return true;
}


}  // namespace lightning::modules
