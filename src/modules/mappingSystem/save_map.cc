#include "modules/mappingSystem/save_map.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <opencv2/opencv.hpp>
#include <pcl/common/point_tests.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <sstream>
#include <yaml-cpp/yaml.h>

#include <glog/logging.h>

#include "common/nav_state.h"
#include "core/maps/tiled_map.h"

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

    pcl::io::savePCDFileBinaryCompressed(save_path + "/global.pcd", *result.global_map);

    TiledMap::Options tm_options;
    tm_options.map_path_ = save_path;
    TiledMap tm(tm_options);
    SE3 start_pose = result.keyframes.front()->GetLIOPose();
    tm.ConvertFromFullPCD(result.global_map, start_pose, save_path);

    if (!SaveBlockMap(save_path, result.global_map, options)) {
        return false;
    }
    if (!SavePoseFile(save_path, result.keyframes)) {
        return false;
    }
    if (result.grid_map) {
        SaveGridMap(save_path, result.grid_map);
    }

    LOG(INFO) << "map saved to: " << save_path;
    return true;
}

bool SaveMap::SaveBlockMap(const std::string& save_path, const CloudPtr& global_map,
                           const SaveMapOptions& options) const {
    if (!global_map || global_map->empty()) {
        return false;
    }
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
        if (!segment_cloud || segment_cloud->empty()) {
            continue;
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (options.block_voxel_size > 0.0) {
            pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
            voxel_filter.setLeafSize(options.block_voxel_size, options.block_voxel_size, options.block_voxel_size);
            voxel_filter.setInputCloud(segment_cloud);
            voxel_filter.filter(*filtered_cloud);
        } else {
            filtered_cloud = segment_cloud;
        }
        if (!filtered_cloud || filtered_cloud->empty()) {
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
                           const std::vector<Keyframe::Ptr>& keyframes) const {
    std::ofstream pose_file(save_path + "/pose.txt");
    if (!pose_file.is_open()) {
        LOG(ERROR) << "failed to open pose.txt";
        return false;
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

bool SaveMap::SaveGridMap(const std::string& save_path,
                          const std::shared_ptr<nav_msgs::msg::OccupancyGrid>& grid_map) const {
    if (!grid_map) {
        return false;
    }
    const auto& map = *grid_map;
    const int width = static_cast<int>(map.info.width);
    const int height = static_cast<int>(map.info.height);
    if (width <= 0 || height <= 0 || static_cast<int>(map.data.size()) < width * height) {
        return false;
    }

    cv::Mat nav_image(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y) {
        const int row_start_index = y * width;
        for (int x = 0; x < width; ++x) {
            const int index = row_start_index + x;
            const int8_t data = map.data[index];
            if (data == 0) {
                nav_image.at<uchar>(height - 1 - y, x) = 255;
            } else if (data == 100) {
                nav_image.at<uchar>(height - 1 - y, x) = 0;
            } else {
                nav_image.at<uchar>(height - 1 - y, x) = 128;
            }
        }
    }
    cv::imwrite(save_path + "/map.pgm", nav_image);

    std::ofstream yaml_file(save_path + "/map.yaml");
    if (!yaml_file.is_open()) {
        LOG(ERROR) << "failed to write map.yaml";
        return false;
    }
    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "image" << YAML::Value << "map.pgm";
    emitter << YAML::Key << "mode" << YAML::Value << "trinary";
    emitter << YAML::Key << "width" << YAML::Value << map.info.width;
    emitter << YAML::Key << "height" << YAML::Value << map.info.height;
    emitter << YAML::Key << "resolution" << YAML::Value << map.info.resolution;
    std::vector<double> orig{map.info.origin.position.x, map.info.origin.position.y, 0.0};
    emitter << YAML::Key << "origin" << YAML::Value << orig;
    emitter << YAML::Key << "negate" << YAML::Value << 0;
    emitter << YAML::Key << "occupied_thresh" << YAML::Value << 0.65;
    emitter << YAML::Key << "free_thresh" << YAML::Value << 0.25;
    emitter << YAML::EndMap;
    yaml_file << emitter.c_str();
    return true;
}

}  // namespace lightning::modules
