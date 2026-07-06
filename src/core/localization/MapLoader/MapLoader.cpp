#include "MapLoader/MapLoader.hpp"

#include <glob.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <glog/logging.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <malloc.h>

namespace MapManager
{

    bool is_pcd_file(const std::string &p)
    {
        size_t last_dot = p.find_last_of(".");
        if (last_dot == std::string::npos)
        {
            return false;
        }

        std::string ext = p.substr(last_dot);
        return (ext == ".pcd" || ext == ".PCD");
    }

    std::vector<std::string> MapLoader::GetPcdPath(const std::string &pcd_directory) const
    {
        std::vector<std::string> pcd_paths;

        if (is_pcd_file(pcd_directory))
        {
            pcd_paths.push_back(pcd_directory);
        }
        else
        {
            // 处理目录情况
            glob_t glob_result;
            std::string pattern = pcd_directory + "/*.pcd";
            if (glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result) == 0)
            {
                for (size_t i = 0; i < glob_result.gl_pathc; ++i)
                {
                    if (is_pcd_file(glob_result.gl_pathv[i]))
                    {
                        pcd_paths.push_back(glob_result.gl_pathv[i]);
                    }
                }
            }
            globfree(&glob_result);

            pattern = pcd_directory + "/*.PCD";
            if (glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result) == 0)
            {
                for (size_t i = 0; i < glob_result.gl_pathc; ++i)
                {
                    if (is_pcd_file(glob_result.gl_pathv[i]))
                    {
                        pcd_paths.push_back(glob_result.gl_pathv[i]);
                    }
                }
            }
            globfree(&glob_result);
        }

        return pcd_paths;
    }

    std::map<std::string, PCDFileMetadata> MapLoader::GetPcdMetadata(const std::string &pcd_metadata_path, const std::vector<std::string> &pcd_paths) const
    {
        std::map<std::string, PCDFileMetadata> pcd_metadata_dict;

        if (!pcd_metadata_path.empty())
        {
            // 加载PCD元数据文件
            auto pcd_metadata = load_pcd_metadata(pcd_metadata_path);

            // 将相对路径替换为绝对路径
            std::set<std::string> missing_pcd_names;
            pcd_metadata_dict = replace_with_absolute_path(pcd_metadata, pcd_paths, missing_pcd_names);

            // 输出缺失的PCD文件信息
            for (const auto &name : missing_pcd_names)
            {
                LOG(WARNING) << "PCD file is not found in the specified directory: " << name;
            }
        }
        else if (pcd_paths.size() == 1)
        {
            // 对于单个PCD文件的特殊情况处理
            pcl::PointCloud<pcl::PointXYZ> single_pcd;
            const auto &pcd_path = pcd_paths.front();
            if (pcl::io::loadPCDFile(pcd_path, single_pcd) == -1)
            {
                LOG(ERROR) << "PCD load failed: " << pcd_path;
            }
            PCDFileMetadata metadata = {};
            pcl::getMinMax3D(single_pcd, metadata.min, metadata.max);
            pcd_metadata_dict[pcd_path] = metadata;
        }
        else
        {
            // 多个PCD文件但没有元数据文件的情况
            for (const auto &pcd_path : pcd_paths)
            {
                pcl::PointCloud<pcl::PointXYZ> single_pcd;
                if (pcl::io::loadPCDFile(pcd_path, single_pcd) == -1)
                {
                    LOG(ERROR) << "PCD load failed: " << pcd_path;
                    continue;
                }
                PCDFileMetadata metadata = {};
                pcl::getMinMax3D(single_pcd, metadata.min, metadata.max);
                pcd_metadata_dict[pcd_path] = metadata;
            }
        }

        return pcd_metadata_dict;
    }

    bool MapLoader::GetDiffMap(DiffMapReqInfo &req, DiffMapResult &res)
    {
        std::lock_guard<std::mutex> lock(map_loader_mtx_);

        // iterate over all the available pcd map grids
        std::vector<bool> should_remove(static_cast<int>(req.cache_ids.size()), true);
        for (const auto &ele : pcd_metadata_dict_)
        {
            std::string path = ele.first;
            PCDFileMetadata metadata = ele.second;

            // assume that the map ID = map path (for now)
            const std::string &map_id = path;

            // skip if the pcd file is not within the queried area
            if (!is_grid_within_queried_area(req, metadata))
                continue;

            auto id_in_cached_list = std::find(req.cache_ids.begin(), req.cache_ids.end(), map_id);
            if (id_in_cached_list != req.cache_ids.end())
            {
                int index = static_cast<int>(id_in_cached_list - req.cache_ids.begin());
                should_remove[index] = false;
            }
            else
            {
                PointCloudMapCellWithID pointcloud_map_cell_with_id = load_point_cloud_map_cell_with_id(path, map_id);
                pointcloud_map_cell_with_id.metadata.min.x = metadata.min.x;
                pointcloud_map_cell_with_id.metadata.min.y = metadata.min.y;
                pointcloud_map_cell_with_id.metadata.max.x = metadata.max.x;
                pointcloud_map_cell_with_id.metadata.max.y = metadata.max.y;
                res.new_pointcloud_with_ids.push_back(pointcloud_map_cell_with_id);
            }
        }

        for (size_t i = 0; i < req.cache_ids.size(); ++i)
        {
            if (should_remove[i])
            {
                res.ids_to_remove.push_back(req.cache_ids[i]);
            }
        }

        return true;
    }

    void MapLoader::Initialize(std::string pcd_metadata_path, std::string pcd_directory)
    {
        LOG(INFO) << "MapLoader init";
        std::vector<std::string> pcd_paths = GetPcdPath(pcd_directory);
        for(auto p: pcd_paths)
        {
            LOG(INFO) << p << std::endl;
        }

        pcd_metadata_dict_ = GetPcdMetadata(pcd_metadata_path, pcd_paths);

    }

    PointCloudMapCellWithID MapLoader::load_point_cloud_map_cell_with_id(const std::string &path, const std::string &map_id) const
    {
        pcl::PointCloud<pcl::PointXYZ> pcd;
        if (pcl::io::loadPCDFile(path, pcd) == -1)
        {
            LOG(ERROR) << "PCD load failed: " << path;
        }
        PointCloudMapCellWithID pointcloud_map_cell_with_id;
        pointcloud_map_cell_with_id.pointcloud = pcd;
        pointcloud_map_cell_with_id.cell_id = map_id;
        return pointcloud_map_cell_with_id;
    }

    MapLoader::MapLoader(std::string pcd_metadata_path, std::string pcd_directory)
    {
        LOG(INFO) << "MapLoader init";
        std::vector<std::string> pcd_paths = GetPcdPath(pcd_directory);
        for(auto p: pcd_paths)
        {
            LOG(INFO) << p << std::endl;
        }

        pcd_metadata_dict_ = GetPcdMetadata(pcd_metadata_path, pcd_paths);

    }

    void MapLoader::LoadMapOnPose(const Eigen::Vector2d &pose)
    {
        DiffMapReqInfo req;
        req.center_x = pose(0);
        req.center_y = pose(1);
        req.radius = 2000.0;

        DiffMapResult res;
        GetDiffMap(req, res);

        auto &maps_to_add = res.new_pointcloud_with_ids;
        auto &map_ids_to_remove = res.ids_to_remove;

        if (maps_to_add.empty() && map_ids_to_remove.empty())
        {
            return;
        }

        map_cloud_.clear();
        // Add pcd
        for (auto &map : maps_to_add)
        {
            map_cloud_ += map.pointcloud;
        }
    }
}

void MapManager::MapLoader::reset()
{
    pcl::PointCloud<pcl::PointXYZ>().swap(map_cloud_);
    malloc_trim(0);
}
