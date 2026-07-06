#ifndef __MAP_MANAGER_HPP__
#define __MAP_MANAGER_HPP__

#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <iostream>
#include <algorithm>
#include <string>
#include <mutex>

#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "MapLoader/utils.hpp"

namespace MapManager
{

    class MapLoader
    {
    private:
        std::string pcd_metadata_path_;
        std::string pcd_path_;
        pcl::PointCloud<pcl::PointXYZ> map_cloud_;

        std::map<std::string, PCDFileMetadata> pcd_metadata_dict_;

        std::vector<std::string> GetPcdPath(const std::string &pcd_directory) const;
        std::map<std::string, PCDFileMetadata> GetPcdMetadata(const std::string &pcd_metadata_path, const std::vector<std::string> &pcd_paths) const;
        PointCloudMapCellWithID load_point_cloud_map_cell_with_id(const std::string &path, const std::string &map_id) const;
        
        std::mutex map_loader_mtx_;
    public:
        MapLoader(){};
        MapLoader(std::string pcd_metadata_path, std::string pcd_directory);
        ~MapLoader(){};

        bool GetDiffMap(DiffMapReqInfo &req, DiffMapResult &res);

        void Initialize(std::string pcd_metadata_path, std::string pcd_directory);

        void LoadMapOnPose(const Eigen::Vector2d &pose);

        void reset();
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr GetMapCloud() {return map_cloud_.makeShared();}
    };

};

#endif