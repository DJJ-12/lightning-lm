// Copyright 2022 The Autoware Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef POINTCLOUD_MAP_LOADER__UTILS_HPP_
#define POINTCLOUD_MAP_LOADER__UTILS_HPP_

#include <pcl/common/common.h>
#include <yaml-cpp/yaml.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace MapManager
{
struct PCDFileMetadata
{
    pcl::PointXYZ min;
    pcl::PointXYZ max;
    bool operator==(const PCDFileMetadata& other) const { return min.x == other.min.x && min.y == other.min.y && min.z == other.min.z && max.x == other.max.x && max.y == other.max.y && max.z == other.max.z; }
};

struct PointCloudMapCellWithID
{
    PointCloudMapCellWithID(): cell_id(), pointcloud(), metadata() {}

    std::string cell_id;
    pcl::PointCloud<pcl::PointXYZ> pointcloud;
    PCDFileMetadata metadata;
};

struct DiffMapResult
{
    std::vector<PointCloudMapCellWithID> new_pointcloud_with_ids;
    std::vector<std::string> ids_to_remove;
};

struct DiffMapReqInfo
{
    DiffMapReqInfo() : center_x(0.0), center_y(0.0), radius(0.0) {}

    std::vector<std::string> cache_ids;
    float center_x;
    float center_y;
    float radius;
};

std::map<std::string, PCDFileMetadata> load_pcd_metadata(const std::string& pcd_metadata_path);
std::map<std::string, PCDFileMetadata> replace_with_absolute_path(const std::map<std::string, PCDFileMetadata>& pcd_metadata_path, const std::vector<std::string>& pcd_paths, std::set<std::string>& missing_pcd_names);

bool cylinder_and_box_overlap_exists(const double center_x, const double center_y, const double radius, const pcl::PointXYZ box_min_point, const pcl::PointXYZ box_max_point);
bool is_grid_within_queried_area(const DiffMapReqInfo& area, const PCDFileMetadata metadata);

}  // namespace autoware::map_loader

#endif  // POINTCLOUD_MAP_LOADER__UTILS_HPP_