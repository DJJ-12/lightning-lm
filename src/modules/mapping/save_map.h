#pragma once

#include <string>

#include "modules/mapping/mapping.h"

namespace lightning::modules {

struct SaveMapOptions {
    int block_resolution = 20;
    double block_voxel_size = 0.1;
    bool overwrite = true;
};

class SaveMap {
   public:
    bool Save(const std::string& save_path, const MappingResult& result,
              const SaveMapOptions& options = SaveMapOptions()) const;

   private:
    bool SaveBlockMap(const std::string& save_path, const CloudPtr& global_map,
                      const SaveMapOptions& options) const;
    bool SavePoseFile(const std::string& save_path,
                      const std::vector<Keyframe::Ptr>& keyframes) const;
    bool SaveGridMap(const std::string& save_path,
                     const std::shared_ptr<nav_msgs::msg::OccupancyGrid>& grid_map) const;
};

}  // namespace lightning::modules
