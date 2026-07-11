#pragma once

#include <string>

#include "modules/mappingSystem/mapping_system.h"

namespace lightning::modules {

struct SaveMapOptions {
    int block_resolution = 20;
    double block_voxel_size = 0.1;
    bool overwrite = true;
};

class SaveMap {
   public:
    bool Save(const std::string& save_path, const MappingSystemResult& result,
              const SaveMapOptions& options = SaveMapOptions()) const;

   private:
    bool SaveBlockMap(const std::string& save_path, const CloudPtr& global_map,
                      const SaveMapOptions& options) const;
    bool SavePoseFile(const std::string& save_path,
                      const std::vector<Keyframe::Ptr>& keyframes) const;
};

}  // namespace lightning::modules
