#pragma once

#include <string>

#include "common/eigen_types.h"

namespace lightning::loc {

class SE3LowPassFilter {
   public:
    void Reset(const SE3& pose);

    bool HasState() const;

    SE3 Update(const SE3& raw_pose,
               double alpha,
               double ignore_trans_threshold,
               double ignore_yaw_threshold_rad,
               bool* accepted,
               std::string* reject_reason);

    SE3 Get() const;

   private:
    bool initialized_ = false;
    SE3 pose_;
};

}  // namespace lightning::loc
