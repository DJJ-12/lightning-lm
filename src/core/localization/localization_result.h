#pragma once

#include <geometry_msgs/msg/transform_stamped.hpp>

#include "common/eigen_types.h"
#include "common/nav_state.h"

namespace lightning::loc {

enum class LocalizationStatus {
    UNKNOWN = 0,
    INITIALIZING = 1,
    GOOD = 2,
    FAIL = 3,
};

struct LocalizationResult {
    double timestamp_ = 0.0;

    bool valid_ = false;
    bool lidar_loc_valid_ = false;

    LocalizationStatus status_ = LocalizationStatus::UNKNOWN;

    double confidence_ = 0.0;

    // map -> base_link
    SE3 pose_;

    geometry_msgs::msg::TransformStamped ToGeoMsg() const;
    NavState ToNavState() const;
};

}  // namespace lightning::loc
