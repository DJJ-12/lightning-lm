#pragma once

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <string>

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
    bool localization_valid_ = false;

    LocalizationStatus status_ = LocalizationStatus::UNKNOWN;

    double confidence_ = 0.0;
    bool reliable_ = false;
    double tp_ = 0.0;
    double nvtl_ = 0.0;
    int iterations_ = 0;
    std::string message_;
    std::string frame_id_ = "map";

    // frame_id -> base_link
    SE3 pose_;
    Eigen::Matrix<double, 6, 6> pose_covariance_ = Eigen::Matrix<double, 6, 6>::Identity();
    bool covariance_valid_ = false;
    Eigen::Vector3d velocity_map_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d angular_velocity_body_ = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 6, 6> twist_covariance_ = Eigen::Matrix<double, 6, 6>::Identity();
    bool twist_covariance_valid_ = false;

    geometry_msgs::msg::TransformStamped ToGeoMsg() const;
    NavState ToNavState() const;
};

}  // namespace lightning::loc
