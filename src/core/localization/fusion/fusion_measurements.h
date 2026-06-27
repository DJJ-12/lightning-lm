#pragma once

#include <Eigen/Core>

#include "common/nav_state.h"

namespace lightning::loc {

struct FusionImuMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double timestamp = 0.0;
    Vec3d acc = Vec3d::Zero();
    Vec3d gyro = Vec3d::Zero();

    // Variance of linear acceleration / angular velocity in IMU frame.
    // If not provided by upstream, backend uses YAML noise parameters and propagates them
    // through a lightweight IMU preintegration covariance model.
    Eigen::Matrix3d acc_covariance = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d gyro_covariance = Eigen::Matrix3d::Zero();
    bool has_acc_covariance = false;
    bool has_gyro_covariance = false;
};

struct FusionLioMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double timestamp = 0.0;

    // T_odom_base_lio
    SE3 odom_base;

    bool pose_valid = false;
    bool is_parked = false;

    double scan2map_ms = 0.0;
    int lm_iterations = 0;

    // 6x6 scan-to-map covariance estimated from LIO-SAM LM Hessian.
    // Order: roll, pitch, yaw, x, y, z.
    Eigen::Matrix<double, 6, 6> pose_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e3;
    bool has_pose_covariance = false;
};

struct FusionNdtMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double timestamp = 0.0;

    // T_map_base_ndt
    SE3 map_base;

    bool converged = false;
    double fitness_score = -1.0;
    double confidence = 1.0;

    // Optional NDT Laplace covariance from pclomp Hessian.
    // Order: roll, pitch, yaw, x, y, z.
    Eigen::Matrix<double, 6, 6> pose_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e3;
    bool has_pose_covariance = false;
};

struct FusionWheelMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double timestamp = 0.0;
    double vx = 0.0;
    double wz = 0.0;

    Eigen::Matrix2d twist_covariance = Eigen::Matrix2d::Identity();
    bool has_twist_covariance = false;
    bool valid = false;
};

}  // namespace lightning::loc
