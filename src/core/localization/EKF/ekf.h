#pragma once

#include <Eigen/Core>

#include "common/eigen_types.h"

namespace lightning::loc {

// Standard planar extended Kalman filter.
//
// State: [x_map, y_map, yaw_map, velocity_x_map, velocity_y_map, yaw_rate].
// The state pose always belongs to the tracking/base reference point. Sensor
// lever arms are handled by the corresponding measurement models.
class EKF {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static constexpr int kStateDim = 6;
    using StateVector = Eigen::Matrix<double, kStateDim, 1>;
    using StateMatrix = Eigen::Matrix<double, kStateDim, kStateDim>;
    using Covariance = StateMatrix;

    enum StateIndex {
        kX = 0,
        kY = 1,
        kYaw = 2,
        kVelocityX = 3,
        kVelocityY = 4,
        kYawRate = 5,
    };

    struct Options {
        // Map-frame linear acceleration and yaw acceleration noise.
        double process_acceleration_std = 1.0;          // m/s^2, x and y
        double process_yaw_acceleration_std = 0.5;      // rad/s^2
        double max_prediction_step = 0.05;              // s

        double min_covariance = 1e-10;
        double max_covariance = 1e8;

        double rtk_position_gate_chi2 = 11.8;
        double ins_yaw_gate_chi2 = 9.0;
        double rtk_velocity_gate_chi2 = 11.8;
        double ndt_pose_gate_chi2 = 16.0;
        double wheel_gate_chi2 = 9.0;
    };

    struct State {
        double x = 0.0;
        double y = 0.0;
        double yaw = 0.0;
        double velocity_x_map = 0.0;
        double velocity_y_map = 0.0;
        double yaw_rate = 0.0;
        double stamp = 0.0;
    };

    EKF();
    explicit EKF(const Options& options);

    void Configure(const Options& options);
    void Reset();
    bool Initialize(double stamp, double x, double y, double yaw,
                    const Eigen::Vector2d& velocity_map, double yaw_rate,
                    const Covariance& covariance);
    bool Initialized() const { return initialized_; }

    // Advances the constant-map-velocity state to stamp. Stale observations
    // are rejected and long intervals are integrated in bounded pieces.
    bool PredictTo(double stamp);

    // GNSS antenna position in map coordinates:
    // z = [x, y] + R(yaw) * lever_arm_tracking.
    bool UpdateRtkPosition(double stamp,
                           const Eigen::Vector2d& sensor_position_map,
                           const Eigen::Vector2d& lever_arm_tracking,
                           const Eigen::Matrix2d& covariance,
                           double gate_chi2 = -1.0,
                           double* mahalanobis = nullptr);

    // Absolute map yaw measurement. This never differentiates yaw manually;
    // yaw_rate is corrected only through EKF covariance coupling.
    bool UpdateYaw(double stamp, double yaw, double variance,
                   double gate_chi2 = -1.0,
                   double* mahalanobis = nullptr);

    // Planar map-frame velocity. /ins/velocity is already an ENU velocity;
    // LocalizationSystem rotates it into map before calling this method.
    bool UpdateMapVelocity(double stamp,
                           const Eigen::Vector2d& velocity_map,
                           const Eigen::Matrix2d& covariance,
                           double gate_chi2 = -1.0,
                           double* mahalanobis = nullptr);

    // NDT tracking pose measurement [x_map, y_map, yaw_map].
    bool UpdateNdtPose(double stamp, const Eigen::Vector3d& pose,
                       const Eigen::Matrix3d& covariance,
                       double gate_chi2 = -1.0,
                       double* mahalanobis = nullptr);

    // Body-frame longitudinal wheel speed:
    // h(x)=cos(yaw)*vx_map+sin(yaw)*vy_map. angular.z is intentionally not
    // inferred to be a gyro measurement merely because the ROS type is Odom.
    bool UpdateWheelOdometry(double stamp, double forward_velocity,
                             double variance,
                             double gate_chi2 = -1.0,
                             double* mahalanobis = nullptr);

    const State& GetState() const { return state_; }
    const Covariance& P() const { return covariance_; }
    SE3 Pose() const;
    Eigen::Vector3d VelocityMap() const;

    static double WrapAngle(double angle);

   private:
    void PredictStep(double dt);
    bool ApplyUpdate(const Eigen::VectorXd& residual,
                     const Eigen::MatrixXd& measurement_jacobian,
                     const Eigen::MatrixXd& measurement_covariance,
                     double gate_chi2, double* mahalanobis);
    StateVector ToVector() const;
    void SetVector(const StateVector& vector);
    void StabilizeCovariance();

    Options options_;
    bool initialized_ = false;
    State state_;
    Covariance covariance_ = Covariance::Identity();
};

}  // namespace lightning::loc
