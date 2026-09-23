#pragma once

#include <Eigen/Core>

#include "common/eigen_types.h"

namespace lightning::loc {

// Twelve-state 3-D pose EKF with an explicit ground-vehicle motion constraint.
//
// State:
//   [p_map(3), rpy_map(3), v_map(3), omega(3)]
//
// Position and orientation are fully three-dimensional. The state storage also
// remains three-dimensional for velocity and angular velocity, but the vehicle
// model enforces v_z=0 and omega_x=omega_y=0 after every prediction/update.
// Therefore only [v_x, v_y, omega_z] participate in deterministic propagation.
class EKF {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static constexpr int kStateDim = 12;
    static constexpr int kPoseDim = 6;
    using StateVector = Eigen::Matrix<double, kStateDim, 1>;
    using StateMatrix = Eigen::Matrix<double, kStateDim, kStateDim>;
    using Covariance = StateMatrix;
    using Matrix6d = Eigen::Matrix<double, 6, 6>;

    enum StateIndex {
        kPositionX = 0,
        kPositionY = 1,
        kPositionZ = 2,
        kRoll = 3,
        kPitch = 4,
        kYaw = 5,
        kVelocityX = 6,
        kVelocityY = 7,
        kVelocityZ = 8,
        kAngularVelocityX = 9,
        kAngularVelocityY = 10,
        kAngularVelocityZ = 11,
    };

    struct Options {
        // Planar acceleration and yaw-acceleration process noise.
        double process_acceleration_std = 1.0;      // m/s^2
        double process_yaw_acceleration_std = 0.5;  // rad/s^2

        // z, roll and pitch have no deterministic rate states. These random
        // walks let full 3-D position/pose observations change them over time.
        double process_vertical_position_rate_std = 1.0;  // m/s
        double process_roll_pitch_rate_std = 0.2;         // rad/s
        double max_prediction_step = 0.05;                // s

        double min_covariance = 1e-10;
        double max_covariance = 1e8;

        double gps_position_gate_chi2 = 16.3;  // 3 position DoF
        double gps_orientation_gate_chi2 = 16.3;  // 3 Euler-angle DoF
        double gps_velocity_gate_chi2 = 11.8;  // 2 DoF
        double ndt_pose_gate_chi2 = 22.5;      // 6 DoF
    };

    struct State {
        Eigen::Vector3d position_map = Eigen::Vector3d::Zero();
        Eigen::Vector3d rpy_map = Eigen::Vector3d::Zero();
        Eigen::Vector3d velocity_map = Eigen::Vector3d::Zero();
        Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
        double stamp = 0.0;
    };

    EKF();
    explicit EKF(const Options& options);

    void Configure(const Options& options);
    void Reset();
    bool Initialize(double stamp,
                    const Eigen::Vector3d& position_map,
                    const Eigen::Vector3d& rpy_map,
                    const Eigen::Vector3d& velocity_map,
                    const Eigen::Vector3d& angular_velocity,
                    const Covariance& covariance);
    bool Initialized() const { return initialized_; }

    // Ground-vehicle prediction:
    //   x,y       += vx,vy * dt
    //   z          = constant
    //   roll,pitch = constant
    //   yaw       += omega_z * dt
    //   vx,vy      = constant
    //   vz          = 0
    //   omega_x,y   = 0
    //   omega_z     = constant.
    bool PredictTo(double stamp);

    // LocalizationSystem converts the device-origin NavSatFix observation and
    // its covariance from ENU into MAP before this update. The GPS device
    // origin is the position observed by the filter, so h(x)=p_map and no
    // lever-arm or attitude term appears in this measurement model.
    bool UpdateGpsPoseMap(
        double stamp,
        const Eigen::Vector3d& gps_position_map,
        const Eigen::Matrix3d& gps_covariance_map,
        double gate_chi2 = -1.0,
        double* mahalanobis = nullptr);

    // LocalizationSystem has already converted the INS observation from
    // ENU<-GPS into MAP<-BODY roll/pitch/yaw.  The measurement and the EKF
    // attitude state therefore have exactly the same physical meaning:
    //   h(x) = [roll_map_body, pitch_map_body, yaw_map_body].
    // Consequently the attitude block of H is the 3x3 identity matrix.
    bool UpdateMapOrientation(
        double stamp,
        const Eigen::Vector3d& measured_rpy_map_body,
        const Eigen::Matrix3d& covariance_map_body,
        double gate_chi2 = -1.0,
        double* mahalanobis = nullptr);

    // Only horizontal map velocity is observable in this vehicle model.
    bool UpdateMapVelocity(double stamp,
                           const Eigen::Vector2d& velocity_map_xy,
                           const Eigen::Matrix2d& covariance,
                           double gate_chi2 = -1.0,
                           double* mahalanobis = nullptr);

    // NDT observes full map<-body position and roll/pitch/yaw.
    bool UpdateNdtPose(double stamp,
                       const SE3& pose_map_body,
                       const Matrix6d& covariance,
                       double gate_chi2 = -1.0,
                       double* mahalanobis = nullptr);

    const State& GetState() const { return state_; }
    const Covariance& P() const { return covariance_; }
    SE3 Pose() const;
    Matrix6d PoseCovariance() const;
    Matrix6d TwistCovarianceBody() const;

    static double WrapAngle(double angle);
    static Eigen::Matrix3d RotationFromRpy(const Eigen::Vector3d& rpy);
    static Eigen::Vector3d RpyFromRotation(const Eigen::Matrix3d& rotation);

   private:
    void PredictStep(double dt);
    bool ApplyUpdate(const Eigen::VectorXd& residual,
                     const Eigen::MatrixXd& measurement_jacobian,
                     const Eigen::MatrixXd& measurement_covariance,
                     double gate_chi2, double* mahalanobis,
                     bool allow_attitude_update);
    StateVector ToVector() const;
    void SetVector(const StateVector& vector);
    void StabilizeCovariance();
    void EnforceMotionConstraints();

    Options options_;
    bool initialized_ = false;
    State state_;
    Covariance covariance_ = Covariance::Identity();
};

}  // namespace lightning::loc
