#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "common/eigen_types.h"

namespace lightning::loc {

// 3-D right-error ESKF for localization-level observations.
// Nominal state: x = {p_map, R_map_body, v_body, omega_body}.
// Rotation error: R_true = R_hat * Exp(delta_theta).
// Error vector: delta_x = [delta_p, delta_theta, delta_v_body, delta_omega].
//
// The motion model is a constant body-twist model. Unlike a constant map-frame
// velocity model, it follows curved vehicle motion when omega_body is nonzero.
class ESKF {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static constexpr int kStateDim = 12;
    using ErrorVector = Eigen::Matrix<double, kStateDim, 1>;
    using Covariance = Eigen::Matrix<double, kStateDim, kStateDim>;

    struct Options {
        double body_acceleration_noise_std = 1.5;
        double angular_acceleration_noise_std = 0.8;
        double max_prediction_step = 0.05;
        double min_covariance = 1e-10;
        double max_covariance = 1e8;
        double pose_gate_chi2 = 20.0;
        double position_gate_chi2 = 16.0;
        double yaw_gate_chi2 = 9.0;
        double twist_gate_chi2 = 20.0;
    };

    struct NominalState {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        double stamp = 0.0;
        Eigen::Vector3d position = Eigen::Vector3d::Zero();
        Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
        Eigen::Vector3d velocity_body = Eigen::Vector3d::Zero();
        Eigen::Vector3d angular_velocity_body = Eigen::Vector3d::Zero();
    };

    ESKF();
    explicit ESKF(const Options& options);

    void Configure(const Options& options);
    void Reset();
    bool Initialize(double stamp, const SE3& pose, const Eigen::Vector3d& velocity_body, const Eigen::Vector3d& angular_velocity_body, const Covariance& covariance);
    bool Initialized() const { return initialized_; }
    bool PredictTo(double stamp);

    bool UpdatePose(double stamp, const SE3& pose, const Eigen::Matrix<double, 6, 6>& covariance, double gate_chi2 = -1.0, double* mahalanobis = nullptr);
    bool UpdatePosition(double stamp, const Eigen::Vector3d& position, const Eigen::Matrix3d& covariance, const Eigen::Array3i& observed_axes, double gate_chi2 = -1.0, double* mahalanobis = nullptr);
    bool UpdatePositionWithLeverArm(double stamp, const Eigen::Vector3d& sensor_position, const Eigen::Vector3d& lever_arm_body, const Eigen::Matrix3d& covariance, const Eigen::Array3i& observed_axes, double gate_chi2 = -1.0, double* mahalanobis = nullptr);
    bool UpdateYaw(double stamp, double yaw, double variance, double gate_chi2 = -1.0, double* mahalanobis = nullptr);
    bool UpdateMapVelocity(double stamp, const Eigen::Vector3d& velocity_map, const Eigen::Matrix3d& covariance, const Eigen::Array3i& observed_axes, double gate_chi2 = -1.0, double* mahalanobis = nullptr);
    bool UpdateMapVelocityWithLeverArm(double stamp, const Eigen::Vector3d& sensor_velocity_map, const Eigen::Vector3d& lever_arm_body, const Eigen::Matrix3d& covariance, const Eigen::Array3i& observed_axes, double gate_chi2 = -1.0, double* mahalanobis = nullptr);
    bool UpdateBodyTwist(double stamp, const Eigen::Vector3d& linear_velocity_body, const Eigen::Vector3d& angular_velocity_body, const Eigen::Matrix<double, 6, 6>& covariance, const Eigen::Array3i& linear_axes, const Eigen::Array3i& angular_axes, double gate_chi2 = -1.0, double* mahalanobis = nullptr);
    bool UpdateAngularVelocity(double stamp, const Eigen::Vector3d& angular_velocity_body, const Eigen::Matrix3d& covariance, const Eigen::Array3i& observed_axes, double gate_chi2 = -1.0, double* mahalanobis = nullptr);

    const NominalState& State() const { return state_; }
    const Covariance& P() const { return covariance_; }
    SE3 Pose() const;
    Eigen::Vector3d VelocityMap() const;

    static Eigen::Matrix3d Exp(const Eigen::Vector3d& angle);
    static Eigen::Vector3d Log(const Eigen::Matrix3d& rotation);
    static Eigen::Matrix3d Skew(const Eigen::Vector3d& vector);
    static Eigen::Matrix3d LeftJacobian(const Eigen::Vector3d& angle);
    static double WrapAngle(double angle);
    static double Yaw(const Eigen::Matrix3d& rotation);

   private:
    bool ApplyUpdate(const Eigen::VectorXd& residual, const Eigen::MatrixXd& measurement_jacobian, const Eigen::MatrixXd& measurement_covariance, double gate_chi2, double* mahalanobis);
    void InjectError(const ErrorVector& correction);
    void PropagateStep(double dt);
    void StabilizeCovariance();
    Eigen::RowVector3d YawJacobian() const;

    Options options_;
    bool initialized_ = false;
    NominalState state_;
    Covariance covariance_ = Covariance::Identity();
};

}  // namespace lightning::loc
