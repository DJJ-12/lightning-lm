#include "core/localization/EKF/ekf.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace lightning::loc {
namespace {

constexpr double kTimeEpsilon = 1e-9;
constexpr double kJacobianEpsilon = 1e-6;

}  // namespace

EKF::EKF() { Reset(); }

EKF::EKF(const Options& options) : options_(options) { Reset(); }

void EKF::Configure(const Options& options) { options_ = options; }

void EKF::Reset() {
    initialized_ = false;
    state_ = State();
    covariance_.setIdentity();
}

bool EKF::Initialize(double stamp,
                     const Eigen::Vector3d& position_map,
                     const Eigen::Vector3d& rpy_map,
                     const Eigen::Vector3d& velocity_map,
                     const Eigen::Vector3d& angular_velocity,
                     const Covariance& covariance) {
    if (!std::isfinite(stamp) || !position_map.allFinite() ||
        !rpy_map.allFinite() || !velocity_map.allFinite() ||
        !angular_velocity.allFinite() || !covariance.allFinite()) {
        return false;
    }

    state_.stamp = stamp;
    state_.position_map = position_map;
    state_.rpy_map = rpy_map.unaryExpr(
        [](double angle) { return WrapAngle(angle); });
    state_.velocity_map = velocity_map;
    state_.angular_velocity = angular_velocity;
    covariance_ = covariance;
    initialized_ = true;
    StabilizeCovariance();
    EnforceMotionConstraints();
    return true;
}

bool EKF::PredictTo(double stamp) {
    if (!initialized_ || !std::isfinite(stamp)) return false;

    double remaining_dt = stamp - state_.stamp;
    if (remaining_dt < -1e-6) return false;
    if (remaining_dt <= kTimeEpsilon) {
        state_.stamp = std::max(state_.stamp, stamp);
        return true;
    }

    const double max_step = std::max(options_.max_prediction_step, 1e-4);
    while (remaining_dt > kTimeEpsilon) {
        const double dt = std::min(remaining_dt, max_step);
        PredictStep(dt);
        remaining_dt -= dt;
    }
    state_.stamp = stamp;
    return true;
}

void EKF::PredictStep(double dt) {
    StateMatrix transition = StateMatrix::Identity();
    transition(kPositionX, kVelocityX) = dt;
    transition(kPositionY, kVelocityY) = dt;
    transition(kYaw, kAngularVelocityZ) = dt;

    state_.position_map.x() += state_.velocity_map.x() * dt;
    state_.position_map.y() += state_.velocity_map.y() * dt;
    state_.rpy_map.z() = WrapAngle(
        state_.rpy_map.z() + state_.angular_velocity.z() * dt);

    // Driving noise:
    // [a_x_map, a_y_map, yaw_acceleration,
    //  vertical_position_rate, roll_rate, pitch_rate].
    Eigen::Matrix<double, kStateDim, 6> noise_input =
        Eigen::Matrix<double, kStateDim, 6>::Zero();
    const double half_dt_squared = 0.5 * dt * dt;
    noise_input(kPositionX, 0) = half_dt_squared;
    noise_input(kVelocityX, 0) = dt;
    noise_input(kPositionY, 1) = half_dt_squared;
    noise_input(kVelocityY, 1) = dt;
    noise_input(kYaw, 2) = half_dt_squared;
    noise_input(kAngularVelocityZ, 2) = dt;
    noise_input(kPositionZ, 3) = dt;
    noise_input(kRoll, 4) = dt;
    noise_input(kPitch, 5) = dt;

    Eigen::Matrix<double, 6, 6> driving_noise =
        Eigen::Matrix<double, 6, 6>::Zero();
    driving_noise(0, 0) = options_.process_acceleration_std *
                          options_.process_acceleration_std;
    driving_noise(1, 1) = driving_noise(0, 0);
    driving_noise(2, 2) = options_.process_yaw_acceleration_std *
                          options_.process_yaw_acceleration_std;
    driving_noise(3, 3) = options_.process_vertical_position_rate_std *
                          options_.process_vertical_position_rate_std;
    driving_noise(4, 4) = options_.process_roll_pitch_rate_std *
                          options_.process_roll_pitch_rate_std;
    driving_noise(5, 5) = driving_noise(4, 4);
    // P- = A * P * A^T + Q
    // Q 传播噪声
    // P- 预测协方差
    covariance_ = transition * covariance_ * transition.transpose() +
                  noise_input * driving_noise * noise_input.transpose();
    StabilizeCovariance();
    EnforceMotionConstraints();
}

bool EKF::UpdategpsPosition(
    double stamp, const Eigen::Vector3d& sensor_position_map,
    const Eigen::Vector3d& lever_arm_tracking,
    const Eigen::Matrix3d& covariance, double gate_chi2,
    double* mahalanobis) {
    if (!PredictTo(stamp) || !sensor_position_map.allFinite() ||
        !lever_arm_tracking.allFinite() || !covariance.allFinite()) {
        return false;
    }
    // 消除杆臂影响
    const Eigen::Matrix3d rotation = RotationFromRpy(state_.rpy_map);
    const Eigen::Vector3d tracking_position_observation =
        sensor_position_map - rotation * lever_arm_tracking;
    const Eigen::Vector3d residual =
        tracking_position_observation - state_.position_map;

    Eigen::Matrix<double, 3, kStateDim> jacobian =
        Eigen::Matrix<double, 3, kStateDim>::Zero();
    jacobian.block<3, 3>(0, kPositionX).setIdentity();

    return ApplyUpdate(
        residual, jacobian, covariance,
        gate_chi2 > 0.0 ? gate_chi2 : options_.gps_position_gate_chi2,
        mahalanobis, false);
}

bool EKF::UpdateMapVelocity(
    double stamp, const Eigen::Vector2d& velocity_map_xy,
    const Eigen::Matrix2d& covariance, double gate_chi2,
    double* mahalanobis) {
    if (!PredictTo(stamp) || !velocity_map_xy.allFinite() ||
        !covariance.allFinite()) {
        return false;
    }

    const Eigen::Vector2d residual =
        velocity_map_xy - state_.velocity_map.head<2>();
    Eigen::Matrix<double, 2, kStateDim> jacobian =
        Eigen::Matrix<double, 2, kStateDim>::Zero();
    jacobian(0, kVelocityX) = 1.0;
    jacobian(1, kVelocityY) = 1.0;

    return ApplyUpdate(
        residual, jacobian, covariance,
        gate_chi2 > 0.0 ? gate_chi2 : options_.gps_velocity_gate_chi2,
        mahalanobis, false);
}

bool EKF::UpdateNdtPose(double stamp, const SE3& pose_map_tracking,
                        const Matrix6d& covariance,
                        double gate_chi2, double* mahalanobis) {
    if (!PredictTo(stamp) || !pose_map_tracking.translation().allFinite() ||
        !pose_map_tracking.unit_quaternion().coeffs().allFinite() ||
        !covariance.allFinite()) {
        return false;
    }
    // 把旋转矩阵 \(R\) 转成欧拉角
    const Eigen::Vector3d measured_rpy = RpyFromRotation(pose_map_tracking.rotationMatrix());
    Eigen::Matrix<double, 6, 1> residual;
    residual.head<3>() =
        pose_map_tracking.translation() - state_.position_map;
    for (int axis = 0; axis < 3; ++axis) {
        residual(3 + axis) =
            WrapAngle(measured_rpy(axis) - state_.rpy_map(axis));
    }

    Eigen::Matrix<double, 6, kStateDim> jacobian =
        Eigen::Matrix<double, 6, kStateDim>::Zero();
    jacobian.block<3, 3>(0, kPositionX).setIdentity();
    jacobian.block<3, 3>(3, kRoll).setIdentity();

    return ApplyUpdate(
        residual, jacobian, covariance,
        gate_chi2 > 0.0 ? gate_chi2 : options_.ndt_pose_gate_chi2,
        mahalanobis, true);
}

bool EKF::ApplyUpdate(const Eigen::VectorXd& residual,
                      const Eigen::MatrixXd& measurement_jacobian,
                      const Eigen::MatrixXd& measurement_covariance,
                      double gate_chi2, double* mahalanobis,
                      bool allow_attitude_update) {
    if (!initialized_ || residual.size() == 0 || !residual.allFinite() ||
        measurement_jacobian.cols() != kStateDim ||
        measurement_jacobian.rows() != residual.size() ||
        measurement_covariance.rows() != residual.size() ||
        measurement_covariance.cols() != residual.size() ||
        !measurement_jacobian.allFinite() ||
        !measurement_covariance.allFinite()) {
        return false;
    }
    // R 观测噪声
    Eigen::MatrixXd noise =
        0.5 * (measurement_covariance + measurement_covariance.transpose());
    for (int index = 0; index < noise.rows(); ++index) {
        noise(index, index) =
            std::max(noise(index, index), options_.min_covariance);
    }
    //(P- + H^T*R*H) * δx = H^T*R*r
    // δx = (P- + H^T*R*H)^-1 * H^T*R*r
    // δx = P- * H^T *(H * P- * H^T + R)^-1 * r = P- * H^T *S^-1 * r = K*r
    // K = P- * H^T *S^-1
    //S= H * P- * H^T + R
    const Eigen::MatrixXd innovation_covariance =
        measurement_jacobian * covariance_ *
            measurement_jacobian.transpose() +
        noise;
    // S = L * D * L^T   3*3 的矩阵
    Eigen::LDLT<Eigen::MatrixXd> decomposition(innovation_covariance);
    if (decomposition.info() != Eigen::Success ||
        !decomposition.isPositive()) {
        return false;
    }
    // 通过解S * y = r  来求解 y= S^-1 * r
    const Eigen::VectorXd solved_residual = decomposition.solve(residual);
    if (decomposition.info() != Eigen::Success ||
        !solved_residual.allFinite()) {
        return false;
    }
   // r^T * y = r^T * S^-1 * r
    const double distance = residual.dot(solved_residual);
    if (mahalanobis) *mahalanobis = distance;
    if (!std::isfinite(distance) || distance < 0.0 ||
        (gate_chi2 > 0.0 && distance > gate_chi2)) {
        return false;
    }
    // H * P-
    const Eigen::MatrixXd right_hand_side = measurement_jacobian * covariance_;
    // （S * y）^T  = (H * P-)^T= P- * H^T
    //y^T= P- * H^T * S^-1 = K
    Eigen::MatrixXd gain = decomposition.solve(right_hand_side).transpose();
    if (decomposition.info() != Eigen::Success || !gain.allFinite()) {
        return false;
    }
    if (!allow_attitude_update) {
        // GNSS position and ENU velocity are not attitude sensors. Even if a
        // previous pose update created numerical cross-covariance, these
        // observations must not rotate RPY or alter angular velocity.
        gain.block(kRoll, 0, 3, gain.cols()).setZero();
        gain.block(kAngularVelocityX, 0, 3, gain.cols()).setZero();
    }

    StateVector vector = ToVector();
    vector += gain * residual;
    SetVector(vector);
    // P+ = (I - K * H) * P- * H^T + K * R
    const Covariance identity = Covariance::Identity();
    const Covariance joseph_left = identity - gain * measurement_jacobian;
    covariance_ =
        joseph_left * covariance_ * joseph_left.transpose() +
        gain * noise * gain.transpose();
    StabilizeCovariance();
    EnforceMotionConstraints();
    return true;
}

EKF::StateVector EKF::ToVector() const {
    StateVector vector;
    vector.segment<3>(kPositionX) = state_.position_map;
    vector.segment<3>(kRoll) = state_.rpy_map;
    vector.segment<3>(kVelocityX) = state_.velocity_map;
    vector.segment<3>(kAngularVelocityX) = state_.angular_velocity;
    return vector;
}

void EKF::SetVector(const StateVector& vector) {
    state_.position_map = vector.segment<3>(kPositionX);
    state_.rpy_map = vector.segment<3>(kRoll);
    for (int axis = 0; axis < 3; ++axis) {
        state_.rpy_map(axis) = WrapAngle(state_.rpy_map(axis));
    }
    state_.velocity_map = vector.segment<3>(kVelocityX);
    state_.angular_velocity = vector.segment<3>(kAngularVelocityX);
    state_.velocity_map.z() = 0.0;
    state_.angular_velocity.x() = 0.0;
    state_.angular_velocity.y() = 0.0;
}

void EKF::StabilizeCovariance() {
    covariance_ = 0.5 * (covariance_ + covariance_.transpose());
    Eigen::SelfAdjointEigenSolver<Covariance> solver(covariance_);
    if (solver.info() != Eigen::Success ||
        !solver.eigenvalues().allFinite()) {
        covariance_.setIdentity();
        covariance_ *= 100.0;
        return;
    }

    Eigen::Matrix<double, kStateDim, 1> eigenvalues = solver.eigenvalues();
    for (int index = 0; index < kStateDim; ++index) {
        eigenvalues(index) = std::clamp(
            eigenvalues(index), options_.min_covariance,
            options_.max_covariance);
    }
    covariance_ = solver.eigenvectors() * eigenvalues.asDiagonal() *
                  solver.eigenvectors().transpose();
    covariance_ = 0.5 * (covariance_ + covariance_.transpose());
}

void EKF::EnforceMotionConstraints() {
    state_.velocity_map.z() = 0.0;
    state_.angular_velocity.x() = 0.0;
    state_.angular_velocity.y() = 0.0;
    constexpr int kConstrainedIndices[] = {
        kVelocityZ, kAngularVelocityX, kAngularVelocityY};
    for (const int index : kConstrainedIndices) {
        covariance_.row(index).setZero();
        covariance_.col(index).setZero();
        covariance_(index, index) = options_.min_covariance;
    }
}

SE3 EKF::Pose() const {
    return SE3(
        Eigen::Quaterniond(RotationFromRpy(state_.rpy_map)),
        state_.position_map);
}

EKF::Matrix6d EKF::PoseCovariance() const {
    Matrix6d output = Matrix6d::Zero();
    output.topLeftCorner<3, 3>() =
        covariance_.block<3, 3>(kPositionX, kPositionX);
    output.topRightCorner<3, 3>() =
        covariance_.block<3, 3>(kPositionX, kRoll);
    output.bottomLeftCorner<3, 3>() =
        covariance_.block<3, 3>(kRoll, kPositionX);
    output.bottomRightCorner<3, 3>() =
        covariance_.block<3, 3>(kRoll, kRoll);
    return output;
}

EKF::Matrix6d EKF::TwistCovarianceBody() const {
    Eigen::Matrix<double, 6, kStateDim> jacobian =
        Eigen::Matrix<double, 6, kStateDim>::Zero();
    const Eigen::Matrix3d rotation = RotationFromRpy(state_.rpy_map);
    jacobian.block<3, 3>(0, kVelocityX) = rotation.transpose();
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::Vector3d plus = state_.rpy_map;
        Eigen::Vector3d minus = state_.rpy_map;
        plus(axis) += kJacobianEpsilon;
        minus(axis) -= kJacobianEpsilon;
        jacobian.block<3, 1>(0, kRoll + axis) =
            (RotationFromRpy(plus).transpose() * state_.velocity_map -
             RotationFromRpy(minus).transpose() * state_.velocity_map) /
            (2.0 * kJacobianEpsilon);
    }
    jacobian.block<3, 3>(3, kAngularVelocityX).setIdentity();
    return jacobian * covariance_ * jacobian.transpose();
}

double EKF::WrapAngle(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

Eigen::Matrix3d EKF::RotationFromRpy(const Eigen::Vector3d& rpy) {
    return (Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()))
        .toRotationMatrix();
}

Eigen::Vector3d EKF::RpyFromRotation(const Eigen::Matrix3d& rotation) {
    const double pitch = std::asin(std::clamp(
        -rotation(2, 0), -1.0, 1.0));
    const double cosine_pitch = std::cos(pitch);
    double roll = 0.0;
    double yaw = 0.0;
    if (std::fabs(cosine_pitch) > 1e-6) {
        roll = std::atan2(rotation(2, 1), rotation(2, 2));
        yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    } else {
        roll = 0.0;
        yaw = std::atan2(-rotation(0, 1), rotation(1, 1));
    }
    return Eigen::Vector3d(
        WrapAngle(roll), WrapAngle(pitch), WrapAngle(yaw));
}

}  // namespace lightning::loc
