#include "core/localization/EKF/ekf.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace lightning::loc {
namespace {

constexpr double kTimeEpsilon = 1e-9;

}  // namespace

EKF::EKF() { Reset(); }

EKF::EKF(const Options& options) : options_(options) { Reset(); }

void EKF::Configure(const Options& options) { options_ = options; }

void EKF::Reset() {
    initialized_ = false;
    state_ = State();
    covariance_.setIdentity();
}

bool EKF::Initialize(double stamp, double x, double y, double yaw,
                     const Eigen::Vector2d& velocity_map, double yaw_rate,
                     const Covariance& covariance) {
    if (!std::isfinite(stamp) || !std::isfinite(x) || !std::isfinite(y) ||
        !std::isfinite(yaw) || !velocity_map.allFinite() ||
        !std::isfinite(yaw_rate) || !covariance.allFinite()) {
        return false;
    }

    state_.stamp = stamp;
    state_.x = x;
    state_.y = y;
    state_.yaw = WrapAngle(yaw);
    state_.velocity_x_map = velocity_map.x();
    state_.velocity_y_map = velocity_map.y();
    state_.yaw_rate = yaw_rate;
    covariance_ = covariance;
    initialized_ = true;
    StabilizeCovariance();
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
    // Constant velocity in map coordinates and constant yaw rate. This model
    // is linear except for wrapping yaw after propagation.
    StateMatrix jacobian = StateMatrix::Identity();
    jacobian(kX, kVelocityX) = dt;
    jacobian(kY, kVelocityY) = dt;
    jacobian(kYaw, kYawRate) = dt;

    state_.x += state_.velocity_x_map * dt;
    state_.y += state_.velocity_y_map * dt;
    state_.yaw = WrapAngle(state_.yaw + state_.yaw_rate * dt);

    // Driving noise is [acceleration_x_map, acceleration_y_map,
    // yaw_acceleration]. The discrete G matrix integrates acceleration into
    // both pose and velocity for this prediction interval.
    Eigen::Matrix<double, kStateDim, 3> noise_input =
        Eigen::Matrix<double, kStateDim, 3>::Zero();
    const double half_dt_squared = 0.5 * dt * dt;
    noise_input(kX, 0) = half_dt_squared;
    noise_input(kY, 1) = half_dt_squared;
    noise_input(kYaw, 2) = half_dt_squared;
    noise_input(kVelocityX, 0) = dt;
    noise_input(kVelocityY, 1) = dt;
    noise_input(kYawRate, 2) = dt;

    Eigen::Matrix3d driving_noise = Eigen::Matrix3d::Zero();
    driving_noise(0, 0) = options_.process_acceleration_std *
                          options_.process_acceleration_std;
    driving_noise(1, 1) = options_.process_acceleration_std *
                          options_.process_acceleration_std;
    driving_noise(2, 2) = options_.process_yaw_acceleration_std *
                          options_.process_yaw_acceleration_std;
    const Covariance process_noise =
        noise_input * driving_noise * noise_input.transpose();

    covariance_ = jacobian * covariance_ * jacobian.transpose() +
                  process_noise;
    StabilizeCovariance();
}

bool EKF::UpdateRtkPosition(
    double stamp, const Eigen::Vector2d& sensor_position_map,
    const Eigen::Vector2d& lever_arm_tracking,
    const Eigen::Matrix2d& covariance, double gate_chi2,
    double* mahalanobis) {
    if (!PredictTo(stamp) || !sensor_position_map.allFinite() ||
        !lever_arm_tracking.allFinite() || !covariance.allFinite()) {
        return false;
    }

    const double cosine = std::cos(state_.yaw);
    const double sine = std::sin(state_.yaw);
    Eigen::Matrix2d rotation;
    rotation << cosine, -sine,
                sine, cosine;
    const Eigen::Vector2d position(state_.x, state_.y);
    const Eigen::Vector2d residual =
        sensor_position_map - (position + rotation * lever_arm_tracking);

    Eigen::Matrix<double, 2, kStateDim> jacobian =
        Eigen::Matrix<double, 2, kStateDim>::Zero();
    jacobian(0, kX) = 1.0;
    jacobian(1, kY) = 1.0;
    jacobian(0, kYaw) =
        -sine * lever_arm_tracking.x() -
        cosine * lever_arm_tracking.y();
    jacobian(1, kYaw) =
        cosine * lever_arm_tracking.x() -
        sine * lever_arm_tracking.y();

    return ApplyUpdate(
        residual, jacobian, covariance,
        gate_chi2 > 0.0 ? gate_chi2 : options_.rtk_position_gate_chi2,
        mahalanobis);
}

bool EKF::UpdateYaw(double stamp, double yaw, double variance,
                    double gate_chi2, double* mahalanobis) {
    if (!PredictTo(stamp) || !std::isfinite(yaw) ||
        !std::isfinite(variance) || variance <= 0.0) {
        return false;
    }

    Eigen::Matrix<double, 1, 1> residual;
    residual(0) = WrapAngle(yaw - state_.yaw);
    Eigen::Matrix<double, 1, kStateDim> jacobian =
        Eigen::Matrix<double, 1, kStateDim>::Zero();
    jacobian(0, kYaw) = 1.0;
    Eigen::Matrix<double, 1, 1> noise;
    noise(0, 0) = variance;
    return ApplyUpdate(
        residual, jacobian, noise,
        gate_chi2 > 0.0 ? gate_chi2 : options_.ins_yaw_gate_chi2,
        mahalanobis);
}

bool EKF::UpdateMapVelocity(
    double stamp, const Eigen::Vector2d& velocity_map,
    const Eigen::Matrix2d& covariance, double gate_chi2,
    double* mahalanobis) {
    if (!PredictTo(stamp) || !velocity_map.allFinite() ||
        !covariance.allFinite()) {
        return false;
    }

    const Eigen::Vector2d predicted(
        state_.velocity_x_map, state_.velocity_y_map);
    const Eigen::Vector2d residual = velocity_map - predicted;

    Eigen::Matrix<double, 2, kStateDim> jacobian =
        Eigen::Matrix<double, 2, kStateDim>::Zero();
    jacobian(0, kVelocityX) = 1.0;
    jacobian(1, kVelocityY) = 1.0;

    return ApplyUpdate(
        residual, jacobian, covariance,
        gate_chi2 > 0.0 ? gate_chi2 : options_.rtk_velocity_gate_chi2,
        mahalanobis);
}

bool EKF::UpdateNdtPose(double stamp, const Eigen::Vector3d& pose,
                        const Eigen::Matrix3d& covariance,
                        double gate_chi2, double* mahalanobis) {
    if (!PredictTo(stamp) || !pose.allFinite() || !covariance.allFinite()) {
        return false;
    }

    Eigen::Vector3d residual;
    residual << pose.x() - state_.x,
                pose.y() - state_.y,
                WrapAngle(pose.z() - state_.yaw);
    Eigen::Matrix<double, 3, kStateDim> jacobian =
        Eigen::Matrix<double, 3, kStateDim>::Zero();
    jacobian(0, kX) = 1.0;
    jacobian(1, kY) = 1.0;
    jacobian(2, kYaw) = 1.0;
    return ApplyUpdate(
        residual, jacobian, covariance,
        gate_chi2 > 0.0 ? gate_chi2 : options_.ndt_pose_gate_chi2,
        mahalanobis);
}

bool EKF::UpdateWheelOdometry(double stamp, double forward_velocity,
                              double variance,
                              double gate_chi2, double* mahalanobis) {
    if (!PredictTo(stamp) || !std::isfinite(forward_velocity) ||
        !std::isfinite(variance) || variance <= 0.0) {
        return false;
    }

    const double cosine = std::cos(state_.yaw);
    const double sine = std::sin(state_.yaw);
    const double predicted_forward_velocity =
        cosine * state_.velocity_x_map +
        sine * state_.velocity_y_map;
    Eigen::Matrix<double, 1, 1> residual;
    residual(0) = forward_velocity - predicted_forward_velocity;
    Eigen::Matrix<double, 1, kStateDim> jacobian =
        Eigen::Matrix<double, 1, kStateDim>::Zero();
    jacobian(0, kYaw) =
        -sine * state_.velocity_x_map +
        cosine * state_.velocity_y_map;
    jacobian(0, kVelocityX) = cosine;
    jacobian(0, kVelocityY) = sine;
    Eigen::Matrix<double, 1, 1> covariance;
    covariance(0, 0) = variance;
    return ApplyUpdate(
        residual, jacobian, covariance,
        gate_chi2 > 0.0 ? gate_chi2 : options_.wheel_gate_chi2,
        mahalanobis);
}

bool EKF::ApplyUpdate(const Eigen::VectorXd& residual,
                      const Eigen::MatrixXd& measurement_jacobian,
                      const Eigen::MatrixXd& measurement_covariance,
                      double gate_chi2, double* mahalanobis) {
    if (!initialized_ || residual.size() == 0 || !residual.allFinite() ||
        measurement_jacobian.cols() != kStateDim ||
        measurement_jacobian.rows() != residual.size() ||
        measurement_covariance.rows() != residual.size() ||
        measurement_covariance.cols() != residual.size() ||
        !measurement_jacobian.allFinite() ||
        !measurement_covariance.allFinite()) {
        return false;
    }

    Eigen::MatrixXd noise =
        0.5 * (measurement_covariance + measurement_covariance.transpose());
    for (int index = 0; index < noise.rows(); ++index) {
        noise(index, index) =
            std::max(noise(index, index), options_.min_covariance);
    }

    const Eigen::MatrixXd innovation_covariance =
        measurement_jacobian * covariance_ *
            measurement_jacobian.transpose() +
        noise;
    Eigen::LDLT<Eigen::MatrixXd> decomposition(innovation_covariance);
    if (decomposition.info() != Eigen::Success ||
        !decomposition.isPositive()) {
        return false;
    }

    const Eigen::VectorXd solved_residual = decomposition.solve(residual);
    if (decomposition.info() != Eigen::Success ||
        !solved_residual.allFinite()) {
        return false;
    }
    const double distance = residual.dot(solved_residual);
    if (mahalanobis) *mahalanobis = distance;
    if (!std::isfinite(distance) || distance < 0.0 ||
        (gate_chi2 > 0.0 && distance > gate_chi2)) {
        return false;
    }

    const Eigen::MatrixXd right_hand_side =
        measurement_jacobian * covariance_;
    const Eigen::MatrixXd gain =
        decomposition.solve(right_hand_side).transpose();
    if (decomposition.info() != Eigen::Success || !gain.allFinite()) {
        return false;
    }

    StateVector vector = ToVector();
    vector += gain * residual;
    SetVector(vector);

    const Covariance identity = Covariance::Identity();
    const Covariance joseph_left =
        identity - gain * measurement_jacobian;
    covariance_ =
        joseph_left * covariance_ * joseph_left.transpose() +
        gain * noise * gain.transpose();
    StabilizeCovariance();
    return true;
}

EKF::StateVector EKF::ToVector() const {
    StateVector vector;
    vector << state_.x, state_.y, state_.yaw,
              state_.velocity_x_map, state_.velocity_y_map,
              state_.yaw_rate;
    return vector;
}

void EKF::SetVector(const StateVector& vector) {
    state_.x = vector(kX);
    state_.y = vector(kY);
    state_.yaw = WrapAngle(vector(kYaw));
    state_.velocity_x_map = vector(kVelocityX);
    state_.velocity_y_map = vector(kVelocityY);
    state_.yaw_rate = vector(kYawRate);
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

SE3 EKF::Pose() const {
    const Eigen::Quaterniond orientation(
        Eigen::AngleAxisd(state_.yaw, Eigen::Vector3d::UnitZ()));
    return SE3(
        orientation, Eigen::Vector3d(state_.x, state_.y, 0.0));
}

Eigen::Vector3d EKF::VelocityMap() const {
    return Eigen::Vector3d(
        state_.velocity_x_map, state_.velocity_y_map, 0.0);
}

double EKF::WrapAngle(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

}  // namespace lightning::loc
