#include "core/localization/EKF/ekf.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace lightning::loc {
namespace {

constexpr double kOmegaEpsilon = 1e-5;
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
                     double velocity, double yaw_rate,
                     const Covariance& covariance) {
    if (!std::isfinite(stamp) || !std::isfinite(x) || !std::isfinite(y) ||
        !std::isfinite(yaw) || !std::isfinite(velocity) ||
        !std::isfinite(yaw_rate) || !covariance.allFinite()) {
        return false;
    }

    state_.stamp = stamp;
    state_.x = x;
    state_.y = y;
    state_.yaw = WrapAngle(yaw);
    state_.velocity = velocity;
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
    const double yaw = state_.yaw;
    const double velocity = state_.velocity;
    const double yaw_rate = state_.yaw_rate;
    const double sine = std::sin(yaw);
    const double cosine = std::cos(yaw);

    StateMatrix jacobian = StateMatrix::Identity();
    if (std::fabs(yaw_rate) >= kOmegaEpsilon) {
        const double yaw2 = yaw + yaw_rate * dt;
        const double sine2 = std::sin(yaw2);
        const double cosine2 = std::cos(yaw2);
        const double inverse_rate = 1.0 / yaw_rate;
        const double inverse_rate_squared = inverse_rate * inverse_rate;
        const double sine_difference = sine2 - sine;
        const double cosine_difference = cosine - cosine2;

        state_.x += velocity * inverse_rate * sine_difference;
        state_.y += velocity * inverse_rate * cosine_difference;
        state_.yaw = WrapAngle(yaw2);

        jacobian(kX, kYaw) =
            velocity * inverse_rate * (cosine2 - cosine);
        jacobian(kX, kVelocity) = inverse_rate * sine_difference;
        jacobian(kX, kYawRate) =
            velocity * inverse_rate_squared *
            (yaw_rate * dt * cosine2 - sine_difference);
        jacobian(kY, kYaw) = velocity * inverse_rate * sine_difference;
        jacobian(kY, kVelocity) = inverse_rate * cosine_difference;
        jacobian(kY, kYawRate) =
            velocity * inverse_rate_squared *
            (yaw_rate * dt * sine2 - cosine_difference);
    } else {
        state_.x += velocity * cosine * dt;
        state_.y += velocity * sine * dt;
        state_.yaw = WrapAngle(yaw + yaw_rate * dt);

        jacobian(kX, kYaw) = -velocity * sine * dt;
        jacobian(kX, kVelocity) = cosine * dt;
        jacobian(kY, kYaw) = velocity * cosine * dt;
        jacobian(kY, kVelocity) = sine * dt;
    }
    jacobian(kYaw, kYawRate) = dt;

    Eigen::Matrix<double, kStateDim, 2> noise_input =
        Eigen::Matrix<double, kStateDim, 2>::Zero();
    const double half_dt_squared = 0.5 * dt * dt;
    noise_input(kX, 0) = cosine * half_dt_squared;
    noise_input(kY, 0) = sine * half_dt_squared;
    noise_input(kYaw, 1) = half_dt_squared;
    noise_input(kVelocity, 0) = dt;
    noise_input(kYawRate, 1) = dt;

    Eigen::Matrix2d driving_noise = Eigen::Matrix2d::Zero();
    driving_noise(0, 0) = options_.process_acceleration_std *
                          options_.process_acceleration_std;
    driving_noise(1, 1) = options_.process_yaw_acceleration_std *
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
    double stamp, const Eigen::Vector2d& sensor_velocity_map,
    const Eigen::Vector2d& lever_arm_tracking,
    const Eigen::Matrix2d& covariance, double gate_chi2,
    double* mahalanobis) {
    if (!PredictTo(stamp) || !sensor_velocity_map.allFinite() ||
        !lever_arm_tracking.allFinite() || !covariance.allFinite()) {
        return false;
    }

    const double cosine = std::cos(state_.yaw);
    const double sine = std::sin(state_.yaw);
    const double a =
        state_.velocity - state_.yaw_rate * lever_arm_tracking.y();
    const double b = state_.yaw_rate * lever_arm_tracking.x();
    const Eigen::Vector2d predicted(
        cosine * a - sine * b,
        sine * a + cosine * b);
    const Eigen::Vector2d residual = sensor_velocity_map - predicted;

    Eigen::Matrix<double, 2, kStateDim> jacobian =
        Eigen::Matrix<double, 2, kStateDim>::Zero();
    jacobian(0, kYaw) = -sine * a - cosine * b;
    jacobian(1, kYaw) = cosine * a - sine * b;
    jacobian(0, kVelocity) = cosine;
    jacobian(1, kVelocity) = sine;
    jacobian(0, kYawRate) =
        -cosine * lever_arm_tracking.y() -
        sine * lever_arm_tracking.x();
    jacobian(1, kYawRate) =
        -sine * lever_arm_tracking.y() +
        cosine * lever_arm_tracking.x();

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
                              double yaw_rate,
                              const Eigen::Matrix2d& covariance,
                              double gate_chi2, double* mahalanobis) {
    if (!PredictTo(stamp) || !std::isfinite(forward_velocity) ||
        !std::isfinite(yaw_rate) || !covariance.allFinite()) {
        return false;
    }

    Eigen::Vector2d residual(
        forward_velocity - state_.velocity,
        yaw_rate - state_.yaw_rate);
    Eigen::Matrix<double, 2, kStateDim> jacobian =
        Eigen::Matrix<double, 2, kStateDim>::Zero();
    jacobian(0, kVelocity) = 1.0;
    jacobian(1, kYawRate) = 1.0;
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
              state_.velocity, state_.yaw_rate;
    return vector;
}

void EKF::SetVector(const StateVector& vector) {
    state_.x = vector(kX);
    state_.y = vector(kY);
    state_.yaw = WrapAngle(vector(kYaw));
    state_.velocity = vector(kVelocity);
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
        state_.velocity * std::cos(state_.yaw),
        state_.velocity * std::sin(state_.yaw), 0.0);
}

double EKF::WrapAngle(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

}  // namespace lightning::loc
