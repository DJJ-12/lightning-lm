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
/*
Eigen::Matrix<double,3,3> EKF::ComputeLeverArmJacobian(
    const Eigen::Vector3d& lever_arm,
    const Eigen::Vector3d& rpy)
{
    const double roll  = rpy.x();
    const double pitch = rpy.y();
    const double yaw   = rpy.z();


    const double cr = cos(roll);
    const double sr = sin(roll);

    const double cp = cos(pitch);
    const double sp = sin(pitch);

    const double cy = cos(yaw);
    const double sy = sin(yaw);

    Eigen::Matrix3d Rx;
    Rx << 1,0,0,
        0,cr,-sr,
        0,sr,cr;

    Eigen::Matrix3d Ry;
    Ry << cp,0,sp,
        0,1,0,
        -sp,0,cp;

    Eigen::Matrix3d Rz;
    Rz << cy,-sy,0,
        sy, cy,0,
        0,0,1;

    Eigen::Matrix3d dRx;
    dRx << 0,0,0,
        0,-sr,-cr,
        0, cr,-sr;

    Eigen::Matrix3d dRy;
    dRy << -sp,0,cp,
        0,0,0,
        -cp,0,-sp;

    Eigen::Matrix3d dRz;
    dRz << -sy,-cy,0,
        cy,-sy,0,
        0,0,0;

    Eigen::Matrix3d dR_roll = Rz * Ry * dRx;

    Eigen::Matrix3d dR_pitch = Rz * dRy * Rx;

    Eigen::Matrix3d dR_yaw = dRz * Ry * Rx;

    Eigen::Matrix<double,3,3> J;

    J.col(0)=dR_roll * lever_arm;

    J.col(1)=dR_pitch * lever_arm;

    J.col(2)=dR_yaw * lever_arm;

    return J;
}

bool EKF::UpdateGpsPoseEnu(
    double stamp,
    const Eigen::Vector3d& gps_position_enu,
    const Eigen::Vector3d& antenna_position_body,
    const Eigen::Matrix3d& gps_covariance_enu,
    const Eigen::Matrix3d& rotation_enu_map,
    const Eigen::Vector3d& translation_enu_map,
    double gate_chi2,
    double* mahalanobis) {
    if (!PredictTo(stamp)) return false;

    // NavSatFix measures antenna point A in ENU. antenna_position_body is
    // p_B_A: that same physical point expressed in body coordinates.
    // Therefore the predicted observation is the antenna, not the body or GPS
    // device origin:
    //   p_E_A = R_E_M * (p_M_B + R_M_B * p_B_A) + t_E_M.
    const Eigen::Matrix3d rotation_map_body =
        RotationFromRpy(state_.rpy_map);
    const Eigen::Vector3d gps_predicted_enu =
        rotation_enu_map *
            (state_.position_map +
             rotation_map_body * antenna_position_body) +
        translation_enu_map;

    const Eigen::Vector3d residual =
        gps_position_enu - gps_predicted_enu;
    Eigen::Matrix<double, 3, kStateDim> jacobian =
        Eigen::Matrix<double, 3, kStateDim>::Zero();

    // d h / d p_map = R_enu_map.
    jacobian.block<3, 3>(0, kPositionX) = rotation_enu_map;

    // d h / d rpy = R_enu_map * d(R_map_body * lever)/d(rpy).
    const Eigen::Matrix3d attitude_jacobian =
        ComputeLeverArmJacobian(antenna_position_body, state_.rpy_map);
    jacobian.block<3, 3>(0, kRoll) =
        rotation_enu_map * attitude_jacobian;

    return ApplyUpdate(
        residual, jacobian, gps_covariance_enu,
        gate_chi2 > 0.0 ? gate_chi2 : options_.gps_position_gate_chi2,
        mahalanobis, true);
}

*/
bool EKF::UpdateGpsPoseMap(
    double stamp,
    const Eigen::Vector3d& gps_position_map,
    const Eigen::Matrix3d& gps_covariance_map,
    double gate_chi2, double* mahalanobis) {
    if (!PredictTo(stamp)) return false;

    const Eigen::Vector3d residual =
        gps_position_map - state_.position_map;
    Eigen::Matrix<double, 3, kStateDim> jacobian =
        Eigen::Matrix<double, 3, kStateDim>::Zero();

    jacobian.block<3, 3>(0, kPositionX).setIdentity();

    return ApplyUpdate(
        residual, jacobian, gps_covariance_map,
        gate_chi2 > 0.0 ? gate_chi2 : options_.gps_position_gate_chi2,
        mahalanobis, true);
}

bool EKF::UpdateMapOrientation(
    double stamp,
    const Eigen::Vector3d& measured_rpy_map_body,
    const Eigen::Matrix3d& covariance_map_body,
    double gate_chi2,
    double* mahalanobis) {
    if (!PredictTo(stamp)) return false;

    // Both sides are MAP<-BODY Euler angles.  Keep each angular innovation on
    // the shortest branch across the +/-pi boundary.
    Eigen::Vector3d residual;
    for (int axis = 0; axis < 3; ++axis) {
        residual(axis) = WrapAngle(
            measured_rpy_map_body(axis) - state_.rpy_map(axis));
    }

    Eigen::Matrix<double, 3, kStateDim> jacobian =
        Eigen::Matrix<double, 3, kStateDim>::Zero();
    jacobian.block<3, 3>(0, kRoll).setIdentity();

    return ApplyUpdate(
        residual, jacobian, covariance_map_body,
        gate_chi2 > 0.0 ? gate_chi2
                        : options_.gps_orientation_gate_chi2,
        mahalanobis, true);
}

bool EKF::UpdateMapVelocity(
    double stamp, const Eigen::Vector2d& velocity_map_xy,
    const Eigen::Matrix2d& covariance, double gate_chi2,
    double* mahalanobis) {
    if (!PredictTo(stamp)) return false;

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

bool EKF::UpdateNdtPose(double stamp, const SE3& pose_map_body,
                        const Matrix6d& covariance,
                        double gate_chi2, double* mahalanobis) {
    if (!PredictTo(stamp)) return false;
    // 把旋转矩阵 \(R\) 转成欧拉角
    const Eigen::Vector3d measured_rpy =
        RpyFromRotation(pose_map_body.rotationMatrix());
    Eigen::Matrix<double, 6, 1> residual;
    residual.head<3>() =
        pose_map_body.translation() - state_.position_map;
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
    // Observation payloads are validated at the LocalizationSystem input
    // boundary. This private function only handles failures produced by the
    // actual EKF calculation.
    // R 观测噪声
    Eigen::MatrixXd noise =
        0.5 * (measurement_covariance + measurement_covariance.transpose());
    for (int index = 0; index < noise.rows(); ++index) {
        noise(index, index) =
            std::max(noise(index, index), options_.min_covariance);
    }
    // Innovation covariance and Kalman gain:
    //   S = H * P- * H^T + R
    //   K = P- * H^T * S^-1
    //   delta_x = K * residual
    const Eigen::MatrixXd innovation_covariance =
        measurement_jacobian * covariance_ *
            measurement_jacobian.transpose() +
        noise;
    // Solve with LDLT instead of explicitly forming S^-1.
    Eigen::LDLT<Eigen::MatrixXd> decomposition(innovation_covariance);
    if (decomposition.info() != Eigen::Success ||
        !decomposition.isPositive()) {
        return false;
    }
    const Eigen::VectorXd solved_residual = decomposition.solve(residual);
    if (!solved_residual.allFinite()) {
        return false;
    }
    // Squared Mahalanobis distance: residual^T * S^-1 * residual.
    const double distance = residual.dot(solved_residual);
    if (mahalanobis) *mahalanobis = distance;
    if (!std::isfinite(distance) || distance < 0.0 ||
        (gate_chi2 > 0.0 && distance > gate_chi2)) {
        return false;
    }
    const Eigen::MatrixXd right_hand_side = measurement_jacobian * covariance_;
    Eigen::MatrixXd gain = decomposition.solve(right_hand_side).transpose();
    if (!gain.allFinite()) return false;
    if (!allow_attitude_update) {
        // Map-frame velocity is not an attitude observation. Even if a pose
        // update created cross-covariance, a velocity-only measurement must
        // not rotate RPY or alter angular velocity. NDT observes RPY directly;
        // a GPS position update may affect it only through existing state
        // cross-covariance, which is standard EKF behavior.
        gain.block(kRoll, 0, 3, gain.cols()).setZero();
        gain.block(kAngularVelocityX, 0, 3, gain.cols()).setZero();
    }

    StateVector vector = ToVector();
    vector += gain * residual;
    SetVector(vector);
    // Joseph form keeps the posterior covariance symmetric and positive:
    //   P+ = (I-KH)P-(I-KH)^T + KRK^T.
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
