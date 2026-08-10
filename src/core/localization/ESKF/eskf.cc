#include "core/localization/ESKF/eskf.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

namespace lightning::loc {
namespace {

constexpr int kPositionIndex = 0;
constexpr int kOrientationIndex = 3;
constexpr int kVelocityIndex = 6;
constexpr int kAngularVelocityIndex = 9;
constexpr double kPi = 3.14159265358979323846;

}  // namespace

ESKF::ESKF() { Reset(); }
ESKF::ESKF(const Options& options) : options_(options) { Reset(); }
void ESKF::Configure(const Options& options) { options_ = options; }

void ESKF::Reset() {
    initialized_ = false;
    state_ = NominalState();
    covariance_.setIdentity();
}

bool ESKF::Initialize(double stamp, const SE3& pose, const Eigen::Vector3d& velocity_body, const Eigen::Vector3d& angular_velocity_body, const Covariance& covariance) {
    if (!std::isfinite(stamp) || !pose.translation().allFinite() || !velocity_body.allFinite() || !angular_velocity_body.allFinite() || !covariance.allFinite()) return false;
    state_.stamp = stamp;
    state_.position = pose.translation();
    state_.orientation = pose.unit_quaternion();
    state_.orientation.normalize();
    state_.velocity_body = velocity_body;
    state_.angular_velocity_body = angular_velocity_body;
    covariance_ = covariance;
    initialized_ = true;
    StabilizeCovariance();
    return true;
}

bool ESKF::PredictTo(double stamp) {
    if (!initialized_ || !std::isfinite(stamp)) return false;
    double remaining = stamp - state_.stamp;
    if (remaining < -1e-6) return false;
    if (remaining <= 1e-9) { state_.stamp = std::max(state_.stamp, stamp); return true; }
    const double max_step = std::max(1e-3, options_.max_prediction_step);
    while (remaining > 1e-9) {
        const double dt = std::min(remaining, max_step);
        PropagateStep(dt);
        remaining -= dt;
    }
    state_.stamp = stamp;
    return true;
}

void ESKF::PropagateStep(double dt) {
    const Eigen::Matrix3d rotation_map_body = state_.orientation.toRotationMatrix(); // R(t)
    const Eigen::Vector3d rotation_increment = state_.angular_velocity_body * dt; // w*dt
    state_.position += rotation_map_body * (LeftJacobian(rotation_increment) * state_.velocity_body * dt); // R(t+dt)*v(t)*dt
    state_.orientation = Eigen::Quaterniond(rotation_map_body * Exp(rotation_increment));//R(t)*exp(w*dt)
    state_.orientation.normalize(); // 归一化

    Covariance continuous_f = Covariance::Zero();
    continuous_f.block<3, 3>(kPositionIndex, kOrientationIndex) = -rotation_map_body * Skew(state_.velocity_body); // -R(t) * [v]×
    continuous_f.block<3, 3>(kPositionIndex, kVelocityIndex) = rotation_map_body; // R
    continuous_f.block<3, 3>(kOrientationIndex, kOrientationIndex) = -Skew(state_.angular_velocity_body); // -[w]×
    continuous_f.block<3, 3>(kOrientationIndex, kAngularVelocityIndex) = Eigen::Matrix3d::Identity(); //1 

    const Covariance identity = Covariance::Identity(); // I
    const Covariance f2 = continuous_f * continuous_f; // 
    const Covariance transition = identity + continuous_f * dt + 0.5 * f2 * dt * dt; // I + F *dt +0.5 * F^2 * dt^2 残差的预测方程

    Covariance continuous_noise = Covariance::Zero();
    continuous_noise.block<3, 3>(kVelocityIndex, kVelocityIndex) = std::pow(options_.body_acceleration_noise_std, 2) * Eigen::Matrix3d::Identity(); // na
    continuous_noise.block<3, 3>(kAngularVelocityIndex, kAngularVelocityIndex) = std::pow(options_.angular_acceleration_noise_std, 2) * Eigen::Matrix3d::Identity(); //nb
    const double dt2 = dt * dt; //平方
    const double dt3 = dt2 * dt; // 立方
    Covariance discrete_noise = continuous_noise * dt; //  G * dt
    discrete_noise += 0.5 * (continuous_f * continuous_noise + continuous_noise * continuous_f.transpose()) * dt2; // 
    discrete_noise += (continuous_f * continuous_noise * continuous_f.transpose()) * dt3 / 3.0;
    discrete_noise = 0.5 * (discrete_noise + discrete_noise.transpose());
    // 先验方差P-(t+dt) = (I + F *dt) * P+(t) * (I + F *dt)T + (G*dt) * noise * (G*dt)T
    covariance_ = transition * covariance_ * transition.transpose() + discrete_noise; 
    StabilizeCovariance();
}

bool ESKF::UpdatePose(double stamp, const SE3& pose, const Eigen::Matrix<double, 6, 6>& covariance, double gate_chi2, double* mahalanobis) {
    if (!PredictTo(stamp) || !pose.translation().allFinite() || !covariance.allFinite()) return false;
    Eigen::Matrix<double, 6, 1> residual; // 位值+姿态 残差
    residual.head<3>() = pose.translation() - state_.position; // 位置
    residual.tail<3>() = Log(state_.orientation.toRotationMatrix().transpose() * pose.so3().matrix()); // log(R(t+dt)-T * Zθ)
    Eigen::Matrix<double, 6, kStateDim> jacobian = Eigen::Matrix<double, 6, kStateDim>::Zero();
    jacobian.block<3, 3>(0, kPositionIndex) = Eigen::Matrix3d::Identity(); // H 观测方程系数矩阵
    jacobian.block<3, 3>(3, kOrientationIndex) = Eigen::Matrix3d::Identity();
    return ApplyUpdate(residual, jacobian, covariance, gate_chi2 > 0.0 ? gate_chi2 : options_.pose_gate_chi2, mahalanobis);
}

bool ESKF::UpdatePosition(double stamp, const Eigen::Vector3d& position, const Eigen::Matrix3d& covariance, const Eigen::Array3i& observed_axes, double gate_chi2, double* mahalanobis) {
    return UpdatePositionWithLeverArm(stamp, position, Eigen::Vector3d::Zero(), covariance, observed_axes, gate_chi2, mahalanobis);
}

bool ESKF::UpdatePositionWithLeverArm(double stamp, const Eigen::Vector3d& sensor_position, const Eigen::Vector3d& lever_arm_body, const Eigen::Matrix3d& covariance, const Eigen::Array3i& observed_axes, double gate_chi2, double* mahalanobis) {
    if (!PredictTo(stamp) || !sensor_position.allFinite() || !lever_arm_body.allFinite() || !covariance.allFinite()) return false;
    std::vector<int> axes;
    for (int axis = 0; axis < 3; ++axis) if (observed_axes(axis) != 0) axes.push_back(axis);
    if (axes.empty()) return false;
    const Eigen::Matrix3d rotation_map_body = state_.orientation.toRotationMatrix(); // R(t)
    const Eigen::Vector3d predicted_sensor_position = state_.position + rotation_map_body * lever_arm_body; //p(t) + R(t)*L  rtk 传感器位置
    const Eigen::Matrix3d orientation_jacobian = -rotation_map_body * Skew(lever_arm_body); //-R[L]×
    Eigen::VectorXd residual(axes.size());
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(axes.size(), kStateDim);
    Eigen::MatrixXd noise(axes.size(), axes.size());
    for (int row = 0; row < static_cast<int>(axes.size()); ++row) {
        const int axis = axes[row];
        residual(row) = sensor_position(axis) - predicted_sensor_position(axis);
        jacobian(row, kPositionIndex + axis) = 1.0;
        jacobian.block<1, 3>(row, kOrientationIndex) = orientation_jacobian.row(axis);
        for (int col = 0; col < static_cast<int>(axes.size()); ++col) noise(row, col) = covariance(axis, axes[col]);
    }
    return ApplyUpdate(residual, jacobian, noise, gate_chi2 > 0.0 ? gate_chi2 : options_.position_gate_chi2, mahalanobis);
}

bool ESKF::UpdateYaw(double stamp, double yaw, double variance, double gate_chi2, double* mahalanobis) {
    if (!PredictTo(stamp) || !std::isfinite(yaw) || !std::isfinite(variance) || variance <= 0.0) return false;
    Eigen::Matrix<double, 1, 1> residual;
    residual(0) = WrapAngle(yaw - Yaw(state_.orientation.toRotationMatrix()));
    Eigen::Matrix<double, 1, kStateDim> jacobian = Eigen::Matrix<double, 1, kStateDim>::Zero();
    jacobian.block<1, 3>(0, kOrientationIndex) = YawJacobian();
    Eigen::Matrix<double, 1, 1> noise;
    noise(0, 0) = variance;
    return ApplyUpdate(residual, jacobian, noise, gate_chi2 > 0.0 ? gate_chi2 : options_.yaw_gate_chi2, mahalanobis);
}

bool ESKF::UpdateMapVelocity(double stamp, const Eigen::Vector3d& velocity_map, const Eigen::Matrix3d& covariance, const Eigen::Array3i& observed_axes, double gate_chi2, double* mahalanobis) {
    return UpdateMapVelocityWithLeverArm(stamp, velocity_map, Eigen::Vector3d::Zero(), covariance, observed_axes, gate_chi2, mahalanobis);
}

bool ESKF::UpdateMapVelocityWithLeverArm(double stamp, const Eigen::Vector3d& sensor_velocity_map, const Eigen::Vector3d& lever_arm_body, const Eigen::Matrix3d& covariance, const Eigen::Array3i& observed_axes, double gate_chi2, double* mahalanobis) {
    if (!PredictTo(stamp) || !sensor_velocity_map.allFinite() || !lever_arm_body.allFinite() || !covariance.allFinite()) return false;
    std::vector<int> axes;
    for (int axis = 0; axis < 3; ++axis) if (observed_axes(axis) != 0) axes.push_back(axis);
    if (axes.empty()) return false;
    const Eigen::Matrix3d rotation_map_body = state_.orientation.toRotationMatrix(); // R-
    const Eigen::Vector3d sensor_velocity_body = state_.velocity_body + state_.angular_velocity_body.cross(lever_arm_body); //v + w × L
    const Eigen::Vector3d predicted_sensor_velocity_map = rotation_map_body * sensor_velocity_body; // R * （ v + w × L ）
    const Eigen::Matrix3d orientation_jacobian = -rotation_map_body * Skew(sensor_velocity_body); // H = 0, -R[v_sensor_body]× , R , -R[l]×
    const Eigen::Matrix3d angular_velocity_jacobian = -rotation_map_body * Skew(lever_arm_body);
    Eigen::VectorXd residual(axes.size());
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(axes.size(), kStateDim);
    Eigen::MatrixXd noise(axes.size(), axes.size());
    for (int row = 0; row < static_cast<int>(axes.size()); ++row) {
        const int axis = axes[row];
        residual(row) = sensor_velocity_map(axis) - predicted_sensor_velocity_map(axis);
        jacobian.block<1, 3>(row, kOrientationIndex) = orientation_jacobian.row(axis);
        jacobian.block<1, 3>(row, kVelocityIndex) = rotation_map_body.row(axis);
        jacobian.block<1, 3>(row, kAngularVelocityIndex) = angular_velocity_jacobian.row(axis);
        for (int col = 0; col < static_cast<int>(axes.size()); ++col) noise(row, col) = covariance(axis, axes[col]);
    }
    return ApplyUpdate(residual, jacobian, noise, gate_chi2 > 0.0 ? gate_chi2 : options_.twist_gate_chi2, mahalanobis);
}

bool ESKF::UpdateBodyTwist(double stamp, const Eigen::Vector3d& linear_velocity_body, const Eigen::Vector3d& angular_velocity_body, const Eigen::Matrix<double, 6, 6>& covariance, const Eigen::Array3i& linear_axes, const Eigen::Array3i& angular_axes, double gate_chi2, double* mahalanobis) {
    if (!PredictTo(stamp) || !linear_velocity_body.allFinite() || !angular_velocity_body.allFinite() || !covariance.allFinite()) return false;
    std::vector<int> selected_indices;
    for (int axis = 0; axis < 3; ++axis) if (linear_axes(axis) != 0) selected_indices.push_back(axis);
    for (int axis = 0; axis < 3; ++axis) if (angular_axes(axis) != 0) selected_indices.push_back(3 + axis);
    if (selected_indices.empty()) return false;
    Eigen::VectorXd residual(selected_indices.size());
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(selected_indices.size(), kStateDim);
    Eigen::MatrixXd noise(selected_indices.size(), selected_indices.size());
    for (int row = 0; row < static_cast<int>(selected_indices.size()); ++row) {
        const int index = selected_indices[row];
        if (index < 3) {
            residual(row) = linear_velocity_body(index) - state_.velocity_body(index);
            jacobian(row, kVelocityIndex + index) = 1.0;
        } else {
            const int axis = index - 3;
            residual(row) = angular_velocity_body(axis) - state_.angular_velocity_body(axis);
            jacobian(row, kAngularVelocityIndex + axis) = 1.0;
        }
        for (int col = 0; col < static_cast<int>(selected_indices.size()); ++col) noise(row, col) = covariance(index, selected_indices[col]);
    }
    return ApplyUpdate(residual, jacobian, noise, gate_chi2 > 0.0 ? gate_chi2 : options_.twist_gate_chi2, mahalanobis);
}

bool ESKF::UpdateAngularVelocity(double stamp, const Eigen::Vector3d& angular_velocity_body, const Eigen::Matrix3d& covariance, const Eigen::Array3i& observed_axes, double gate_chi2, double* mahalanobis) {
    Eigen::Matrix<double, 6, 6> twist_covariance = Eigen::Matrix<double, 6, 6>::Zero();
    twist_covariance.block<3, 3>(0, 0) = 1e8 * Eigen::Matrix3d::Identity();
    twist_covariance.block<3, 3>(3, 3) = covariance;
    return UpdateBodyTwist(stamp, Eigen::Vector3d::Zero(), angular_velocity_body, twist_covariance, Eigen::Array3i(0, 0, 0), observed_axes, gate_chi2, mahalanobis);
}

bool ESKF::ApplyUpdate(const Eigen::VectorXd& residual, const Eigen::MatrixXd& measurement_jacobian, const Eigen::MatrixXd& measurement_covariance, double gate_chi2, double* mahalanobis) {
    if (!initialized_ || residual.size() == 0 || !residual.allFinite() || !measurement_jacobian.allFinite() || !measurement_covariance.allFinite()) return false;
    Eigen::MatrixXd noise = 0.5 * (measurement_covariance + measurement_covariance.transpose()); // 观测噪声
    for (int i = 0; i < noise.rows(); ++i) noise(i, i) = std::max(noise(i, i), options_.min_covariance); // 给观测噪声的设置下限
    const Eigen::MatrixXd innovation_covariance = measurement_jacobian * covariance_ * measurement_jacobian.transpose() + noise; // S = H*P-*HT+Σ_Z
    Eigen::LDLT<Eigen::MatrixXd> decomposition(innovation_covariance);
    if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) return false;
    const double distance = residual.dot(decomposition.solve(residual)); // 马氏距离r' * S^-1 * r
    if (mahalanobis) *mahalanobis = distance;
    if (!std::isfinite(distance) || (gate_chi2 > 0.0 && distance > gate_chi2)) return false; // 马氏距离不能过大， 马氏距离如果太大，概率太低，越不可靠
    // K = P- * H' * S^-1 
    const Eigen::MatrixXd gain = covariance_ * measurement_jacobian.transpose() * decomposition.solve(Eigen::MatrixXd::Identity(residual.size(), residual.size()));
    const ErrorVector correction = gain * residual; // K * r
    const Covariance identity = Covariance::Identity(); //I
    const Covariance joseph_left = identity - gain * measurement_jacobian; // I - k * H
    // 后验方差P+ = ( I - k * H) * P- *  (I - k * H)' + K*Σ_Z*k'
    covariance_ = joseph_left * covariance_ * joseph_left.transpose() + gain * noise * gain.transpose();
    InjectError(correction); // x(k+1)+ = x(k+1)- + K *r 
    StabilizeCovariance();
    return true;
}

void ESKF::InjectError(const ErrorVector& correction) {
    state_.position += correction.segment<3>(kPositionIndex); // p(k+1)+ = p(k+1)- + K *r 
    const Eigen::Vector3d orientation_correction = correction.segment<3>(kOrientationIndex); // δθ
    state_.orientation = Eigen::Quaterniond(state_.orientation.toRotationMatrix() * Exp(orientation_correction)); // R(K+1)- * exp(δθ)
    state_.orientation.normalize(); // 归一化
    state_.velocity_body += correction.segment<3>(kVelocityIndex); // v(k+1)+ = v(k+1)- + K *r
    state_.angular_velocity_body += correction.segment<3>(kAngularVelocityIndex); //w(k+1)+ = w(k+1)- + K *r
    Covariance reset_jacobian = Covariance::Identity();
    reset_jacobian.block<3, 3>(kOrientationIndex, kOrientationIndex) = Eigen::Matrix3d::Identity() - 0.5 * Skew(orientation_correction);
    covariance_ = reset_jacobian * covariance_ * reset_jacobian.transpose();
}

void ESKF::StabilizeCovariance() {
    covariance_ = 0.5 * (covariance_ + covariance_.transpose());
    Eigen::SelfAdjointEigenSolver<Covariance> solver(covariance_);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) { covariance_.setIdentity(); covariance_ *= 100.0; return; }
    Eigen::Matrix<double, kStateDim, 1> eigenvalues = solver.eigenvalues();
    for (int i = 0; i < kStateDim; ++i) eigenvalues(i) = std::clamp(eigenvalues(i), options_.min_covariance, options_.max_covariance);
    covariance_ = solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
    covariance_ = 0.5 * (covariance_ + covariance_.transpose());
}

Eigen::RowVector3d ESKF::YawJacobian() const {
    const Eigen::Matrix3d rotation = state_.orientation.toRotationMatrix();
    const double a = rotation(1, 0);
    const double b = rotation(0, 0);
    const double denominator = std::max(1e-12, a * a + b * b);
    Eigen::RowVector3d jacobian;
    // 航向角ψ 对 δΦ，δθ，δψ 的导数
    jacobian << 0.0, (-b * rotation(1, 2) + a * rotation(0, 2)) / denominator, (b * rotation(1, 1) - a * rotation(0, 1)) / denominator;
    return jacobian;
}

SE3 ESKF::Pose() const { return SE3(state_.orientation, state_.position); }
Eigen::Vector3d ESKF::VelocityMap() const { return state_.orientation.toRotationMatrix() * state_.velocity_body; }

Eigen::Matrix3d ESKF::Skew(const Eigen::Vector3d& vector) {
    Eigen::Matrix3d result;
    result << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(), -vector.y(), vector.x(), 0.0;
    return result;
}

Eigen::Matrix3d ESKF::Exp(const Eigen::Vector3d& angle) {
    const double theta = angle.norm();
    const Eigen::Matrix3d skew = Skew(angle);
    if (theta < 1e-8) return Eigen::Matrix3d::Identity() + skew + 0.5 * skew * skew;
    return Eigen::Matrix3d::Identity() + std::sin(theta) / theta * skew + (1.0 - std::cos(theta)) / (theta * theta) * skew * skew;
}

Eigen::Matrix3d ESKF::LeftJacobian(const Eigen::Vector3d& angle) {
    const double theta = angle.norm();
    const Eigen::Matrix3d skew = Skew(angle);
    if (theta < 1e-8) return Eigen::Matrix3d::Identity() + 0.5 * skew + skew * skew / 6.0;
    return Eigen::Matrix3d::Identity() + (1.0 - std::cos(theta)) / (theta * theta) * skew + (theta - std::sin(theta)) / (theta * theta * theta) * skew * skew;
}

Eigen::Vector3d ESKF::Log(const Eigen::Matrix3d& rotation) {
    const double cosine = std::clamp(0.5 * (rotation.trace() - 1.0), -1.0, 1.0);
    const double theta = std::acos(cosine);
    Eigen::Vector3d vector(rotation(2, 1) - rotation(1, 2), rotation(0, 2) - rotation(2, 0), rotation(1, 0) - rotation(0, 1));
    if (theta < 1e-8) return 0.5 * vector;
    const double sine = std::sin(theta);
    if (std::fabs(sine) < 1e-8) return Eigen::AngleAxisd(rotation).axis() * theta;
    return 0.5 * theta / sine * vector;
}

double ESKF::WrapAngle(double angle) {
    while (angle > kPi) angle -= 2.0 * kPi;
    while (angle < -kPi) angle += 2.0 * kPi;
    return angle;
}

double ESKF::Yaw(const Eigen::Matrix3d& rotation) { return std::atan2(rotation(1, 0), rotation(0, 0)); }

}  // namespace lightning::loc
