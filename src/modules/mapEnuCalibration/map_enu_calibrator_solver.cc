#include "modules/mapEnuCalibration/map_enu_calibrator.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>

namespace lightning::modules {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;

double Clamp(double value, double lower, double upper) {
    return std::max(lower, std::min(upper, value));
}

bool IsFiniteRotation(const Eigen::Matrix3d& rotation) {
    return rotation.allFinite() &&
           std::fabs(rotation.determinant() - 1.0) < 1e-3 &&
           (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() < 1e-3;
}

template <int Dimension>
Eigen::Matrix<double, Dimension, Dimension> SymmetricPositiveDefinite(
    const Eigen::Matrix<double, Dimension, Dimension>& input,
    double minimum_eigenvalue = 1e-10,
    double maximum_eigenvalue = 1e8) {
    using Matrix = Eigen::Matrix<double, Dimension, Dimension>;
    const Matrix symmetric = 0.5 * (input + input.transpose());
    Eigen::SelfAdjointEigenSolver<Matrix> solver(symmetric);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
        return Matrix::Identity();
    }
    auto eigenvalues = solver.eigenvalues();
    for (int i = 0; i < Dimension; ++i) {
        eigenvalues(i) = Clamp(eigenvalues(i), minimum_eigenvalue, maximum_eigenvalue);
    }
    return solver.eigenvectors() * eigenvalues.asDiagonal() *
           solver.eigenvectors().transpose();
}

template <int Dimension>
Eigen::Matrix<double, Dimension, Dimension> InverseSpd(
    const Eigen::Matrix<double, Dimension, Dimension>& covariance) {
    using Matrix = Eigen::Matrix<double, Dimension, Dimension>;
    Eigen::LDLT<Matrix> ldlt(SymmetricPositiveDefinite<Dimension>(covariance));
    return (ldlt.info() == Eigen::Success && ldlt.isPositive())
               ? ldlt.solve(Matrix::Identity())
               : Matrix::Identity();
}

double Percentile95(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t index = std::min(
        values.size() - 1,
        static_cast<std::size_t>(std::ceil(0.95 * values.size()) - 1.0));
    return values[index];
}

}  // namespace

bool MapEnuCalibrator::EstimateRotation(
    const std::vector<CalibrationSample>& samples,
    Eigen::Matrix3d* rotation_enu_map,
    double* condition_number,
    double* singular_ratio) {
    if (!rotation_enu_map || samples.size() < 2) return false;
    Eigen::Matrix3d correlation = Eigen::Matrix3d::Zero();
    double total_weight = 0.0;
    for (const CalibrationSample& sample : samples) {
        const double baseline_length = sample.baseline_enu.norm();
        if (sample.baseline_map.norm() < 1e-9 || baseline_length < 1e-9) {
            continue;
        }
        const Eigen::Matrix3d baseline_covariance =
            sample.main_gnss_covariance + sample.slave_gnss_covariance;
        const double rotation_variance =
            sample.ndt_pose_covariance.block<3, 3>(3, 3).trace() / 3.0;
        const double variance = std::max(
            1e-10,
            baseline_covariance.trace() / 3.0 +
                baseline_length * baseline_length * rotation_variance);
        const double weight = 1.0 / variance;
        const Eigen::Vector3d direction_map =
            sample.baseline_map.normalized();
        const Eigen::Vector3d direction_enu =
            sample.baseline_enu.normalized();
        correlation +=
            weight * direction_enu * direction_map.transpose();
        total_weight += weight;
    }
    if (total_weight <= 0.0 || !correlation.allFinite()) return false;
    correlation /= total_weight;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        correlation, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.info() != Eigen::Success) return false;
    const Eigen::Vector3d singular_values = svd.singularValues();
    if (!singular_values.allFinite() || singular_values(0) <= 1e-12) {
        return false;
    }
    const double second_ratio =
        singular_values(1) / singular_values(0);
    if (singular_ratio) *singular_ratio = second_ratio;
    if (condition_number) {
        *condition_number = singular_values(0) /
            std::max(1e-12, singular_values(1));
    }

    Eigen::Matrix3d correction = Eigen::Matrix3d::Identity();
    correction(2, 2) =
        (svd.matrixU() * svd.matrixV().transpose()).determinant();
    *rotation_enu_map =
        svd.matrixU() * correction * svd.matrixV().transpose();
    return IsFiniteRotation(*rotation_enu_map);
}

Eigen::Matrix3d MapEnuCalibrator::Skew(
    const Eigen::Vector3d& vector) {
    Eigen::Matrix3d matrix;
    matrix << 0.0, -vector.z(), vector.y(),
              vector.z(), 0.0, -vector.x(),
              -vector.y(), vector.x(), 0.0;
    return matrix;
}

Eigen::Matrix3d MapEnuCalibrator::ExpSo3(
    const Eigen::Vector3d& angle) {
    const double norm = angle.norm();
    if (norm < 1e-12) {
        return Eigen::Matrix3d::Identity() + Skew(angle);
    }
    return Eigen::AngleAxisd(norm, angle / norm).toRotationMatrix();
}

Eigen::Matrix3d MapEnuCalibrator::PointCovarianceInMap(
    const CalibrationSample& sample,
    const Eigen::Vector3d& antenna_in_body,
    double antenna_position_std_m) {
    Eigen::Matrix<double, 3, 6> jacobian =
        Eigen::Matrix<double, 3, 6>::Zero();
    jacobian.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    jacobian.block<3, 3>(0, 3) =
        -sample.rotation_map_body * Skew(antenna_in_body);
    Eigen::Matrix3d covariance =
        jacobian * sample.ndt_pose_covariance * jacobian.transpose();
    covariance += Eigen::Matrix3d::Identity() *
        antenna_position_std_m * antenna_position_std_m;
    return SymmetricPositiveDefinite<3>(covariance);
}

Eigen::Matrix3d MapEnuCalibrator::PointInformationInEnu(
    const CalibrationSample& sample,
    const Eigen::Vector3d& antenna_in_body,
    const Eigen::Matrix3d& gnss_covariance,
    const Eigen::Matrix3d& rotation_enu_map,
    double antenna_position_std_m) {
    const Eigen::Matrix3d covariance_map = PointCovarianceInMap(
        sample, antenna_in_body, antenna_position_std_m);
    const Eigen::Matrix3d covariance_enu =
        gnss_covariance + rotation_enu_map * covariance_map *
                              rotation_enu_map.transpose();
    return InverseSpd<3>(covariance_enu);
}

bool MapEnuCalibrator::EstimateTranslation(
    const std::vector<CalibrationSample>& samples,
    const Options& options,
    const Eigen::Matrix3d& rotation_enu_map,
    Eigen::Vector3d* translation_enu_map) {
    if (!translation_enu_map || samples.empty()) return false;
    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d right_hand_side = Eigen::Vector3d::Zero();
    const auto accumulate_antenna = [&](
        const CalibrationSample& sample,
        const Eigen::Vector3d& antenna_body,
        const Eigen::Vector3d& position_enu,
        const Eigen::Matrix3d& gnss_covariance) {
        const Eigen::Vector3d position_map =
            sample.rotation_map_body * antenna_body +
            sample.translation_map_body;
        const Eigen::Vector3d translation_candidate =
            position_enu - rotation_enu_map * position_map;
        const Eigen::Matrix3d information = PointInformationInEnu(
            sample, antenna_body, gnss_covariance, rotation_enu_map,
            options.antenna_position_std_m);
        normal += information;
        right_hand_side += information * translation_candidate;
    };
    for (const CalibrationSample& sample : samples) {
        accumulate_antenna(
            sample, options.main_antenna_in_body, sample.main_enu,
            sample.main_gnss_covariance);
        accumulate_antenna(
            sample, options.slave_antenna_in_body, sample.slave_enu,
            sample.slave_gnss_covariance);
    }
    Eigen::LDLT<Eigen::Matrix3d> decomposition(
        SymmetricPositiveDefinite<3>(normal));
    if (decomposition.info() != Eigen::Success ||
        !decomposition.isPositive()) {
        return false;
    }
    *translation_enu_map = decomposition.solve(right_hand_side);
    return translation_enu_map->allFinite();
}

bool MapEnuCalibrator::RefineTransform(
    const std::vector<CalibrationSample>& samples,
    const Options& options,
    Eigen::Matrix3d* rotation_enu_map,
    Eigen::Vector3d* translation_enu_map,
    MapEnuCovariance* covariance_enu_map) {
    if (!rotation_enu_map || !translation_enu_map ||
        !covariance_enu_map || samples.empty()) {
        return false;
    }

    const auto build_system = [&](const Eigen::Matrix3d& rotation,
                                  const Eigen::Vector3d& translation,
                                  MapEnuCovariance* normal,
                                  Eigen::Matrix<double, 6, 1>* rhs) {
        normal->setZero();
        rhs->setZero();
        const auto accumulate_antenna = [&](
            const CalibrationSample& sample,
            const Eigen::Vector3d& antenna_body,
            const Eigen::Vector3d& observed_enu,
            const Eigen::Matrix3d& gnss_covariance) {
            const Eigen::Vector3d point_map =
                sample.rotation_map_body * antenna_body +
                sample.translation_map_body;
            const Eigen::Vector3d residual =
                observed_enu - (rotation * point_map + translation);
            const Eigen::Matrix3d information = PointInformationInEnu(
                sample, antenna_body, gnss_covariance, rotation,
                options.antenna_position_std_m);
            const double normalized_residual = std::sqrt(std::max(
                0.0, residual.dot(information * residual)));
            const double huber_weight =
                normalized_residual <= options.huber_delta_sigma ||
                    normalized_residual < 1e-12
                ? 1.0
                : options.huber_delta_sigma / normalized_residual;

            Eigen::Matrix<double, 3, 6> jacobian =
                Eigen::Matrix<double, 3, 6>::Zero();
            // Prediction Jacobian for R <- R Exp(dtheta), t <- t + dt.
            jacobian.block<3, 3>(0, 0) = -rotation * Skew(point_map);
            jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
            // ((J^TWJ) Delta x = J^TWr
            *normal += huber_weight *
                jacobian.transpose() * information * jacobian;
            *rhs += huber_weight *
                jacobian.transpose() * information * residual;
        };

        for (const CalibrationSample& sample : samples) {
            accumulate_antenna(
                sample, options.main_antenna_in_body, sample.main_enu,
                sample.main_gnss_covariance);
            accumulate_antenna(
                sample, options.slave_antenna_in_body, sample.slave_enu,
                sample.slave_gnss_covariance);
        }
        *normal = 0.5 * (*normal + normal->transpose());
    };

    MapEnuCovariance final_normal = MapEnuCovariance::Zero();
    for (int iteration = 0;
         iteration < options.maximum_refinement_iterations; ++iteration) {
        Eigen::Matrix<double, 6, 1> right_hand_side = Eigen::Matrix<double, 6, 1>::Zero();
        build_system(
            *rotation_enu_map, *translation_enu_map, &final_normal,
            &right_hand_side);
        final_normal += MapEnuCovariance::Identity() * 1e-9;
        Eigen::LDLT<MapEnuCovariance> decomposition(final_normal);
        if (decomposition.info() != Eigen::Success ||
            !decomposition.isPositive()) {
            return false;
        }
        Eigen::Matrix<double, 6, 1> increment = decomposition.solve(right_hand_side);
        if (!increment.allFinite()) return false;

        const double rotation_norm = increment.head<3>().norm();
        if (rotation_norm > 0.20) {
            increment.head<3>() *= 0.20 / rotation_norm;
        }
        const double translation_norm = increment.tail<3>().norm();
        if (translation_norm > 2.0) {
            increment.tail<3>() *= 2.0 / translation_norm;
        }
        *rotation_enu_map =
            *rotation_enu_map * ExpSo3(increment.head<3>());
        *translation_enu_map += increment.tail<3>();
        Eigen::Quaterniond normalized_rotation(*rotation_enu_map);
        normalized_rotation.normalize();
        *rotation_enu_map = normalized_rotation.toRotationMatrix();

        if (increment.head<3>().norm() < 1e-9 &&
            increment.tail<3>().norm() < 1e-7) {
            break;
        }
    }

    Eigen::Matrix<double, 6, 1> unused_rhs;
    build_system(
        *rotation_enu_map, *translation_enu_map, &final_normal,
        &unused_rhs);
    Eigen::SelfAdjointEigenSolver<MapEnuCovariance> solver(final_normal);
    if (solver.info() != Eigen::Success ||
        !solver.eigenvalues().allFinite() ||
        solver.eigenvalues().minCoeff() <= 1e-12) {
        return false;
    }
    *covariance_enu_map = solver.eigenvectors() *
        solver.eigenvalues().cwiseInverse().asDiagonal() *
        solver.eigenvectors().transpose();
    *covariance_enu_map = 0.5 *
        (*covariance_enu_map + covariance_enu_map->transpose());
    return IsFiniteRotation(*rotation_enu_map) &&
        translation_enu_map->allFinite() &&
        covariance_enu_map->allFinite();
}

void MapEnuCalibrator::ComputeQuality(
    const std::vector<CalibrationSample>& samples,
    const Eigen::Matrix3d& rotation_enu_map,
    MapEnuCalibrationResult* result) {
    if (!result || samples.empty()) return;
    std::vector<double> rotation_errors_deg;
    rotation_errors_deg.reserve(samples.size());
    double rotation_squared_sum = 0.0;
    double rotation_max = 0.0;
    double baseline_squared_sum = 0.0;

    for (const CalibrationSample& sample : samples) {
        const Eigen::Vector3d predicted_direction =
            (rotation_enu_map * sample.baseline_map).normalized();
        const Eigen::Vector3d observed_direction =
            sample.baseline_enu.normalized();
        const double angle_deg = std::acos(Clamp(
            predicted_direction.dot(observed_direction), -1.0, 1.0)) /
            kDegToRad;
        rotation_errors_deg.push_back(angle_deg);
        rotation_squared_sum += angle_deg * angle_deg;
        rotation_max = std::max(rotation_max, angle_deg);

        const Eigen::Vector3d baseline_body =
            sample.rotation_map_body.transpose() * sample.baseline_map;
        const double length_error =
            sample.baseline_enu.norm() - baseline_body.norm();
        baseline_squared_sum += length_error * length_error;

    }
    result->baseline_rms_m =
        std::sqrt(baseline_squared_sum / samples.size());
    result->rotation_rms_deg =
        std::sqrt(rotation_squared_sum / samples.size());
    result->rotation_p95_deg = Percentile95(rotation_errors_deg);
    result->rotation_max_deg = rotation_max;
}


}  // namespace lightning::modules
