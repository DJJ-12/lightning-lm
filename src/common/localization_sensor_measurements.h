#pragma once

#include <cmath>
#include <deque>
#include <vector>

#include <Eigen/Core>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>

#include "core/lightning_math.hpp"

namespace lightning {

struct RtkPositionMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double stamp = 0.0;
    // Time used only for pairing NavSatFix with Pose2D. Online uses receive
    // time because Pose2D has no Header; offline uses rosbag message time.
    double sync_stamp = 0.0;
    Eigen::Vector3d position_enu = Eigen::Vector3d::Zero();
    Eigen::Matrix3d position_covariance_enu = Eigen::Matrix3d::Zero();
    bool position_covariance_valid = false;
    bool valid = false;
};

struct RtkHeadingMeasurement {
    double stamp = 0.0;
    // Same time basis as RtkPositionMeasurement::sync_stamp.
    double sync_stamp = 0.0;
    double yaw_enu = 0.0;
    double yaw_variance = 0.0;
    bool valid = false;
};

// Local Cartesian navigation observation used by localization ESKF. RTK
// position comes from NavSatFix latitude/longitude/altitude projected to UTM
// ENU; RTK heading comes from Pose2D.theta. Position and heading may arrive as
// separate ROS messages, but only a synchronized complete measurement is sent
// to LocalizationSystem.
struct RtkInsMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double stamp = 0.0;
    Eigen::Vector3d position_enu = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity_enu = Eigen::Vector3d::Zero();
    Eigen::Matrix3d position_covariance_enu = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d velocity_covariance_enu = Eigen::Matrix3d::Zero();
    double yaw_enu = 0.0;
    double yaw_variance = 0.0;
    bool position_covariance_valid = false;
    bool velocity_covariance_valid = false;
    bool yaw_valid = false;
    bool position_valid = false;
    bool velocity_valid = false;
};

// External wheel/vehicle odometry boundary. The measurement is expressed in
// the tracking/body frame (ROS FLU by default). This is intentionally separate
// from DETA100 /system_speed, which is an INS-derived body velocity rather than
// a dedicated wheel encoder observation and has no message timestamp.
struct WheelOdometryMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double stamp = 0.0;
    Eigen::Vector3d linear_velocity_body = Eigen::Vector3d::Zero();
    Eigen::Vector3d angular_velocity_body = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Zero();
    bool covariance_valid = false;
    bool valid = true;
};

namespace localization_adapter {

inline bool Covariance3Valid(const Eigen::Matrix3d& covariance) {
    if (!covariance.allFinite()) return false;
    const Eigen::Matrix3d symmetric = 0.5 * (covariance + covariance.transpose());
    return symmetric.diagonal().minCoeff() >= 0.0 && symmetric.diagonal().maxCoeff() > 0.0;
}

inline bool OdometryToWheelMeasurement(
    const nav_msgs::msg::Odometry& msg,
    WheelOdometryMeasurement* output) {
    if (!output) return false;
    output->stamp = rclcpp::Time(msg.header.stamp).seconds();
    output->linear_velocity_body = Eigen::Vector3d(
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y,
        msg.twist.twist.linear.z);
    output->angular_velocity_body = Eigen::Vector3d(
        msg.twist.twist.angular.x,
        msg.twist.twist.angular.y,
        msg.twist.twist.angular.z);
    output->covariance.setZero();
    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 6; ++col) {
            output->covariance(row, col) = msg.twist.covariance[row * 6 + col];
        }
    }
    output->covariance_valid =
        output->covariance.allFinite() &&
        output->covariance.diagonal().minCoeff() >= 0.0 &&
        output->covariance.diagonal().maxCoeff() > 0.0;
    output->valid = std::isfinite(output->stamp) &&
                    output->linear_velocity_body.allFinite() &&
                    output->angular_velocity_body.allFinite();
    return output->valid;
}

inline bool NavSatFixToRtkPositionMeasurement(
    const sensor_msgs::msg::NavSatFix& msg,
    int utm_zone,
    RtkPositionMeasurement* output) {
    if (!output) return false;
    RtkPositionMeasurement measurement;
    measurement.stamp = rclcpp::Time(msg.header.stamp).seconds();
    measurement.sync_stamp = measurement.stamp;
    measurement.valid = std::isfinite(measurement.stamp) &&
        std::isfinite(measurement.sync_stamp) &&
        lightning::math::JsbsimWgs84Enu::ForwardUtmDegrees(
            msg.latitude, msg.longitude, msg.altitude, utm_zone,
            &measurement.position_enu);

    measurement.position_covariance_enu = Eigen::Matrix3d::Zero();
    measurement.position_covariance_valid =
        msg.position_covariance_type !=
        sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN;
    if (measurement.position_covariance_valid) {
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                measurement.position_covariance_enu(row, col) =
                    msg.position_covariance[row * 3 + col];
            }
        }
        measurement.position_covariance_valid =
            Covariance3Valid(measurement.position_covariance_enu);
    }

    *output = measurement;
    return output->valid;
}

inline bool Pose2DToRtkHeadingMeasurement(
    const geometry_msgs::msg::Pose2D& msg,
    double stamp,
    double yaw_variance,
    RtkHeadingMeasurement* output) {
    if (!output) return false;
    RtkHeadingMeasurement measurement;
    measurement.stamp = stamp;
    measurement.sync_stamp = stamp;
    measurement.yaw_enu = std::atan2(std::sin(msg.theta), std::cos(msg.theta));
    measurement.yaw_variance = yaw_variance;
    measurement.valid =
        std::isfinite(measurement.stamp) &&
        std::isfinite(measurement.sync_stamp) &&
        std::isfinite(measurement.yaw_enu) &&
        std::isfinite(measurement.yaw_variance) &&
        measurement.yaw_variance > 0.0;
    *output = measurement;
    return output->valid;
}

inline bool MakeRtkInsMeasurement(
    const RtkPositionMeasurement& position,
    const RtkHeadingMeasurement& heading,
    RtkInsMeasurement* output) {
    if (!output || !position.valid || !heading.valid) return false;

    RtkInsMeasurement measurement;
    measurement.stamp = position.stamp >= heading.stamp ? position.stamp : heading.stamp;
    measurement.position_enu = position.position_enu;
    measurement.position_covariance_enu = position.position_covariance_enu;
    measurement.position_covariance_valid = position.position_covariance_valid;
    measurement.position_valid = true;
    measurement.yaw_enu = heading.yaw_enu;
    measurement.yaw_variance = heading.yaw_variance;
    measurement.yaw_valid = true;
    *output = measurement;
    return true;
}

class RtkMeasurementSynchronizer {
   public:
    RtkMeasurementSynchronizer() = default;
    explicit RtkMeasurementSynchronizer(double max_time_difference_sec) {
        SetMaxTimeDifference(max_time_difference_sec);
    }

    void SetMaxTimeDifference(double max_time_difference_sec) {
        max_time_difference_sec_ =
            std::isfinite(max_time_difference_sec) && max_time_difference_sec >= 0.0
                ? max_time_difference_sec
                : 0.20;
    }

    void Clear() {
        positions_.clear();
        headings_.clear();
    }

    void AddPosition(
        const RtkPositionMeasurement& position,
        std::vector<RtkInsMeasurement>* output) {
        if (!output || !position.valid) return;
        positions_.push_back(position);
        TrySync(output);
    }

    void AddHeading(
        const RtkHeadingMeasurement& heading,
        std::vector<RtkInsMeasurement>* output) {
        if (!output || !heading.valid) return;
        headings_.push_back(heading);
        TrySync(output);
    }

   private:
    void TrySync(std::vector<RtkInsMeasurement>* output) {
        while (!positions_.empty() && !headings_.empty()) {
            const double time_diff = positions_.front().sync_stamp - headings_.front().sync_stamp;
            if (std::fabs(time_diff) <= max_time_difference_sec_) {
                RtkInsMeasurement measurement;
                if (MakeRtkInsMeasurement(positions_.front(), headings_.front(), &measurement)) {
                    output->push_back(measurement);
                }
                positions_.pop_front();
                headings_.pop_front();
                continue;
            }

            if (time_diff < 0.0) {
                positions_.pop_front();
            } else {
                headings_.pop_front();
            }
        }
    }

    double max_time_difference_sec_ = 0.20;
    std::deque<RtkPositionMeasurement> positions_;
    std::deque<RtkHeadingMeasurement> headings_;
};

}  // namespace localization_adapter

}  // namespace lightning
