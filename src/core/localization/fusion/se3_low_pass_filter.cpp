#include "core/localization/fusion/se3_low_pass_filter.h"

#include <cmath>

#include <glog/logging.h>

namespace lightning::loc {

namespace {

double NormalizeAngle(double angle) {
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

double YawOf(const SE3& pose) {
    const Mat3d R = pose.so3().matrix();
    return std::atan2(R(1, 0), R(0, 0));
}

double RadToDeg(double rad) { return rad * 180.0 / M_PI; }

}  // namespace

void SE3LowPassFilter::Reset(const SE3& pose) {
    pose_ = pose;
    initialized_ = true;
}

bool SE3LowPassFilter::HasState() const { return initialized_; }

SE3 SE3LowPassFilter::Update(const SE3& raw_pose,
                             double alpha,
                             double ignore_trans_threshold,
                             double ignore_yaw_threshold_rad,
                             bool* accepted,
                             std::string* reject_reason) {
    (void)ignore_trans_threshold;
    (void)ignore_yaw_threshold_rad;

    if (!initialized_) {
        Reset(raw_pose);
        if (accepted) {
            *accepted = true;
        }
        if (reject_reason) {
            *reject_reason = "init";
        }
        return pose_;
    }

    const SE3 delta = pose_.inverse() * raw_pose;
    const Vec6d xi = delta.log();
    const double trans_norm = xi.head<3>().norm();
    const double yaw_delta = std::abs(NormalizeAngle(YawOf(raw_pose) - YawOf(pose_)));

    if (ignore_trans_threshold > 0.0 && trans_norm > ignore_trans_threshold) {
        if (accepted) {
            *accepted = false;
        }
        if (reject_reason) {
            *reject_reason = "trans_jump";
        }
        LOG(WARNING) << "[MAP_ODOM_FILTER_REJECT] reason=trans_jump"
                     << ", raw_delta_trans=" << trans_norm
                     << ", threshold=" << ignore_trans_threshold
                     << ", raw_delta_yaw_deg=" << RadToDeg(yaw_delta);
        return pose_;
    }
    if (ignore_yaw_threshold_rad > 0.0 && yaw_delta > ignore_yaw_threshold_rad) {
        if (accepted) {
            *accepted = false;
        }
        if (reject_reason) {
            *reject_reason = "yaw_jump";
        }
        LOG(WARNING) << "[MAP_ODOM_FILTER_REJECT] reason=yaw_jump"
                     << ", raw_delta_trans=" << trans_norm
                     << ", raw_delta_yaw_deg=" << RadToDeg(yaw_delta)
                     << ", threshold_deg=" << RadToDeg(ignore_yaw_threshold_rad);
        return pose_;
    }

    const double a = std::max(0.0, std::min(1.0, alpha));
    pose_ = pose_ * SE3::exp(a * xi);

    if (accepted) {
        *accepted = true;
    }
    if (reject_reason) {
        *reject_reason = "accepted";
    }
    LOG(INFO) << "[MAP_ODOM_FILTER] raw_delta_trans=" << trans_norm
              << ", raw_delta_yaw_deg=" << RadToDeg(yaw_delta)
              << ", alpha=" << a
              << ", accepted=1, reason=accepted";
    return pose_;
}

SE3 SE3LowPassFilter::Get() const { return pose_; }

}  // namespace lightning::loc
