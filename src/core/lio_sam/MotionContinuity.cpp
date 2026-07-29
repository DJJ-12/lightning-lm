#include "core/lio_sam/MotionContinuity.h"

double mapOptimization::yawFromAffine(const Eigen::Affine3f& affine){
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(affine, x, y, z, roll, pitch, yaw);
        return yaw;
    }

double mapOptimization::xyDistance(const Eigen::Affine3f& lhs, const Eigen::Affine3f& rhs){
        float lx, ly, lz, lr, lp, lyaw;
        float rx, ry, rz, rr, rp, ryaw;
        pcl::getTranslationAndEulerAngles(lhs, lx, ly, lz, lr, lp, lyaw);
        pcl::getTranslationAndEulerAngles(rhs, rx, ry, rz, rr, rp, ryaw);
        const double dx = static_cast<double>(lx) - rx;
        const double dy = static_cast<double>(ly) - ry;
        return std::sqrt(dx * dx + dy * dy);
    }

mapOptimization::MotionContinuityInfo mapOptimization::evaluateMotionContinuity(
    const Eigen::Affine3f& candidateAffine, const char* tag){
        MotionContinuityInfo info;
        const bool logThisCall =
            debugTiming && ((diagnosticExecutedCalls + 1) % 20 == 0);

        if (!candidateAffine.matrix().allFinite())
        {
            info.continuous = false;
            RCLCPP_WARN(get_logger(),
                "[MOTION_GATE][%s] candidate pose contains NaN or Inf; reject.",
                tag);
            return info;
        }

        if (!hasLastOutputPose)
        {
            if (logThisCall)
            {
                RCLCPP_INFO(get_logger(),
                    "[MOTION_GATE][%s] no last published pose; treat as continuous.",
                    tag);
            }
            return info;
        }

        info.dt = lastOutputTime > 0.0 ? std::max(1e-3, timeLaserInfoCur - lastOutputTime) : 0.0;
        info.ds = xyDistance(lastOutputAffine, candidateAffine);
        info.dyawDeg = std::abs(
            pcl::rad2deg(normalizeAngleRad(yawFromAffine(candidateAffine) - yawFromAffine(lastOutputAffine))));
        info.speed = info.ds / std::max(1e-3, info.dt);
        info.yawRate = info.dyawDeg / std::max(1e-3, info.dt);

        const double curvatureDs = std::max(info.ds, 0.10);
        info.curvature = (info.dyawDeg * M_PI / 180.0) / curvatureDs;

        if (speedHist.empty() || yawRateHist.empty() || curvatureHist.empty())
        {
            if (logThisCall)
            {
                RCLCPP_INFO(get_logger(),
                    "[MOTION_GATE][%s] warmup hist=%zu dt=%.3f ds=%.3f dyaw=%.2fdeg "
                    "v=%.3f omega=%.2f kappa=%.3f; continuous.",
                    tag,
                    speedHist.size(),
                    info.dt,
                    info.ds,
                    info.dyawDeg,
                    info.speed,
                    info.yawRate,
                    info.curvature);
            }
            return info;
        }

        info.ready = true;
        info.speedRef = speedHist.back();
        info.yawRateRef = yawRateHist.back();
        info.curvatureRef = curvatureHist.back();
        info.speedJump = std::abs(info.speed - info.speedRef);
        info.yawRateJump = std::abs(info.yawRate - info.yawRateRef);
        info.curvatureJump = std::abs(info.curvature - info.curvatureRef);
        info.accel = info.speedJump / std::max(1e-3, info.dt);
        info.yawAccel = info.yawRateJump / std::max(1e-3, info.dt);

        const bool speedBad = info.speed > 1.50;
        const bool accelBad = info.accel > 1.20 && info.speedJump > 0.30;
        const bool yawRateBad = info.yawRate > 80.0;
        const bool yawAccelBad = info.yawAccel > 180.0 && info.yawRateJump > 18.0;
        const bool curvatureBad = info.curvatureJump > 1.80 && info.dyawDeg > 1.0 && info.ds > 0.03;

        info.continuous = !(speedBad || accelBad || yawRateBad || yawAccelBad || curvatureBad);

        if (logThisCall)
        {
            RCLCPP_INFO(get_logger(),
                "[MOTION_GATE][%s] ok=%d dt=%.3f ds=%.3f dyaw=%.2fdeg "
                "v=%.3f(prev=%.3f,dv=%.3f,a=%.3f) "
                "omega=%.2f(prev=%.2f,d=%.2f,alpha=%.2f) "
                "kappa=%.3f(prev=%.3f,d=%.3f) bad(speed=%d accel=%d omega=%d alpha=%d kappa=%d)",
                tag,
                int(info.continuous),
                info.dt,
                info.ds,
                info.dyawDeg,
                info.speed,
                info.speedRef,
                info.speedJump,
                info.accel,
                info.yawRate,
                info.yawRateRef,
                info.yawRateJump,
                info.yawAccel,
                info.curvature,
                info.curvatureRef,
                info.curvatureJump,
                int(speedBad),
                int(accelBad),
                int(yawRateBad),
                int(yawAccelBad),
                int(curvatureBad));
        }

        return info;
    }

void mapOptimization::pushMotionHistory(double speed, double yawRateDeg, double curvature){
        speedHist.push_back(speed);
        yawRateHist.push_back(yawRateDeg);
        curvatureHist.push_back(curvature);
        while (static_cast<int>(speedHist.size()) > motionHistoryWindow)
            speedHist.pop_front();
        while (static_cast<int>(yawRateHist.size()) > motionHistoryWindow)
            yawRateHist.pop_front();
        while (static_cast<int>(curvatureHist.size()) > motionHistoryWindow)
            curvatureHist.pop_front();
    }

void mapOptimization::updateOutputTrajectoryHistory(){
        const Eigen::Affine3f outputAffine = trans2Affine3f(transformTobeMapped);
        if (!outputAffine.matrix().allFinite())
            return;

        if (hasLastOutputPose)
        {
            const double dt = std::max(1e-3, timeLaserInfoCur - lastOutputTime);
            const double ds = xyDistance(lastOutputAffine, outputAffine);
            const double dyawDeg = std::abs(
                pcl::rad2deg(normalizeAngleRad(yawFromAffine(outputAffine) - yawFromAffine(lastOutputAffine))));
            const double speed = ds / dt;
            const double yawRate = dyawDeg / dt;
            const double curvature = (dyawDeg * M_PI / 180.0) / std::max(ds, 0.10);
            pushMotionHistory(speed, yawRate, curvature);
        }

        lastOutputAffine = outputAffine;
        lastOutputTime = timeLaserInfoCur;
        hasLastOutputPose = true;
    }

bool mapOptimization::Finite6(const std::array<float, 6>& transform) const {
        return std::all_of(
            transform.begin(),
            transform.end(),
            [](float value) { return std::isfinite(value); });
    }

// 这些状态必须属于当前 mapOptimization 对象。
// 不能使用函数内 static，否则后续任务会继承上一个任务的时间戳和 IMU 初值。


Eigen::Affine3f mapOptimization::BuildFallbackAffine(
    const Eigen::Affine3f& priorAffine,
    const Eigen::Affine3f& lmAffine,
    float* rawDeltaNorm,
    float* usedDeltaNorm) const {
    Eigen::Affine3f Affine = priorAffine;

    if (!lmAffine.matrix().allFinite())
    {
        if (rawDeltaNorm)
            *rawDeltaNorm = std::numeric_limits<float>::infinity();
        if (usedDeltaNorm)
            *usedDeltaNorm = 0.0f;
        return Affine;
    }

    Eigen::Vector3f delta = lmAffine.translation() - priorAffine.translation();
    const float deltaNorm = delta.norm();
    if (rawDeltaNorm)
        *rawDeltaNorm = deltaNorm;

    if (std::isfinite(static_cast<double>(deltaNorm)) &&
        deltaNorm > kFallbackMaxTranslationStep &&
        deltaNorm > 1e-6f) {
        delta *= kFallbackMaxTranslationStep / deltaNorm;
    }

    if (usedDeltaNorm)
        *usedDeltaNorm = delta.norm();

    // Keep prior rotation. Only borrow the bounded LM translation trend.
    Affine.translation() = priorAffine.translation() + delta;
    return Affine;
}

const char* mapOptimization::trackingStateName() const{
        if (mappingTrackingState == MappingTrackingState::LOST)
            return "LOST";
        return "TRACKING";
    }

void mapOptimization::resetFrameQuality(){
        mappingPoseReliable = false;
        lidarCorrectionFlag = 2;
        mappingPoseSource = "FAIL";
        lastLMIterationCount = 0;
        currentOdomCov = 2;
        isDegenerate = false;
    }

void mapOptimization::acceptMappingPose(const std::string& source){
        mappingTrackingState = MappingTrackingState::TRACKING;
        mappingPoseReliable = true;
        lidarCorrectionFlag = 0;
        mappingPoseSource = source;
        currentOdomCov = 0;
        isDegenerate = false;
        mappingFailureCount = 0;
        mappingFirstFailureTime = -1.0;
        surroundingKeyFrameIndices.clear();

        //  rule:
        // Only reliable INIT/LM/ICP poses update the trusted pose cache used in normal tracking.
        // Unreliable fallback poses are published by scan2MapOptimization(), but never enter this cache.
        TrustedPose& trusted = trusted_pose_;
        if (trusted.has_last && timeLaserInfoCur > trusted.last_time + 1e-3)
        {
            trusted.prev = trusted.last;
            trusted.prev_time = trusted.last_time;
            trusted.has_prev = true;
        }

        for (int i = 0; i < 6; ++i)
            trusted.last[i] = transformTobeMapped[i];
        trusted.last_time = timeLaserInfoCur;
        trusted.has_last = true;

        trusted.last_imu[0] = cloudInfo->imu_roll_init;
        trusted.last_imu[1] = cloudInfo->imu_pitch_init;
        trusted.last_imu[2] = cloudInfo->imu_yaw_init;
        trusted.last_imu[3] = 0.0f;
        trusted.last_imu[4] = 0.0f;
        trusted.last_imu[5] = 0.0f;
        trusted.has_last_imu = Finite6(trusted.last_imu);
    }

void mapOptimization::rejectMappingPose(const std::string& source){
        mappingTrackingState = MappingTrackingState::LOST;
        mappingPoseReliable = false;
        lidarCorrectionFlag = 2;
        mappingPoseSource = source;
        currentOdomCov = 2;
        isDegenerate = true;

        if (mappingFailureCount == 0)
            mappingFirstFailureTime = timeLaserInfoCur;
        mappingFailureCount++;
    }



