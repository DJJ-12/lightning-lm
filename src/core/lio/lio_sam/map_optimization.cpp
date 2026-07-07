#include "core/lio/lio_sam/map_optimization.h"

mapOptimization::mapOptimization(const rclcpp::NodeOptions & options) : ParamServer("lio_sam_mapOptimization", options){
        cout << "------------build version - 20260516-1739 -------------------\n" << endl;
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterRawICP.setLeafSize(0.30f, 0.30f, 0.30f);

#ifdef _OPENMP
        RCLCPP_INFO(get_logger(),
            "[OPENMP] enabled, omp_get_max_threads=%d, numberOfCores=%d",
            omp_get_max_threads(),
            numberOfCores);
#else
        RCLCPP_WARN(get_logger(), "[OPENMP] disabled at compile time");
#endif

        allocateMemory();

        if (loopClosureEnableFlag)
        {
            loopClosureThreadRunning_.store(true);
            loopClosureThread_ = std::thread(&mapOptimization::loopClosureThread, this);
        }
    }

mapOptimization::~mapOptimization() {
    loopClosureThreadRunning_.store(false);
    if (loopClosureThread_.joinable())
        loopClosureThread_.join();

    if (isam != nullptr) {
        delete isam;
        isam = nullptr;
    }
}

void mapOptimization::allocateMemory(){
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());


        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization
        laserCloudRawLast.reset(new pcl::PointCloud<PointType>());

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudRawFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudRawFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP.setZero();
        currentOdomCov = 0;
        incrementalOdometryAffineFront = Eigen::Affine3f::Identity();
        incrementalOdometryAffineBack = Eigen::Affine3f::Identity();
        hasLastOutputPose = false;
        lastOutputAffine = Eigen::Affine3f::Identity();
        lastOutputTime = -1.0;
        speedHist.clear();
        yawRateHist.clear();
        curvatureHist.clear();
    }

bool mapOptimization::Run(LioSamCloudInfo& msgIn){
        createdNewKeyframe = false;

        // extract time stamp
        timeLaserInfoCur = msgIn.timestamp;

        // extract info and feature cloud
        cloudInfoPtr = &msgIn;
        laserCloudCornerLast = msgIn.cloud_corner;
        laserCloudSurfLast = msgIn.cloud_surface;

        std::lock_guard<std::mutex> lock(mtx);

        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;

            const auto t0 = std::chrono::steady_clock::now();
            resetFrameQuality();
            const auto t_reset = std::chrono::steady_clock::now();

            updateInitialGuess();
            const auto t_guess = std::chrono::steady_clock::now();

            extractSurroundingKeyFrames();
            const auto t_extract = std::chrono::steady_clock::now();

            downsampleCurrentScan();
            const auto t_downsample = std::chrono::steady_clock::now();

            scan2MapOptimization();
            const auto t_scan2map = std::chrono::steady_clock::now();

            const size_t keyframeCountBefore = cloudKeyPoses6D->size();
            saveKeyFramesAndFactor();
            createdNewKeyframe = cloudKeyPoses6D->size() > keyframeCountBefore;
            const auto t_save = std::chrono::steady_clock::now();

            correctPoses();
            const auto t_correct = std::chrono::steady_clock::now();

            updateOdometryState();
            const auto t_state = std::chrono::steady_clock::now();

            static int map_timing_count = 0;
            static double map_timing_total_ms = 0.0;
            auto ms = [](const auto& a, const auto& b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            const double total_ms = ms(t0, t_state);
            map_timing_total_ms += total_ms;
            if (debugTiming && ++map_timing_count % 20 == 0)
            {
                RCLCPP_INFO(get_logger(),
                    "[MAP_OPT_TIMING] reset=%.3f ms, guess=%.3f ms, extract=%.3f ms, "
                    "downsample=%.3f ms, scan2map=%.3f ms, save=%.3f ms, "
                    "correct=%.3f ms, state=%.3f ms, total=%.3f ms, avg_total=%.3f ms, "
                    "keyframes=%zu, corner_last_ds=%d, surf_last_ds=%d, corner_map=%d, surf_map=%d",
                    ms(t0, t_reset),
                    ms(t_reset, t_guess),
                    ms(t_guess, t_extract),
                    ms(t_extract, t_downsample),
                    ms(t_downsample, t_scan2map),
                    ms(t_scan2map, t_save),
                    ms(t_save, t_correct),
                    ms(t_correct, t_state),
                    total_ms,
                    map_timing_total_ms / static_cast<double>(map_timing_count),
                    cloudKeyPoses6D->size(),
                    laserCloudCornerLastDSNum,
                    laserCloudSurfLastDSNum,
                    laserCloudCornerFromMapDSNum,
                    laserCloudSurfFromMapDSNum);
            }

            // int keyframeAllowed = int(mappingPoseReliable && transformIsFinite(transformTobeMapped));
            /* 0617 娴嬭瘯棰戠巼 鏁呮娉ㄩ噴鎺?
            RCLCPP_WARN(get_logger(),
                "[T2M] time=%.6f roll=%.3f pitch=%.3f yaw=%.3f "
                "x=%.3f y=%.3f z=%.3f degenerate=%d reliable=%d keyframeAllowed=%d "
                "state=%s source=%s keyposes=%zu corner=%zu surf=%zu",
                timeLaserInfoCur,
                transformTobeMapped[0] * 180.0 / M_PI,
                transformTobeMapped[1] * 180.0 / M_PI,
                transformTobeMapped[2] * 180.0 / M_PI,
                transformTobeMapped[3],
                transformTobeMapped[4],
                transformTobeMapped[5],
                int(isDegenerate),
                int(mappingPoseReliable),
                keyframeAllowed,
                trackingStateName(),
                mappingPoseSource.c_str(),
                cloudKeyPoses3D->size(),
                laserCloudCornerLastDS->size(),
                laserCloudSurfLastDS->size());
            */
            cloudInfoPtr = nullptr;
            return true;
        }

        cloudInfoPtr = nullptr;
        return true;
    }

double mapOptimization::TimeLaserInfoCur() const{ return timeLaserInfoCur; }

const float* mapOptimization::TransformTobeMapped() const{ return transformTobeMapped; }

bool mapOptimization::CreatedNewKeyframe() const{ return createdNewKeyframe; }

void mapOptimization::ClearCreatedNewKeyframe(){ createdNewKeyframe = false; }

size_t mapOptimization::KeyPoseSize() const{
        return cloudKeyPoses6D ? cloudKeyPoses6D->size() : 0;
    }

PointTypePose mapOptimization::KeyPose(size_t idx) const{
        return cloudKeyPoses6D->points[idx];
    }

pcl::PointCloud<PointType>::Ptr mapOptimization::LatestRawCloudKeyFrame() const{
        if (rawCloudKeyFrames.empty())
            return nullptr;
        return rawCloudKeyFrames.back();
    }

void mapOptimization::pointAssociateToMap(PointType const * const pi, PointType * const po){
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

pcl::PointCloud<PointType>::Ptr mapOptimization::transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn){
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

gtsam::Pose3 mapOptimization::pclPointTogtsamPose3(PointTypePose thisPoint){
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

gtsam::Pose3 mapOptimization::trans2gtsamPose(float transformIn[]){
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

Eigen::Affine3f mapOptimization::pclPointToAffine3f(PointTypePose thisPoint){
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

Eigen::Affine3f mapOptimization::trans2Affine3f(const float transformIn[6]){
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

PointTypePose mapOptimization::trans2PointTypePose(float transformIn[]){
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

double mapOptimization::normalizeAngleRad(double angle){
        while (angle > M_PI)
            angle -= 2.0 * M_PI;
        while (angle < -M_PI)
            angle += 2.0 * M_PI;
        return angle;
    }

void mapOptimization::setTransformFromAffine(const Eigen::Affine3f& affine){
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(affine, x, y, z, roll, pitch, yaw);
        transformTobeMapped[0] = roll;
        transformTobeMapped[1] = pitch;
        transformTobeMapped[2] = yaw;
        transformTobeMapped[3] = x;
        transformTobeMapped[4] = y;
        transformTobeMapped[5] = z;
    }

bool mapOptimization::isFinitePoint(const PointType& p) const{
        return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
    }

void mapOptimization::filterInvalidAndRangeInPlace(
    pcl::PointCloud<PointType>::Ptr& cloud, const char* tag, bool alsoLimitSourceRange){
        (void)tag;
        if (!cloud || cloud->empty())
            return;

        pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>());
        filtered->reserve(cloud->size());
        const float maxRange2 = maxRawIcpSourceRange * maxRawIcpSourceRange;
        for (const auto& p : cloud->points)
        {
            if (!isFinitePoint(p))
                continue;
            if (alsoLimitSourceRange)
            {
                const float r2 = p.x * p.x + p.y * p.y + p.z * p.z;
                if (r2 > maxRange2)
                    continue;
            }
            filtered->push_back(p);
        }
        filtered->width = filtered->size();
        filtered->height = 1;
        filtered->is_dense = true;
        cloud.swap(filtered);
    }

void mapOptimization::cropMapCloudAroundPriorInPlace(
    pcl::PointCloud<PointType>::Ptr& cloud, const Eigen::Vector3f& center){
        if (!cloud || cloud->empty())
            return;

        pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>());
        filtered->reserve(cloud->size());
        const float maxRadius2 = maxRawIcpTargetRadius * maxRawIcpTargetRadius;
        for (const auto& p : cloud->points)
        {
            if (!isFinitePoint(p))
                continue;
            const float dx = p.x - center.x();
            const float dy = p.y - center.y();
            const float dz = p.z - center.z();
            if (dx * dx + dy * dy + dz * dz <= maxRadius2)
                filtered->push_back(p);
        }
        filtered->width = filtered->size();
        filtered->height = 1;
        filtered->is_dense = true;
        cloud.swap(filtered);
    }

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
        static int motion_gate_log_count = 0;
        const bool logThisCall = debugTiming && (++motion_gate_log_count % 20 == 0);

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

void mapOptimization::updateOutputTrajectoryHistory(const Eigen::Affine3f& outputAffine){
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

bool mapOptimization::prepareCurrentRawCloudForRegistration(){
        laserCloudRawLast->clear();
        if (!cloudInfoPtr || !cloudInfoPtr->cloud_deskewed || cloudInfoPtr->cloud_deskewed->empty())
            return false;

        pcl::copyPointCloud(*cloudInfoPtr->cloud_deskewed, *laserCloudRawLast);
        filterInvalidAndRangeInPlace(laserCloudRawLast, "raw_current_registration", true);
        return laserCloudRawLast->size() >= 300;
    }

bool mapOptimization::buildRawLocalMapForRegistration(){
        laserCloudRawFromMap->clear();
        laserCloudRawFromMapDS->clear();

        if (!cloudKeyPoses3D || cloudKeyPoses3D->empty() || rawCloudKeyFrames.empty())
            return false;

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        PointType searchPoint;
        searchPoint.x = transformTobeMapped[3];
        searchPoint.y = transformTobeMapped[4];
        searchPoint.z = transformTobeMapped[5];

        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);
        kdtreeSurroundingKeyPoses->radiusSearch(
            searchPoint,
            static_cast<double>(surroundingKeyframeSearchRadius),
            pointSearchInd,
            pointSearchSqDis);

        constexpr int maxRawKeyframes = 20;
        if (pointSearchInd.empty())
        {
            const int cloudSize = static_cast<int>(cloudKeyPoses3D->size());
            const int historyNum = std::min(maxRawKeyframes, cloudSize);
            for (int i = cloudSize - historyNum; i < cloudSize; ++i)
                pointSearchInd.push_back(i);
        }
        else if (static_cast<int>(pointSearchInd.size()) > maxRawKeyframes)
        {
            pointSearchInd.resize(maxRawKeyframes);
        }

        Eigen::Vector3f priorCenter(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]);
        for (const int idx : pointSearchInd)
        {
            if (idx < 0 ||
                idx >= static_cast<int>(rawCloudKeyFrames.size()) ||
                idx >= static_cast<int>(cloudKeyPoses6D->size()) ||
                !rawCloudKeyFrames[idx] ||
                rawCloudKeyFrames[idx]->empty())
                continue;

            pcl::PointCloud<PointType>::Ptr transformed =
                transformPointCloud(rawCloudKeyFrames[idx], &cloudKeyPoses6D->points[idx]);
            cropMapCloudAroundPriorInPlace(transformed, priorCenter);
            if (!transformed->empty())
                *laserCloudRawFromMap += *transformed;
        }

        if (laserCloudRawFromMap->size() < 800)
            return false;

        downSizeFilterRawICP.setInputCloud(laserCloudRawFromMap);
        downSizeFilterRawICP.filter(*laserCloudRawFromMapDS);
        return laserCloudRawFromMapDS->size() >= 500;
    }

bool mapOptimization::rawCloudICPFallback(
    const Eigen::Affine3f& initialGuess, Eigen::Affine3f& resultAffine, double& fitnessScore){
        fitnessScore = std::numeric_limits<double>::infinity();
        static int raw_icp_log_count = 0;
        const bool logThisCall = debugTiming && (++raw_icp_log_count % 20 == 0);

        if (!cloudKeyPoses3D || cloudKeyPoses3D->empty() || rawCloudKeyFrames.empty())
            return false;

        setTransformFromAffine(initialGuess);
        if (!buildRawLocalMapForRegistration())
            return false;

        if (!prepareCurrentRawCloudForRegistration())
            return false;

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setInputSource(laserCloudRawLast);
        icp.setInputTarget(laserCloudRawFromMapDS);
        icp.setMaxCorrespondenceDistance(1.0);
        icp.setMaximumIterations(35);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-4);
        icp.setRANSACIterations(0);

        pcl::PointCloud<PointType>::Ptr aligned(new pcl::PointCloud<PointType>());
        icp.align(*aligned, initialGuess.matrix());

        fitnessScore = icp.hasConverged() ?
            icp.getFitnessScore(2.0) :
            std::numeric_limits<double>::infinity();
        if (!icp.hasConverged())
        {
            if (logThisCall)
            {
                RCLCPP_INFO(get_logger(),
                    "[ICP][FAIL] did not converge. source=%zu target=%zu",
                    laserCloudRawLast->size(),
                    laserCloudRawFromMapDS->size());
            }
            return false;
        }

        resultAffine = Eigen::Affine3f::Identity();
        resultAffine.matrix() = icp.getFinalTransformation();
        if (fitnessScore > 0.35)
        {
            if (logThisCall)
            {
                RCLCPP_INFO(get_logger(),
                    "[ICP][FAIL] fitness %.6f > 0.350000. source=%zu target=%zu",
                    fitnessScore,
                    laserCloudRawLast->size(),
                    laserCloudRawFromMapDS->size());
            }
            return false;
        }

        if (logThisCall)
        {
            RCLCPP_INFO(get_logger(),
                "[ICP][CANDIDATE] fitness=%.6f source=%zu target=%zu",
                fitnessScore,
                laserCloudRawLast->size(),
                laserCloudRawFromMapDS->size());
        }
        return true;
    }

void mapOptimization::copyTransform(const float src[6], float dst[6]){
        for (int i = 0; i < 6; ++i)
            dst[i] = src[i];
    }

bool mapOptimization::transformIsFinite(const float transformIn[6]){
        for (int i = 0; i < 6; ++i)
        {
            if (!std::isfinite(static_cast<double>(transformIn[i])))
                return false;
        }
        return true;
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
        lastLMCloudSelNum = 0;
        lastLMIterationCount = 0;
        lastLMRan = false;
        lastLMConverged = false;
        currentOdomCov = 2;
        isDegenerate = false;
    }

bool mapOptimization::acceptMappingPose(const std::string& source){
        mappingTrackingState = MappingTrackingState::TRACKING;
        mappingPoseReliable = true;
        lidarCorrectionFlag = 0;
        mappingPoseSource = source;
        currentOdomCov = 0;
        isDegenerate = false;
        mappingFailureCount = 0;
        mappingFirstFailureTime = -1.0;
        surroundingKeyFrameIndices.clear();
        return true;
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

void mapOptimization::loopClosureThread(){
        if (loopClosureEnableFlag == false)
            return;

        const double frequency = loopClosureFrequency > 0.0f ? loopClosureFrequency : 1.0;
        rclcpp::Rate rate(frequency);
        while (rclcpp::ok() && loopClosureThreadRunning_.load())
        {
            rate.sleep();
            if (!loopClosureThreadRunning_.load())
                break;

            performLoopClosure();
        }
    }

void mapOptimization::performLoopClosure(){
        if (loopClosureEnableFlag == false)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        {
            std::lock_guard<std::mutex> lock(mtx);
            *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
            *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        }

        // find keys
        int loopKeyCur;
        int loopKeyPre;
        if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
            return;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
        }

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        {
            std::lock_guard<std::mutex> lock(mtx);
            loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
            loopPoseQueue.push_back(poseFrom.between(poseTo));
            loopNoiseQueue.push_back(constraintNoise);

            // add loop constriant
            loopIndexContainer[loopKeyCur] = loopKeyPre;
        }
    }

bool mapOptimization::detectLoopClosureDistance(int *latestID, int *closestID){
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

void mapOptimization::loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum){
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

void mapOptimization::updateInitialGuess(){
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        static Eigen::Affine3f lastImuTransformation = Eigen::Affine3f::Identity();
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation = Eigen::Affine3f::Identity();

        if (cloudKeyPoses3D->points.empty())
        {
            transformTobeMapped[0] = 0.0f;
            transformTobeMapped[1] = 0.0f;
            transformTobeMapped[2] = 0.0f;

            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0.0f;

            if (cloudInfoPtr && cloudInfoPtr->imu_available)
            {
                lastImuTransformation = pcl::getTransformation(
                    0.0f, 0.0f, 0.0f,
                    cloudInfoPtr->imu_roll_init,
                    cloudInfoPtr->imu_pitch_init,
                    cloudInfoPtr->imu_yaw_init);
            }
            else
            {
                lastImuTransformation = Eigen::Affine3f::Identity();
            }
            lastImuPreTransAvailable = false;

            copyTransform(transformTobeMapped, frameInitialGuessTransform);
            return;
        }

        if (cloudInfoPtr && cloudInfoPtr->odom_available)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(
                cloudInfoPtr->initial_guess_x,
                cloudInfoPtr->initial_guess_y,
                cloudInfoPtr->initial_guess_z,
                cloudInfoPtr->initial_guess_roll,
                cloudInfoPtr->initial_guess_pitch,
                cloudInfoPtr->initial_guess_yaw);
            if (!lastImuPreTransAvailable)
            {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            }
            else
            {
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                Eigen::Affine3f transFinal = transTobe * transIncre;
                setTransformFromAffine(transFinal);
                lastImuPreTransformation = transBack;

                if (cloudInfoPtr->imu_available)
                {
                    lastImuTransformation = pcl::getTransformation(
                        0.0f, 0.0f, 0.0f,
                        cloudInfoPtr->imu_roll_init,
                        cloudInfoPtr->imu_pitch_init,
                        cloudInfoPtr->imu_yaw_init);
                }
                copyTransform(transformTobeMapped, frameInitialGuessTransform);
                return;
            }
        }

        if (cloudInfoPtr && cloudInfoPtr->imu_available)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(
                0.0f, 0.0f, 0.0f,
                cloudInfoPtr->imu_roll_init,
                cloudInfoPtr->imu_pitch_init,
                cloudInfoPtr->imu_yaw_init);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;
            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            setTransformFromAffine(transFinal);
            lastImuTransformation = transBack;
            copyTransform(transformTobeMapped, frameInitialGuessTransform);
            return;
        }

        if (!transformIsFinite(transformTobeMapped))
        {
            for (int i = 0; i < 6; ++i)
                transformTobeMapped[i] = 0.0f;
        }
        copyTransform(transformTobeMapped, frameInitialGuessTransform);
    }

void mapOptimization::extractNearby(){
        const auto t0 = std::chrono::steady_clock::now();
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        for (auto& pt : surroundingKeyPosesDS->points)
        {
            if (kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis) > 0)
                pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        const int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses - 1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }
        const auto t_search = std::chrono::steady_clock::now();
        (void)t_search;
        (void)t0;
        extractCloud(surroundingKeyPosesDS);
    }

void mapOptimization::extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract){
        const auto t_build0 = std::chrono::steady_clock::now();
        // fuse the map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        surroundingKeyFrameIndices.clear();
        const size_t candidateCount = cloudToExtract ? cloudToExtract->size() : 0;
        for (int i = 0; cloudToExtract && i < (int)cloudToExtract->size(); ++i)
        {
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            if (thisKeyInd < 0 ||
                thisKeyInd >= (int)cornerCloudKeyFrames.size() ||
                thisKeyInd >= (int)surfCloudKeyFrames.size() ||
                thisKeyInd >= (int)cloudKeyPoses6D->size())
                continue;

            surroundingKeyFrameIndices.push_back(thisKeyInd);
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }
        const auto t_build = std::chrono::steady_clock::now();

        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // 0428 鏂板
        if(laserCloudCornerFromMapDSNum < 500)
        {
            pcl::copyPointCloud(*laserCloudCornerFromMap, *laserCloudCornerFromMapDS);
            laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        }
        //  0428 end
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();
        //0428 
        if(laserCloudSurfFromMapDSNum < 500)
        {
            pcl::copyPointCloud(*laserCloudSurfFromMap, *laserCloudSurfFromMapDS);
            laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        }
        //0428 end
        const auto t_voxel = std::chrono::steady_clock::now();

        kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
        kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
        const auto t_kdtree = std::chrono::steady_clock::now();
        lastLocalMapKdtreeMs =
            std::chrono::duration<double, std::milli>(t_kdtree - t_voxel).count();

        static int local_map_timing_count = 0;
        if (debugTiming && ++local_map_timing_count % 20 == 0)
        {
            auto ms = [](const auto& a, const auto& b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            RCLCPP_INFO(get_logger(),
                "[LOCAL_MAP_TIMING] keyframes=%zu, candidates=%zu, selected=%zu, "
                "corner_raw=%zu, surf_raw=%zu, corner_ds=%d, surf_ds=%d, "
                "build_ms=%.3f, voxel_ms=%.3f, kdtree_ms=%.3f, cache=%zu",
                cloudKeyPoses6D->size(),
                candidateCount,
                surroundingKeyFrameIndices.size(),
                laserCloudCornerFromMap->size(),
                laserCloudSurfFromMap->size(),
                laserCloudCornerFromMapDSNum,
                laserCloudSurfFromMapDSNum,
                ms(t_build0, t_build),
                ms(t_build, t_voxel),
                lastLocalMapKdtreeMs,
                laserCloudMapContainer.size());
        }

        // clear map cache if too large
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

void mapOptimization::extractSurroundingKeyFrames(){
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        extractNearby();
    }

void mapOptimization::limitPointCloudUniform(pcl::PointCloud<PointType>::Ptr cloud, int maxNum){
        if (!cloud || maxNum <= 0 || static_cast<int>(cloud->size()) <= maxNum)
            return;

        pcl::PointCloud<PointType> limited;
        limited.reserve(maxNum);
        const double step = static_cast<double>(cloud->size()) / static_cast<double>(maxNum);
        for (int i = 0; i < maxNum; ++i)
        {
            const int index = static_cast<int>(i * step);
            limited.push_back(cloud->points[index]);
        }
        limited.height = 1;
        limited.width = limited.size();
        limited.is_dense = cloud->is_dense;
        cloud->swap(limited);
    }

void mapOptimization::downsampleCurrentScan(){
        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();
        if(laserCloudCornerLastDSNum < 500)
        {
            pcl::copyPointCloud(*laserCloudCornerLast, *laserCloudCornerLastDS);
            laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();
        }


        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
        if(laserCloudSurfLastDSNum < 500)
        {
            pcl::copyPointCloud(*laserCloudSurfLast, *laserCloudSurfLastDS);
            laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
        }

        if (isOnlineMapping && onlineLimitOptimizationPoints)
        {
            limitPointCloudUniform(laserCloudSurfLastDS, onlineMaxSurfOptimizationPoints);
            laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
            limitPointCloudUniform(laserCloudCornerLastDS, onlineMaxCornerOptimizationPoints);
            laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();
        }
    }

void mapOptimization::updatePointAssociateToMap(){
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

void mapOptimization::cornerOptimization(){
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
                    
            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

void mapOptimization::surfOptimization(){
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

void mapOptimization::combineOptimizationCoeffs(){
        // combine corner coeffs
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

bool mapOptimization::LMOptimization(int iterCount){
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        lastLMCloudSelNum = laserCloudSelNum;
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // lidar -> camera
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }

void mapOptimization::scan2MapOptimization(){
        {
            static int scan2map_log_count = 0;
            const bool logThisFrame = debugTiming && (++scan2map_log_count % 20 == 0);
            currentOdomCov = 0;
            isDegenerate = false;
            copyTransform(transformTobeMapped, frameInitialGuessTransform);
            Eigen::Affine3f priorAffine = trans2Affine3f(transformTobeMapped);

            if (cloudKeyPoses3D->points.empty())
            {
                acceptMappingPose("INIT");
                return;
            }

            bool lmRan = false;
            bool lmConverged = false;
            bool lmFinite = false;
            bool lmMotionOk = false;
            int finalCoeffNum = 0;
            Eigen::Affine3f lmAffine = priorAffine;
            const int maxIterations =
                isOnlineMapping ? onlineMaxOptimizationIterations : maxOptimizationIterations;

            if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum &&
                laserCloudSurfLastDSNum > surfFeatureMinValidNum)
            {
                lmRan = true;
                lastLMRan = true;
                kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
                kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

                for (int iterCount = 0; iterCount < maxIterations; iterCount++)
                {
                    laserCloudOri->clear();
                    coeffSel->clear();

                    cornerOptimization();
                    surfOptimization();
                    combineOptimizationCoeffs();

                    finalCoeffNum = static_cast<int>(laserCloudOri->size());
                    lmConverged = LMOptimization(iterCount);
                    lastLMIterationCount = iterCount + 1;

                    if (lmConverged)
                    {
                        lastLMConverged = true;
                        break;
                    }
                }

                transformUpdate();
                lmFinite = transformIsFinite(transformTobeMapped);
                if (lmFinite)
                {
                    lmAffine = trans2Affine3f(transformTobeMapped);
                    MotionContinuityInfo lmMotion = evaluateMotionContinuity(lmAffine, "LM");
                    lmMotionOk = lmMotion.continuous;
                }

                if (lmConverged && !isDegenerate && finalCoeffNum >= 80 && lmMotionOk)
                {
                    currentOdomCov = 0;
                    mappingPoseReliable = true;
                    mappingPoseSource = "LM";
                    acceptMappingPose("LM");
                    if (logThisFrame)
                    {
                        RCLCPP_INFO(get_logger(),
                            "[LM][USE_HIGH] converged=%d degenerate=%d coeff=%d motionOK=%d cov=0 save=yes",
                            int(lmConverged),
                            int(isDegenerate),
                            finalCoeffNum,
                            int(lmMotionOk));
                    }
                    return;
                }
            }
            else
            {
                if (logThisFrame)
                {
                    RCLCPP_INFO(get_logger(),
                        "[LM][FEATURE_WEAK] cornerDS=%d/%d surfDS=%d/%d. Try raw ICP fallback.",
                        laserCloudCornerLastDSNum,
                        edgeFeatureMinValidNum,
                        laserCloudSurfLastDSNum,
                        surfFeatureMinValidNum);
                }
            }

            if (logThisFrame)
            {
                RCLCPP_INFO(get_logger(),
                    "[LM][SUSPECT] ran=%d converged=%d degenerate=%d coeff=%d motionOK=%d. Try raw ICP fallback.",
                    int(lmRan),
                    int(lmConverged),
                    int(isDegenerate),
                    finalCoeffNum,
                    int(lmMotionOk));
            }

            setTransformFromAffine(priorAffine);
            Eigen::Affine3f icpAffine = priorAffine;
            double icpFitness = std::numeric_limits<double>::infinity();
            if (rawCloudICPFallback(priorAffine, icpAffine, icpFitness))
            {
                MotionContinuityInfo icpMotion = evaluateMotionContinuity(icpAffine, "ICP");
                const bool icpHigh = icpMotion.continuous && icpFitness < 0.3;
                if (icpHigh)
                {
                    setTransformFromAffine(icpAffine);
                    transformUpdate();
                    currentOdomCov = 0;
                    isDegenerate = false;
                    mappingPoseReliable = true;
                    mappingPoseSource = "ICP";
                    acceptMappingPose("ICP");

                    if (logThisFrame)
                    {
                        RCLCPP_INFO(get_logger(),
                            "[ICP][USE_HIGH] fitness=%.6f motionOK=%d cov=0 save=yes",
                            icpFitness,
                            int(icpMotion.continuous));
                    }
                    return;
                }

                if (logThisFrame)
                {
                    RCLCPP_INFO(get_logger(),
                        "[ICP][REJECT_WEAK] fitness=%.6f motionOK=%d. Continue fallback.",
                        icpFitness,
                        int(icpMotion.continuous));
                }
            }
            else
            {
                if (logThisFrame)
                    RCLCPP_INFO(get_logger(), "[ICP][FAILED] Continue fallback.");
            }

            if (lmRan && lmFinite)
            {
                setTransformFromAffine(lmAffine);
                transformUpdate();
                currentOdomCov = lmMotionOk ? 1 : 2;
                isDegenerate = true;
                mappingPoseReliable = false;
                lidarCorrectionFlag = 2;
                mappingPoseSource = "FALLBACK_LM";
                mappingTrackingState = MappingTrackingState::LOST;
                if (mappingFailureCount == 0)
                    mappingFirstFailureTime = timeLaserInfoCur;
                ++mappingFailureCount;

                if (logThisFrame)
                {
                    RCLCPP_INFO(get_logger(),
                        "[FALLBACK][PUBLISH_LM] ICP failed. lmMotionOK=%d cov=%d save=no",
                        int(lmMotionOk),
                        currentOdomCov);
                }
                return;
            }

            setTransformFromAffine(priorAffine);
            transformUpdate();
            currentOdomCov = 2;
            isDegenerate = true;
            mappingPoseReliable = false;
            lidarCorrectionFlag = 2;
            mappingPoseSource = "FALLBACK_PRIOR";
            mappingTrackingState = MappingTrackingState::LOST;
            if (mappingFailureCount == 0)
                mappingFirstFailureTime = timeLaserInfoCur;
            ++mappingFailureCount;

            if (logThisFrame)
            {
                RCLCPP_INFO(get_logger(),
                    "[FALLBACK][PUBLISH_PRIOR] LM unavailable and ICP failed. cov=2 save=no");
            }
            return;
        }
    }

void mapOptimization::transformUpdate(){
        if (cloudInfoPtr->imu_available == true)
        {
            if (std::abs(cloudInfoPtr->imu_pitch_init) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf2::Quaternion imuQuaternion;
                tf2::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfoPtr->imu_roll_init, 0, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfoPtr->imu_pitch_init, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

float mapOptimization::constraintTransformation(float value, float limit){
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

bool mapOptimization::saveFrame(){
        if (currentOdomCov != 0)
            return false;

        if (cloudKeyPoses3D->points.empty())
            return true;

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

void mapOptimization::addOdomFactor(){
        if (cloudKeyPoses3D->points.empty())
        {
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            noiseModel::Diagonal::shared_ptr odometryNoise =
                noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

void mapOptimization::addLoopFactor(){
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

void mapOptimization::saveKeyFramesAndFactor(){
        const auto t0 = std::chrono::steady_clock::now();
        auto t_save_frame = t0;
        auto t_odom = t0;
        auto t_loop = t0;
        auto t_isam = t0;
        auto t_estimate = t0;
        auto t_copy = t0;
        bool saved_keyframe = false;
        auto maybeLogIsamTiming = [&](const char* result) {
            if (!debugTiming)
                return;
            static int isam_timing_count = 0;
            if (++isam_timing_count % 20 != 0)
                return;

            const auto t_end = std::chrono::steady_clock::now();
            auto ms = [](const auto& a, const auto& b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            RCLCPP_INFO(get_logger(),
                "[ISAM_TIMING] result=%s, saveFrame_ms=%.3f, addOdom_ms=%.3f, "
                "addLoop_ms=%.3f, isam_update_ms=%.3f, estimate_ms=%.3f, "
                "copy_keyframe_ms=%.3f, total_ms=%.3f, saved=%d, keyframes=%zu",
                result,
                ms(t0, t_save_frame),
                ms(t_save_frame, t_odom),
                ms(t_odom, t_loop),
                ms(t_loop, t_isam),
                ms(t_isam, t_estimate),
                ms(t_estimate, t_copy),
                ms(t0, t_end),
                int(saved_keyframe),
                cloudKeyPoses6D->size());
        };

        if (!mappingPoseReliable  || !transformIsFinite(transformTobeMapped)) //  || isDegenerate
        {
            static int keyframe_skip_log_count = 0;
            if (debugTiming && ++keyframe_skip_log_count % 20 == 0)
            {
                RCLCPP_INFO(get_logger(),
                    "[KEYFRAME_SKIP_FINAL] trackingAccepted=%d source=%s degenerate=%d finite=%d. "
                    "Skip keyframe and pose factor only.",
                    int(mappingPoseReliable),
                    mappingPoseSource.c_str(),
                    int(isDegenerate),
                    int(transformIsFinite(transformTobeMapped)));
            }
            t_save_frame = std::chrono::steady_clock::now();
            t_odom = t_save_frame;
            t_loop = t_save_frame;
            t_isam = t_save_frame;
            t_estimate = t_save_frame;
            t_copy = t_save_frame;
            maybeLogIsamTiming("POSE_NOT_RELIABLE");
            return;
        }

        const bool should_save_frame = saveFrame();
        t_save_frame = std::chrono::steady_clock::now();
        if (should_save_frame == false)
        {
            t_odom = t_save_frame;
            t_loop = t_save_frame;
            t_isam = t_save_frame;
            t_estimate = t_save_frame;
            t_copy = t_save_frame;
            maybeLogIsamTiming("SAVE_FRAME_FALSE");
            return;
        }

        // odom factor
        addOdomFactor();
        t_odom = std::chrono::steady_clock::now();

        // loop factor
        addLoopFactor();
        t_loop = std::chrono::steady_clock::now();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }
        t_isam = std::chrono::steady_clock::now();

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        t_estimate = std::chrono::steady_clock::now();
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        // Temporarily disabled for speed. Keep this line for future re-enable if covariance is needed.
        // poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisRawKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        if (cloudInfoPtr && cloudInfoPtr->cloud_deskewed && !cloudInfoPtr->cloud_deskewed->empty())
        {
            pcl::copyPointCloud(*cloudInfoPtr->cloud_deskewed, *thisRawKeyFrame);
            filterInvalidAndRangeInPlace(thisRawKeyFrame, "raw_keyframe_save", true);
        }

        // save key frame cloud
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);
        rawCloudKeyFrames.push_back(thisRawKeyFrame);
        t_copy = std::chrono::steady_clock::now();
        saved_keyframe = true;
        maybeLogIsamTiming("SAVED");
    }

void mapOptimization::correctPoses(){
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear map cache
            laserCloudMapContainer.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
            }

            aLoopIsClosed = false;
        }
    }

void mapOptimization::updateOdometryState(){
        if (!transformIsFinite(transformTobeMapped))
            return;

        updateOutputTrajectoryHistory(trans2Affine3f(transformTobeMapped));
    }
