#include "core/lio_sam/map_optimization.h"
#include "core/lio_sam/loop.h"

mapOptimization::mapOptimization(const rclcpp::NodeOptions & options) : ParamServer("lio_sam_mapOptimization", options){
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

        RCLCPP_INFO(get_logger(),
            "[跨任务状态修复][mapOptimization] 构造完成 this=%p, time_member_addr=%p, object_last_stamp=%.9f, mappingProcessInterval=%.6f",
            static_cast<void*>(this), static_cast<void*>(&timeLastProcessing_),
            timeLastProcessing_, mappingProcessInterval);

        if (loopClosureEnableFlag)
        {
            loopClosureThreadRunning_.store(true);
            loopClosureThread_ = std::thread(&mapOptimization::loopClosureThread, this);
        }
    }

mapOptimization::~mapOptimization() {
    RCLCPP_INFO(get_logger(),
        "[跨任务状态修复][mapOptimization] 开始析构 this=%p, object_last_stamp=%.9f, run_calls=%llu, executed=%llu, skipped=%llu, keyposes=%zu",
        static_cast<void*>(this), timeLastProcessing_,
        static_cast<unsigned long long>(diagnosticRunCalls),
        static_cast<unsigned long long>(diagnosticExecutedCalls),
        static_cast<unsigned long long>(diagnosticSkippedCalls),
        cloudKeyPoses6D ? cloudKeyPoses6D->size() : 0U);
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
        lastRunExecuted = false;
        ++diagnosticRunCalls;

        timeLaserInfoCur = msgIn.timestamp;

        // extract info and feature cloud
        cloudInfo = msgIn;
        if (!msgIn.cloud_corner || !msgIn.cloud_surface || !msgIn.cloud_deskewed) {
            return false;
        }
        *laserCloudCornerLast = *msgIn.cloud_corner;
        *laserCloudSurfLast = *msgIn.cloud_surface;

        std::lock_guard<std::mutex> lock(mtx);

        if (timeLaserInfoCur - timeLastProcessing_ >= mappingProcessInterval)
        {
            timeLastProcessing_ = timeLaserInfoCur;

            updateInitialGuess();
            currentOdomCov = 0;

            extractSurroundingKeyFrames();

            downsampleCurrentScan();

            scan2MapOptimization();

            saveKeyFramesAndFactor();

            correctPoses();

            updateOdometryState();
            lastRunExecuted = true;
            ++diagnosticExecutedCalls;
        }
        else
        {
            ++diagnosticSkippedCalls;
        }
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

void mapOptimization::pointAssociateToMap(PointType const * const pi, PointType * const po)
{
    po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
    po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
    po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
    po->intensity = pi->intensity;
}

pcl::PointCloud<PointType>::Ptr mapOptimization::transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
{
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

gtsam::Pose3 mapOptimization::pclPointTogtsamPose3(PointTypePose thisPoint)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
}

gtsam::Pose3 mapOptimization::trans2gtsamPose(float transformIn[])
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

Eigen::Affine3f mapOptimization::pclPointToAffine3f(PointTypePose thisPoint)
{
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

Eigen::Affine3f mapOptimization::trans2Affine3f(const float transformIn[6])
{
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}

PointTypePose mapOptimization::trans2PointTypePose(float transformIn[])
{
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll  = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw   = transformIn[2];
    return thisPose6D;
}


void mapOptimization::updateInitialGuess()
{
    // save current transformation before any processing
    incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

    // initialization
    if (cloudKeyPoses3D->points.empty())
    {
        //transformTobeMapped[0] = cloudInfo.imu_roll_init;
        //transformTobeMapped[1] = cloudInfo.imu_pitch_init;
        //transformTobeMapped[2] = cloudInfo.imu_yaw_init;
        transformTobeMapped[0] = 0.0;  // roll
        transformTobeMapped[1] = 0.0;  // pitch
        transformTobeMapped[2] = 0.0;  // yaw

        if (!useImuHeadingInitialization)
            transformTobeMapped[2] = 0;

        lastImuTransformation_ = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
        
        {     //0429
            if (std::abs(pcl::rad2deg(transformTobeMapped[0])) > debugRollPitchWarnDeg ||
            std::abs(pcl::rad2deg(transformTobeMapped[1])) > debugRollPitchWarnDeg)
            {
                RCLCPP_WARN(this->get_logger(),
                    "[RP-THRESH][updateInitialGuess-init] roll=%.3f pitch=%.3f thresh=%.3f deg",
                    pcl::rad2deg(transformTobeMapped[0]),
                    pcl::rad2deg(transformTobeMapped[1]),
                    debugRollPitchWarnDeg);
            }
        
        }
        return;
    }

    // use imu pre-integration estimation for pose guess
    if (cloudInfo.odom_available == true)
    {
        Eigen::Affine3f transBack = pcl::getTransformation(
            cloudInfo.initial_guess_x, cloudInfo.initial_guess_y, cloudInfo.initial_guess_z,
            cloudInfo.initial_guess_roll, cloudInfo.initial_guess_pitch, cloudInfo.initial_guess_yaw);
        if (lastImuPreTransAvailable_ == false)
        {
            lastImuPreTransformation_ = transBack;
            lastImuPreTransAvailable_ = true;
        } else {
            Eigen::Affine3f transIncre = lastImuPreTransformation_.inverse() * transBack;
            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuPreTransformation_ = transBack;

            lastImuTransformation_ = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
            
            if (std::abs(pcl::rad2deg(transformTobeMapped[0])) > debugRollPitchWarnDeg ||
                std::abs(pcl::rad2deg(transformTobeMapped[1])) > debugRollPitchWarnDeg)
            {
                RCLCPP_WARN(this->get_logger(),
                    "[RP-THRESH][updateInitialGuess-odom] roll=%.3f pitch=%.3f guess_rp=(%.3f, %.3f) thresh=%.3f deg",
                    pcl::rad2deg(transformTobeMapped[0]),
                    pcl::rad2deg(transformTobeMapped[1]),
                    pcl::rad2deg(cloudInfo.initial_guess_roll),
                    pcl::rad2deg(cloudInfo.initial_guess_pitch),
                    debugRollPitchWarnDeg);
            }
            
            return;
        }
    }

    // use imu incremental estimation for pose guess (only rotation)
    if (cloudInfo.imu_available == true)
    {
        Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init);
        Eigen::Affine3f transIncre = lastImuTransformation_.inverse() * transBack;

        Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
        Eigen::Affine3f transFinal = transTobe * transIncre;
        pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

        lastImuTransformation_ = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
        return;
    }
}

void mapOptimization::extractForLoopClosure()
{
    pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
    int numPoses = cloudKeyPoses3D->size();
    for (int i = numPoses-1; i >= 0; --i)
    {
        if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
            cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
        else
            break;
    }

    extractCloud(cloudToExtract);
}

void mapOptimization::extractNearby()
{
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
    for(auto& pt : surroundingKeyPosesDS->points)
    {
        kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
        pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
    }

    // also extract some latest key frames in case the robot rotates in one position
    int numPoses = cloudKeyPoses3D->size();
    for (int i = numPoses-1; i >= 0; --i)
    {
        if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
            surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
        else
            break;
    }

    extractCloud(surroundingKeyPosesDS);
}

void mapOptimization::extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
{
    // fuse the map
    laserCloudCornerFromMap->clear();
    laserCloudSurfFromMap->clear(); 
    for (int i = 0; i < (int)cloudToExtract->size(); ++i)
    {
        if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
            continue;

        int thisKeyInd = (int)cloudToExtract->points[i].intensity;
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

    // Downsample the surrounding corner key frames (or map)
    downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
    downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
    laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
    // 0428 新增
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

    // clear map cache if too large
    if (laserCloudMapContainer.size() > 1000)
        laserCloudMapContainer.clear();
}

void mapOptimization::extractSurroundingKeyFrames()
{
    if (cloudKeyPoses3D->points.empty() == true)
        return; 
    
    // if (loopClosureEnableFlag == true)
    // {
    //     extractForLoopClosure();    
    // } else {
    //     extractNearby();
    // }

    extractNearby();
}

void mapOptimization::downsampleCurrentScan()
{
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
}

void mapOptimization::updatePointAssociateToMap()
{
    transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
}

void mapOptimization::cornerOptimization()
{
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

void mapOptimization::surfOptimization()
{
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

void mapOptimization::combineOptimizationCoeffs()
{
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

bool mapOptimization::LMOptimization(int iterCount)
{
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


double mapOptimization::normalizeAngleRad(double angle)
{
    while (angle > M_PI)  angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

void mapOptimization::setTransformFromAffine(const Eigen::Affine3f& affine)
{
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(affine, x, y, z, roll, pitch, yaw);
    transformTobeMapped[0] = roll;
    transformTobeMapped[1] = pitch;
    transformTobeMapped[2] = yaw;
    transformTobeMapped[3] = x;
    transformTobeMapped[4] = y;
    transformTobeMapped[5] = z;
}

void mapOptimization::copyTransform(const float src[6], float dst[6])
{
    std::copy(src, src + 6, dst);
}

bool mapOptimization::transformIsFinite(const float transformIn[6])
{
    for (int i = 0; i < 6; ++i)
    {
        if (!std::isfinite(transformIn[i]))
            return false;
    }
    return true;
}

const char* mapOptimization::trackingStateName() const
{
    return mappingTrackingState == MappingTrackingState::TRACKING ? "TRACKING" : "LOST";
}

void mapOptimization::resetFrameQuality()
{
    mappingPoseReliable = true;
    lidarCorrectionFlag = 0;
    mappingPoseSource = "INIT";
}

bool mapOptimization::acceptMappingPose(const std::string& source)
{
    mappingPoseReliable = true;
    lidarCorrectionFlag = 0;
    mappingPoseSource = source;
    mappingTrackingState = MappingTrackingState::TRACKING;
    mappingFailureCount = 0;
    mappingFirstFailureTime = -1.0;
    return true;
}

void mapOptimization::rejectMappingPose(const std::string& source)
{
    mappingPoseReliable = false;
    lidarCorrectionFlag = 2;
    mappingPoseSource = source;
    mappingTrackingState = MappingTrackingState::LOST;
    if (mappingFailureCount++ == 0)
        mappingFirstFailureTime = timeLaserInfoCur;
}

bool mapOptimization::isFinitePoint(const PointType& p) const
{
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

void mapOptimization::filterInvalidAndRangeInPlace(pcl::PointCloud<PointType>::Ptr& cloud, const char* tag, bool alsoLimitSourceRange)
{
    if (!cloud || cloud->empty()) return;
    pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>());
    filtered->reserve(cloud->size());
    const float maxRange2 = maxRawIcpSourceRange * maxRawIcpSourceRange;
    for (const auto& p : cloud->points)
    {
        if (!isFinitePoint(p)) continue;
        if (alsoLimitSourceRange)
        {
            const float r2 = p.x*p.x + p.y*p.y + p.z*p.z;
            if (r2 > maxRange2) continue;
        }
        filtered->push_back(p);
    }
    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = true;
    cloud.swap(filtered);
}

void mapOptimization::cropMapCloudAroundPriorInPlace(pcl::PointCloud<PointType>::Ptr& cloud, const Eigen::Vector3f& center)
{
    if (!cloud || cloud->empty()) return;
    pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>());
    filtered->reserve(cloud->size());
    const float r2max = maxRawIcpTargetRadius * maxRawIcpTargetRadius;
    for (const auto& p : cloud->points)
    {
        if (!isFinitePoint(p)) continue;
        const float dx = p.x - center.x();
        const float dy = p.y - center.y();
        const float dz = p.z - center.z();
        if (dx*dx + dy*dy + dz*dz <= r2max) filtered->push_back(p);
    }
    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = true;
    cloud.swap(filtered);
}

bool mapOptimization::poseCloseToPrior(const Eigen::Affine3f& priorAffine,
                                       const Eigen::Affine3f& candidateAffine,
                                       const double maxTrans,
                                       const double maxYawDeg,
                                       const double maxZ,
                                       const char* tag)
{
    Eigen::Affine3f rel = priorAffine.inverse() * candidateAffine;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles(rel, dx, dy, dz, droll, dpitch, dyaw);

    const double dTrans = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double dYawDeg = std::abs(pcl::rad2deg(normalizeAngleRad(dyaw)));
    const double dZ = std::abs(dz);

    float cx, cy, cz, cr, cp, cyaw;
    pcl::getTranslationAndEulerAngles(candidateAffine, cx, cy, cz, cr, cp, cyaw);
    const double rollAbsDeg = std::abs(pcl::rad2deg(cr));
    const double pitchAbsDeg = std::abs(pcl::rad2deg(cp));

    const bool ok = (dTrans <= maxTrans && dYawDeg <= maxYawDeg && dZ <= maxZ &&
                        rollAbsDeg < debugRollPitchWarnDeg && pitchAbsDeg < debugRollPitchWarnDeg);
    if (!ok)
    {
        RCLCPP_WARN(this->get_logger(),
            "[BAD FRAME][%s] candidate is not consistent with current prior. "
            "dTrans=%.3f/%.3f dYaw=%.2f/%.2fdeg dZ=%.3f/%.3f "
            "candidate xyz/rpy=(%.3f %.3f %.3f | %.2f %.2f %.2f deg)",
            tag, dTrans, maxTrans, dYawDeg, maxYawDeg, dZ, maxZ,
            cx, cy, cz, pcl::rad2deg(cr), pcl::rad2deg(cp), pcl::rad2deg(cyaw));
    }
    return ok;
}

double mapOptimization::yawFromAffine(const Eigen::Affine3f& a)
{
    float x, y, z, r, p, yaw;
    pcl::getTranslationAndEulerAngles(a, x, y, z, r, p, yaw);
    return yaw;
}

double mapOptimization::xyDistance(const Eigen::Affine3f& a, const Eigen::Affine3f& b)
{
    float ax, ay, az, ar, ap, ayaw;
    float bx, by, bz, br, bp, byaw;
    pcl::getTranslationAndEulerAngles(a, ax, ay, az, ar, ap, ayaw);
    pcl::getTranslationAndEulerAngles(b, bx, by, bz, br, bp, byaw);
    const double dx = ax - bx;
    const double dy = ay - by;
    return std::sqrt(dx * dx + dy * dy);
}

mapOptimization::MotionContinuityInfo mapOptimization::evaluateMotionContinuity(const Eigen::Affine3f& candidateAffine, const char* tag)
{
    MotionContinuityInfo info;

    if (!hasLastOutputPose)
    {
        RCLCPP_WARN(this->get_logger(),
            "[MOTION_GATE][%s] no last published pose; treat as continuous.", tag);
        return info;
    }

    info.dt = (lastOutputTime > 0.0) ? std::max(1e-3, timeLaserInfoCur - lastOutputTime) : 0.0;
    info.ds = xyDistance(lastOutputAffine, candidateAffine);
    info.dyawDeg = std::abs(pcl::rad2deg(normalizeAngleRad(yawFromAffine(candidateAffine) - yawFromAffine(lastOutputAffine))));
    info.speed = info.ds / std::max(1e-3, info.dt);
    info.yawRate = info.dyawDeg / std::max(1e-3, info.dt);

    // Low-speed 10 Hz vehicle: single-frame ds can be very small, so use a floor.
    // This makes curvature a smooth geometric indicator, not a division-by-noise detector.
    const double curvatureDs = std::max(info.ds, 0.10);
    info.curvature = (info.dyawDeg * M_PI / 180.0) / curvatureDs;

    // Warm-up: before one previous published motion sample exists, do not reject.
    // The previous sample is computed from the previous published pose to the current last published pose.
    // This is NOT based on keyframes; it uses every published LiDAR odometry frame.
    if (speedHist.empty() || yawRateHist.empty() || curvatureHist.empty())
    {
        RCLCPP_WARN(this->get_logger(),
            "[MOTION_GATE][%s] warmup hist=%zu dt=%.3f ds=%.3f dyaw=%.2fdeg v=%.3f omega=%.2f kappa=%.3f; continuous.",
            tag, speedHist.size(), info.dt, info.ds, info.dyawDeg, info.speed, info.yawRate, info.curvature);
        return info;
    }

    info.ready = true;

    // Adjacent-frame continuity:
    //   acceleration        = |v_k     - v_{k-1}|     / dt_k
    //   angular acceleration= |omega_k - omega_{k-1}| / dt_k
    //   curvature jump      = |kappa_k - kappa_{k-1}|
    // Here k-1 is the immediately previous published odometry interval, not a keyframe interval.
    // This matches the physical meaning of a jump: a sudden change relative to the adjacent trajectory segment.
    info.speedRef = speedHist.back();
    info.yawRateRef = yawRateHist.back();
    info.curvatureRef = curvatureHist.back();

    info.speedJump = std::abs(info.speed - info.speedRef);
    info.yawRateJump = std::abs(info.yawRate - info.yawRateRef);
    info.curvatureJump = std::abs(info.curvature - info.curvatureRef);
    info.accel = info.speedJump / std::max(1e-3, info.dt);
    info.yawAccel = info.yawRateJump / std::max(1e-3, info.dt);

    // Physical continuity gates for a <0.5 m/s ground vehicle.
    // These are intentionally permissive: they should catch sudden LM false convergence, not normal slow turning.
    // Since dt may be affected by mappingProcessInterval/CPU, require both a derivative jump and an absolute jump.
    const bool speedBad = (info.speed > 1.50);
    const bool accelBad = (info.accel > 1.20 && info.speedJump > 0.30);
    const bool yawRateBad = (info.yawRate > 80.0);
    const bool yawAccelBad = (info.yawAccel > 180.0 && info.yawRateJump > 18.0);
    const bool curvatureBad = (info.curvatureJump > 1.80 && info.dyawDeg > 1.0 && info.ds > 0.03);

    info.continuous = !(speedBad || accelBad || yawRateBad || yawAccelBad || curvatureBad);

    RCLCPP_WARN(this->get_logger(),
        "[MOTION_GATE][%s] ok=%d dt=%.3f ds=%.3f dyaw=%.2fdeg "
        "v=%.3f(prev=%.3f,dv=%.3f,a=%.3f) "
        "omega=%.2f(prev=%.2f,d=%.2f,alpha=%.2f) "
        "kappa=%.3f(prev=%.3f,d=%.3f) bad(speed=%d accel=%d omega=%d alpha=%d kappa=%d)",
        tag, (int)info.continuous, info.dt, info.ds, info.dyawDeg,
        info.speed, info.speedRef, info.speedJump, info.accel,
        info.yawRate, info.yawRateRef, info.yawRateJump, info.yawAccel,
        info.curvature, info.curvatureRef, info.curvatureJump,
        (int)speedBad, (int)accelBad, (int)yawRateBad, (int)yawAccelBad, (int)curvatureBad);

    return info;
}

void mapOptimization::pushMotionHistory(double v, double omegaDeg, double kappa)
{
    speedHist.push_back(v); // 速度
    yawRateHist.push_back(omegaDeg); // 角速度
    curvatureHist.push_back(kappa); // 曲率
    while ((int)speedHist.size() > motionHistoryWindow) speedHist.pop_front();
    while ((int)yawRateHist.size() > motionHistoryWindow) yawRateHist.pop_front();
    while ((int)curvatureHist.size() > motionHistoryWindow) curvatureHist.pop_front();
}

void mapOptimization::updateOutputTrajectoryHistory(const Eigen::Affine3f& outputAffine)
{
    if (hasLastOutputPose)
    {
        const double dt = std::max(1e-3, timeLaserInfoCur - lastOutputTime);
        const double ds = xyDistance(lastOutputAffine, outputAffine);
        const double dyawDeg = std::abs(pcl::rad2deg(normalizeAngleRad(yawFromAffine(outputAffine) - yawFromAffine(lastOutputAffine))));
        const double v = ds / dt;
        const double omega = dyawDeg / dt;
        const double kappa = (dyawDeg * M_PI / 180.0) / std::max(ds, 0.10);
        pushMotionHistory(v, omega, kappa);
    }

    lastOutputAffine = outputAffine;
    lastOutputTime = timeLaserInfoCur;
    hasLastOutputPose = true;
}

bool mapOptimization::prepareCurrentRawCloudForRegistration()
{
    laserCloudRawLast->clear();
    if (!cloudInfo.cloud_deskewed)
        return false;
    *laserCloudRawLast = *cloudInfo.cloud_deskewed;
    if (laserCloudRawLast->empty())
        return false;

    // Current frame source: keep full valid resolution. Only remove NaN/Inf and >80m points.
    filterInvalidAndRangeInPlace(laserCloudRawLast, "raw_current_registration", true);
    return laserCloudRawLast->size() >= 300;
}

bool mapOptimization::buildRawLocalMapForRegistration()
{
    laserCloudRawFromMap->clear();
    laserCloudRawFromMapDS->clear();

    if (cloudKeyPoses3D->empty() || rawCloudKeyFrames.empty())
        return false;

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    PointType searchPoint;
    searchPoint.x = transformTobeMapped[3];
    searchPoint.y = transformTobeMapped[4];
    searchPoint.z = transformTobeMapped[5];

    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);
    kdtreeSurroundingKeyPoses->radiusSearch(searchPoint, surroundingKeyframeSearchRadius,
                                            pointSearchInd, pointSearchSqDis, 0);

    const int maxRawKeyframes = 20;
    if (pointSearchInd.empty())
    {
        const int cloudSize = cloudKeyPoses3D->size();
        const int historyNum = std::min(maxRawKeyframes, cloudSize);
        for (int i = cloudSize - historyNum; i < cloudSize; ++i)
            pointSearchInd.push_back(i);
    }
    else if ((int)pointSearchInd.size() > maxRawKeyframes)
    {
        pointSearchInd.resize(maxRawKeyframes);
    }

    Eigen::Vector3f priorCenter(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]);
    for (const int idx : pointSearchInd)
    {
        if (idx < 0 || idx >= (int)rawCloudKeyFrames.size())
            continue;
        if (rawCloudKeyFrames[idx]->empty())
            continue;

        pcl::PointCloud<PointType>::Ptr transformed = transformPointCloud(rawCloudKeyFrames[idx], &cloudKeyPoses6D->points[idx]);
        cropMapCloudAroundPriorInPlace(transformed, priorCenter);
        if (!transformed->empty())
            *laserCloudRawFromMap += *transformed;
    }

    if (laserCloudRawFromMap->size() < 800)
        return false;

    // Only the submap target is downsampled. The current source scan is not voxel-filtered.
    downSizeFilterRawICP.setInputCloud(laserCloudRawFromMap);
    downSizeFilterRawICP.filter(*laserCloudRawFromMapDS);
    return laserCloudRawFromMapDS->size() >= 500;
}

bool mapOptimization::rawCloudICPFallback(const Eigen::Affine3f& initialGuess,
                            Eigen::Affine3f& resultAffine,
                            double& fitnessScore)
{
    if (!prepareCurrentRawCloudForRegistration())
    {
        RCLCPP_WARN(this->get_logger(),
            "[ICP][FAIL] current raw deskewed cloud is too small after invalid/range filtering: raw=%zu",
            laserCloudRawLast->size());
        return false;
    }

    setTransformFromAffine(initialGuess);
    if (!buildRawLocalMapForRegistration())
    {
        RCLCPP_WARN(this->get_logger(),
            "[ICP][FAIL] raw local map is too small: rawMap=%zu rawMapDS=%zu rawKeyFrames=%zu",
            laserCloudRawFromMap->size(), laserCloudRawFromMapDS->size(), rawCloudKeyFrames.size());
        return false;
    }

    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setInputSource(laserCloudRawLast);        // full current scan, not downsampled
    icp.setInputTarget(laserCloudRawFromMapDS);   // downsampled local raw submap
    icp.setMaxCorrespondenceDistance(1.0);
    icp.setMaximumIterations(35);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-4);
    icp.setRANSACIterations(0);

    pcl::PointCloud<PointType>::Ptr aligned(new pcl::PointCloud<PointType>());
    icp.align(*aligned, initialGuess.matrix());

    fitnessScore = icp.hasConverged() ? icp.getFitnessScore(2.0) : std::numeric_limits<double>::infinity();
    if (!icp.hasConverged())
    {
        RCLCPP_WARN(this->get_logger(),
            "[ICP][FAIL] did not converge. source=%zu target=%zu",
            laserCloudRawLast->size(), laserCloudRawFromMapDS->size());
        return false;
    }

    resultAffine = Eigen::Affine3f(icp.getFinalTransformation());

    const double maxIcpFitness = 0.35;
    if (fitnessScore > maxIcpFitness)
    {
        RCLCPP_WARN(this->get_logger(),
            "[ICP][FAIL] fitness %.6f > %.6f. source=%zu target=%zu",
            fitnessScore, maxIcpFitness, laserCloudRawLast->size(), laserCloudRawFromMapDS->size());
        return false;
    }

    RCLCPP_WARN(this->get_logger(),
        "[ICP][CANDIDATE] fitness=%.6f source=%zu target=%zu",
        fitnessScore, laserCloudRawLast->size(), laserCloudRawFromMapDS->size());
    return true;
}

void mapOptimization::scan2MapOptimization()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    currentOdomCov = 0;
    isDegenerate = false;

    Eigen::Affine3f priorAffine = trans2Affine3f(transformTobeMapped);
    bool lmRan = false;
    bool lmMotionOk = false;
    bool lmConverged = false;
    int finalCoeffNum = 0;
    Eigen::Affine3f lmAffine = priorAffine;

    if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum &&
        laserCloudSurfLastDSNum   > surfFeatureMinValidNum)
    {
        lmRan = true;
        kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
        kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

        for (int iterCount = 0; iterCount < 30; iterCount++)
        {
            laserCloudOri->clear();
            coeffSel->clear();

            cornerOptimization();
            surfOptimization();
            combineOptimizationCoeffs();

            finalCoeffNum = (int)laserCloudOri->size();
            lmConverged = LMOptimization(iterCount);

            if (lmConverged)
                break;
        }

        transformUpdate();
        lmAffine = trans2Affine3f(transformTobeMapped);
        MotionContinuityInfo lmMotion = evaluateMotionContinuity(lmAffine, "LM");
        lmMotionOk = lmMotion.continuous;
    }
    else
    {
        RCLCPP_WARN(this->get_logger(),
            "[LM][FEATURE_WEAK] cornerDS=%d/%d surfDS=%d/%d. Try raw ICP fallback.",
            laserCloudCornerLastDSNum, edgeFeatureMinValidNum,
            laserCloudSurfLastDSNum, surfFeatureMinValidNum);
    }
    
    if (lmConverged && !isDegenerate && finalCoeffNum >= 80 && lmMotionOk)
    {
        currentOdomCov = 0;
        RCLCPP_WARN(this->get_logger(),
            "[LM][USE_HIGH] converged=%d degenerate=%d coeff=%d motionOK=%d cov=0 save=yes",
            (int)lmConverged, (int)isDegenerate, finalCoeffNum, (int)lmMotionOk);
        return;
    }
    // LM 优化失败
    RCLCPP_WARN(this->get_logger(),
        "[LM][SUSPECT] ran=%d converged=%d degenerate=%d coeff=%d motionOK=%d. Try raw ICP fallback.",
        (int)lmRan, (int)lmConverged, (int)isDegenerate, finalCoeffNum, (int)lmMotionOk);

    setTransformFromAffine(priorAffine);
    Eigen::Affine3f icpAffine = priorAffine;
    double icpFitness = std::numeric_limits<double>::infinity();
    if (rawCloudICPFallback(priorAffine, icpAffine, icpFitness))
    {
        MotionContinuityInfo icpMotion = evaluateMotionContinuity(icpAffine, "ICP");
        const bool icpHigh = icpMotion.continuous && icpFitness < 0.3;
        if (icpHigh){
            setTransformFromAffine(icpAffine);
            transformUpdate();
            currentOdomCov = 0;
            isDegenerate = false;

            RCLCPP_WARN(this->get_logger(),
                "[ICP][USE_HIGH] fitness=%.6f motionOK=%d cov=0 save=yes",
                icpFitness, (int)icpMotion.continuous);
            return;
        }
        RCLCPP_WARN(this->get_logger(),
            "[ICP][REJECT_WEAK] fitness=%.6f motionOK=%d. Continue fallback.",
            icpFitness, (int)icpMotion.continuous);
    }else{
        RCLCPP_WARN(this->get_logger(),
            "[ICP][FAILED] Continue fallback.");
    }
    // icp 也失败 ， 就使用LM 优化结果 ，但是要注意这个结果的位姿并不可靠
    if (lmRan)
    {
        setTransformFromAffine(lmAffine);
        transformUpdate();
        currentOdomCov = lmMotionOk ? 1 : 2;
        isDegenerate = true;
        RCLCPP_WARN(this->get_logger(),
            "[FALLBACK][PUBLISH_LM] ICP failed. lmMotionOK=%d cov=%d save=no",
            (int)lmMotionOk, currentOdomCov);
    }
    else
    {
        setTransformFromAffine(priorAffine);
        transformUpdate();
        currentOdomCov = 2;
        isDegenerate = true;
        RCLCPP_WARN(this->get_logger(),
            "[FALLBACK][PUBLISH_PRIOR] LM unavailable and ICP failed. cov=2 save=no");
    }
}

void mapOptimization::transformUpdate()
{
    if (cloudInfo.imu_available == true)
    {
        if (std::abs(cloudInfo.imu_pitch_init) < 1.4)
        {
            double imuWeight = imuRPYWeight;
            tf2::Quaternion imuQuaternion;
            tf2::Quaternion transformQuaternion;
            double rollMid, pitchMid, yawMid;

            // slerp roll
            transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
            imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
            tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
            transformTobeMapped[0] = rollMid;

            // slerp pitch
            transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
            imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
            tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
            transformTobeMapped[1] = pitchMid;
        }
    }

    transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
    transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
    transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

    incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
}

float mapOptimization::constraintTransformation(float value, float limit)
{
    if (value < -limit)
        value = -limit;
    if (value > limit)
        value = limit;

    return value;
}

bool mapOptimization::saveFrame()
{
    // Only trajectory-curvature-continuous high-confidence poses enter the keyframe map.
    // cov=1/2 are still published to IMUPreintegration with larger noise but never update the submap.
    if (currentOdomCov != 0)
        return false;
    if (cloudKeyPoses3D->points.empty())
        return true;

    if (lidarType == 1)
    {
        if (timeLaserInfoCur - cloudKeyPoses6D->back().time > 1.0)
            return true;
    }

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

void mapOptimization::addOdomFactor()
{
    if (cloudKeyPoses3D->points.empty())
    {
        noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
        gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
    }else{
        noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
        gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
        gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
        initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    }
}

void mapOptimization::addLoopFactor()
{
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

void mapOptimization::saveKeyFramesAndFactor()
{
    if (saveFrame() == false)
        return;

    // odom factor
    addOdomFactor();

    // loop factor
    addLoopFactor();

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

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    //save key poses
    PointType thisPose3D;
    PointTypePose thisPose6D;
    Pose3 latestEstimate;

    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
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
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

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
    pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
    pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

    // Save the full-resolution raw deskewed keyframe for future ICP fallback.
    // Do NOT voxel-filter here. The feature keyframes are already based on
    // laserCloudCornerLastDS / laserCloudSurfLastDS; for raw ICP fallback we
    // keep the original deskewed scan and downsample only when constructing
    // ICP source/target clouds.
    pcl::PointCloud<PointType>::Ptr thisRawKeyFrame(new pcl::PointCloud<PointType>());
    if (cloudInfo.cloud_deskewed) {
        *thisRawKeyFrame = *cloudInfo.cloud_deskewed;
    }
    filterInvalidAndRangeInPlace(thisRawKeyFrame, "raw_keyframe_save", true);

    // save key frame cloud
    cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);
    rawCloudKeyFrames.push_back(thisRawKeyFrame);
    createdNewKeyframe = true;

    // save path for visualization (这里可以不用了，因为我们在 publishOdometry 中每一帧都更新轨迹了)
    // updatePath(thisPose6D);
}

void mapOptimization::correctPoses()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (aLoopIsClosed == true)
    {
        // clear map cache
        laserCloudMapContainer.clear();
        // update key poses (但不清空轨迹)
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

        // 注意：我们绝不清空或修改 globalPath！
        // 因为你要的是原始的、每一帧的轨迹，不需要优化后的轨迹！
        
        aLoopIsClosed = false;
    }
}

void mapOptimization::updateOdometryState(){
        if (!transformIsFinite(transformTobeMapped))
            return;

        updateOutputTrajectoryHistory(trans2Affine3f(transformTobeMapped));
    }

