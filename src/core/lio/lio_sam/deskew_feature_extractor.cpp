#include "core/lio/lio_sam/deskew_feature_extractor.h"

bool DeskewFeatureExtractor::by_value::operator()(
    const DeskewFeatureExtractor::smoothness_t& left,
    const DeskewFeatureExtractor::smoothness_t& right) const {
    return left.value < right.value;
}

void DeskewFeatureExtractor::DeskewState::resize(size_t size) {
    imuTime.resize(size);
    imuRotX.resize(size);
    imuRotY.resize(size);
    imuRotZ.resize(size);
}

void DeskewFeatureExtractor::DeskewState::reset() {
    imuAvailable = false;
    timeScanCur = 0.0;
    imuPointerCur = 0;
    firstPointFlag = true;
    transStartInverse = Eigen::Affine3f::Identity();
    std::fill(imuTime.begin(), imuTime.end(), 0.0);
    std::fill(imuRotX.begin(), imuRotX.end(), 0.0);
    std::fill(imuRotY.begin(), imuRotY.end(), 0.0);
    std::fill(imuRotZ.begin(), imuRotZ.end(), 0.0);
}

DeskewFeatureExtractor::DeskewFeatureExtractor(const rclcpp::NodeOptions& options)
        : ParamServer("lio_sam_deskewFeatureExtractor", options){
        const size_t cloudSize = static_cast<size_t>(N_SCAN) * Horizon_SCAN;
        rangeMat.resize(cloudSize);
        fullCloud.reset(new PointCloudType());
        fullCloud->points.resize(cloudSize);
        extractedCloud.reset(new PointCloudType());
        cornerCloud.reset(new PointCloudType());
        surfaceCloud.reset(new PointCloudType());
        surfaceCloudScan.reset(new PointCloudType());
        surfaceCloudScanDS.reset(new PointCloudType());
        extractedCloud->reserve(cloudSize);
        cornerCloud->reserve(cloudSize);
        surfaceCloud->reserve(cloudSize);
        surfaceCloudScan->reserve(cloudSize);
        surfaceCloudScanDS->reserve(cloudSize);

        startRingIndex.resize(N_SCAN);
        endRingIndex.resize(N_SCAN);
        pointColInd.resize(cloudSize);
        pointRange.resize(cloudSize);
        cloudSmoothness.resize(cloudSize);
        cloudCurvature.resize(cloudSize);
        cloudNeighborPicked.resize(cloudSize);
        cloudLabel.resize(cloudSize);

        currentDeskewState.resize(queueLength);
        savedDeskewState.resize(queueLength);
        downSizeFilter.setLeafSize(
            odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);
        resetParameters();
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

bool DeskewFeatureExtractor::Run(const PointCloudType::Ptr& inputCloud,
             const std::vector<sensor_msgs::msg::Imu>& imuWindow,
             double lidarBeginTime,
             double lidarEndTime,
             const std::string& frameId,
             bool keepDeskewedCloud,
             LioSamCloudInfo& cloudInfoOut){
        resetParameters();

        laserCloudIn = inputCloud;
        timeScanCur = lidarBeginTime;
        timeScanEnd = lidarEndTime;
        currentFrameId = frameId;

        if (!imuDeskewInfo(imuWindow))
            return false;
        cacheRawKeyframeDeskewInfo();

        projectPointCloud();

        cloudExtraction();

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        packCloudInfo_packFeatureCloud(keepDeskewedCloud, cloudInfoOut);
        return true;
    }

void DeskewFeatureExtractor::resetParameters(){
        std::fill(rangeMat.begin(), rangeMat.end(), FLT_MAX);
        std::fill(fullCloud->points.begin(), fullCloud->points.end(), PointType());
        extractedCloud->clear();
        cornerCloud->clear();
        surfaceCloud->clear();
        surfaceCloudScan->clear();
        surfaceCloudScanDS->clear();
        std::fill(startRingIndex.begin(), startRingIndex.end(), 0);
        std::fill(endRingIndex.begin(), endRingIndex.end(), 0);
        std::fill(pointColInd.begin(), pointColInd.end(), 0);
        std::fill(pointRange.begin(), pointRange.end(), 0.0f);
        std::fill(cloudNeighborPicked.begin(), cloudNeighborPicked.end(), 0);
        std::fill(cloudLabel.begin(), cloudLabel.end(), 0);
        std::fill(cloudCurvature.begin(), cloudCurvature.end(), 0.0f);
        for (size_t i = 0; i < cloudSmoothness.size(); ++i)
        {
            cloudSmoothness[i].value = 0.0f;
            cloudSmoothness[i].ind = static_cast<int>(i);
        }
        currentDeskewState.reset();
        imuRollInit = 0.0f;
        imuPitchInit = 0.0f;
        imuYawInit = 0.0f;
        laserCloudIn.reset();
        timeScanCur = 0.0;
        timeScanEnd = 0.0;
        currentFrameId.clear();
    }

bool DeskewFeatureExtractor::imuDeskewInfo(const std::vector<sensor_msgs::msg::Imu>& imuWindow){
        if (imuWindow.empty() ||
            stamp2Sec(imuWindow.front().header.stamp) > timeScanCur ||
            stamp2Sec(imuWindow.back().header.stamp) < timeScanEnd)
        {
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for IMU data ...");
            return false;
        }

        currentDeskewState.timeScanCur = timeScanCur;
        size_t firstImu = 0;
        while (firstImu < imuWindow.size() &&
               stamp2Sec(imuWindow[firstImu].header.stamp) < timeScanCur - 0.05)
        {
            ++firstImu;
        }
        if (firstImu >= imuWindow.size())
            return true;

        for (size_t i = firstImu; i < imuWindow.size(); ++i)
        {
            sensor_msgs::msg::Imu thisImuMsg = imuConverter(imuWindow[i]);
            const double currentImuTime = stamp2Sec(thisImuMsg.header.stamp);

            if (currentImuTime <= timeScanCur)
            {
                imuRPY2rosRPY(
                    &thisImuMsg, &imuRollInit, &imuPitchInit, &imuYawInit);
                if (useImuAccelRollPitchInitialization)
                {
                    double accRoll = 0.0;
                    double accPitch = 0.0;
                    if (imuAccel2rosRollPitch(&thisImuMsg, &accRoll, &accPitch))
                    {
                        imuRollInit = accRoll;
                        imuPitchInit = accPitch;
                    }
                }
            }
            if (currentImuTime > timeScanEnd + 0.01)
                break;

            if (currentDeskewState.imuPointerCur >=
                static_cast<int>(currentDeskewState.imuTime.size()))
            {
                break;
            }

            const int pointer = currentDeskewState.imuPointerCur;
            if (pointer == 0)
            {
                currentDeskewState.imuRotX[0] = 0.0;
                currentDeskewState.imuRotY[0] = 0.0;
                currentDeskewState.imuRotZ[0] = 0.0;
                currentDeskewState.imuTime[0] = currentImuTime;
                ++currentDeskewState.imuPointerCur;
                continue;
            }

            double angularX, angularY, angularZ;
            imuAngular2rosAngular(&thisImuMsg, &angularX, &angularY, &angularZ);
            const double timeDiff =
                currentImuTime - currentDeskewState.imuTime[pointer - 1];
            currentDeskewState.imuRotX[pointer] =
                currentDeskewState.imuRotX[pointer - 1] + angularX * timeDiff;
            currentDeskewState.imuRotY[pointer] =
                currentDeskewState.imuRotY[pointer - 1] + angularY * timeDiff;
            currentDeskewState.imuRotZ[pointer] =
                currentDeskewState.imuRotZ[pointer - 1] + angularZ * timeDiff;
            currentDeskewState.imuTime[pointer] = currentImuTime;
            ++currentDeskewState.imuPointerCur;
        }

        --currentDeskewState.imuPointerCur;
        if (currentDeskewState.imuPointerCur <= 0)
            return true;
        currentDeskewState.imuAvailable = true;
        return true;
    }

void DeskewFeatureExtractor::findRotation(double pointTime,
                      const DeskewFeatureExtractor::DeskewState& state,
                      float* rotXCur,
                      float* rotYCur,
                      float* rotZCur) const{
        *rotXCur = 0.0f;
        *rotYCur = 0.0f;
        *rotZCur = 0.0f;
        if (!state.imuAvailable)
            return;

        int imuPointerFront = 0;
        while (imuPointerFront < state.imuPointerCur)
        {
            if (pointTime < state.imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        if (pointTime > state.imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = state.imuRotX[imuPointerFront];
            *rotYCur = state.imuRotY[imuPointerFront];
            *rotZCur = state.imuRotZ[imuPointerFront];
        }
        else
        {
            const int imuPointerBack = imuPointerFront - 1;
            const double ratioFront =
                (pointTime - state.imuTime[imuPointerBack]) /
                (state.imuTime[imuPointerFront] - state.imuTime[imuPointerBack]);
            const double ratioBack =
                (state.imuTime[imuPointerFront] - pointTime) /
                (state.imuTime[imuPointerFront] - state.imuTime[imuPointerBack]);
            *rotXCur = state.imuRotX[imuPointerFront] * ratioFront +
                       state.imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = state.imuRotY[imuPointerFront] * ratioFront +
                       state.imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = state.imuRotZ[imuPointerFront] * ratioFront +
                       state.imuRotZ[imuPointerBack] * ratioBack;
        }
    }

PointType DeskewFeatureExtractor::deskewPoint(const PointType* point,
                          double relTime,
                          DeskewFeatureExtractor::DeskewState& state){
        if (!state.imuAvailable)
            return *point;

        const double pointTime = state.timeScanCur + relTime;
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, state, &rotXCur, &rotYCur, &rotZCur);

        if (state.firstPointFlag)
        {
            state.transStartInverse =
                pcl::getTransformation(0.0f, 0.0f, 0.0f, rotXCur, rotYCur, rotZCur)
                    .inverse();
            state.firstPointFlag = false;
        }

        const Eigen::Affine3f transFinal =
            pcl::getTransformation(0.0f, 0.0f, 0.0f, rotXCur, rotYCur, rotZCur);
        const Eigen::Affine3f transBt = state.transStartInverse * transFinal;
        PointType newPoint;
        newPoint.x = transBt(0, 0) * point->x + transBt(0, 1) * point->y +
                     transBt(0, 2) * point->z + transBt(0, 3);
        newPoint.y = transBt(1, 0) * point->x + transBt(1, 1) * point->y +
                     transBt(1, 2) * point->z + transBt(1, 3);
        newPoint.z = transBt(2, 0) * point->x + transBt(2, 1) * point->y +
                     transBt(2, 2) * point->z + transBt(2, 3);
        newPoint.intensity = point->intensity;
        newPoint.ring = point->ring;
        newPoint.time = point->time;
        return newPoint;
    }

PointType DeskewFeatureExtractor::deskewPoint(const PointType* point, double relTime){
        return deskewPoint(point, relTime, currentDeskewState);
    }

void DeskewFeatureExtractor::cacheRawKeyframeDeskewInfo(){
        savedDeskewState.imuAvailable = currentDeskewState.imuAvailable;
        savedDeskewState.timeScanCur = currentDeskewState.timeScanCur;
        savedDeskewState.imuPointerCur = currentDeskewState.imuPointerCur;
        savedDeskewState.firstPointFlag = true;
        savedDeskewState.transStartInverse = Eigen::Affine3f::Identity();
        const int count = std::max(0, currentDeskewState.imuPointerCur + 1);
        std::copy_n(currentDeskewState.imuTime.begin(), count, savedDeskewState.imuTime.begin());
        std::copy_n(currentDeskewState.imuRotX.begin(), count, savedDeskewState.imuRotX.begin());
        std::copy_n(currentDeskewState.imuRotY.begin(), count, savedDeskewState.imuRotY.begin());
        std::copy_n(currentDeskewState.imuRotZ.begin(), count, savedDeskewState.imuRotZ.begin());
        savedFrameId = currentFrameId;
    }

void DeskewFeatureExtractor::projectPointCloud(){
        const float angResX = 360.0f / static_cast<float>(Horizon_SCAN);
        for (const auto& point : laserCloudIn->points)
        {
            const int rowIdn = static_cast<int>(point.ring);
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            if (rowIdn % downsampleRate != 0)
                continue;

            const float horizonAngle =
                std::atan2(point.x, point.y) * 180.0f / M_PI;
            int columnIdn =
                -static_cast<int>(std::round((horizonAngle - 90.0f) / angResX)) +
                Horizon_SCAN / 2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            const float range = pointDistance(point);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            const int cloudIndex = rowIdn * Horizon_SCAN + columnIdn;
            if (rangeMat[cloudIndex] != FLT_MAX)
                continue;

            rangeMat[cloudIndex] = range;
            fullCloud->points[cloudIndex] = deskewPoint(&point, point.time);
        }
    }

void DeskewFeatureExtractor::cloudExtraction(){
        int count = 0;
        for (int i = 0; i < N_SCAN; ++i)
        {
            startRingIndex[i] = count - 1 + 5;
            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                const int cloudIndex = i * Horizon_SCAN + j;
                if (rangeMat[cloudIndex] == FLT_MAX)
                    continue;
                pointColInd[count] = j;
                pointRange[count] = rangeMat[cloudIndex];
                extractedCloud->push_back(fullCloud->points[cloudIndex]);
                ++count;
            }
            endRingIndex[i] = count - 1 - 5;
        }
    }

void DeskewFeatureExtractor::calculateSmoothness(){
        const int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; ++i)
        {
            const float diffRange = pointRange[i - 5] + pointRange[i - 4] +
                                    pointRange[i - 3] + pointRange[i - 2] +
                                    pointRange[i - 1] - pointRange[i] * 10 +
                                    pointRange[i + 1] + pointRange[i + 2] +
                                    pointRange[i + 3] + pointRange[i + 4] +
                                    pointRange[i + 5];
            cloudCurvature[i] = diffRange * diffRange;
            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

void DeskewFeatureExtractor::markOccludedPoints(){
        const int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            const float depth1 = pointRange[i];
            const float depth2 = pointRange[i + 1];
            const int columnDiff = std::abs(pointColInd[i + 1] - pointColInd[i]);
            if (columnDiff < 10)
            {
                if (depth1 - depth2 > 0.3)
                {
                    for (int offset = -5; offset <= 0; ++offset)
                        cloudNeighborPicked[i + offset] = 1;
                }
                else if (depth2 - depth1 > 0.3)
                {
                    for (int offset = 1; offset <= 6; ++offset)
                        cloudNeighborPicked[i + offset] = 1;
                }
            }

            const float diff1 = std::abs(pointRange[i - 1] - pointRange[i]);
            const float diff2 = std::abs(pointRange[i + 1] - pointRange[i]);
            if (diff1 > 0.02 * pointRange[i] && diff2 > 0.02 * pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

void DeskewFeatureExtractor::extractFeatures(){
        cornerCloud->clear();
        surfaceCloud->clear();
        surfaceCloudScan->clear();
        surfaceCloudScanDS->clear();

        const int cloudSize = static_cast<int>(extractedCloud->points.size());
        const int validPointCount = std::min(
            {cloudSize,
             static_cast<int>(pointColInd.size()),
             static_cast<int>(cloudCurvature.size()),
             static_cast<int>(cloudNeighborPicked.size()),
             static_cast<int>(cloudLabel.size()),
             static_cast<int>(cloudSmoothness.size())});
        if (validPointCount <= 0)
            return;

        for (int i = 0; i < N_SCAN; ++i)
        {
            surfaceCloudScan->clear();
            for (int j = 0; j < 6; ++j)
            {
                const int sp =
                    (startRingIndex[i] * (6 - j) + endRingIndex[i] * j) / 6;
                const int ep =
                    (startRingIndex[i] * (5 - j) + endRingIndex[i] * (j + 1)) / 6 - 1;
                if (sp >= ep)
                    continue;
                if (sp < 0 || ep < 0 || sp >= validPointCount || ep >= validPointCount)
                    continue;

                std::sort(
                    cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, by_value());

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; --k)
                {
                    const int ind = cloudSmoothness[k].ind;
                    if (ind < 0 || ind >= validPointCount)
                        continue;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        ++largestPickedNum;
                        if (largestPickedNum <= 20)
                        {
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        }
                        else
                        {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; ++l)
                        {
                            const int currentIndex = ind + l;
                            const int previousIndex = currentIndex - 1;
                            if (currentIndex < 0 || currentIndex >= validPointCount ||
                                previousIndex < 0 || previousIndex >= validPointCount)
                                break;
                            const int columnDiff =
                                std::abs(pointColInd[currentIndex] - pointColInd[previousIndex]);
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[currentIndex] = 1;
                        }
                        for (int l = -1; l >= -5; --l)
                        {
                            const int currentIndex = ind + l;
                            const int nextIndex = currentIndex + 1;
                            if (currentIndex < 0 || currentIndex >= validPointCount ||
                                nextIndex < 0 || nextIndex >= validPointCount)
                                break;
                            const int columnDiff =
                                std::abs(pointColInd[currentIndex] - pointColInd[nextIndex]);
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[currentIndex] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; ++k)
                {
                    const int ind = cloudSmoothness[k].ind;
                    if (ind < 0 || ind >= validPointCount)
                        continue;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {
                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; ++l)
                        {
                            const int currentIndex = ind + l;
                            const int previousIndex = currentIndex - 1;
                            if (currentIndex < 0 || currentIndex >= validPointCount ||
                                previousIndex < 0 || previousIndex >= validPointCount)
                                break;
                            const int columnDiff =
                                std::abs(pointColInd[currentIndex] - pointColInd[previousIndex]);
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[currentIndex] = 1;
                        }
                        for (int l = -1; l >= -5; --l)
                        {
                            const int currentIndex = ind + l;
                            const int nextIndex = currentIndex + 1;
                            if (currentIndex < 0 || currentIndex >= validPointCount ||
                                nextIndex < 0 || nextIndex >= validPointCount)
                                break;
                            const int columnDiff =
                                std::abs(pointColInd[currentIndex] - pointColInd[nextIndex]);
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[currentIndex] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; ++k)
                {
                    if (cloudLabel[k] <= 0)
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                }
            }

            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);
            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

void DeskewFeatureExtractor::packCloudInfo_packFeatureCloud(bool keepDeskewedCloud,
                                        LioSamCloudInfo& cloudInfoOut){
        extractedCloud->header.stamp =
            static_cast<std::uint64_t>(std::llround(timeScanCur * 1e9));
        extractedCloud->header.frame_id = currentFrameId;
        extractedCloud->height = 1;
        extractedCloud->width = extractedCloud->size();
        extractedCloud->is_dense = true;

        cornerCloud->header = extractedCloud->header;
        cornerCloud->height = 1;
        cornerCloud->width = cornerCloud->size();
        cornerCloud->is_dense = true;

        surfaceCloud->header = extractedCloud->header;
        surfaceCloud->height = 1;
        surfaceCloud->width = surfaceCloud->size();
        surfaceCloud->is_dense = true;

        cloudInfoOut.timestamp = timeScanCur;
        cloudInfoOut.frame_id = currentFrameId;
        cloudInfoOut.imu_available = currentDeskewState.imuAvailable;
        cloudInfoOut.imu_roll_init = imuRollInit;
        cloudInfoOut.imu_pitch_init = imuPitchInit;
        cloudInfoOut.imu_yaw_init = imuYawInit;
        cloudInfoOut.start_ring_index = startRingIndex;
        cloudInfoOut.end_ring_index = endRingIndex;

        const size_t cloudSize = extractedCloud->size();
        cloudInfoOut.point_col_ind.resize(cloudSize);
        cloudInfoOut.point_range.resize(cloudSize);
        std::copy_n(pointColInd.begin(), cloudSize, cloudInfoOut.point_col_ind.begin());
        std::copy_n(pointRange.begin(), cloudSize, cloudInfoOut.point_range.begin());
        cloudInfoOut.cloud_corner = cornerCloud;
        cloudInfoOut.cloud_surface = surfaceCloud;
        if (keepDeskewedCloud)
            cloudInfoOut.cloud_deskewed = extractedCloud;
        else
            cloudInfoOut.cloud_deskewed.reset();

        const size_t cloudCapacity = static_cast<size_t>(N_SCAN) * Horizon_SCAN;
        extractedCloud.reset(new PointCloudType());
        cornerCloud.reset(new PointCloudType());
        surfaceCloud.reset(new PointCloudType());
        extractedCloud->reserve(cloudCapacity);
        cornerCloud->reserve(cloudCapacity);
        surfaceCloud->reserve(cloudCapacity);
    }
