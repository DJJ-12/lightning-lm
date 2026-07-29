#include "core/lio_sam/featureExtraction.h"

#include <algorithm>
#include <cmath>

#include <pcl/console/print.h>

bool FeatureExtractor::by_value::operator()(
    const FeatureExtractor::smoothness_t& left,
    const FeatureExtractor::smoothness_t& right) const {
    return left.value < right.value;
}

FeatureExtractor::FeatureExtractor(const rclcpp::NodeOptions& options)
    : ParamServer("lio_sam_featureExtractor", options) {
    const size_t cloudSize = static_cast<size_t>(N_SCAN) * Horizon_SCAN;
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

    downSizeFilter.setLeafSize(
        odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);
    resetParameters();
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
}

FeatureExtractor::~FeatureExtractor() {
    RCLCPP_INFO(get_logger(),
        "[FeatureExtractor] destroy this=%p",
        static_cast<void*>(this));
}

bool FeatureExtractor::Run(
    const std::shared_ptr<LioSamCloudInfo>& cloudInfo) {
    if (!cloudInfo ||
        !cloudInfo->cloud_deskewed ||
        cloudInfo->cloud_deskewed->empty()) {
        RCLCPP_WARN(get_logger(), "[FeatureExtractor] empty deskewed cloud");
        return false;
    }

    const size_t cloudSize = cloudInfo->cloud_deskewed->size();
    if (cloudSize < 11) {
        RCLCPP_WARN(get_logger(),
            "[FeatureExtractor] too few projected points: %zu",
            cloudSize);
        return false;
    }
    if (cloudInfo->start_ring_index.size() <
            static_cast<size_t>(N_SCAN) ||
        cloudInfo->end_ring_index.size() <
            static_cast<size_t>(N_SCAN) ||
        cloudInfo->point_col_ind.size() < cloudSize ||
        cloudInfo->point_range.size() < cloudSize) {
        RCLCPP_WARN(get_logger(),
            "[FeatureExtractor] incomplete projected cloud info: cloud=%zu rings=%zu/%zu col=%zu range=%zu",
            cloudSize,
            cloudInfo->start_ring_index.size(),
            cloudInfo->end_ring_index.size(),
            cloudInfo->point_col_ind.size(),
            cloudInfo->point_range.size());
        return false;
    }

    extractedCloud.reset();
    resetParameters();
    extractedCloud = cloudInfo->cloud_deskewed;
    startRingIndex = cloudInfo->start_ring_index;
    endRingIndex = cloudInfo->end_ring_index;
    pointColInd = cloudInfo->point_col_ind;
    pointRange = cloudInfo->point_range;

    if (cloudSmoothness.size() < cloudSize) {
        cloudSmoothness.resize(cloudSize);
        cloudCurvature.resize(cloudSize);
        cloudNeighborPicked.resize(cloudSize);
        cloudLabel.resize(cloudSize);
    }

    calculateSmoothness();
    markOccludedPoints();
    extractFeatures();

    cloudInfo->cloud_corner.reset(
        new PointCloudType(*cornerCloud));
    cloudInfo->cloud_surface.reset(
        new PointCloudType(*surfaceCloud));
    cloudInfo->cloud_corner->header =
        cloudInfo->cloud_deskewed->header;
    cloudInfo->cloud_surface->header =
        cloudInfo->cloud_deskewed->header;
    return true;
}

void FeatureExtractor::resetParameters() {
    if (extractedCloud) {
        extractedCloud->clear();
    }
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
    for (size_t i = 0; i < cloudSmoothness.size(); ++i) {
        cloudSmoothness[i].value = 0.0f;
        cloudSmoothness[i].ind = static_cast<int>(i);
    }
}

void FeatureExtractor::calculateSmoothness() {
    const int cloudSize = static_cast<int>(extractedCloud->points.size());
    for (int i = 5; i < cloudSize - 5; i++) {
        const float diffRange = pointRange[i-5] + pointRange[i-4]
                              + pointRange[i-3] + pointRange[i-2]
                              + pointRange[i-1] - pointRange[i] * 10
                              + pointRange[i+1] + pointRange[i+2]
                              + pointRange[i+3] + pointRange[i+4]
                              + pointRange[i+5];

        cloudCurvature[i] = diffRange * diffRange;
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;
        cloudSmoothness[i].value = cloudCurvature[i];
        cloudSmoothness[i].ind = i;
    }
}

void FeatureExtractor::markOccludedPoints() {
    const int cloudSize = static_cast<int>(extractedCloud->points.size());
    for (int i = 5; i < cloudSize - 6; ++i) {
        const float depth1 = pointRange[i];
        const float depth2 = pointRange[i+1];
        const int columnDiff = std::abs(pointColInd[i+1] - pointColInd[i]);
        if (columnDiff < 10) {
            if (depth1 - depth2 > 0.3f) {
                cloudNeighborPicked[i - 5] = 1;
                cloudNeighborPicked[i - 4] = 1;
                cloudNeighborPicked[i - 3] = 1;
                cloudNeighborPicked[i - 2] = 1;
                cloudNeighborPicked[i - 1] = 1;
                cloudNeighborPicked[i] = 1;
            } else if (depth2 - depth1 > 0.3f) {
                cloudNeighborPicked[i + 1] = 1;
                cloudNeighborPicked[i + 2] = 1;
                cloudNeighborPicked[i + 3] = 1;
                cloudNeighborPicked[i + 4] = 1;
                cloudNeighborPicked[i + 5] = 1;
                cloudNeighborPicked[i + 6] = 1;
            }
        }

        const float diff1 = std::abs(pointRange[i-1] - pointRange[i]);
        const float diff2 = std::abs(pointRange[i+1] - pointRange[i]);
        if (diff1 > 0.02f * pointRange[i] && diff2 > 0.02f * pointRange[i]) {
            cloudNeighborPicked[i] = 1;
        }
    }
}

void FeatureExtractor::extractFeatures() {
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
            const int sp = (startRingIndex[i] * (6 - j) + endRingIndex[i] * j) / 6;
            const int ep = (startRingIndex[i] * (5 - j) + endRingIndex[i] * (j + 1)) / 6 - 1;
            if (sp >= ep)
                continue;
            if (sp < 0 || ep < 0 || sp >= validPointCount || ep >= validPointCount)
                continue;

            std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, by_value());

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
                        const int columnDiff = std::abs(pointColInd[currentIndex] - pointColInd[previousIndex]);
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
                        const int columnDiff =std::abs(pointColInd[currentIndex] - pointColInd[nextIndex]);
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
                        const int columnDiff = std::abs(pointColInd[currentIndex] - pointColInd[previousIndex]);
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
                        const int columnDiff = std::abs(pointColInd[currentIndex] - pointColInd[nextIndex]);
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
