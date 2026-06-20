#pragma once

#include "core/lio/lio_sam/offline/utility_offline.hpp"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloudScan;
    pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS;

    pcl::VoxelGrid<PointType> downSizeFilter;

    LioSamCloudInfo* cloudInfoPtr = nullptr;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    FeatureExtraction(const rclcpp::NodeOptions & options) :
        ParamServer("lio_sam_featureExtraction", options)
    {
        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloudScan.reset(new pcl::PointCloud<PointType>());
        surfaceCloudScanDS.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    bool Run(LioSamCloudInfo& cloudInfoInOut)
    {
        cloudInfoPtr = &cloudInfoInOut;
        extractedCloud = cloudInfoInOut.cloud_deskewed; // new cloud for extraction

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        packFeatureCloud(cloudInfoInOut);
        cloudInfoPtr = nullptr;
        return true;
    }

    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {
            float diffRange = cloudInfoPtr->point_range[i-5] + cloudInfoPtr->point_range[i-4]
                            + cloudInfoPtr->point_range[i-3] + cloudInfoPtr->point_range[i-2]
                            + cloudInfoPtr->point_range[i-1] - cloudInfoPtr->point_range[i] * 10
                            + cloudInfoPtr->point_range[i+1] + cloudInfoPtr->point_range[i+2]
                            + cloudInfoPtr->point_range[i+3] + cloudInfoPtr->point_range[i+4]
                            + cloudInfoPtr->point_range[i+5];

            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            float depth1 = cloudInfoPtr->point_range[i];
            float depth2 = cloudInfoPtr->point_range[i+1];
            int columnDiff = std::abs(int(cloudInfoPtr->point_col_ind[i+1] - cloudInfoPtr->point_col_ind[i]));
            if (columnDiff < 10){
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            float diff1 = std::abs(float(cloudInfoPtr->point_range[i-1] - cloudInfoPtr->point_range[i]));
            float diff2 = std::abs(float(cloudInfoPtr->point_range[i+1] - cloudInfoPtr->point_range[i]));

            if (diff1 > 0.02 * cloudInfoPtr->point_range[i] && diff2 > 0.02 * cloudInfoPtr->point_range[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();
        surfaceCloudScan->clear();
        surfaceCloudScanDS->clear();

        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();

            for (int j = 0; j < 6; j++)
            {

                int sp = (cloudInfoPtr->start_ring_index[i] * (6 - j) + cloudInfoPtr->end_ring_index[i] * j) / 6;
                int ep = (cloudInfoPtr->start_ring_index[i] * (5 - j) + cloudInfoPtr->end_ring_index[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= 20){
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfoPtr->point_col_ind[ind + l] - cloudInfoPtr->point_col_ind[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfoPtr->point_col_ind[ind + l] - cloudInfoPtr->point_col_ind[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {

                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(int(cloudInfoPtr->point_col_ind[ind + l] - cloudInfoPtr->point_col_ind[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(int(cloudInfoPtr->point_col_ind[ind + l] - cloudInfoPtr->point_col_ind[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    void freeCloudInfoMemory()
    {
    }

    void packFeatureCloud(LioSamCloudInfo& cloudInfoInOut)
    {
        // save newly extracted features
        cloudInfoInOut.cloud_corner = cornerCloud;
        cloudInfoInOut.cloud_surface = surfaceCloud;
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());
    }
};
