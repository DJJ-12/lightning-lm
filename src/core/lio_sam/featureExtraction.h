#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include "core/lio_sam/utility.hpp"

class FeatureExtractor : public ParamServer
{
public:
    explicit FeatureExtractor(const rclcpp::NodeOptions& options);
    ~FeatureExtractor();
    bool Run(
        const std::shared_ptr<LioSamCloudInfo>& cloudInfo);
    void resetParameters();

private:
    struct smoothness_t
    {
        float value = 0.0f;
        int ind = 0;
    };

    struct by_value
    {
        bool operator()(const smoothness_t& left, const smoothness_t& right) const;
    };

    void calculateSmoothness();
    void markOccludedPoints();
    void extractFeatures();

    PointCloudType::Ptr extractedCloud;

    std::vector<int> startRingIndex;
    std::vector<int> endRingIndex;
    std::vector<int> pointColInd;
    std::vector<float> pointRange;

    std::vector<smoothness_t> cloudSmoothness;
    std::vector<float> cloudCurvature;
    std::vector<int> cloudNeighborPicked;
    std::vector<int> cloudLabel;

    PointCloudType::Ptr cornerCloud;
    PointCloudType::Ptr surfaceCloud;
    PointCloudType::Ptr surfaceCloudScan;
    PointCloudType::Ptr surfaceCloudScanDS;
    pcl::VoxelGrid<PointType> downSizeFilter;
};
