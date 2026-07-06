#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/lio/lio_sam/offline/utility_offline.hpp"

class DeskewFeatureExtractor : public ParamServer
{
public:
    explicit DeskewFeatureExtractor(const rclcpp::NodeOptions& options);
    bool Run(const PointCloudType::Ptr& inputCloud,
             const std::vector<sensor_msgs::msg::Imu>& imuWindow,
             double lidarBeginTime,
             double lidarEndTime,
             const std::string& frameId,
             bool keepDeskewedCloud,
             LioSamCloudInfo& cloudInfoOut);
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

    struct DeskewState
    {
        void resize(size_t size);

        void reset();

        bool imuAvailable = false;
        double timeScanCur = 0.0;
        std::vector<double> imuTime;
        std::vector<double> imuRotX;
        std::vector<double> imuRotY;
        std::vector<double> imuRotZ;
        int imuPointerCur = 0;
        bool firstPointFlag = true;
        Eigen::Affine3f transStartInverse = Eigen::Affine3f::Identity();
    };

    bool imuDeskewInfo(const std::vector<sensor_msgs::msg::Imu>& imuWindow);
    void findRotation(double pointTime,
                      const DeskewState& state,
                      float* rotXCur,
                      float* rotYCur,
                      float* rotZCur) const;
    PointType deskewPoint(const PointType* point,
                          double relTime,
                          DeskewState& state);
    PointType deskewPoint(const PointType* point, double relTime);
    void cacheRawKeyframeDeskewInfo();
    void projectPointCloud();
    void cloudExtraction();
    void calculateSmoothness();
    void markOccludedPoints();
    void extractFeatures();
    void packCloudInfo_packFeatureCloud(bool keepDeskewedCloud,
                                        LioSamCloudInfo& cloudInfoOut);

    PointCloudType::Ptr laserCloudIn;
    std::vector<float> rangeMat;
    PointCloudType::Ptr fullCloud;
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

    DeskewState currentDeskewState;
    DeskewState savedDeskewState;
    float imuRollInit = 0.0f;
    float imuPitchInit = 0.0f;
    float imuYawInit = 0.0f;
    double timeScanCur = 0.0;
    double timeScanEnd = 0.0;
    std::string currentFrameId;
    std::string savedFrameId;
};
