#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <iostream>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>
#include <deque>
#include <mutex>
#include <string>
#include <limits>
#include <iomanip>
#include <stdexcept>
#include <array>
#include <thread>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <opencv2/opencv.hpp>

#include <pcl/kdtree/kdtree_flann.h>

namespace flann {
namespace serialization {

template <typename Key, typename Value, typename Hash, typename Eq, typename Alloc>
struct Serializer<std::unordered_map<Key, Value, Hash, Eq, Alloc>> {
    template <typename InputArchive>
    static inline void load(InputArchive& ar,
                            std::unordered_map<Key, Value, Hash, Eq, Alloc>& map)
    {
        size_t size = 0;
        ar & size;
        map.clear();

        for (size_t i = 0; i < size; ++i) {
            Key key;
            Value value;
            ar & key;
            ar & value;
            map.emplace(std::move(key), std::move(value));
        }
    }

    template <typename OutputArchive>
    static inline void save(OutputArchive& ar,
                            const std::unordered_map<Key, Value, Hash, Eq, Alloc>& map)
    {
        size_t size = map.size();
        ar & size;

        for (const auto& kv : map) {
            Key key = kv.first;
            Value value = kv.second;
            ar & key;
            ar & value;
        }
    }
};

}  // namespace serialization
}  // namespace flann

#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

#include "common/point_def.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/angles.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace std;

using PointType = lightning::PointType;
using PointCloudType = lightning::PointCloudType;
using CloudPtr = lightning::CloudPtr;

inline constexpr int queueLength = 2000;

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x) (float, y, y)
                                  (float, z, z) (float, intensity, intensity)
                                  (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                  (double, time, time))

typedef PointXYZIRPYT PointTypePose;

struct LioSamCloudInfo
{
    double timestamp = 0.0;
    std::string frame_id;

    bool imu_available = false;
    float imu_roll_init = 0.0f;
    float imu_pitch_init = 0.0f;
    float imu_yaw_init = 0.0f;

    bool odom_available = false;
    float initial_guess_x = 0.0f;
    float initial_guess_y = 0.0f;
    float initial_guess_z = 0.0f;
    float initial_guess_roll = 0.0f;
    float initial_guess_pitch = 0.0f;
    float initial_guess_yaw = 0.0f;

    std::vector<int> start_ring_index;
    std::vector<int> end_ring_index;
    std::vector<int> point_col_ind;
    std::vector<float> point_range;

    PointCloudType::Ptr cloud_deskewed{new PointCloudType()};
    PointCloudType::Ptr cloud_corner{new PointCloudType()};
    PointCloudType::Ptr cloud_surface{new PointCloudType()};
};

class ParamServer : public rclcpp::Node
{
public:
    // IMU initialization
    bool useImuHeadingInitialization;

    // Lidar Sensor Configuration
    int N_SCAN;
    int Horizon_SCAN;
    int lidarType;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;

    // IMU
    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity;
    float imuRPYWeight;
    bool useImuAccelRollPitchInitialization;
    vector<double> extRotV;
    vector<double> extRPYV;
    Eigen::Matrix3d extRot;
    Eigen::Matrix3d extRPY;
    Eigen::Quaterniond extQRPY;

    // LOAM
    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;

    float z_tollerance;
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;
    double mappingProcessInterval;
    bool isOnlineMapping = false;
    int maxOptimizationIterations = 30;
    int onlineMaxOptimizationIterations = 10;
    bool onlineLimitOptimizationPoints = true;
    int onlineMaxSurfOptimizationPoints = 4000;
    int onlineMaxCornerOptimizationPoints = 900;
    bool debugTiming = false;


    // Surrounding map
    float surroundingkeyframeAddingDistThreshold;
    float surroundingkeyframeAddingAngleThreshold;
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;

    // Loop closure
    bool  loopClosureEnableFlag;
    float loopClosureFrequency;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    ParamServer(std::string node_name, const rclcpp::NodeOptions & options) : Node(node_name, options)
    {
        declare_parameter("useImuHeadingInitialization", false);
        get_parameter("useImuHeadingInitialization", useImuHeadingInitialization);

        declare_parameter("N_SCAN", 64);
        get_parameter("N_SCAN", N_SCAN);
        declare_parameter("Horizon_SCAN", 512);
        get_parameter("Horizon_SCAN", Horizon_SCAN);
        declare_parameter("lidarType", 2);
        get_parameter("lidarType", lidarType);
        declare_parameter("downsampleRate", 1);
        get_parameter("downsampleRate", downsampleRate);
        declare_parameter("lidarMinRange", 5.5);
        get_parameter("lidarMinRange", lidarMinRange);
        declare_parameter("lidarMaxRange", 1000.0);
        get_parameter("lidarMaxRange", lidarMaxRange);

        declare_parameter("imuAccNoise", 9e-4);
        get_parameter("imuAccNoise", imuAccNoise);
        declare_parameter("imuGyrNoise", 1.6e-4);
        get_parameter("imuGyrNoise", imuGyrNoise);
        declare_parameter("imuAccBiasN", 5e-4);
        get_parameter("imuAccBiasN", imuAccBiasN);
        declare_parameter("imuGyrBiasN", 7e-5);
        get_parameter("imuGyrBiasN", imuGyrBiasN);
        declare_parameter("imuGravity", 9.80511);
        get_parameter("imuGravity", imuGravity);
        declare_parameter("imuRPYWeight", 0.01);
        get_parameter("imuRPYWeight", imuRPYWeight);
        declare_parameter("useImuAccelRollPitchInitialization", false);
        get_parameter("useImuAccelRollPitchInitialization", useImuAccelRollPitchInitialization);

        double ida[] = { 1.0,  0.0,  0.0,
                         0.0,  1.0,  0.0,
                         0.0,  0.0,  1.0};
        std::vector < double > id(ida, std::end(ida));
        declare_parameter("extrinsicRot", id);
        get_parameter("extrinsicRot", extRotV);
        declare_parameter("extrinsicRPY", id);
        get_parameter("extrinsicRPY", extRPYV);

        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extQRPY = Eigen::Quaterniond(extRPY);

        declare_parameter("edgeThreshold", 1.0);
        get_parameter("edgeThreshold", edgeThreshold);
        declare_parameter("surfThreshold", 0.1);
        get_parameter("surfThreshold", surfThreshold);
        declare_parameter("edgeFeatureMinValidNum", 10);
        get_parameter("edgeFeatureMinValidNum", edgeFeatureMinValidNum);
        declare_parameter("surfFeatureMinValidNum", 100);
        get_parameter("surfFeatureMinValidNum", surfFeatureMinValidNum);

        declare_parameter("odometrySurfLeafSize", 0.4);
        get_parameter("odometrySurfLeafSize", odometrySurfLeafSize);
        declare_parameter("mappingCornerLeafSize", 0.2);
        get_parameter("mappingCornerLeafSize", mappingCornerLeafSize);
        declare_parameter("mappingSurfLeafSize", 0.4);
        get_parameter("mappingSurfLeafSize", mappingSurfLeafSize);

        declare_parameter("z_tollerance", 1000.0);
        get_parameter("z_tollerance", z_tollerance);
        declare_parameter("rotation_tollerance", 1000.0);
        get_parameter("rotation_tollerance", rotation_tollerance);

        declare_parameter("numberOfCores", 4);
        get_parameter("numberOfCores", numberOfCores);
        declare_parameter("mappingProcessInterval", 0.1);
        get_parameter("mappingProcessInterval", mappingProcessInterval);
        declare_parameter("isOnlineMapping", false);
        get_parameter("isOnlineMapping", isOnlineMapping);
        declare_parameter("maxOptimizationIterations", 30);
        get_parameter("maxOptimizationIterations", maxOptimizationIterations);
        declare_parameter("onlineMaxOptimizationIterations", 10);
        get_parameter("onlineMaxOptimizationIterations", onlineMaxOptimizationIterations);
        declare_parameter("onlineLimitOptimizationPoints", true);
        get_parameter("onlineLimitOptimizationPoints", onlineLimitOptimizationPoints);
        declare_parameter("onlineMaxSurfOptimizationPoints", 4000);
        get_parameter("onlineMaxSurfOptimizationPoints", onlineMaxSurfOptimizationPoints);
        declare_parameter("onlineMaxCornerOptimizationPoints", 900);
        get_parameter("onlineMaxCornerOptimizationPoints", onlineMaxCornerOptimizationPoints);
        declare_parameter("debugTiming", false);
        get_parameter("debugTiming", debugTiming);

        declare_parameter("surroundingkeyframeAddingDistThreshold", 1.0);
        get_parameter("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold);
        declare_parameter("surroundingkeyframeAddingAngleThreshold", 0.2);
        get_parameter("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold);
        declare_parameter("surroundingKeyframeDensity", 2.0);
        get_parameter("surroundingKeyframeDensity", surroundingKeyframeDensity);
        declare_parameter("surroundingKeyframeSearchRadius", 50.0);
        get_parameter("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius);

        declare_parameter("loopClosureEnableFlag", true);
        get_parameter("loopClosureEnableFlag", loopClosureEnableFlag);
        declare_parameter("loopClosureFrequency", 1.0);
        get_parameter("loopClosureFrequency", loopClosureFrequency);
        declare_parameter("surroundingKeyframeSize", 50);
        get_parameter("surroundingKeyframeSize", surroundingKeyframeSize);
        declare_parameter("historyKeyframeSearchRadius", 15.0);
        get_parameter("historyKeyframeSearchRadius", historyKeyframeSearchRadius);
        declare_parameter("historyKeyframeSearchTimeDiff", 30.0);
        get_parameter("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff);
        declare_parameter("historyKeyframeSearchNum", 25);
        get_parameter("historyKeyframeSearchNum", historyKeyframeSearchNum);
        declare_parameter("historyKeyframeFitnessScore", 0.3);
        get_parameter("historyKeyframeFitnessScore", historyKeyframeFitnessScore);

        usleep(100);
    }
};

template<typename T>
double stamp2Sec(const T& stamp)
{
    return rclcpp::Time(stamp).seconds();
}


template<typename T>
void imuAngular2rosAngular(sensor_msgs::msg::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}


template<typename T>
void imuRPY2rosRPY(sensor_msgs::msg::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf2::Quaternion orientation;
    tf2::fromMsg(thisImuMsg->orientation, orientation);
    tf2::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}

template<typename T>
bool imuAccel2rosRollPitch(sensor_msgs::msg::Imu *thisImuMsg, T *rosRoll, T *rosPitch)
{
    double ax = thisImuMsg->linear_acceleration.x;
    double ay = thisImuMsg->linear_acceleration.y;
    double az = thisImuMsg->linear_acceleration.z;
    double accNorm = std::sqrt(ax * ax + ay * ay + az * az);

    if (!std::isfinite(accNorm) || accNorm < 1.0)
        return false;

    *rosRoll = std::atan2(ay, az);
    *rosPitch = std::atan2(-ax, std::sqrt(ay * ay + az * az));
    return true;
}


inline float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


inline float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

inline rmw_qos_profile_t qos_profile{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    1,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

inline auto qos = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile.history,
        qos_profile.depth
    ),
    qos_profile);

inline rmw_qos_profile_t qos_profile_imu{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    2000,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

inline auto qos_imu = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_imu.history,
        qos_profile_imu.depth
    ),
    qos_profile_imu);

inline rmw_qos_profile_t qos_profile_lidar{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    5,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

inline auto qos_lidar = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_lidar.history,
        qos_profile_lidar.depth
    ),
    qos_profile_lidar);

#endif
