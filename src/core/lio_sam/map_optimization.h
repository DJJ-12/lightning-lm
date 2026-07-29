#pragma once

#include "core/lio_sam/utility.hpp"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <limits>
#include <thread>
#include <unordered_set>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    LioSamCloudInfo cloudInfo;
    bool createdNewKeyframe = false;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> rawCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudRawLast;

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<std::uint8_t> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<std::uint8_t> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    std::vector<int> surroundingKeyFrameIndices;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudRawFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudRawFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterRawICP;

    double timeLaserInfoCur;

    static constexpr int kMaxConsecutiveMappingFailures = 15;

    float transformTobeMapped[6];
    int currentOdomCov = 0;

    std::mutex mtx;
    
    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

    struct MotionContinuityInfo
    {
        bool ready = false;
        bool continuous = true;
        double dt = 0.0;
        double ds = 0.0;
        double dyawDeg = 0.0;
        double speed = 0.0;
        double yawRate = 0.0;
        double accel = 0.0;
        double yawAccel = 0.0;
        double curvature = 0.0;
        double speedRef = 0.0;
        double yawRateRef = 0.0;
        double curvatureRef = 0.0;
        double speedJump = 0.0;
        double yawRateJump = 0.0;
        double curvatureJump = 0.0;
    };

    struct HybridTrustedPoseState {
        bool has_last = false;
        bool has_prev = false;
        bool has_last_imu = false;
        double last_time = -1.0;
        double prev_time = -1.0;
        std::array<float, 6> last{};
        std::array<float, 6> prev{};
        std::array<float, 6> last_imu{};
    };

    enum class MappingTrackingState
    {
        TRACKING,
        LOST
    };

    bool mappingPoseReliable = true;
    int lidarCorrectionFlag = 0; // 0: reliable pose, 2: failed pose, IMU side skips LiDAR pose factor
    std::string mappingPoseSource = "INIT";
    MappingTrackingState mappingTrackingState = MappingTrackingState::TRACKING;
    int mappingFailureCount = 0;
    double mappingFirstFailureTime = -1.0;
    float frameInitialGuessTransform[6] = {0, 0, 0, 0, 0, 0};
    int lastLMCloudSelNum = 0;
    int lastLMIterationCount = 0;
    bool lastLMRan = false;
    bool lastLMConverged = false;
    bool hasLastOutputPose = false;
    bool lastRunExecuted = false;

    // 这些状态必须属于当前 mapOptimization 对象。
    // 不能使用函数内 static，否则后续任务会继承上一个任务的时间戳和 IMU 初值。
    double timeLastProcessing_ = -1.0;
    Eigen::Affine3f lastImuTransformation_ = Eigen::Affine3f::Identity();
    bool lastImuPreTransAvailable_ = false;
    Eigen::Affine3f lastImuPreTransformation_ = Eigen::Affine3f::Identity();

    std::uint64_t diagnosticRunCalls = 0;
    std::uint64_t diagnosticExecutedCalls = 0;
    std::uint64_t diagnosticSkippedCalls = 0;
    std::uint64_t diagnosticInitialGuessCalls = 0;
    Eigen::Affine3f lastOutputAffine = Eigen::Affine3f::Identity();
    double lastOutputTime = -1.0;
    HybridTrustedPoseState hybrid_trusted_pose_;
    std::deque<double> speedHist;
    std::deque<double> yawRateHist;
    std::deque<double> curvatureHist;
    const int motionHistoryWindow = 8;
    const float maxRawIcpSourceRange = 80.0f;
    const float maxRawIcpTargetRadius = 80.0f;
    const float debugRollPitchWarnDeg = 45.0f;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;
    double lastLocalMapKdtreeMs = 0.0;
    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    std::thread loopClosureThread_;
    std::atomic_bool loopClosureThreadRunning_{false};

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront = Eigen::Affine3f::Identity();
    Eigen::Affine3f incrementalOdometryAffineBack = Eigen::Affine3f::Identity();

    mapOptimization(const rclcpp::NodeOptions & options);
    ~mapOptimization();
    void allocateMemory();
    bool Run(LioSamCloudInfo& msgIn);
    bool LastRunExecuted() const { return lastRunExecuted; }
    double TimeLaserInfoCur() const;
    const float* TransformTobeMapped() const;
    bool CreatedNewKeyframe() const;
    void ClearCreatedNewKeyframe();
    size_t KeyPoseSize() const;
    PointTypePose KeyPose(size_t idx) const;
    pcl::PointCloud<PointType>::Ptr LatestRawCloudKeyFrame() const;
    void pointAssociateToMap(PointType const * const pi, PointType * const po);
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn);
    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint);
    gtsam::Pose3 trans2gtsamPose(float transformIn[]);
    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint);
    Eigen::Affine3f trans2Affine3f(const float transformIn[6]);
    PointTypePose trans2PointTypePose(float transformIn[]);
    double normalizeAngleRad(double angle);
    void setTransformFromAffine(const Eigen::Affine3f& affine);
    void copyTransform(const float src[6], float dst[6]);
    bool transformIsFinite(const float transformIn[6]);
    const char* trackingStateName() const;
    void resetFrameQuality();
    bool acceptMappingPose(const std::string& source);
    void rejectMappingPose(const std::string& source);
    bool isFinitePoint(const PointType& p) const;
    void filterInvalidAndRangeInPlace(pcl::PointCloud<PointType>::Ptr& cloud,
                                      const char* tag,
                                      bool alsoLimitSourceRange = false);
    void cropMapCloudAroundPriorInPlace(pcl::PointCloud<PointType>::Ptr& cloud,
                                        const Eigen::Vector3f& center);
    bool prepareCurrentRawCloudForRegistration();
    bool buildRawLocalMapForRegistration();
    bool rawCloudICPFallback(const Eigen::Affine3f& initialGuess,
                             Eigen::Affine3f& resultAffine,
                             double& fitnessScore);
    bool poseCloseToPrior(const Eigen::Affine3f& priorAffine,
                          const Eigen::Affine3f& candidateAffine,
                          double maxTrans,
                          double maxYawDeg,
                          double maxZ,
                          const char* tag);
    double yawFromAffine(const Eigen::Affine3f& affine);
    double xyDistance(const Eigen::Affine3f& lhs, const Eigen::Affine3f& rhs);
    MotionContinuityInfo evaluateMotionContinuity(const Eigen::Affine3f& candidateAffine,
                                                  const char* tag);
    void pushMotionHistory(double speed, double yawRateDeg, double curvature);
    void updateOutputTrajectoryHistory(const Eigen::Affine3f& outputAffine);
    void loopClosureThread();
    void performLoopClosure();
    bool detectLoopClosureDistance(int *latestID, int *closestID);
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum);
    void updateInitialGuess();
    void extractForLoopClosure();
    void extractNearby();
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract);
    void extractSurroundingKeyFrames();
    void downsampleCurrentScan();
    void updatePointAssociateToMap();
    void cornerOptimization();
    void surfOptimization();
    void combineOptimizationCoeffs();
    bool LMOptimization(int iterCount);
    void scan2MapOptimization();
    void transformUpdate();
    float constraintTransformation(float value, float limit);
    bool saveFrame();
    void addOdomFactor();
    void addLoopFactor();
    void saveKeyFramesAndFactor();
    void correctPoses();
    void updateOdometryState();

};
