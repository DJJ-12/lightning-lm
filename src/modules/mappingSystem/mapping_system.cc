#include "modules/mappingSystem/mapping_system.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <utility>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl_conversions/pcl_conversions.h>

#include "core/lio_sam/lio_sam_mapping.h"
#include "ui/pangolin_window.h"
#include "wrapper/ros_utils.h"

namespace lightning::modules {

MappingSystem::~MappingSystem() {
    LOG(INFO) << "[MappingSystem] destroy begin"
              << ", this=" << this
              << ", lio_sam=" << lio_sam_.get()
              << ", ui=" << ui_.get();
    Reset();
    LOG(INFO) << "[MappingSystem] destroy finished, this=" << this;
}

void MappingSystem::LoadMappingParams(const YAML::Node& yaml) {
    const YAML::Node common = yaml["common"];
    const YAML::Node lio_sam = yaml["lio_sam"];
    if (!common) {
        return;
    }

    const std::string sensor = common["sensor"].as<std::string>();
    if (sensor == "velodyne") {
        sensor_ = SensorType::VELODYNE;
    } else if (sensor == "ouster") {
        sensor_ = SensorType::OUSTER;
    } else if (sensor == "livox") {
        sensor_ = SensorType::LIVOX;
    } else if (sensor == "hesai") {
        sensor_ = SensorType::HESAI;
    }
    N_SCAN_ = common["N_SCAN"] ? common["N_SCAN"].as<int>() : N_SCAN_;
    Horizon_SCAN_ = common["Horizon_SCAN"] ? common["Horizon_SCAN"].as<int>() : Horizon_SCAN_;
    lidarMinRange_ = common["lidarMinRange"] ? common["lidarMinRange"].as<double>() : lidarMinRange_;
    lidarMaxRange_ = common["lidarMaxRange"] ? common["lidarMaxRange"].as<double>() : lidarMaxRange_;
    point_filter_num_ = common["point_filter_num"] ? common["point_filter_num"].as<int>() : point_filter_num_;
    if (lio_sam && lio_sam["downsampleRate"]) {
        downsampleRate_ = lio_sam["downsampleRate"].as<int>();
    }
    useImuAccelRollPitchInitialization =
        lio_sam["useImuAccelRollPitchInitialization"].as<bool>();
    std::vector<double> imu_extrinsic_rot{1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0};
    std::vector<double> imu_extrinsic_rpy = imu_extrinsic_rot;
    if (lio_sam && lio_sam["extrinsicRot"]) {
        imu_extrinsic_rot = lio_sam["extrinsicRot"].as<std::vector<double>>();
    }
    if (lio_sam && lio_sam["extrinsicRPY"]) {
        imu_extrinsic_rpy = lio_sam["extrinsicRPY"].as<std::vector<double>>();
    }
    CHECK_EQ(imu_extrinsic_rot.size(), 9);
    CHECK_EQ(imu_extrinsic_rpy.size(), 9);
    extRot =
        Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(
            imu_extrinsic_rot.data(), 3, 3);
    Mat3d imu_extrinsic_rpy_matrix =
        Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(
            imu_extrinsic_rpy.data(), 3, 3);
    extQRPY = Quatd(imu_extrinsic_rpy_matrix);

    const size_t cloud_size = static_cast<size_t>(N_SCAN_) * static_cast<size_t>(Horizon_SCAN_);
    fullCloud_.reset(new PointCloudType());
    fullCloud_->points.resize(cloud_size);
    extractedCloud_.reset(new PointCloudType());
    rangeMat_.resize(cloud_size);
    columnIdnCountVec_.resize(N_SCAN_);
    imuTime_.resize(queueLength);
    imuRotX_.resize(queueLength);
    imuRotY_.resize(queueLength);
    imuRotZ_.resize(queueLength);
}

bool MappingSystem::Init(const std::string& yaml_path, const MappingSystemOptions& options) {
    Reset();
    options_ = options;

    YAML::Node yaml = YAML::LoadFile(yaml_path);
    LoadMappingParams(yaml);
    if (yaml["system"] && yaml["system"]["with_ui"]) {
        options_.with_ui = yaml["system"]["with_ui"].as<bool>();
    }
    if (yaml["common"] && yaml["common"]["base_link_frame"]) {
        base_link_frame_ = yaml["common"]["base_link_frame"].as<std::string>();
    }

    std::vector<double> base_lidar_t{0.0, 0.0, 0.0};
    std::vector<double> base_lidar_R{1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0};
    if (yaml["common"] && yaml["common"]["extrinsicBaseLidarTrans"]) {
        base_lidar_t = yaml["common"]["extrinsicBaseLidarTrans"].as<std::vector<double>>();
    }
    if (yaml["common"] && yaml["common"]["extrinsicBaseLidarRot"]) {
        base_lidar_R = yaml["common"]["extrinsicBaseLidarRot"].as<std::vector<double>>();
    }
    CHECK_EQ(base_lidar_t.size(), 3);
    CHECK_EQ(base_lidar_R.size(), 9);

    Mat3d R_base_lidar;
    R_base_lidar << base_lidar_R[0], base_lidar_R[1], base_lidar_R[2],
        base_lidar_R[3], base_lidar_R[4], base_lidar_R[5],
        base_lidar_R[6], base_lidar_R[7], base_lidar_R[8];
    Quatd q_base_lidar(R_base_lidar);
    q_base_lidar.normalize();
    T_base_lidar_ = SE3(q_base_lidar, Vec3d(base_lidar_t[0], base_lidar_t[1], base_lidar_t[2]));
    LOG(INFO) << "[Mapping][BASE_LIDAR] T_base_lidar trans="
              << T_base_lidar_.translation().transpose();

    LioSamMapping::Options lio_options;
    lio_options.mapping_mode_ = options_.online_input
        ? LioSamMapping::MappingRuntimeMode::ONLINE_MAPPING
        : LioSamMapping::MappingRuntimeMode::OFFLINE_MAPPING;
    lio_sam_ = std::make_shared<LioSamMapping>(lio_options);
    if (!lio_sam_->Init(yaml_path)) {
        LOG(ERROR) << "failed to init LIO-SAM mapping";
        return false;
    }

    if (options_.with_ui) {
        ui_ = std::make_shared<ui::PangolinWindow>();
        ui_->Init();
        lio_sam_->SetUI(ui_);
    }

    return true;
}

bool MappingSystem::Start() {
    std::lock_guard<std::mutex> lock(mtx_);
    cur_kf_.reset();
    mapping_update_pending_ = false;
    keyframe_count_ = 0;
    keyframe_cloud_bytes_ = 0;
    cloudQueue_.clear();
    livoxCloudQueue_.clear();
    imuQueue_.clear();
    last_timestamp_imu_ = -1.0;
    last_timestamp_lidar_ = -1.0;
    ringFlag_ = 0;
    deskewFlag_ = 0;
    running_ = true;
    return true;
}

void MappingSystem::Stop() {
    std::lock_guard<std::mutex> lock(mtx_);
    running_ = false;
}

void MappingSystem::Reset() {
    std::shared_ptr<ui::PangolinWindow> ui;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        running_ = false;
        cur_kf_.reset();
        mapping_update_pending_ = false;
        keyframe_count_ = 0;
        keyframe_cloud_bytes_ = 0;
        cloudQueue_.clear();
        livoxCloudQueue_.clear();
        imuQueue_.clear();
        ringFlag_ = 0;
        deskewFlag_ = 0;

        if (lio_sam_) {
            lio_sam_->SetUI(nullptr);
        }
        ui = std::move(ui_);
        lio_sam_.reset();
    }

    if (ui) {
        ui->Quit();
    }
}

void MappingSystem::ProcessIMU(const sensor_msgs::msg::Imu::SharedPtr& imu) {
    if (!imu) {
        return;
    }

    std::lock_guard<std::mutex> lock(mtx_);
    if (!running_) {
        return;
    }
    sensor_msgs::msg::Imu converted_imu = *imu;
    if (!imuConverter(*imu, converted_imu)) {
        return;
    }
    const double timestamp = ToSec(converted_imu.header.stamp);
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "mapping imu loop back, clear buffer";
        imuQueue_.clear();
    }
    imuQueue_.push_back(converted_imu);
    while (imuQueue_.size() > queueLength) {
        imuQueue_.pop_front();
    }
    last_timestamp_imu_ = timestamp;
}

void MappingSystem::ResetProjectionState() {
    const size_t cloud_size = static_cast<size_t>(N_SCAN_) * static_cast<size_t>(Horizon_SCAN_);
    if (rangeMat_.size() != cloud_size) {
        rangeMat_.resize(cloud_size);
    }
    std::fill(rangeMat_.begin(), rangeMat_.end(), FLT_MAX);
    std::fill(columnIdnCountVec_.begin(), columnIdnCountVec_.end(), 0);

    if (!fullCloud_) {
        fullCloud_.reset(new PointCloudType());
    }
    fullCloud_->clear();
    fullCloud_->points.resize(cloud_size);

    extractedCloud_.reset(new PointCloudType());
    extractedCloud_->reserve(cloud_size);

    cloudInfo_ = std::make_shared<LioSamCloudInfo>();
    cloudInfo_->start_ring_index.assign(N_SCAN_, 0);
    cloudInfo_->end_ring_index.assign(N_SCAN_, 0);
    cloudInfo_->point_col_ind.assign(cloud_size, 0);
    cloudInfo_->point_range.assign(cloud_size, 0.0f);
    firstPointFlag_ = true;
    imuPointerCur_ = 0;
}

bool MappingSystem::cachePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& laserCloudMsg) {
    // cache point cloud
    cloudQueue_.push_back(*laserCloudMsg);
    if (cloudQueue_.size() <= 2)
        return false;

    // convert cloud
    sensor_msgs::msg::PointCloud2 currentCloudMsg = std::move(cloudQueue_.front());
    cloudQueue_.pop_front();
    cloudHeader_ = currentCloudMsg.header;
    timeScanHeader_ = ToSec(cloudHeader_.stamp);
    ResetProjectionState();
    if (sensor_ == SensorType::VELODYNE || sensor_ == SensorType::LIVOX)
    {
        pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn_);
    }
    else if (sensor_ == SensorType::OUSTER)
    {
        // Convert to Velodyne format
        pcl::PointCloud<ouster_ros::Point>::Ptr tmpOusterCloudIn(
            new pcl::PointCloud<ouster_ros::Point>());
        pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
        laserCloudIn_->points.resize(tmpOusterCloudIn->size());
        laserCloudIn_->is_dense = tmpOusterCloudIn->is_dense;
        for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
        {
            auto &src = tmpOusterCloudIn->points[i];
            auto &dst = laserCloudIn_->points[i];
            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.intensity = src.intensity;
            dst.ring = src.ring;
            dst.time = src.t * 1e-9f;
        }
    }
    else if (sensor_ == SensorType::HESAI)
    {
        pcl::PointCloud<PointRobotSense>::Ptr tmpHesaiCloudIn(
            new pcl::PointCloud<PointRobotSense>());
        pcl::moveFromROSMsg(currentCloudMsg, *tmpHesaiCloudIn);

        laserCloudIn_->points.resize(tmpHesaiCloudIn->size());
        laserCloudIn_->is_dense = tmpHesaiCloudIn->is_dense;

        for (size_t i = 0; i < tmpHesaiCloudIn->size(); i++)
        {
            auto &src = tmpHesaiCloudIn->points[i];
            auto &dst = laserCloudIn_->points[i];

            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.intensity = src.intensity;
            dst.ring = src.ring;

            // Hesai time: absolute timestamp in seconds.
            // Convert to relative scan time expected by LIO-SAM.
            double rel_time = src.time - timeScanHeader_;

            dst.time = static_cast<float>(rel_time);
        }
    }

    // get timestamp
    timeScanCur_ = ToSec(cloudHeader_.stamp) + laserCloudIn_->points.front().time;
    timeScanEnd_ = ToSec(cloudHeader_.stamp) + laserCloudIn_->points.back().time;

    // remove Nan
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*laserCloudIn_, *laserCloudIn_, indices);

    // check dense flag
    if (laserCloudIn_->is_dense == false)
    {
        LOG(ERROR) << "Point cloud is not in dense format, please remove NaN points first!";
        return false;
    }

    // check ring channel
    // we will skip the ring check in case of velodyne - as we calculate the ring value downstream (line 572)
    if (ringFlag_ == 0)
    {
        ringFlag_ = -1;
        for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
        {
            if (currentCloudMsg.fields[i].name == "ring")
            {
                ringFlag_ = 1;
                break;
            }
        }
        if (ringFlag_ == -1)
        {
            if (sensor_ == SensorType::VELODYNE) {
                ringFlag_ = 2;
            } else {
                LOG(ERROR) << "Point cloud ring channel not available, please configure your point cloud data!";
                return false;
            }
        }
    }

    // check point time
    if (deskewFlag_ == 0)
    {
        deskewFlag_ = -1;
        for (auto &field : currentCloudMsg.fields)
        {
            if (field.name == "time" || field.name == "t")
            {
                deskewFlag_ = 1;
                break;
            }
        }
        if (deskewFlag_ == -1)
            LOG(WARNING) << "Point cloud timestamp not available, deskew function disabled, system will drift significantly!";
    }

    cloudInfo_->timestamp = timeScanCur_;
    cloudInfo_->frame_id =
        cloudHeader_.frame_id.empty()
            ? base_link_frame_
            : cloudHeader_.frame_id;
    last_timestamp_lidar_ = timeScanHeader_;
    return true;
}

bool MappingSystem::cachePointCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& laserCloudMsg) {
    if (!laserCloudMsg) {
        return false;
    }

    livoxCloudQueue_.push_back(*laserCloudMsg);
    if (livoxCloudQueue_.size() <= 2) {
        return false;
    }

    livox_ros_driver2::msg::CustomMsg currentCloudMsg = std::move(livoxCloudQueue_.front());
    livoxCloudQueue_.pop_front();
    cloudHeader_ = currentCloudMsg.header;
    timeScanHeader_ = ToSec(cloudHeader_.stamp);
    if (last_timestamp_lidar_ >= 0.0 && timeScanHeader_ < last_timestamp_lidar_) {
        LOG(WARNING) << "mapping lidar loop back, clear Livox buffer";
        livoxCloudQueue_.clear();
        return false;
    }
    ResetProjectionState();

    laserCloudIn_.reset(new PointCloudType());
    const size_t livox_point_num =
        std::min(static_cast<size_t>(currentCloudMsg.point_num), currentCloudMsg.points.size());
    laserCloudIn_->reserve(livox_point_num);
    for (size_t i = 0; i < livox_point_num; ++i) {
        if (point_filter_num_ > 1 && i % point_filter_num_ != 0) {
            continue;
        }
        const auto& src = currentCloudMsg.points[i];
        PointType dst;
        dst.x = src.x;
        dst.y = src.y;
        dst.z = src.z;
        dst.intensity = src.reflectivity;
        dst.ring = src.line;
        dst.time = static_cast<double>(src.offset_time) * 1e-9;
        laserCloudIn_->push_back(dst);
    }
    laserCloudIn_->height = 1;
    laserCloudIn_->width = laserCloudIn_->size();
    laserCloudIn_->is_dense = true;

    if (laserCloudIn_->empty()) {
        LOG(WARNING) << "[MappingSystem] cached empty Livox cloud";
        return false;
    }

    ringFlag_ = 1;
    deskewFlag_ = 1;
    timeScanCur_ = timeScanHeader_;
    timeScanEnd_ = timeScanCur_ + laserCloudIn_->points.back().time;
    cloudInfo_->timestamp = timeScanCur_;
    cloudInfo_->frame_id =
        cloudHeader_.frame_id.empty()
            ? base_link_frame_
            : cloudHeader_.frame_id;
    last_timestamp_lidar_ = timeScanHeader_;
    return true;
}

bool MappingSystem::deskewInfo() {
    if (imuQueue_.empty() ||
        ToSec(imuQueue_.front().header.stamp) > timeScanCur_ ||
        ToSec(imuQueue_.back().header.stamp) < timeScanEnd_) {
        LOG(INFO) << "Waiting for IMU data ...";
        return false;
    }

    imuDeskewInfo();
    return true;
}

bool MappingSystem::imuConverter(const sensor_msgs::msg::Imu& imu_in,
                               sensor_msgs::msg::Imu& imu_out) const {
    // rotate acceleration
    Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
    acc = extRot * acc;
    imu_out.linear_acceleration.x = acc.x();
    imu_out.linear_acceleration.y = acc.y();
    imu_out.linear_acceleration.z = acc.z();
    // rotate gyroscope
    Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
    gyr = extRot * gyr;
    imu_out.angular_velocity.x = gyr.x();
    imu_out.angular_velocity.y = gyr.y();
    imu_out.angular_velocity.z = gyr.z();
    // rotate roll pitch yaw
    Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
    //Eigen::Quaterniond q_final = extQRPY ; // 0428
    Eigen::Quaterniond q_final = q_from * extQRPY ; 
    q_final.normalize(); //0428

    imu_out.orientation.x = q_final.x();
    imu_out.orientation.y = q_final.y();
    imu_out.orientation.z = q_final.z();
    imu_out.orientation.w = q_final.w();

    if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
    {
        RCLCPP_ERROR(get_logger(), "Invalid quaternion, please use a 9-axis IMU!");
        throw std::runtime_error("invalid IMU quaternion");
    }

    return true;
};


void MappingSystem::imuDeskewInfo() {

    while (!imuQueue_.empty()) {
        if (ToSec(imuQueue_.front().header.stamp) < timeScanCur_ - 0.01) {
            imuQueue_.pop_front();
        } else {
            break;
        }
    }
    if (imuQueue_.empty()) {
        return;
    }

    imuPointerCur_ = 0;
    for (size_t i = 0; i < imuQueue_.size(); ++i) {
        sensor_msgs::msg::Imu thisImuMsg = imuQueue_[i];
        const double currentImuTime = ToSec(thisImuMsg.header.stamp);

        if (currentImuTime <= timeScanCur_)             
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

        if (currentImuTime > timeScanEnd_ + 0.01) {
            break;
        }
        if (imuPointerCur_ >= static_cast<int>(imuTime_.size())) {
            break;
        }

        if (imuPointerCur_ == 0) {
            imuRotX_[0] = 0.0;
            imuRotY_[0] = 0.0;
            imuRotZ_[0] = 0.0;
            imuTime_[0] = currentImuTime;
            ++imuPointerCur_;
            continue;
        }

        double angular_x = 0.0;
        double angular_y = 0.0;
        double angular_z = 0.0;
        imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

        const double timeDiff = currentImuTime - imuTime_[imuPointerCur_ - 1];
        imuRotX_[imuPointerCur_] = imuRotX_[imuPointerCur_ - 1] + angular_x * timeDiff;
        imuRotY_[imuPointerCur_] = imuRotY_[imuPointerCur_ - 1] + angular_y * timeDiff;
        imuRotZ_[imuPointerCur_] = imuRotZ_[imuPointerCur_ - 1] + angular_z * timeDiff;
        imuTime_[imuPointerCur_] = currentImuTime;
        ++imuPointerCur_;
    }

    --imuPointerCur_;
    if (imuPointerCur_ <= 0) {
        return;
    }

    cloudInfo_->imu_roll_init = imuRollInit;
    cloudInfo_->imu_pitch_init = imuPitchInit;
    cloudInfo_->imu_yaw_init = imuYawInit;
}

void MappingSystem::findRotation(double pointTime, float* rotXCur, float* rotYCur, float* rotZCur) {
    *rotXCur = 0.0f;
    *rotYCur = 0.0f;
    *rotZCur = 0.0f;

    int imuPointerFront = 0;
    while (imuPointerFront < imuPointerCur_) {
        if (pointTime < imuTime_[imuPointerFront]) {
            break;
        }
        ++imuPointerFront;
    }

    if (pointTime > imuTime_[imuPointerFront] || imuPointerFront == 0) {
        *rotXCur = imuRotX_[imuPointerFront];
        *rotYCur = imuRotY_[imuPointerFront];
        *rotZCur = imuRotZ_[imuPointerFront];
    } else {
        const int imuPointerBack = imuPointerFront - 1;
        const double ratioFront =
            (pointTime - imuTime_[imuPointerBack]) / (imuTime_[imuPointerFront] - imuTime_[imuPointerBack]);
        const double ratioBack =
            (imuTime_[imuPointerFront] - pointTime) / (imuTime_[imuPointerFront] - imuTime_[imuPointerBack]);
        *rotXCur = imuRotX_[imuPointerFront] * ratioFront + imuRotX_[imuPointerBack] * ratioBack;
        *rotYCur = imuRotY_[imuPointerFront] * ratioFront + imuRotY_[imuPointerBack] * ratioBack;
        *rotZCur = imuRotZ_[imuPointerFront] * ratioFront + imuRotZ_[imuPointerBack] * ratioBack;
    }
}

void MappingSystem::findPosition(double /*relTime*/, float* posXCur, float* posYCur, float* posZCur) {
    *posXCur = 0.0f;
    *posYCur = 0.0f;
    *posZCur = 0.0f;
}

PointType MappingSystem::deskewPoint(PointType* point, double relTime) {
    if (deskewFlag_ == -1 ) {
        return *point;
    }

    const double pointTime = timeScanCur_ + relTime;
    float rotXCur = 0.0f;
    float rotYCur = 0.0f;
    float rotZCur = 0.0f;
    findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

    float posXCur = 0.0f;
    float posYCur = 0.0f;
    float posZCur = 0.0f;
    findPosition(relTime, &posXCur, &posYCur, &posZCur);

    if (firstPointFlag_ == true) {
        transStartInverse_ =
            pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur).inverse();
        firstPointFlag_ = false;
    }

    const Eigen::Affine3f transFinal =
        pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
    const Eigen::Affine3f transBt = transStartInverse_ * transFinal;

    PointType newPoint = *point;
    newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
    newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
    newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
    return newPoint;
}

void MappingSystem::projectPointCloud() {
    const int cloudSize = static_cast<int>(laserCloudIn_->points.size());
    for (int i = 0; i < cloudSize; ++i) {
        PointType thisPoint = laserCloudIn_->points[i];
        const float range = pointDistance(thisPoint);
        if (range < lidarMinRange_ || range > lidarMaxRange_) {
            continue;
        }

        int rowIdn = thisPoint.ring;
        if (ringFlag_ == 2) {
            const float verticalAngle =
                std::atan2(thisPoint.z, std::sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) *
                180.0f / static_cast<float>(M_PI);
            rowIdn = static_cast<int>((verticalAngle + (N_SCAN_ - 1)) / 2.0f);
        }
        if (rowIdn < 0 || rowIdn >= N_SCAN_) {
            continue;
        }
        if (rowIdn % downsampleRate_ != 0) {
            continue;
        }

        int columnIdn = -1;
        if (sensor_ == SensorType::LIVOX) {
            columnIdn = columnIdnCountVec_[rowIdn]++;
        } else {
            const float horizonAngle = std::atan2(thisPoint.x, thisPoint.y) * 180.0f / static_cast<float>(M_PI);
            const float ang_res_x = 360.0f / static_cast<float>(Horizon_SCAN_);
            columnIdn = -std::round((horizonAngle - 90.0f) / ang_res_x) + Horizon_SCAN_ / 2;
            if (columnIdn >= Horizon_SCAN_) {
                columnIdn -= Horizon_SCAN_;
            }
        }
        if (columnIdn < 0 || columnIdn >= Horizon_SCAN_) {
            continue;
        }

        const int index = columnIdn + rowIdn * Horizon_SCAN_;
        if (rangeMat_[index] != FLT_MAX) {
            continue;
        }

        thisPoint = deskewPoint(&thisPoint, laserCloudIn_->points[i].time);
        rangeMat_[index] = range;
        fullCloud_->points[index] = thisPoint;
    }
}

void MappingSystem::cloudExtraction() {
    int count = 0;
    for (int i = 0; i < N_SCAN_; ++i) {
        cloudInfo_->start_ring_index[i] = count - 1 + 5;
        for (int j = 0; j < Horizon_SCAN_; ++j) {
            const int index = j + i * Horizon_SCAN_;
            if (rangeMat_[index] != FLT_MAX) {
                if (count >= static_cast<int>(
                        cloudInfo_->point_col_ind.size())) {
                    break;
                }
                cloudInfo_->point_col_ind[count] = j;
                cloudInfo_->point_range[count] =
                    rangeMat_[index];
                extractedCloud_->push_back(fullCloud_->points[index]);
                ++count;
            }
        }
        cloudInfo_->end_ring_index[i] = count - 1 - 5;
    }

    extractedCloud_->height = 1;
    extractedCloud_->width = extractedCloud_->size();
    extractedCloud_->is_dense = true;
    extractedCloud_->header.frame_id = cloudInfo_->frame_id;
    extractedCloud_->header.stamp = static_cast<std::uint64_t>(std::llround(timeScanHeader_ * 1e9));

    cloudInfo_->point_col_ind.resize(count);
    cloudInfo_->point_range.resize(count);
    cloudInfo_->cloud_deskewed = extractedCloud_;
}

void MappingSystem::ProcessCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
    std::shared_ptr<LioSamCloudInfo> cloud_info;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!running_ || !lio_sam_) {
            return;
        }
        if (!cachePointCloud(cloud)) {
            return;
        }
        if (!deskewInfo()) {
            return;
        }
        projectPointCloud();
        cloudExtraction();
        if (!cloudInfo_->cloud_deskewed ||
            cloudInfo_->cloud_deskewed->size() < 11) {
            LOG(WARNING) << "[MappingSystem] too few deskewed points: "
                         << (cloudInfo_->cloud_deskewed
                                 ? cloudInfo_->cloud_deskewed->size()
                                 : 0);
            return;
        }
        cloud_info = cloudInfo_;
    }
    RunLioSamFrame(cloud_info);
}

void MappingSystem::ProcessCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud) {
    std::shared_ptr<LioSamCloudInfo> cloud_info;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!running_ || !lio_sam_) {
            return;
        }
        if (!cachePointCloud(cloud)) {
            return;
        }
        if (!deskewInfo()) {
            return;
        }
        projectPointCloud();
        cloudExtraction();
        if (!cloudInfo_->cloud_deskewed ||
            cloudInfo_->cloud_deskewed->size() < 11) {
            LOG(WARNING) << "[MappingSystem] too few deskewed Livox points: "
                         << (cloudInfo_->cloud_deskewed
                                 ? cloudInfo_->cloud_deskewed->size()
                                 : 0);
            return;
        }
        cloud_info = cloudInfo_;
    }
    RunLioSamFrame(cloud_info);
}

void MappingSystem::RunLioSamFrame(
    const std::shared_ptr<LioSamCloudInfo>& cloud_info) {
    if (!cloud_info) {
        return;
    }
    std::shared_ptr<LioSamMapping> lio_sam;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!running_ || !lio_sam_) {
            return;
        }
        lio_sam = lio_sam_;
    }

    if (!lio_sam->Run(cloud_info)) {
        return;
    }
    HandleProcessedKeyframe(lio_sam->GetKeyframe());
}

void MappingSystem::HandleProcessedKeyframe(const Keyframe::Ptr& kf) {
    if (!kf) {
        return;
    }

    std::lock_guard<std::mutex> lock(mtx_);
    if (kf == cur_kf_) {
        return;
    }
    cur_kf_ = kf;
    mapping_update_pending_ = true;
    const CloudPtr cloud = kf->GetCloud();
    if (cloud) {
        keyframe_cloud_bytes_ += cloud->points.capacity() * sizeof(PointType);
    }
    ++keyframe_count_;
    if (ui_) {
        ui_->UpdateKF(cur_kf_);
    }
}

CloudPtr MappingSystem::BuildCurrentMapInBaseFrame() {
    std::shared_ptr<LioSamMapping> lio_sam;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        lio_sam = lio_sam_;
    }
    if (!lio_sam) {
        return nullptr;
    }

    lio_sam->SyncOptimizedKeyframePoses();
    const std::vector<Keyframe::Ptr> keyframes = lio_sam->GetAllKeyframes();
    if (keyframes.empty()) {
        return nullptr;
    }

    CloudPtr map_base(new PointCloudType());
    const Eigen::Matrix4f T_base_lidar = T_base_lidar_.matrix().cast<float>();

    for (const auto& kf : keyframes) {
        if (!kf) {
            continue;
        }
        CloudPtr cloud = kf->GetCloud();
        if (!cloud || cloud->empty()) {
            continue;
        }

        const Eigen::Matrix4f pose_base =
            T_base_lidar * kf->GetOptPose().matrix().cast<float>();
        CloudPtr cloud_base(new PointCloudType());
        pcl::transformPointCloud(*cloud, *cloud_base, pose_base);
        *map_base += *cloud_base;
    }

    if (map_base->empty()) {
        return nullptr;
    }
    map_base->header.frame_id = "map";
    map_base->height = 1;
    map_base->width = map_base->size();
    map_base->is_dense = false;
    return map_base;
}

nav_msgs::msg::Path MappingSystem::BuildCurrentPath() {
    std::shared_ptr<LioSamMapping> lio_sam;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        lio_sam = lio_sam_;
    }

    std::vector<Keyframe::Ptr> keyframes;
    if (lio_sam) {
        lio_sam->SyncOptimizedKeyframePoses();
        keyframes = lio_sam->GetAllKeyframes();
    }

    nav_msgs::msg::Path path;
    path.header.frame_id = "map";
    path.poses.reserve(keyframes.size());
    for (const auto& kf : keyframes) {
        if (!kf) {
            continue;
        }

        const SE3 pose = T_base_lidar_ * kf->GetOptPose();
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.frame_id = "map";
        pose_msg.pose.position.x = pose.translation().x();
        pose_msg.pose.position.y = pose.translation().y();
        pose_msg.pose.position.z = pose.translation().z();
        const Quatd q = pose.unit_quaternion();
        pose_msg.pose.orientation.x = q.x();
        pose_msg.pose.orientation.y = q.y();
        pose_msg.pose.orientation.z = q.z();
        pose_msg.pose.orientation.w = q.w();
        path.poses.push_back(pose_msg);
    }
    return path;
}

bool MappingSystem::ConsumeMappingUpdate() {
    std::lock_guard<std::mutex> lock(mtx_);
    const bool pending = mapping_update_pending_;
    mapping_update_pending_ = false;
    return pending;
}

MappingSystemResult MappingSystem::GetResult() {
    std::shared_ptr<LioSamMapping> lio_sam;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        lio_sam = lio_sam_;
    }

    MappingSystemResult result;
    result.T_base_lidar = T_base_lidar_;
    if (lio_sam) {
        lio_sam->SyncOptimizedKeyframePoses();
        result.keyframes = lio_sam->GetAllKeyframes();
        result.global_map = lio_sam->GetGlobalMap(true);
        result.global_map_is_lidar_frame = true;
    }

    result.valid = result.global_map && !result.global_map->empty() && !result.keyframes.empty();
    return result;
}

}  // namespace lightning::modules
