#include "runtime/bag_input.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <vector>

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include "wrapper/bag_io.h"
#include "common/localization_sensor_measurements.h"

namespace lightning::runtime {
namespace {

std::string NormalizeMode(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

constexpr double kPi = 3.14159265358979323846;

}  // namespace

bool BagInput::Run(const std::string& bag_path, const std::string& yaml_path,
                   ImuCallback imu_cb, CloudCallback cloud_cb, LivoxCallback livox_cb,
                   RtkInsCallback rtk_ins_cb, WheelOdometryCallback wheel_odometry_cb,
                   ProgressCallback progress_cb, CancelCallback cancel_requested) {
    if (bag_path.empty()) {
        LOG(ERROR) << "BagInput failed: bag_path is empty";
        return false;
    }
    YAML::Node yaml = YAML::LoadFile(yaml_path);
    if (!yaml["common"]) {
        LOG(ERROR) << "BagInput failed: missing common config";
        return false;
    }
    const YAML::Node common = yaml["common"];
    const std::string imu_topic = common["imu_topic"] ? common["imu_topic"].as<std::string>() : std::string();
    const std::string cloud_topic = common["lidar_topic"] ? common["lidar_topic"].as<std::string>() : std::string();
    const std::string livox_topic = common["livox_lidar_topic"] ? common["livox_lidar_topic"].as<std::string>() : std::string();
    const YAML::Node localization = yaml["localization"];
    const YAML::Node localization_rtk = localization && localization["rtk_ins"] ? localization["rtk_ins"] : YAML::Node();
    const YAML::Node eskf = localization && localization["eskf"] ? localization["eskf"] : YAML::Node();
    const std::string localization_mode = NormalizeMode(localization && localization["mode"] ? localization["mode"].as<std::string>() : "ndt_only");
    const bool localization_filter_enabled = localization_mode != "ndt_only";
    const bool localization_rtk_enabled = localization_filter_enabled && localization_rtk && localization_rtk["enabled"] ? localization_rtk["enabled"].as<bool>() : false;
    const bool rtk_enabled = localization_rtk_enabled;
    const bool wheel_enabled = localization_filter_enabled && eskf && eskf["use_wheel_odometry"] ? eskf["use_wheel_odometry"].as<bool>() : false;
    const std::string rtk_fix_topic = common["rtk_fix_topic"] ? common["rtk_fix_topic"].as<std::string>() : "/fdilink/gnss_fix";
    const std::string rtk_heading_topic = common["rtk_heading_topic"] ? common["rtk_heading_topic"].as<std::string>() : "/fdilink/mag_pose_2d";
    const int rtk_utm_zone = common["rtk_utm_zone"] ? common["rtk_utm_zone"].as<int>() : 0;
    const double rtk_sync_max_dt_sec = common["rtk_sync_max_dt_sec"] ? common["rtk_sync_max_dt_sec"].as<double>() : 0.20;
    const double rtk_heading_sigma_rad = localization_rtk && localization_rtk["heading_sigma_deg"] ? localization_rtk["heading_sigma_deg"].as<double>() * kPi / 180.0 : 10.0 * kPi / 180.0;
    const double rtk_heading_variance = rtk_heading_sigma_rad * rtk_heading_sigma_rad;
    const std::string wheel_odometry_topic = common["wheel_odometry_topic"] ? common["wheel_odometry_topic"].as<std::string>() : std::string();
    if (rtk_enabled && rtk_fix_topic.empty()) {
        LOG(ERROR) << "BagInput failed: localization rtk_fix_topic is empty";
        return false;
    }
    if (rtk_enabled && rtk_heading_topic.empty()) {
        LOG(ERROR) << "BagInput failed: localization rtk_heading_topic is empty";
        return false;
    }
    if (wheel_enabled && wheel_odometry_topic.empty()) {
        LOG(ERROR) << "BagInput failed: wheel odometry topic is empty";
        return false;
    }

    RosbagIO rosbag(bag_path);
    std::set<std::string> counted_topics;
    if (!cloud_topic.empty() && cloud_cb) counted_topics.insert(cloud_topic);
    if (!livox_topic.empty() && livox_cb) counted_topics.insert(livox_topic);
    if (rtk_enabled && rtk_ins_cb && !rtk_fix_topic.empty()) counted_topics.insert(rtk_fix_topic);
    if (rtk_enabled && rtk_ins_cb && !rtk_heading_topic.empty()) counted_topics.insert(rtk_heading_topic);
    if (wheel_enabled && wheel_odometry_cb && !wheel_odometry_topic.empty()) counted_topics.insert(wheel_odometry_topic);
    BagInputProgress progress;
    progress.total_frames = rosbag.CountMessagesFromMetadata(counted_topics);
    if (progress_cb) progress_cb(progress);

    if (!imu_topic.empty() && imu_cb) rosbag.AddImuHandle(imu_topic, [imu_cb, cancel_requested](sensor_msgs::msg::Imu::SharedPtr msg) -> bool {
        if (cancel_requested && cancel_requested()) return false;
        imu_cb(msg);
        return true;
    });
    if (!cloud_topic.empty() && cloud_cb) rosbag.AddPointCloud2Handle(cloud_topic, [cloud_cb, progress_cb, cancel_requested, &progress](sensor_msgs::msg::PointCloud2::SharedPtr msg) -> bool {
        if (cancel_requested && cancel_requested()) return false;
        cloud_cb(msg);
        ++progress.processed_frames;
        if (progress_cb) progress_cb(progress);
        return true;
    });
    if (!livox_topic.empty() && livox_cb) rosbag.AddLivoxCloudHandle(livox_topic, [livox_cb, progress_cb, cancel_requested, &progress](livox_ros_driver2::msg::CustomMsg::SharedPtr msg) -> bool {
        if (cancel_requested && cancel_requested()) return false;
        livox_cb(msg);
        ++progress.processed_frames;
        if (progress_cb) progress_cb(progress);
        return true;
    });

    localization_adapter::RtkMeasurementSynchronizer rtk_sync(rtk_sync_max_dt_sec);
    auto emit_synced_rtk = [rtk_ins_cb](const std::vector<RtkInsMeasurement>& measurements) {
        for (const RtkInsMeasurement& measurement : measurements) {
            rtk_ins_cb(measurement);
        }
    };

    if (rtk_enabled && rtk_ins_cb && !rtk_fix_topic.empty()) rosbag.AddNavSatFixHandle(rtk_fix_topic, [rtk_utm_zone, progress_cb, cancel_requested, &progress, &rtk_sync, &emit_synced_rtk](sensor_msgs::msg::NavSatFix::SharedPtr msg, double bag_stamp) -> bool {
        if (cancel_requested && cancel_requested()) return false;
        RtkPositionMeasurement position;
        std::vector<RtkInsMeasurement> synced_measurements;
        if (localization_adapter::NavSatFixToRtkPositionMeasurement(*msg, rtk_utm_zone, &position)) {
            position.sync_stamp = bag_stamp;
            rtk_sync.AddPosition(position, &synced_measurements);
            emit_synced_rtk(synced_measurements);
        }
        ++progress.processed_frames;
        if (progress_cb) progress_cb(progress);
        return true;
    });
    if (rtk_enabled && rtk_ins_cb && !rtk_heading_topic.empty()) rosbag.AddPose2DHandle(rtk_heading_topic, [rtk_heading_variance, progress_cb, cancel_requested, &progress, &rtk_sync, &emit_synced_rtk](geometry_msgs::msg::Pose2D::SharedPtr msg, double stamp) -> bool {
        if (cancel_requested && cancel_requested()) return false;
        RtkHeadingMeasurement heading;
        std::vector<RtkInsMeasurement> synced_measurements;
        if (localization_adapter::Pose2DToRtkHeadingMeasurement(*msg, stamp, rtk_heading_variance, &heading)) {
            rtk_sync.AddHeading(heading, &synced_measurements);
            emit_synced_rtk(synced_measurements);
        }
        ++progress.processed_frames;
        if (progress_cb) progress_cb(progress);
        return true;
    });
    if (wheel_enabled && wheel_odometry_cb && !wheel_odometry_topic.empty()) rosbag.AddOdometryHandle(wheel_odometry_topic, [wheel_odometry_cb, progress_cb, cancel_requested, &progress](nav_msgs::msg::Odometry::SharedPtr msg) -> bool {
        if (cancel_requested && cancel_requested()) return false;
        WheelOdometryMeasurement measurement;
        if (localization_adapter::OdometryToWheelMeasurement(*msg, &measurement)) wheel_odometry_cb(measurement);
        ++progress.processed_frames;
        if (progress_cb) progress_cb(progress);
        return true;
    });

    rosbag.Go();
    return !(cancel_requested && cancel_requested());
}

}  // namespace lightning::runtime
