#include "runtime/bag_input.h"

#include <algorithm>
#include <cctype>

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include "wrapper/bag_io.h"
#include "common/localization_message_adapter.h"

namespace lightning::runtime {
namespace {

std::string NormalizeMode(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

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
    const YAML::Node mapping_rtk = yaml["mapping_rtk"] ? yaml["mapping_rtk"] : YAML::Node();
    const bool mapping_rtk_enabled = mapping_rtk && mapping_rtk["enabled"] ? mapping_rtk["enabled"].as<bool>() : false;
    const YAML::Node localization = yaml["localization"];
    const YAML::Node localization_rtk = localization && localization["rtk_ins"] ? localization["rtk_ins"] : YAML::Node();
    const YAML::Node localization_eskf = localization && localization["eskf"] ? localization["eskf"] : YAML::Node();
    const std::string localization_mode = NormalizeMode(localization && localization["mode"] ? localization["mode"].as<std::string>() : "ndt_only");
    const bool localization_filter_enabled = localization_mode != "ndt_only";
    const bool localization_rtk_enabled = localization_filter_enabled && localization_rtk && localization_rtk["enabled"] ? localization_rtk["enabled"].as<bool>() : false;
    const bool rtk_ins_enabled = mapping_rtk_enabled || localization_rtk_enabled;
    const bool wheel_enabled = localization_filter_enabled && localization_eskf && localization_eskf["use_wheel_odometry"] ? localization_eskf["use_wheel_odometry"].as<bool>() : false;
    const std::string rtk_ins_topic = common["rtk_local_odometry_topic"] ? common["rtk_local_odometry_topic"].as<std::string>() : (common["localization_rtk_ned_odometry_topic"] ? common["localization_rtk_ned_odometry_topic"].as<std::string>() : std::string());
    const std::string rtk_frame_name = NormalizeMode(common["rtk_local_odometry_frame"] ? common["rtk_local_odometry_frame"].as<std::string>() : "ned");
    if (rtk_frame_name != "ned" && rtk_frame_name != "enu") {
        LOG(ERROR) << "BagInput failed: rtk_local_odometry_frame must be 'ned' or 'enu'";
        return false;
    }
    const auto rtk_input_frame = rtk_frame_name == "enu" ? localization_adapter::LocalNavigationFrame::ENU : localization_adapter::LocalNavigationFrame::NED;
    const std::string wheel_odometry_topic = common["wheel_odometry_topic"] ? common["wheel_odometry_topic"].as<std::string>() : std::string();
    if (rtk_ins_enabled && rtk_ins_topic.empty()) {
        LOG(ERROR) << "BagInput failed: mapping/localization RTK local odometry topic is empty";
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
    if (rtk_ins_enabled && rtk_ins_cb && !rtk_ins_topic.empty()) counted_topics.insert(rtk_ins_topic);
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

    if (rtk_ins_enabled && rtk_ins_cb && !rtk_ins_topic.empty()) rosbag.AddOdometryHandle(rtk_ins_topic, [rtk_ins_cb, rtk_input_frame, progress_cb, cancel_requested, &progress](nav_msgs::msg::Odometry::SharedPtr msg) -> bool {
        if (cancel_requested && cancel_requested()) return false;
        RtkInsMeasurement measurement;
        if (localization_adapter::LocalOdometryToRtkInsMeasurement(*msg, rtk_input_frame, &measurement)) rtk_ins_cb(measurement);
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
