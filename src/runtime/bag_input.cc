#include "runtime/bag_input.h"

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include "wrapper/bag_io.h"

namespace lightning::runtime {
namespace {

std::string ReadTopic(const YAML::Node& common, const char* name) {
    if (!common) return std::string();
    const YAML::Node value = common[name];
    return value && value.IsScalar()
        ? value.as<std::string>()
        : std::string();
}

}  // namespace

bool BagInput::Run(const std::string& bag_path, const std::string& yaml_path,
                   ImuCallback imu_cb, CloudCallback cloud_cb, LivoxCallback livox_cb,
                   RtkPositionCallback rtk_position_cb,
                   RtkVelocityCallback rtk_velocity_cb,
                   WheelOdometryCallback wheel_odometry_cb,
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
    const std::string imu_topic = ReadTopic(common, "imu_topic");
    const std::string cloud_topic = ReadTopic(common, "lidar_topic");
    const std::string livox_topic = ReadTopic(common, "livox_lidar_topic");
    const std::string rtk_fix_topic = ReadTopic(common, "rtk_fix_topic");
    const std::string rtk_velocity_topic =
        ReadTopic(common, "rtk_velocity_topic");
    const std::string wheel_odometry_topic =
        ReadTopic(common, "wheel_odometry_topic");

    RosbagIO rosbag(bag_path);
    std::set<std::string> counted_topics;
    if (!cloud_topic.empty() && cloud_cb) {
        counted_topics.insert(cloud_topic);
    }
    if (!livox_topic.empty() && livox_cb) {
        counted_topics.insert(livox_topic);
    }
    if (!rtk_fix_topic.empty() && rtk_position_cb) counted_topics.insert(rtk_fix_topic);
    if (!rtk_velocity_topic.empty() && rtk_velocity_cb) counted_topics.insert(rtk_velocity_topic);
    if (!wheel_odometry_topic.empty() && wheel_odometry_cb) {
        counted_topics.insert(wheel_odometry_topic);
    }
    BagInputProgress progress;
    progress.total_frames = rosbag.CountMessagesFromMetadata(counted_topics);
    if (progress_cb) progress_cb(progress);

    if (!imu_topic.empty() && imu_cb) {
        rosbag.AddImuHandle(
            imu_topic,
            [imu_cb, cancel_requested](
                sensor_msgs::msg::Imu::SharedPtr msg) -> bool {
                if (cancel_requested && cancel_requested()) return false;
                imu_cb(msg);
                return true;
            });
    }
    if (!cloud_topic.empty() && cloud_cb) {
        rosbag.AddPointCloud2Handle(
            cloud_topic,
            [cloud_cb, progress_cb, cancel_requested, &progress](
                sensor_msgs::msg::PointCloud2::SharedPtr msg) -> bool {
                if (cancel_requested && cancel_requested()) return false;
                cloud_cb(msg);
                ++progress.processed_frames;
                if (progress_cb) progress_cb(progress);
                return true;
            });
    }
    if (!livox_topic.empty() && livox_cb) {
        rosbag.AddLivoxCloudHandle(
            livox_topic,
            [livox_cb, progress_cb, cancel_requested, &progress](
                livox_ros_driver2::msg::CustomMsg::SharedPtr msg) -> bool {
                if (cancel_requested && cancel_requested()) return false;
                livox_cb(msg);
                ++progress.processed_frames;
                if (progress_cb) progress_cb(progress);
                return true;
            });
    }

    if (!rtk_fix_topic.empty() && rtk_position_cb) {
        rosbag.AddNavSatFixHandle(
            rtk_fix_topic,
            [rtk_position_cb, progress_cb, cancel_requested, &progress](
                sensor_msgs::msg::NavSatFix::SharedPtr msg) -> bool {
                if (cancel_requested && cancel_requested()) return false;
                rtk_position_cb(msg);
                ++progress.processed_frames;
                if (progress_cb) progress_cb(progress);
                return true;
            });
    }

    if (!rtk_velocity_topic.empty() && rtk_velocity_cb) {
        rosbag.AddTwistHandle(
            rtk_velocity_topic,
            [rtk_velocity_cb, progress_cb, cancel_requested, &progress](
                geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg)
                -> bool {
                if (cancel_requested && cancel_requested()) return false;
                rtk_velocity_cb(msg);
                ++progress.processed_frames;
                if (progress_cb) progress_cb(progress);
                return true;
            });
    }

    if (!wheel_odometry_topic.empty() && wheel_odometry_cb) {
        rosbag.AddOdometryHandle(
            wheel_odometry_topic,
            [wheel_odometry_cb, progress_cb, cancel_requested, &progress](
                nav_msgs::msg::Odometry::SharedPtr msg) -> bool {
                if (cancel_requested && cancel_requested()) return false;

                wheel_odometry_cb(msg);
                ++progress.processed_frames;
                if (progress_cb) progress_cb(progress);
                return true;
            });
    }

    rosbag.Go();
    return !(cancel_requested && cancel_requested());
}

}  // namespace lightning::runtime
