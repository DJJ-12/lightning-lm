#include "runtime/bag_input.h"

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include "wrapper/bag_io.h"

namespace lightning::runtime {

bool BagInput::Run(const std::string& bag_path, const std::string& yaml_path,
                   ImuCallback imu_cb, CloudCallback cloud_cb, LivoxCallback livox_cb,
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
    const std::string imu_topic = yaml["common"]["imu_topic"].as<std::string>();
    const std::string cloud_topic = yaml["common"]["lidar_topic"].as<std::string>();
    const std::string livox_topic = yaml["common"]["livox_lidar_topic"].as<std::string>();

    RosbagIO rosbag(bag_path);
    const std::set<std::string> lidar_topics{cloud_topic, livox_topic};
    BagInputProgress progress;
    progress.total_frames = rosbag.CountMessagesFromMetadata(lidar_topics);
    if (progress_cb) {
        progress_cb(progress);
    }

    rosbag.AddImuHandle(imu_topic, [imu_cb, cancel_requested](sensor_msgs::msg::Imu::SharedPtr msg) -> bool {
        if (cancel_requested && cancel_requested()) {
            return false;
        }
        if (imu_cb) {
            imu_cb(msg);
        }
        return true;
    });
    rosbag.AddPointCloud2Handle(cloud_topic, [cloud_cb, progress_cb, cancel_requested, &progress](sensor_msgs::msg::PointCloud2::SharedPtr msg) -> bool {
        if (cancel_requested && cancel_requested()) {
            return false;
        }
        if (cloud_cb) {
            cloud_cb(msg);
        }
        ++progress.processed_frames;
        if (progress_cb) {
            progress_cb(progress);
        }
        return true;
    });
    rosbag.AddLivoxCloudHandle(livox_topic, [livox_cb, progress_cb, cancel_requested, &progress](livox_ros_driver2::msg::CustomMsg::SharedPtr msg) -> bool {
        if (cancel_requested && cancel_requested()) {
            return false;
        }
        if (livox_cb) {
            livox_cb(msg);
        }
        ++progress.processed_frames;
        if (progress_cb) {
            progress_cb(progress);
        }
        return true;
    });

    rosbag.Go();
    return !(cancel_requested && cancel_requested());
}

}  // namespace lightning::runtime
