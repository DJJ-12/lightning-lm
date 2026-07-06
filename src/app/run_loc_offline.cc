//
// Created by xiang on 25-3-18.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <fstream>
#include <iomanip>
#include <rclcpp/time.hpp>

#include "core/localization/localization.h"
#include "ui/pangolin_window.h"
#include "utils/timer.h"
#include "wrapper/bag_io.h"
#include "wrapper/ros_utils.h"

#include "io/yaml_io.h"
#include <yaml-cpp/yaml.h>

DEFINE_string(input_bag, "", "输入数据包");
DEFINE_string(config, "./config/default.yaml", "配置文件");
DEFINE_string(map_path, "", "地图路径");
DEFINE_string(output_pose, "./loc_result.txt", "Output localization trajectory file");

/// 运行定位的测试
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = google::INFO;

    google::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_input_bag.empty()) {
        LOG(ERROR) << "未指定输入数据";
        return -1;
    }

    using namespace lightning;

    RosbagIO rosbag(FLAGS_input_bag);

    loc::Localization::Options options;
    options.online_mode_ = false;

    loc::Localization loc(options);
    YAML::Node yaml_node = YAML::LoadFile(FLAGS_config);
    std::string map_path = FLAGS_map_path;
    if (map_path.empty() && yaml_node["localization"] && yaml_node["localization"]["map_path"]) {
        map_path = yaml_node["localization"]["map_path"].as<std::string>();
    }
    if (map_path.empty()) {
        map_path = "./data/new_map/";
    }

    std::ofstream pose_file(FLAGS_output_pose);
    if (!pose_file.is_open()) {
        LOG(ERROR) << "failed to open output pose file: " << FLAGS_output_pose;
        return -1;
    }
    pose_file << "# timestamp tx ty tz qx qy qz qw\n";
    loc.SetTFCallback(
        [&pose_file](const geometry_msgs::msg::TransformStamped& tf_msg) {
            const double timestamp = rclcpp::Time(tf_msg.header.stamp).seconds();
            pose_file << std::setprecision(18)
                      << timestamp << " "
                      << tf_msg.transform.translation.x << " "
                      << tf_msg.transform.translation.y << " "
                      << tf_msg.transform.translation.z << " "
                      << tf_msg.transform.rotation.x << " "
                      << tf_msg.transform.rotation.y << " "
                      << tf_msg.transform.rotation.z << " "
                      << tf_msg.transform.rotation.w << "\n";
        });

    if (!loc.Init(FLAGS_config, map_path)) {
        LOG(ERROR) << "failed to init localization";
        return -1;
    }

    lightning::YAML_IO yaml(FLAGS_config);
    std::string lidar_topic = yaml.GetValue<std::string>("common", "lidar_topic");
    std::string livox_topic = yaml.GetValue<std::string>("common", "livox_lidar_topic");
    rosbag
        .AddPointCloud2Handle(lidar_topic,
            [&loc](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
                loc.ProcessLidarMsg(cloud);
                usleep(1000);
                return true;
            })
        .AddLivoxCloudHandle(livox_topic,
                             [&loc](livox_ros_driver2::msg::CustomMsg::SharedPtr cloud) {
                                 loc.ProcessLivoxLidarMsg(cloud);
                                 usleep(1000);
                                 return true;
                             })
        .Go();

    Timer::PrintAll();
    loc.Finish();
    pose_file.close();
    LOG(INFO) << "localization trajectory saved to " << FLAGS_output_pose;

    LOG(INFO) << "done";

    return 0;
}
