//
// Created by xiang on 25-3-18.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

#include <rclcpp/executors/single_threaded_executor.hpp>

#include "core/system/slam.h"
#include "lightning_interfaces/srv/get_offline_mapping_progress.hpp"
#include "ui/pangolin_window.h"
#include "utils/timer.h"
#include "wrapper/bag_io.h"
#include "wrapper/ros_utils.h"

#include "io/yaml_io.h"
#include <yaml-cpp/yaml.h>

DEFINE_string(input_bag, "", "输入数据包");
DEFINE_string(config, "./config/default.yaml", "配置文件");

struct OfflineProgressState {
    std::atomic_bool running{false};
    std::atomic_bool finished{false};
    std::atomic_bool success{false};

    std::atomic<uint64_t> total_frames{0};
    std::atomic<uint64_t> processed_frames{0};

    std::mutex message_mutex;
    std::string message = "offline mapping not started";
};

/// 运行一个LIO前端，带可视化
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

    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("lightning_slam_offline");
    auto progress_state = std::make_shared<OfflineProgressState>();

    auto progress_srv =
        node->create_service<lightning_interfaces::srv::GetOfflineMappingProgress>(
            "/lightning/get_offline_mapping_progress",
            [progress_state](
                const lightning_interfaces::srv::GetOfflineMappingProgress::Request::SharedPtr request,
                lightning_interfaces::srv::GetOfflineMappingProgress::Response::SharedPtr response) {
                (void)request;

                const uint64_t total = progress_state->total_frames.load();
                const uint64_t processed = progress_state->processed_frames.load();

                response->running = progress_state->running.load();
                response->finished = progress_state->finished.load();
                response->success = progress_state->success.load();
                response->total_frames = total;
                response->processed_frames = processed;
                response->progress =
                    total > 0
                        ? static_cast<float>(processed) / static_cast<float>(total)
                        : 0.0f;

                {
                    std::lock_guard<std::mutex> lock(progress_state->message_mutex);
                    response->message = progress_state->message;
                }
            });

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::thread spin_thread([&executor]() {
        executor.spin();
    });

    SlamSystem::Options options;
    options.online_mode_ = false;

    SlamSystem slam(options);

    /// 实时模式好像掉帧掉的比较厉害？

    if (!slam.Init(FLAGS_config)) {
        LOG(ERROR) << "failed to init slam";
        executor.cancel();
        if (spin_thread.joinable()) {
            spin_thread.join();
        }
        rclcpp::shutdown();
        return -1;
    }

    slam.StartSLAM("new_map");

    lightning::YAML_IO yaml(FLAGS_config);
    std::string lidar_topic = yaml.GetValue<std::string>("common", "lidar_topic");
    std::string imu_topic = yaml.GetValue<std::string>("common", "imu_topic");
    std::string livox_lidar_topic = yaml.GetValue<std::string>("common", "livox_lidar_topic");

    YAML::Node yaml_node = YAML::LoadFile(FLAGS_config);
    std::string save_map_path = "./data/new_map/";
    if (yaml_node["localization"] && yaml_node["localization"]["map_path"]) {
        save_map_path = yaml_node["localization"]["map_path"].as<std::string>();
    }

    RosbagIO rosbag(FLAGS_input_bag);
    std::set<std::string> lidar_topics;
    if (!lidar_topic.empty()) {
        lidar_topics.insert(lidar_topic);
    }
    if (!livox_lidar_topic.empty()) {
        lidar_topics.insert(livox_lidar_topic);
    }

    progress_state->total_frames = rosbag.CountMessagesFromMetadata(lidar_topics);
    if (progress_state->total_frames.load() == 0) {
        LOG(WARNING)
            << "failed to read lidar frame count from metadata.yaml; "
            << "progress percentage will stay 0";
    }

    progress_state->running = true;
    progress_state->finished = false;
    progress_state->success = false;
    progress_state->processed_frames = 0;
    {
        std::lock_guard<std::mutex> lock(progress_state->message_mutex);
        progress_state->message = "offline mapping is running";
    }

    rosbag
        /// IMU 的处理
        /*
        .AddImuHandle(imu_topic,
                      [&slam](IMUPtr imu) {
                          slam.ProcessIMU(imu);
                          return true;
                      })
        */
        .AddImuHandle(imu_topic,
                      [&slam](sensor_msgs::msg::Imu::SharedPtr imu) {
                          slam.ProcessIMU(imu);
                          return true;
                      })
        /// lidar 的处理
        .AddPointCloud2Handle(lidar_topic,
                              [&slam, progress_state](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                                  slam.ProcessLidar(msg);
                                  ++progress_state->processed_frames;
                                  return true;
                              })
        /// livox 的处理
        .AddLivoxCloudHandle(livox_lidar_topic,
                             [&slam, progress_state](livox_ros_driver2::msg::CustomMsg::SharedPtr cloud) {
                                 slam.ProcessLidar(cloud);
                                 ++progress_state->processed_frames;
                                 return true;
                             })
        .Go();

    const bool ok = slam.SaveMap(save_map_path);
    progress_state->running = false;
    progress_state->finished = true;
    progress_state->success = ok;
    {
        std::lock_guard<std::mutex> lock(progress_state->message_mutex);
        progress_state->message = ok
            ? "offline mapping finished and map saved: " + save_map_path
            : "offline mapping finished but failed to save map";
    }

    Timer::PrintAll();

    executor.cancel();
    if (spin_thread.joinable()) {
        spin_thread.join();
    }
    rclcpp::shutdown();

    LOG(INFO) << "done";

    return 0;
}
