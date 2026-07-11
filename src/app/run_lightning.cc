#include <gflags/gflags.h>
#include <glog/logging.h>
#include <rclcpp/rclcpp.hpp>

#include "runtime/lightning.h"
#include "runtime/service.h"

DEFINE_string(config, "./config/default.yaml", "config yaml path");

int main(int argc, char** argv) {
    LOG(INFO) << "=================build version : 2026-0711-1645===============================";
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>("lightning");
    auto lightning = std::make_shared<lightning::runtime::Lightning>();
    if (!lightning->Init(node, FLAGS_config)) {
        LOG(ERROR) << "failed to init lightning with config: " << FLAGS_config;
        rclcpp::shutdown();
        return 1;
    }

    lightning::runtime::Service service;
    if (!service.Init(node, lightning)) {
        LOG(ERROR) << "failed to init lightning services";
        rclcpp::shutdown();
        return 1;
    }

    LOG(INFO) << "lightning started. Use /lightning/set_mode and other services to control it.";

    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
