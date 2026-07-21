#include <gflags/gflags.h>
#include <glog/logging.h>
#include <rclcpp/rclcpp.hpp>

#include <csignal>
#include <execinfo.h>
#include <unistd.h>

#include "runtime/lightning.h"
#include "runtime/service.h"

DEFINE_string(config, "./config/default.yaml", "config yaml path");

namespace {

void CrashSignalHandler(int signal_number) {
    static constexpr char message[] =
        "\n[崩溃诊断] 捕获到致命信号，下面打印当前线程原生调用栈：\n";
    ::write(STDERR_FILENO, message, sizeof(message) - 1);
    void* frames[64];
    const int frame_count = ::backtrace(frames, 64);
    ::backtrace_symbols_fd(frames, frame_count, STDERR_FILENO);
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}

void InstallCrashSignalHandlers() {
    std::signal(SIGABRT, CrashSignalHandler);
    std::signal(SIGSEGV, CrashSignalHandler);
}

}  // namespace

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "=================build version : 2026-0720-dedicated-input=================";
    InstallCrashSignalHandlers();
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
        lightning->Shutdown();
        rclcpp::shutdown();
        return 1;
    }

    LOG(INFO) << "lightning started. Use /lightning/set_mode and other services to control it.";

    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    executor.spin();

    // 主服务executor已经停止，此时再按固定顺序停止业务线程和Topic专用executor。
    LOG(INFO) << "[主程序退出] 主服务executor已经停止";
    lightning->Shutdown();
    rclcpp::shutdown();
    LOG(INFO) << "[主程序退出] 完成";
    return 0;
}
