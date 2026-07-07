//
// Created by xiang on 25-5-6.
//

#ifndef LIGHTNING_SLAM_H
#define LIGHTNING_SLAM_H

#include <atomic>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>

#include "lightning_interfaces/srv/finish_mapping.hpp"
#include "lightning_interfaces/srv/get_grid_map.hpp"
#include "lightning_interfaces/srv/start_mapping.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/keyframe.h"

namespace lightning {

class LaserMapping;  //  lio 前端
class LioSamMapping;
class PointCloudPreprocess;

namespace ui {
class PangolinWindow;
}

namespace g2p5 {
class G2P5;
}

/**
 * SLAM 系统调用接口
 */
class SlamSystem {
   public:
    struct Options {
        Options() {}

        bool online_mode_ = true;  // 在线模式，在线模式下会起一些子线程来做异步处理

        bool with_cc_ = true;               // 是否需要带交叉验证
        bool with_gridmap_ = true;          // 是否需要2D栅格
        bool with_visualization_ = true;    // 是否需要可视化UI
        bool with_2dvisualization_ = true;  // 是否需要2D可视化UI

        bool step_on_kf_ = true;  // 是否在关键帧处暂停p
    };

    using StartMappingService = lightning_interfaces::srv::StartMapping;
    using FinishMappingService = lightning_interfaces::srv::FinishMapping;
    using GetGridMapService = lightning_interfaces::srv::GetGridMap;

    SlamSystem(Options options);
    ~SlamSystem();

    /// 初始化
    bool Init(const std::string& yaml_path);

    /// 对外部交互接口
    /// 开始建图，输入地图名称
    void StartSLAM(std::string map_name);

    /// 保存地图，默认保存至./data/地图名/ 下方
    bool SaveMap(const std::string& path = "");

    /// 处理IMU
    void ProcessIMU(const sensor_msgs::msg::Imu::SharedPtr& imu);
    void ProcessIMU(const lightning::IMUPtr& imu);

    /// 处理点云
    void ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud);
    void ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud);

    /// 实时模式下的spin
    void Spin();

   private:
    Options options_;
    std::atomic_bool running_ = false;
    std::atomic_bool mapping_has_started_{false};
    bool use_lio_sam_ = false;
    SE3 T_base_lidar_ = SE3();

    rclcpp::Service<StartMappingService>::SharedPtr start_mapping_srv_ = nullptr;
    rclcpp::Service<FinishMappingService>::SharedPtr finish_mapping_srv_ = nullptr;
    rclcpp::Service<GetGridMapService>::SharedPtr get_grid_map_srv_ = nullptr;
    std::mutex map_save_mutex_;

    std::string map_name_;  // 地图名
    std::string base_link_frame_ = "base_link";

    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;
    std::shared_ptr<LioSamMapping> lio_sam_ = nullptr;
    std::shared_ptr<LaserMapping> lio_ = nullptr;       // lio 前端
    std::shared_ptr<ui::PangolinWindow> ui_ = nullptr;  // ui
    std::shared_ptr<g2p5::G2P5> g2p5_ = nullptr;        // 栅格地图

    Keyframe::Ptr cur_kf_ = nullptr;

    /// 实时模式下的ros2 node, subscribers
    rclcpp::Node::SharedPtr node_;
    std::string imu_topic_;
    std::string cloud_topic_;
    std::string livox_topic_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_ = nullptr;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_ = nullptr;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_sub_ = nullptr;
};
}  // namespace lightning

#endif  // LIGHTNING_SLAM_H
