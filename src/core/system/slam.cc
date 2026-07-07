//
// Created by xiang on 25-5-6.
//

#include "core/system/slam.h"
#include "core/g2p5/g2p5.h"
#include "core/lio/laser_mapping.h"
#include "core/lio/lio_sam/lio_sam_mapping.h"
#include "core/lio/pointcloud_preprocess.h"
#include "core/maps/tiled_map.h"
#include "ui/pangolin_window.h"
#include "wrapper/ros_utils.h"

#include <cmath>
#include <pcl/common/point_tests.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <system_error>

namespace lightning {

SlamSystem::SlamSystem(lightning::SlamSystem::Options options) : options_(options) {
    /// handle ctrl-c
    signal(SIGINT, lightning::debug::SigHandle);
}

bool SlamSystem::Init(const std::string& yaml_path) {
    auto yaml = YAML::LoadFile(yaml_path);
    std::string frontend = "laser_mapping";
    if (yaml["system"]["frontend"]) {
        frontend = yaml["system"]["frontend"].as<std::string>();
    }
    use_lio_sam_ = frontend == "lio_sam" || frontend == "liosam";
    if (yaml["common"] && yaml["common"]["base_link_frame"]) {
        base_link_frame_ = yaml["common"]["base_link_frame"].as<std::string>();
    }

    std::vector<double> base_lidar_t{0.0, 0.0, 0.0};
    std::vector<double> base_lidar_R{1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0};
    if (yaml["extrinsicBaseLidarTrans"]) {
        base_lidar_t = yaml["extrinsicBaseLidarTrans"].as<std::vector<double>>();
    } else if (yaml["common"] && yaml["common"]["extrinsicBaseLidarTrans"]) {
        base_lidar_t = yaml["common"]["extrinsicBaseLidarTrans"].as<std::vector<double>>();
    }
    if (yaml["extrinsicBaseLidarRot"]) {
        base_lidar_R = yaml["extrinsicBaseLidarRot"].as<std::vector<double>>();
    } else if (yaml["common"] && yaml["common"]["extrinsicBaseLidarRot"]) {
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
    LOG(INFO) << "[BASE_LIDAR] T_base_lidar trans=" << T_base_lidar_.translation().transpose();

    preprocess_ = std::make_shared<PointCloudPreprocess>();
    if (!preprocess_->Init(yaml_path)) {
        LOG(ERROR) << "failed to init input preprocess";
        return false;
    }

    if (use_lio_sam_) {
        LioSamMapping::Options lio_options;
        lio_options.mapping_mode_ = options_.online_mode_
            ? LioSamMapping::MappingRuntimeMode::ONLINE_MAPPING
            : LioSamMapping::MappingRuntimeMode::OFFLINE_MAPPING;
        lio_sam_ = std::make_shared<LioSamMapping>(lio_options);
        if (!lio_sam_->Init(yaml_path)) {
            LOG(ERROR) << "failed to init lio_sam_ module";
            return false;
        }
    } else {
        lio_ = std::make_shared<LaserMapping>();
        if (!lio_->Init(yaml_path)) {
            LOG(ERROR) << "failed to init lio module";
            return false;
        }
    }

    options_.with_visualization_ = yaml["system"]["with_ui"].as<bool>();
    options_.with_2dvisualization_ = yaml["system"]["with_2dui"].as<bool>();
    options_.with_gridmap_ = yaml["system"]["with_g2p5"].as<bool>();
    options_.step_on_kf_ = yaml["system"]["step_on_kf"].as<bool>();

    if (options_.with_visualization_) {
        LOG(INFO) << "slam with 3D UI";
        ui_ = std::make_shared<ui::PangolinWindow>();
        ui_->Init();
        if (use_lio_sam_) {
            lio_sam_->SetUI(ui_);
        } else {
            lio_->SetUI(ui_);
        }
    }

    if (options_.with_gridmap_) {
        g2p5::G2P5::Options opt;
        opt.online_mode_ = options_.online_mode_;

        g2p5_ = std::make_shared<g2p5::G2P5>(opt);
        g2p5_->Init(yaml_path);

        if (options_.with_2dvisualization_) {
            g2p5_->SetMapUpdateCallback([this](g2p5::G2P5MapPtr map) {
                cv::Mat image = map->ToCV();
                cv::imshow("map", image);

                if (options_.step_on_kf_) {
                    cv::waitKey(0);

                } else {
                    cv::waitKey(10);
                }
            });
        }
    }

    if (options_.online_mode_) {
        LOG(INFO) << "online mode, creating ros2 node ... ";

        /// subscribers
        node_ = std::make_shared<rclcpp::Node>("lightning_slam");

        imu_topic_ = yaml["common"]["imu_topic"].as<std::string>();
        cloud_topic_ = yaml["common"]["lidar_topic"].as<std::string>();
        livox_topic_ = yaml["common"]["livox_lidar_topic"].as<std::string>();

        rclcpp::QoS qos(10);
        // qos.best_effort();

        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, qos, [this](sensor_msgs::msg::Imu::SharedPtr msg) { ProcessIMU(msg); });

        cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic_, qos, [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
                Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
            });

        livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            livox_topic_, qos, [this](livox_ros_driver2::msg::CustomMsg::SharedPtr cloud) {
                Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
            });

        start_mapping_srv_ = node_->create_service<StartMappingService>(
            "/lightning/start_mapping",
            [this](const StartMappingService::Request::SharedPtr request,
                   StartMappingService::Response::SharedPtr response) {
                if (running_.load()) {
                    response->success = false;
                    response->message = "mapping already running";
                    return;
                }

                if (mapping_has_started_.load()) {
                    response->success = false;
                    response->message =
                        "mapping already completed once; restart node to start a new map";
                    return;
                }

                const std::string map_id =
                    request->map_id.empty() ? "new_map" : request->map_id;
                mapping_has_started_ = true;
                StartSLAM(map_id);

                response->success = true;
                response->message = "mapping started: " + map_id;
            });

        finish_mapping_srv_ = node_->create_service<FinishMappingService>(
            "/lightning/finish_mapping",
            [this](const FinishMappingService::Request::SharedPtr request,
                   FinishMappingService::Response::SharedPtr response) {
                if (!running_.load()) {
                    response->success = false;
                    response->message = "mapping is not running";
                    return;
                }

                running_ = false;

                if (!request->save_map) {
                    response->success = true;
                    response->message = "mapping finished without saving";
                    return;
                }

                const std::string save_path =
                    request->save_path.empty()
                        ? "./data/" + map_name_ + "/"
                        : request->save_path;

                std::lock_guard<std::mutex> lock(map_save_mutex_);
                const bool ok = SaveMap(save_path);
                response->success = ok;
                response->message = ok
                    ? "mapping finished and map saved: " + save_path
                    : "mapping finished but failed to save map";
            });

        get_grid_map_srv_ = node_->create_service<GetGridMapService>(
            "/lightning/get_grid_map",
            [this](const GetGridMapService::Request::SharedPtr request,
                   GetGridMapService::Response::SharedPtr response) {
                (void)request;
                if (!g2p5_) {
                    response->success = false;
                    response->message = "2D grid map is disabled";
                    return;
                }
                auto map = g2p5_->GetNewestMap();
                if (!map) {
                    response->success = false;
                    response->message = "2D grid map is empty";
                    return;
                }
                response->map = map->ToROS();
                response->success = true;
                response->message = "ok";
            });

        LOG(INFO) << "online slam node has been created.";
    }

    return true;
}

SlamSystem::~SlamSystem() {
    if (ui_) {
        ui_->Quit();
    }
}

void SlamSystem::StartSLAM(std::string map_name) {
    map_name_ = map_name.empty() ? "new_map" : map_name;
    cur_kf_ = nullptr;
    running_ = true;
}

bool SlamSystem::SaveMap(const std::string& path) {
    std::string save_path = path;
    if (save_path.empty()) {
        save_path = "./data/" + map_name_ + "/";
    }

    LOG(INFO) << "slam map saving to " << save_path;

    if (use_lio_sam_) {
        lio_sam_->SyncOptimizedKeyframePoses();
    }

    std::vector<Keyframe::Ptr> keyframes = use_lio_sam_ ? lio_sam_->GetAllKeyframes() : lio_->GetAllKeyframes();
    if (keyframes.empty()) {
        LOG(WARNING) << "no keyframes, skip map saving";
        return false;
    }

    // auto global_map_no_loop = lio_->GetGlobalMap(true);
    auto global_map = use_lio_sam_ ? lio_sam_->GetGlobalMap(true) : lio_->GetGlobalMap(true);
    if (!global_map || global_map->empty()) {
        LOG(WARNING) << "global map is empty, skip map saving";
        return false;
    }

    if (std::filesystem::exists(save_path)) {
        std::filesystem::remove_all(save_path);
    }

    std::error_code ec;
    std::filesystem::create_directories(save_path, ec);
    if (ec) {
        LOG(ERROR) << "failed to create save path: " << save_path
                   << ", error: " << ec.message();
        return false;
    }

    TiledMap::Options tm_options;
    tm_options.map_path_ = save_path;
    TiledMap tm(tm_options);

    SE3 start_pose = keyframes.front()->GetLIOPose();
    tm.ConvertFromFullPCD(global_map, start_pose, save_path);

    pcl::io::savePCDFileBinaryCompressed(save_path + "/global.pcd", *global_map);
    {
        const int block_resolution = 20;
        const double voxel_size = 0.1;

        const std::string block_map_dir = save_path + "/BlockMap";
        const std::string pcd_dir = block_map_dir + "/pointcloud_map";
        const std::string metadata_path =
            block_map_dir + "/pointcloud_map_metadata.yaml";

        std::filesystem::create_directories(block_map_dir);
        std::filesystem::create_directories(pcd_dir);

        using SegmentIndex = std::pair<int, int>;
        std::map<SegmentIndex, pcl::PointCloud<pcl::PointXYZ>::Ptr> segment_clouds;

        for (const auto& pt : global_map->points) {
            if (!pcl::isFinite(pt)) {
                continue;
            }

            const int seg_x =
                static_cast<int>(
                    std::floor(pt.x / static_cast<double>(block_resolution))) *
                block_resolution;
            const int seg_y =
                static_cast<int>(
                    std::floor(pt.y / static_cast<double>(block_resolution))) *
                block_resolution;

            SegmentIndex seg{seg_x, seg_y};
            auto& segment_cloud = segment_clouds[seg];
            if (!segment_cloud) {
                segment_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
            }

            pcl::PointXYZ p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = pt.z;
            segment_cloud->push_back(p);
        }

        std::ofstream metadata_file(metadata_path);
        if (!metadata_file.is_open()) {
            LOG(ERROR) << "failed to open BlockMap metadata file: " << metadata_path;
            return false;
        }

        metadata_file << "x_resolution: " << block_resolution << "\n";
        metadata_file << "y_resolution: " << block_resolution << "\n";

        int segment_num = 0;
        for (auto& item : segment_clouds) {
            const SegmentIndex& seg = item.first;
            auto& segment_cloud = item.second;
            if (!segment_cloud || segment_cloud->empty()) {
                continue;
            }

            pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(
                new pcl::PointCloud<pcl::PointXYZ>());
            if (voxel_size > 0.0) {
                pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
                voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
                voxel_filter.setInputCloud(segment_cloud);
                voxel_filter.filter(*filtered_cloud);
            } else {
                filtered_cloud = segment_cloud;
            }

            if (!filtered_cloud || filtered_cloud->empty()) {
                continue;
            }

            filtered_cloud->width = filtered_cloud->size();
            filtered_cloud->height = 1;
            filtered_cloud->is_dense = false;

            std::ostringstream filename_stream;
            filename_stream << "seg_" << seg.first << "_" << seg.second << ".pcd";

            const std::string filename = filename_stream.str();
            const std::string pcd_path = pcd_dir + "/" + filename;
            pcl::io::savePCDFileBinary(pcd_path, *filtered_cloud);

            metadata_file << filename << ": [" << seg.first << ", " << seg.second
                          << "]\n";
            ++segment_num;
        }

        LOG(INFO) << "RobotLocalize BlockMap saved to " << block_map_dir
                  << ", segment num: " << segment_num;
    }
    std::ofstream pose_file(save_path + "/pose.txt");
    pose_file << "# id timestamp tx ty tz qx qy qz qw\n";
    for (const auto& kf : keyframes) {
        SE3 pose = kf->GetLIOPose();
        NavState state = kf->GetState();
        Vec3d t = pose.translation();
        Quatd q = pose.unit_quaternion();
        pose_file << std::setprecision(18) << kf->GetID() << " " << state.timestamp_ << " " << t.x() << " "
                  << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                  << "\n";
    }
    // pcl::io::savePCDFileBinaryCompressed(save_path + "/global_no_loop.pcd", *global_map_no_loop);
    // pcl::io::savePCDFileBinaryCompressed(save_path + "/global_raw.pcd", *global_map_raw);

    if (options_.with_gridmap_ && g2p5_) {
        /// 存为ROS兼容的模式
        auto newest_map = g2p5_->GetNewestMap();
        if (!newest_map) {
            LOG(WARNING) << "grid map is empty, skip saving 2D map";
        } else {
            auto map = newest_map->ToROS();
            const int width = map.info.width;
            const int height = map.info.height;

            cv::Mat nav_image(height, width, CV_8UC1);
            for (int y = 0; y < height; ++y) {
                const int rowStartIndex = y * width;
                for (int x = 0; x < width; ++x) {
                    const int index = rowStartIndex + x;
                    int8_t data = map.data[index];
                    if (data == 0) {                                   // Free
                        nav_image.at<uchar>(height - 1 - y, x) = 255;  // White
                    } else if (data == 100) {                          // Occupied
                        nav_image.at<uchar>(height - 1 - y, x) = 0;    // Black
                    } else {                                           // Unknown
                        nav_image.at<uchar>(height - 1 - y, x) = 128;  // Gray
                    }
                }
            }

            cv::imwrite(save_path + "/map.pgm", nav_image);

            /// yaml
            std::ofstream yamlFile(save_path + "/map.yaml");
            if (!yamlFile.is_open()) {
                LOG(ERROR) << "failed to write map.yaml";
                return false;
            }

            try {
                YAML::Emitter emitter;
                emitter << YAML::BeginMap;
                emitter << YAML::Key << "image" << YAML::Value << "map.pgm";
                emitter << YAML::Key << "mode" << YAML::Value << "trinary";
                emitter << YAML::Key << "width" << YAML::Value << map.info.width;
                emitter << YAML::Key << "height" << YAML::Value << map.info.height;
                emitter << YAML::Key << "resolution" << YAML::Value << float(0.05);
                std::vector<double> orig{map.info.origin.position.x, map.info.origin.position.y, 0};
                emitter << YAML::Key << "origin" << YAML::Value << orig;
                emitter << YAML::Key << "negate" << YAML::Value << 0;
                emitter << YAML::Key << "occupied_thresh" << YAML::Value << 0.65;
                emitter << YAML::Key << "free_thresh" << YAML::Value << 0.25;

                emitter << YAML::EndMap;

                yamlFile << emitter.c_str();
                yamlFile.close();
            } catch (...) {
                yamlFile.close();
                return false;
            }
        }
    }

    LOG(INFO) << "map saved to: " << save_path;
    return true;
}

void SlamSystem::ProcessIMU(const sensor_msgs::msg::Imu::SharedPtr& imu) {
    if (running_ == false) {
        return;
    }
    IMUPtr input = std::make_shared<IMU>();
    input->timestamp = ToSec(imu->header.stamp);
    input->angular_velocity =
        Vec3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
    input->linear_acceleration =
        Vec3d(imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z);
    input->orientation =
        Quatd(imu->orientation.w, imu->orientation.x, imu->orientation.y, imu->orientation.z);
    if (use_lio_sam_) {
        lio_sam_->ProcessIMU(input);
    } else {
        lio_->ProcessIMU(input);
    }
}

void SlamSystem::ProcessIMU(const lightning::IMUPtr& imu) {
    if (running_ == false) {
        return;
    }
    if (use_lio_sam_) {
        lio_sam_->ProcessIMU(imu);
    } else {
        lio_->ProcessIMU(imu);
    }
}

void SlamSystem::ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
    if (running_ == false) {
        return;
    }
    CloudPtr input(new PointCloudType);
    preprocess_->Process(cloud, input);
    CloudPtr input_base(new PointCloudType);
    pcl::transformPointCloud(*input, *input_base, T_base_lidar_.matrix().cast<float>());
    input_base->header = input->header;
    input_base->header.frame_id = base_link_frame_;
    input = input_base;

    Keyframe::Ptr kf;
    if (use_lio_sam_) {
        lio_sam_->ProcessPointCloud2(input);
        if (!lio_sam_->Run()) {
            return;
        }        
        kf = lio_sam_->GetKeyframe();
    } else {
        lio_->ProcessPointCloud2(input);
        if (!lio_->Run()) {
            return;
        }
        kf = lio_->GetKeyframe();
    }
    if (kf != cur_kf_) {
        cur_kf_ = kf;
    } else {
        return;
    }

    if (cur_kf_ == nullptr) {
        return;
    }

    if (options_.with_gridmap_) {
        g2p5_->PushKeyframe(cur_kf_);
    }

    if (ui_) {
        ui_->UpdateKF(cur_kf_);
    }
}

void SlamSystem::ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud) {
    if (running_ == false) {
        return;
    }

    CloudPtr input(new PointCloudType);
    preprocess_->Process(cloud, input);
    CloudPtr input_base(new PointCloudType);
    pcl::transformPointCloud(*input, *input_base, T_base_lidar_.matrix().cast<float>());
    input_base->header = input->header;
    input_base->header.frame_id = base_link_frame_;
    input = input_base;

    Keyframe::Ptr kf;
    if (use_lio_sam_) {
        lio_sam_->ProcessPointCloud2(input);
        if (!lio_sam_->Run()) {
            return;
        }
        kf = lio_sam_->GetKeyframe();
    } else {
        lio_->ProcessPointCloud2(input);
        if (!lio_->Run()) {
            return;
        }        
        kf = lio_->GetKeyframe();
    }
    if (kf != cur_kf_) {
        cur_kf_ = kf;
    } else {
        return;
    }

    if (cur_kf_ == nullptr) {
        return;
    }

    if (options_.with_gridmap_) {
        g2p5_->PushKeyframe(cur_kf_);
    }

    if (ui_) {
        ui_->UpdateKF(cur_kf_);
    }
}

void SlamSystem::Spin() {
    if (options_.online_mode_ && node_ != nullptr) {
        spin(node_);
    }
}

}  // namespace lightning
