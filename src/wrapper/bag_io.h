#ifndef LIGHTNING_BAG_IO_H
#define LIGHTNING_BAG_IO_H

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include "livox_ros_driver2/msg/custom_msg.hpp"

#include "common/imu.h"
#include "wrapper/ros_utils.h"

namespace lightning {

class RosbagIO {
   public:
    explicit RosbagIO(std::string bag_file) : bag_file_(std::move(bag_file)) {}

    using MsgType = std::shared_ptr<rosbag2_storage::SerializedBagMessage>;
    using MessageProcessFunction = std::function<bool(const MsgType& m)>;
    using PointCloud2Handle = std::function<bool(sensor_msgs::msg::PointCloud2::SharedPtr)>;
    using LivoxCloud2Handle = std::function<bool(livox_ros_driver2::msg::CustomMsg::SharedPtr)>;
    using ImuHandle = std::function<bool(IMUPtr)>;
    using RawImuHandle = std::function<bool(sensor_msgs::msg::Imu::SharedPtr)>;
    using NavSatFixHandle = std::function<bool(sensor_msgs::msg::NavSatFix::SharedPtr, double)>;
    using Pose2DHandle = std::function<bool(geometry_msgs::msg::Pose2D::SharedPtr, double)>;
    using OdometryHandle = std::function<bool(nav_msgs::msg::Odometry::SharedPtr)>;

    void Go(int sleep_usec = 0);
    uint64_t CountMessagesFromMetadata(const std::set<std::string>& topics) const;

    RosbagIO& AddHandle(const std::string& topic_name, MessageProcessFunction func) {
        process_func_.emplace(topic_name, std::move(func));
        return *this;
    }

    RosbagIO& AddPointCloud2Handle(const std::string& topic_name, PointCloud2Handle f) {
        return AddHandle(topic_name, [f = std::move(f), this](const MsgType& m) -> bool {
            auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
            rclcpp::SerializedMessage data(*m->serialized_data);
            seri_cloud2_.deserialize_message(&data, msg.get());
            return f(msg);
        });
    }

    RosbagIO& AddLivoxCloudHandle(const std::string& topic_name, LivoxCloud2Handle f) {
        return AddHandle(topic_name, [f = std::move(f), this](const MsgType& m) -> bool {
            auto msg = std::make_shared<livox_ros_driver2::msg::CustomMsg>();
            rclcpp::SerializedMessage data(*m->serialized_data);
            seri_livox_.deserialize_message(&data, msg.get());
            return f(msg);
        });
    }

    RosbagIO& AddImuHandle(const std::string& topic_name, ImuHandle f) {
        return AddHandle(topic_name, [f = std::move(f), this](const MsgType& m) -> bool {
            auto msg = std::make_shared<sensor_msgs::msg::Imu>();
            rclcpp::SerializedMessage data(*m->serialized_data);
            seri_imu_.deserialize_message(&data, msg.get());

            IMUPtr imu = std::make_shared<IMU>();
            imu->timestamp = ToSec(msg->header.stamp);
            imu->linear_acceleration =
                Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
            imu->angular_velocity =
                Vec3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
            imu->orientation =
                Quatd(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
            return f(imu);
        });
    }

    RosbagIO& AddNavSatFixHandle(const std::string& topic_name, NavSatFixHandle f) {
        return AddHandle(topic_name, [f = std::move(f), this](const MsgType& m) -> bool {
            auto msg = std::make_shared<sensor_msgs::msg::NavSatFix>();
            rclcpp::SerializedMessage data(*m->serialized_data);
            seri_navsat_fix_.deserialize_message(&data, msg.get());
            return f(msg, static_cast<double>(m->time_stamp) * 1e-9);
        });
    }

    RosbagIO& AddPose2DHandle(const std::string& topic_name, Pose2DHandle f) {
        return AddHandle(topic_name, [f = std::move(f), this](const MsgType& m) -> bool {
            auto msg = std::make_shared<geometry_msgs::msg::Pose2D>();
            rclcpp::SerializedMessage data(*m->serialized_data);
            seri_pose2d_.deserialize_message(&data, msg.get());
            return f(msg, static_cast<double>(m->time_stamp) * 1e-9);
        });
    }

    RosbagIO& AddOdometryHandle(const std::string& topic_name, OdometryHandle f) {
        return AddHandle(topic_name, [f = std::move(f), this](const MsgType& m) -> bool {
            auto msg = std::make_shared<nav_msgs::msg::Odometry>();
            rclcpp::SerializedMessage data(*m->serialized_data);
            seri_odometry_.deserialize_message(&data, msg.get());
            return f(msg);
        });
    }

    RosbagIO& AddImuHandle(const std::string& topic_name, RawImuHandle f) {
        return AddHandle(topic_name, [f = std::move(f), this](const MsgType& m) -> bool {
            auto msg = std::make_shared<sensor_msgs::msg::Imu>();
            rclcpp::SerializedMessage data(*m->serialized_data);
            seri_imu_.deserialize_message(&data, msg.get());
            return f(msg);
        });
    }

   private:
    std::map<std::string, MessageProcessFunction> process_func_;

    rclcpp::Serialization<sensor_msgs::msg::Imu> seri_imu_;
    rclcpp::Serialization<sensor_msgs::msg::PointCloud2> seri_cloud2_;
    rclcpp::Serialization<sensor_msgs::msg::NavSatFix> seri_navsat_fix_;
    rclcpp::Serialization<geometry_msgs::msg::Pose2D> seri_pose2d_;
    rclcpp::Serialization<nav_msgs::msg::Odometry> seri_odometry_;
    rclcpp::Serialization<livox_ros_driver2::msg::CustomMsg> seri_livox_;

    std::string bag_file_;
};

}  // namespace lightning

#endif  // LIGHTNING_BAG_IO_H
