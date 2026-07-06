#pragma once

#ifndef LIGHTNING_MODULE_OPTIONS_H
#define LIGHTNING_MODULE_OPTIONS_H

#include <string>

#include <common/eigen_types.h>

#include <rclcpp/rclcpp.hpp>

namespace lightning {

namespace debug {

extern bool flg_exit;
extern bool flg_pause;
extern bool flg_next;
extern float play_speed;

inline void SigHandle(int sig) {
    (void)sig;
    debug::flg_exit = true;
    rclcpp::shutdown();
}

}  // namespace debug

namespace lo {
extern float lidar_time_interval;
}  // namespace lo

namespace map {
extern std::string map_path;
extern Vec3d map_origin;
}  // namespace map

namespace ui {
extern float opacity;
}  // namespace ui

namespace fasterlio {

constexpr double INIT_TIME = 0.1;
constexpr int NUM_MATCH_POINTS = 5;
constexpr int MIN_NUM_MATCH_POINTS = 3;

extern int NUM_MAX_ITERATIONS;
extern float ESTI_PLANE_THRESHOLD;

}  // namespace fasterlio

}  // namespace lightning

#endif
