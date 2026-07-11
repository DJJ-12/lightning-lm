#pragma once

#ifndef LIGHTNING_MODULE_OPTIONS_H
#define LIGHTNING_MODULE_OPTIONS_H

namespace lightning {

namespace lo {
extern float lidar_time_interval;
}  // namespace lo

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
