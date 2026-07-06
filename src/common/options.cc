#include <common/options.h>

namespace lightning {

namespace debug {

bool flg_exit = false;
bool flg_pause = false;
bool flg_next = false;
float play_speed = 10.0;

}  // namespace debug

namespace lo {
float lidar_time_interval = 0.1;
}  // namespace lo

namespace fasterlio {
int NUM_MAX_ITERATIONS = 8;
float ESTI_PLANE_THRESHOLD = 0.1;
}  // namespace fasterlio

namespace map {
std::string map_path = "";
Vec3d map_origin = Vec3d::Zero();
}  // namespace map

namespace ui {
float opacity = 0.2;
}  // namespace ui

}  // namespace lightning
