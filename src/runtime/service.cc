#include "runtime/service.h"

#include <Eigen/Geometry>
#include <algorithm>
#include <utility>

namespace lightning::runtime {

namespace {
SE3 PoseFromRequest(const lightning_interfaces::srv::SetLocation::Request& request) {
    constexpr double kDegToRad = 0.017453292519943295;
    Eigen::AngleAxisd roll_angle(request.roll * kDegToRad, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch_angle(request.pitch * kDegToRad, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw_angle(request.yaw * kDegToRad, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q(yaw_angle * pitch_angle * roll_angle);
    Eigen::Vector3d t(request.x, request.y, request.z);
    return SE3(q, t);
}
}  // namespace

bool Service::Init(rclcpp::Node::SharedPtr node, std::shared_ptr<Lightning> lightning) {
    node_ = node;
    lightning_ = std::move(lightning);
    if (!node_ || !lightning_) {
        return false;
    }

    set_mode_srv_ = node_->create_service<lightning_interfaces::srv::SetMode>(
        "/lightning/set_mode",
        [this](const lightning_interfaces::srv::SetMode::Request::SharedPtr request,
               lightning_interfaces::srv::SetMode::Response::SharedPtr response) {
            auto result = lightning_->SetMode(request->mode);
            const auto status = lightning_->GetStatus();
            response->success = result.success;
            response->current_mode = ModeToString(lightning_->CurrentMode());
            response->task_state = TaskStateToString(status.state);
            response->message = result.message;
        });

    get_status_srv_ = node_->create_service<lightning_interfaces::srv::GetStatus>(
        "/lightning/get_status",
        [this](const lightning_interfaces::srv::GetStatus::Request::SharedPtr,
               lightning_interfaces::srv::GetStatus::Response::SharedPtr response) {
            const auto status = lightning_->GetStatus();
            response->success = true;
            response->mode = ModeToString(lightning_->CurrentMode());
            response->task_state = TaskStateToString(status.state);
            response->running = status.running;
            response->finished = status.finished;
            response->task_success = status.success;
            response->total_frames = status.total_frames;
            response->processed_frames = status.processed_frames;
            response->progress = status.progress;
            response->message = status.message;
        });

    cancel_task_srv_ = node_->create_service<lightning_interfaces::srv::CancelTask>(
        "/lightning/cancel_task",
        [this](const lightning_interfaces::srv::CancelTask::Request::SharedPtr,
               lightning_interfaces::srv::CancelTask::Response::SharedPtr response) {
            auto result = lightning_->CancelTask();
            const auto status = lightning_->GetStatus();
            response->success = result.success;
            response->mode = ModeToString(lightning_->CurrentMode());
            response->task_state = TaskStateToString(status.state);
            response->message = result.message;
        });

    get_offline_progress_srv_ = node_->create_service<lightning_interfaces::srv::GetOfflineMappingProgress>(
        "/lightning/mapping/get_offline_mapping_progress",
        [this](const lightning_interfaces::srv::GetOfflineMappingProgress::Request::SharedPtr,
               lightning_interfaces::srv::GetOfflineMappingProgress::Response::SharedPtr response) {
            const auto progress = lightning_->GetOfflineMappingProgress();
            response->running = progress.running;
            response->finished = progress.finished;
            response->success = progress.success;
            response->total_frames = progress.total_frames;
            response->processed_frames = progress.processed_frames;
            response->progress = progress.progress;
            response->message = progress.message;
        });

    start_mapping_srv_ = node_->create_service<lightning_interfaces::srv::StartMapping>(
        "/lightning/mapping/start_mapping",
        [this](const lightning_interfaces::srv::StartMapping::Request::SharedPtr request,
               lightning_interfaces::srv::StartMapping::Response::SharedPtr response) {
            auto result = lightning_->StartMapping(request->save_path);
            response->success = result.success;
            response->message = result.message;
        });

    load_mapping_bag_srv_ = node_->create_service<lightning_interfaces::srv::LoadBag>(
        "/lightning/mapping/load_bag",
        [this](const lightning_interfaces::srv::LoadBag::Request::SharedPtr request,
               lightning_interfaces::srv::LoadBag::Response::SharedPtr response) {
            auto result = lightning_->LoadBag(request->bag_path);
            response->success = result.success;
            response->message = result.message;
        });

    load_localization_bag_srv_ = node_->create_service<lightning_interfaces::srv::LoadBag>(
        "/lightning/localization/load_bag",
        [this](const lightning_interfaces::srv::LoadBag::Request::SharedPtr request,
               lightning_interfaces::srv::LoadBag::Response::SharedPtr response) {
            auto result = lightning_->LoadBag(request->bag_path);
            response->success = result.success;
            response->message = result.message;
        });

    set_localization_map_path_srv_ = node_->create_service<lightning_interfaces::srv::SetMapPath>(
        "/lightning/localization/set_map_path",
        [this](const lightning_interfaces::srv::SetMapPath::Request::SharedPtr request,
               lightning_interfaces::srv::SetMapPath::Response::SharedPtr response) {
            auto result = lightning_->SetMapPath(request->map_path);
            response->success = result.success;
            response->message = result.message;
        });

    finish_mapping_srv_ = node_->create_service<lightning_interfaces::srv::FinishMapping>(
        "/lightning/mapping/finish_mapping",
        [this](const lightning_interfaces::srv::FinishMapping::Request::SharedPtr request,
               lightning_interfaces::srv::FinishMapping::Response::SharedPtr response) {
            auto result = lightning_->FinishMapping(request->save_map);
            response->success = result.success;
            response->message = result.message;
        });

    finish_localization_srv_ = node_->create_service<lightning_interfaces::srv::FinishLocalization>(
        "/lightning/localization/finish_localization",
        [this](const lightning_interfaces::srv::FinishLocalization::Request::SharedPtr,
               lightning_interfaces::srv::FinishLocalization::Response::SharedPtr response) {
            auto result = lightning_->FinishLocalization();
            response->success = result.success;
            response->message = result.message;
        });

    get_mapping_map_path_srv_ = node_->create_service<lightning_interfaces::srv::GetMapPath>(
        "/lightning/mapping/get_map_path",
        [this](const lightning_interfaces::srv::GetMapPath::Request::SharedPtr,
               lightning_interfaces::srv::GetMapPath::Response::SharedPtr response) {
            std::string map_path;
            auto result = lightning_->GetMapPath(&map_path);
            response->success = result.success;
            response->map_path = map_path;
            response->message = result.message;
        });

    get_localization_map_path_srv_ = node_->create_service<lightning_interfaces::srv::GetMapPath>(
        "/lightning/localization/get_map_path",
        [this](const lightning_interfaces::srv::GetMapPath::Request::SharedPtr,
               lightning_interfaces::srv::GetMapPath::Response::SharedPtr response) {
            std::string map_path;
            auto result = lightning_->GetMapPath(&map_path);
            response->success = result.success;
            response->map_path = map_path;
            response->message = result.message;
        });

    set_location_srv_ = node_->create_service<lightning_interfaces::srv::SetLocation>(
        "/lightning/localization/set_location",
        [this](const lightning_interfaces::srv::SetLocation::Request::SharedPtr request,
               lightning_interfaces::srv::SetLocation::Response::SharedPtr response) {
            bool initialized_now = false;
            auto result = lightning_->SetLocation(PoseFromRequest(*request), &initialized_now);
            response->success = result.success;
            response->initialized = initialized_now;
            response->message = result.message;
        });

    get_localization_quality_srv_ = node_->create_service<lightning_interfaces::srv::GetLocalizationQuality>(
        "/lightning/localization/get_localization_quality",
        [this](const lightning_interfaces::srv::GetLocalizationQuality::Request::SharedPtr,
               lightning_interfaces::srv::GetLocalizationQuality::Response::SharedPtr response) {
            const auto result = lightning_->GetLocalizationQuality();
            response->valid = result.valid_;
            response->reliable = result.reliable_;
            response->status = static_cast<uint8_t>(result.status_);
            response->confidence = result.confidence_;

            const Vec3d t = result.pose_.translation();
            const Mat3d R = result.pose_.unit_quaternion().toRotationMatrix();
            const Eigen::Vector3d rpy = R.eulerAngles(0, 1, 2);
            response->x = t.x();
            response->y = t.y();
            response->z = t.z();
            response->roll = rpy.x();
            response->pitch = rpy.y();
            response->yaw = rpy.z();
            response->tp = result.tp_;
            response->nvtl = result.nvtl_;
            response->iterations = static_cast<uint32_t>(std::max(0, result.iterations_));
            response->message = result.valid_ ? result.message_ : "no localization result yet";
        });

    return true;
}

}  // namespace lightning::runtime
