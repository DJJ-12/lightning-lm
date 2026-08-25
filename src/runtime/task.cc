#include "runtime/task.h"

#include <algorithm>

namespace lightning::runtime {

std::string TaskStateToString(TaskState state) {
    switch (state) {
        case TaskState::IDLE:
            return "idle";
        case TaskState::READY:
            return "ready";
        case TaskState::WAIT_CLOUD:
            return "wait_cloud";
        case TaskState::RUNNING:
            return "running";
        case TaskState::SAVING:
            return "saving";
        case TaskState::FINISHED:
            return "finished";
        case TaskState::FAILED:
            return "failed";
        case TaskState::CANCELLED:
            return "cancelled";
        default:
            return "unknown";
    }
}

void Task::Reset(TaskState state, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_ = TaskSnapshot();
    snapshot_.state = state;
    snapshot_.running = state == TaskState::RUNNING || state == TaskState::SAVING;
    snapshot_.finished = state == TaskState::FINISHED;
    snapshot_.message = message;
    cancel_requested_ = false;
}

void Task::SetState(TaskState state, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.state = state;
    snapshot_.running = state == TaskState::RUNNING || state == TaskState::SAVING;
    snapshot_.finished = state == TaskState::FINISHED || state == TaskState::FAILED || state == TaskState::CANCELLED;
    if (state == TaskState::FAILED || state == TaskState::CANCELLED) {
        snapshot_.task_success = false;
    }
    if (!message.empty()) {
        snapshot_.message = message;
    }
}

void Task::SetProgress(std::uint64_t processed, std::uint64_t total, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.processed_frames = processed;
    snapshot_.total_frames = total;
    if (total > 0) {
        snapshot_.progress = std::min(100.0f, static_cast<float>(processed) * 100.0f / static_cast<float>(total));
    } else {
        snapshot_.progress = 0.0f;
    }
    if (!message.empty()) {
        snapshot_.message = message;
    }
}

void Task::SetFinished(bool task_success, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.task_success = task_success;
    snapshot_.running = false;
    snapshot_.finished = true;
    snapshot_.state = task_success ? TaskState::FINISHED : TaskState::FAILED;
    snapshot_.progress = task_success ? 100.0f : snapshot_.progress;
    snapshot_.message = message;
}

void Task::RequestCancel() {
    cancel_requested_ = true;
    SetState(TaskState::CANCELLED, "cancel requested");
}

bool Task::CancelRequested() const {
    return cancel_requested_.load();
}

TaskSnapshot Task::Snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
}

TaskState Task::State() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_.state;
}

}  // namespace lightning::runtime
