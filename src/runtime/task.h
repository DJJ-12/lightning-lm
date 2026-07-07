#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace lightning::runtime {

enum class TaskState {
    IDLE = 0,
    READY,
    WAIT_CLOUD,
    RUNNING,
    SAVING,
    FINISHED,
    FAILED,
    CANCELLED
};

std::string TaskStateToString(TaskState state);

struct TaskSnapshot {
    TaskState state = TaskState::IDLE;
    bool running = false;
    bool finished = false;
    bool success = false;
    std::uint64_t total_frames = 0;
    std::uint64_t processed_frames = 0;
    float progress = 0.0f;
    std::string message;
};

class Task {
   public:
    void Reset(TaskState state = TaskState::IDLE, const std::string& message = "");
    void SetState(TaskState state, const std::string& message = "");
    void SetProgress(std::uint64_t processed, std::uint64_t total, const std::string& message = "");
    void SetFinished(bool success, const std::string& message);
    void RequestCancel();
    bool CancelRequested() const;
    TaskSnapshot Snapshot() const;
    TaskState State() const;

   private:
    mutable std::mutex mutex_;
    TaskSnapshot snapshot_;
    std::atomic_bool cancel_requested_{false};
};

}  // namespace lightning::runtime
