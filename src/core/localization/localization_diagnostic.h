#pragma once

#include <cstdint>

namespace lightning::loc {

enum class LocalizationFrameOutcome : std::uint8_t {
    SYSTEM_NOT_READY = 0,
    MAP_NOT_LOADED,
    EMPTY_AFTER_CONVERT,
    WAITING_INITIAL_POSE,
    INITIALIZED_WITH_FRAME,
    INITIALIZATION_IN_PROGRESS,
    EMPTY_AFTER_VOXEL,
    STATE_NOT_READY,
    NDT_EXECUTED
};

struct LocalizationInputDiagnostic {
    std::uint64_t pipeline_sequence = 0;
    std::uint64_t topic_sequence = 0;
    bool online = false;
    double topic_receive_steady_sec = 0.0;
    double worker_begin_steady_sec = 0.0;
    double header_stamp = 0.0;
};

inline const char* LocalizationFrameOutcomeName(LocalizationFrameOutcome outcome) {
    switch (outcome) {
        case LocalizationFrameOutcome::SYSTEM_NOT_READY:
            return "system_not_ready";
        case LocalizationFrameOutcome::MAP_NOT_LOADED:
            return "map_not_loaded";
        case LocalizationFrameOutcome::EMPTY_AFTER_CONVERT:
            return "empty_after_convert";
        case LocalizationFrameOutcome::WAITING_INITIAL_POSE:
            return "waiting_initial_pose";
        case LocalizationFrameOutcome::INITIALIZED_WITH_FRAME:
            return "initialized_with_frame";
        case LocalizationFrameOutcome::INITIALIZATION_IN_PROGRESS:
            return "initialization_in_progress";
        case LocalizationFrameOutcome::EMPTY_AFTER_VOXEL:
            return "empty_after_voxel";
        case LocalizationFrameOutcome::STATE_NOT_READY:
            return "state_not_ready";
        case LocalizationFrameOutcome::NDT_EXECUTED:
            return "ndt_executed";
    }
    return "unknown";
}

}  // namespace lightning::loc
