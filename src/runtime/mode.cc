#include "runtime/mode.h"

#include <algorithm>

namespace lightning::runtime {

namespace {
std::string Normalize(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}
}  // namespace

Mode ModeFromString(const std::string& mode) {
    const std::string v = Normalize(mode);
    if (v == "idle") {
        return Mode::IDLE;
    }
    if (v == "offline_mapping" || v == "offline") {
        return Mode::OFFLINE_MAPPING;
    }
    if (v == "online_mapping" || v == "mapping" || v == "online") {
        return Mode::ONLINE_MAPPING;
    }
    if (v == "localization" || v == "loc") {
        return Mode::LOCALIZATION;
    }
    return Mode::IDLE;
}

std::string ModeToString(Mode mode) {
    switch (mode) {
        case Mode::IDLE:
            return "idle";
        case Mode::OFFLINE_MAPPING:
            return "offline_mapping";
        case Mode::ONLINE_MAPPING:
            return "online_mapping";
        case Mode::LOCALIZATION:
            return "localization";
        default:
            return "idle";
    }
}

bool IsMappingMode(Mode mode) {
    return mode == Mode::OFFLINE_MAPPING || mode == Mode::ONLINE_MAPPING;
}

}  // namespace lightning::runtime
