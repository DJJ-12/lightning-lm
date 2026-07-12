#include "runtime/mode.h"

#include <algorithm>
#include <cctype>

namespace lightning::runtime {

namespace {
std::string Normalize(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c);
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c);
    }).base();
    if (first >= last) {
        return "";
    }
    value = std::string(first, last);
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
    if (v == "offline_mapping") {
        return Mode::OFFLINE_MAPPING;
    }
    if (v == "online_mapping") {
        return Mode::ONLINE_MAPPING;
    }
    if (v == "offline_localization") {
        return Mode::OFFLINE_LOCALIZATION;
    }
    if (v == "online_localization") {
        return Mode::ONLINE_LOCALIZATION;
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
        case Mode::OFFLINE_LOCALIZATION:
            return "offline_localization";
        case Mode::ONLINE_LOCALIZATION:
            return "online_localization";
        default:
            return "idle";
    }
}

bool IsMappingMode(Mode mode) {
    return mode == Mode::OFFLINE_MAPPING || mode == Mode::ONLINE_MAPPING;
}

bool IsLocalizationMode(Mode mode) {
    return mode == Mode::OFFLINE_LOCALIZATION || mode == Mode::ONLINE_LOCALIZATION;
}

bool IsKnownModeName(const std::string& mode) {
    const std::string v = Normalize(mode);
    return v == "idle" || v == "offline_mapping" ||
           v == "online_mapping" || v == "offline_localization" ||
           v == "online_localization";
}

}  // namespace lightning::runtime
