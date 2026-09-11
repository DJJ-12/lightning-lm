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
    const std::string value = Normalize(mode);
    if (value == "offline_mapping") return Mode::OFFLINE_MAPPING;
    if (value == "online_mapping") return Mode::ONLINE_MAPPING;
    if (value == "offline_localization") return Mode::OFFLINE_LOCALIZATION;
    if (value == "online_localization") return Mode::ONLINE_LOCALIZATION;
    return Mode::IDLE;
}

std::string ModeToString(Mode mode) {
    switch (mode) {
        case Mode::OFFLINE_MAPPING:
            return "offline_mapping";
        case Mode::ONLINE_MAPPING:
            return "online_mapping";
        case Mode::OFFLINE_LOCALIZATION:
            return "offline_localization";
        case Mode::ONLINE_LOCALIZATION:
            return "online_localization";
        case Mode::IDLE:
        default:
            return "idle";
    }
}

bool IsKnownModeName(const std::string& mode) {
    const std::string value = Normalize(mode);
    return value == "idle" || value == "offline_mapping" ||
           value == "online_mapping" || value == "offline_localization" ||
           value == "online_localization";
}

bool IsMappingMode(Mode mode) {
    return mode == Mode::OFFLINE_MAPPING || mode == Mode::ONLINE_MAPPING;
}

bool IsLocalizationMode(Mode mode) {
    return mode == Mode::OFFLINE_LOCALIZATION ||
           mode == Mode::ONLINE_LOCALIZATION;
}

}  // namespace lightning::runtime
