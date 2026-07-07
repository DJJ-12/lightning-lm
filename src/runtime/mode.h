#pragma once

#include <string>

namespace lightning::runtime {

enum class Mode {
    IDLE = 0,
    OFFLINE_MAPPING,
    ONLINE_MAPPING,
    LOCALIZATION
};

Mode ModeFromString(const std::string& mode);
std::string ModeToString(Mode mode);
bool IsMappingMode(Mode mode);

}  // namespace lightning::runtime
