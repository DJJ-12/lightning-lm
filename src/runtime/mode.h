#pragma once

#include <string>

namespace lightning::runtime {

enum class Mode {
    IDLE = 0,
    OFFLINE_MAPPING,
    ONLINE_MAPPING,
    OFFLINE_LOCALIZATION,
    ONLINE_LOCALIZATION
};

Mode ModeFromString(const std::string& mode);
std::string ModeToString(Mode mode);
bool IsKnownModeName(const std::string& mode);
bool IsMappingMode(Mode mode);
bool IsLocalizationMode(Mode mode);

}  // namespace lightning::runtime
