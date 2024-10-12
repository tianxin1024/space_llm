#include "utils/logger.h"
#include <cuda_runtime.h>

namespace space_llm {

Logger::Logger() {
    char *is_first_rank_only_char = std::getenv("QK_LOG_FIRST_RANK_ONLY");
    bool is_first_rank_only =
        (is_first_rank_only_char != nullptr && std::string(is_first_rank_only_char) == "ON") ? true : false;

    int device_id;
    cudaGetDevice(&device_id);

    char *level_name = std::getenv("QK_LOG_LEVEL");
    if (level_name != nullptr) {
        std::map<std::string, Level> name_to_level = {
            {"TRACE", TRACE},
            {"DEBUG", DEBUG},
            {"INFO", INFO},
            {"WARNING", WARNING},
            {"ERROR", ERROR},
        };
        auto level = name_to_level.find(level_name);
        // If QK_LOG_FIRST_RANK_ONLY=ON, set LOG LEVEL of other device to ERROR
        if (is_first_rank_only && device_id != 0) {
            level = name_to_level.find("ERROR");
        }
        if (level != name_to_level.end()) {
            setLevel(level->second);
        } else {
            fprintf(stderr,
                    "[QK][WARNING] Invalid logger level QK_LOG_LEVEL=%s. "
                    "Ignore the environment variable and use a default "
                    "logging level.\n",
                    level_name);
            level_name = nullptr;
        }
    }
}

} // namespace space_llm
