#pragma once

#include <cstdlib>
#include <map>
#include <string>

#include "utils/string_utils.h"

// #define NDEBUG 0

namespace space_llm {

class Logger {
public:
    enum Level {
        TRACE = 0,
        DEBUG = 10,
        INFO = 20,
        WARNING = 30,
        ERROR = 40
    };

    static Logger &getLogger() {
        thread_local Logger instance;
        return instance;
    }
    Logger(Logger const &) = delete;
    void operator=(Logger const &) = delete;

    template <typename... Args>
    void log(const Level level, const std::string format, const Args &... args) {
        if (level_ <= level) {
            std::string fmt = getPrefix(level) + format + "\n";
            FILE *out = level_ < WARNING ? stdout : stderr;
            std::string logstr = fmtstr(fmt, args...);
            fprintf(out, "%s", logstr.c_str());
        }
    }

    template <typename... Args>
    void log(const Level level, const int rank, const std::string format, const Args &... args) {
        if (level_ <= level) {
            std::string fmt = getPrefix(level, rank) + format + "\n";
            FILE *out = level_ < WARNING ? stdout : stderr;
            std::string logstr = fmtstr(fmt, args...);
            fprintf(out, "%s", logstr.c_str());
        }
    }

    void setLevel(const Level level) {
        level_ = level;
        log(INFO, "Set logger level by %s", getLevelName(level).c_str());
    }

    int getLevel() const {
        return level_;
    }

private:
    const std::string PREFIX = "[QK]";
    const std::map<const Level, const std::string> level_name_ = {
        {TRACE, "TRACE"},
        {DEBUG, "DEBUG"},
        {INFO, "INFO"},
        {WARNING, "WARNING"},
        {ERROR, "ERROR"}};

#ifndef NDEBUG
    const Level DEFAULT_LOG_LEVEL = DEBUG;
#else
    const Level DEFAULT_LOG_LEVEL = INFO;
#endif

    Level level_ = DEFAULT_LOG_LEVEL;

    Logger();

    inline const std::string getLevelName(const Level level) {
        return level_name_.at(level);
    }

    inline const std::string getPrefix(const Level level) {
        return PREFIX + "[" + getLevelName(level) + "] ";
    }

    inline const std::string getPrefix(const Level level, const int rank) {
        return PREFIX + "[" + getLevelName(level) + "][" + std::to_string(rank) + "] ";
    }
};

#define QK_LOG(level, ...)                                          \
    do {                                                            \
        if (space_llm::Logger::getLogger().getLevel() <= level) {   \
            space_llm::Logger::getLogger().log(level, __VA_ARGS__); \
        }                                                           \
    } while (0)

#define QK_LOG_TRACE(...) QK_LOG(space_llm::Logger::TRACE, __VA_ARGS__)
#define QK_LOG_DEBUG(...) QK_LOG(space_llm::Logger::DEBUG, __VA_ARGS__)
#define QK_LOG_INFO(...) QK_LOG(space_llm::Logger::INFO, __VA_ARGS__)
#define QK_LOG_WARNING(...) QK_LOG(space_llm::Logger::WARNING, __VA_ARGS__)
#define QK_LOG_ERROR(...) QK_LOG(space_llm::Logger::ERROR, __VA_ARGS__)

} // namespace space_llm
