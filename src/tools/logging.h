#ifndef LOGGING_HEADER
#define LOGGING_HEADER

#include <format>

#define LOGGING_ENABLED

#ifdef LOGGING_ENABLED

#define LOG_INFO(str, ...) std::cout << "[Log info message] : \"" << std::format(str, __VA_ARGS__) << "\"\n (" << "File - " << __FILE__ << ", Line - " << __LINE__ << ')' << std::endl;
#define LOG_WARNING(str, ...) std::cout << "[Log warning message] : \"" << std::format(str, __VA_ARGS__) << "\"\n (" << "File - " << __FILE__ << ", Line - " << __LINE__ << ')' << std::endl;
#define LOG_IF_INFO(cond, str, ...) if (cond) {std::cout << "[Log info message] : \"" << std::format(str, __VA_ARGS__) << "\"\n (" << "File - " << __FILE__ << ", Line - " << __LINE__ << ')' << std::endl;}
#define LOG_IF_WARNING(cond, str, ...) if (cond) {std::cout << "[Log warning message] : \"" << std::format(str, __VA_ARGS__) << "\"\n (" << "File - " << __FILE__ << ", Line - " << __LINE__ << ')' << std::endl;}

#else

#define LOG_INFO(str, ...)
#define LOG_WARNING()
#define LOG_IF_INFO()
#define LOG_IF_WARNING()

#endif

/*
LOG_FILE_INFO
LOG_FILE_WARNING
*/

#endif