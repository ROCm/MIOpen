#pragma once

#include <exception>
#include <iostream>
#include <string>

namespace fin {

struct Exception : std::exception
{
    std::string message;
    Exception(const std::string& msg = "") : message(msg) {}

    Exception SetContext(const std::string& file, int line)
    {
        message = file + ":" + std::to_string(line) + ": " + message;
        return *this;
    }

    const char* what() const noexcept override { return message.c_str(); }
};
} // namespace fin
#define FIN_THROW(...)                                                    \
    do                                                                    \
    {                                                                     \
        throw fin::Exception(__VA_ARGS__).SetContext(__FILE__, __LINE__); \
    } while(false)
