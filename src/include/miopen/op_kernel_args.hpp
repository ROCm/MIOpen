#pragma once

#include <type_traits>
#include <cstdint>

#include <boost/container/small_vector.hpp>
struct OpKernelArg
{
    template <typename T>
    OpKernelArg(T arg)
    {
        assert(std::is_fundamental<T>::value);
        char* chptr = reinterpret_cast<char*>(&arg);
        for(size_t idx = 0; idx < sizeof(T); idx++)
        {
            buffer.push_back(*(chptr + idx));
        }
    }
    template <typename T>
    OpKernelArg(T* arg)
    {
        auto intptr = reinterpret_cast<std::uintptr_t>(arg);
        char* chptr = reinterpret_cast<char*>(&intptr);
        for(size_t idx = 0; idx < sizeof(std::uintptr_t); idx++)
        {
            buffer.push_back(*(chptr + idx));
        }
    }
    OpKernelArg(const OpKernelArg& other) : buffer(other.buffer){};
    std::size_t size() { return buffer.size(); };
    boost::container::small_vector<char, 8> buffer;
};
