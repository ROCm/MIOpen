#pragma once

#include <type_traits>
#include <cstdint>

#include <boost/container/small_vector.hpp>
struct OpKernelArg
{
    OpKernelArg(char val, size_t sz)
    {
        for(size_t idx = 0; idx < sz; idx++)
        {
            buffer.push_back(val);
        }
    }
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
        is_ptr = true;
    }
    OpKernelArg(const OpKernelArg& other) : buffer(other.buffer), is_ptr(other.is_ptr){};
    std::size_t size() const { return buffer.size(); };
    boost::container::small_vector<char, 8> buffer;
    bool is_ptr = false;
};
