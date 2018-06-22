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
        auto chptr = reinterpret_cast<char*>(&arg);
        for(size_t idx = 0; idx < sizeof(T); idx++)
        {
            buffer.push_back(*(chptr + idx));
        }
    }
    template <typename T>
    OpKernelArg(T* arg)
    {
        auto intptr = reinterpret_cast<std::uintptr_t>(arg);
        auto chptr  = reinterpret_cast<char*>(&intptr);
        for(size_t idx = 0; idx < sizeof(std::uintptr_t); idx++)
        {
            buffer.push_back(*(chptr + idx));
        }
    }
    OpKernelArg(const OpKernelArg& other) : buffer(other.buffer){};
    OpKernelArg& operator=(const OpKernelArg& other)
    { 
        if(buffer.size() < other.buffer.size())
        {
            buffer.resize(other.buffer.size());
        }
        std::copy(other.buffer.begin(), other.buffer.end(), buffer.begin());
        return *this;
    }
    std::size_t size() { return buffer.size(); };
    boost::container::small_vector<char, 8> buffer;
};
