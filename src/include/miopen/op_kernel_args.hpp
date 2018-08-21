#pragma once

#include <type_traits>
#include <cstdint>

#include <boost/container/small_vector.hpp>
struct OpKernelArg
{

    OpKernelArg(char val, size_t sz)
    : buffer(sz)
    {
        std::fill(buffer.begin(), buffer.end(), val);
    }

    template <typename T>
    OpKernelArg(T arg)
    : buffer(sizeof(T))
    {
        static_assert(std::is_trivial<T>{}, "Only for trivial types");
        *(reinterpret_cast<T*>(buffer.data())) = arg;
    }

    template <typename T>
    OpKernelArg(T* arg) // NOLINT
    : buffer(sizeof(T))
    {
        static_assert(std::is_trivial<T>{}, "Only for trivial types");
        *(reinterpret_cast<T*>(buffer.data())) = arg;
        is_ptr = true;
    }

    std::size_t size() const { return buffer.size(); };
    boost::container::small_vector<char, 8> buffer;
    bool is_ptr = false;
};
