#ifndef OLC_GUARD_MLOPEN_OP_KERNEL_ARGS_HPP
#define OLC_GUARD_MLOPEN_OP_KERNEL_ARGS_HPP

#include <type_traits>
#include <cstdint>
#include <half.hpp>

#include <boost/container/small_vector.hpp>

namespace online_compile {

struct OpKernelArg
{

    OpKernelArg(char val, size_t sz) : buffer(sz) { std::fill(buffer.begin(), buffer.end(), val); }

    template <typename T>
    OpKernelArg(T arg) : buffer(sizeof(T))
    {
        static_assert(std::is_trivial<T>{} || std::is_same<T, half_float::half>{},
                      "Only for trivial types");
        *(reinterpret_cast<T*>(buffer.data())) = arg;
    }

    template <typename T>
    OpKernelArg(T* arg) // NOLINT
        : buffer(sizeof(T*))
    {
        *(reinterpret_cast<T**>(buffer.data())) = arg;
        is_ptr                                  = true;
    }

    std::size_t size() const { return buffer.size(); };
    boost::container::small_vector<char, 8> buffer;
    bool is_ptr = false;
};

} // namespace online_compile

#endif
