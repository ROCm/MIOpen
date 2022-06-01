#ifndef MIOPEN_GUARD_MLOPEN_OP_KERNEL_ARGS_HPP
#define MIOPEN_GUARD_MLOPEN_OP_KERNEL_ARGS_HPP

#include <miopen/config.h> /// WORKAROUND_BOOST_ISSUE_392

/// The puzzling problem: WORKAROUND_BOOST_ISSUE_392 does not help
/// with ROCm 5.1.x: even with config.h is included, build fails
/// and complains the same thing about noinline.
/// Therefore we avoid use of boost in this file for 5.1.x.
/// NOTE: This W/A should be removed altogether with WORKAROUND_BOOST_ISSUE_392.
#define WORKAROUND_BOOST_ISSUE_392_AVOID_BOOST (HIP_PACKAGE_VERSION_MAJOR == 5 && HIP_PACKAGE_VERSION_MINOR == 1)

#include <type_traits>
#include <cstdint>
#include <half.hpp>

#if WORKAROUND_BOOST_ISSUE_392_AVOID_BOOST
#include <vector>
#else
#include <boost/container/small_vector.hpp>
#endif

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
#if WORKAROUND_BOOST_ISSUE_392_AVOID_BOOST
    std::vector<char> buffer;
#else
    boost::container::small_vector<char, 8> buffer;
#endif
    bool is_ptr = false;
};

#endif
