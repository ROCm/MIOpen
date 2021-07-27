#ifndef CK_MAGIC_DIVISION_HPP
#define CK_MAGIC_DIVISION_HPP

#include "config.hpp"
#include "integral_constant.hpp"
#include "number.hpp"
#include "type.hpp"
#include "tuple.hpp"

namespace ck {

// magic number division
// Caution:
//   1. For uint32_t as dividend: magic number division implementation being used would produce
//   correct result if the dividend is uint32_t and its value is within 31-bit value range.
//   2. For int32_t as dividendd: magic number division for int32_t dividened has not been
//   implemented, the int32_t dividend would be bit-wise interpreted as uint32_t and magic number
//   division implementation for uint32_t is then used. Therefore, dividend value need to be
//   non-negative.
// TODO:
//   1. Implement magic number divison for int32_t
//   2. Implement magic number divison for unit32_t with 32-bit value range
struct MagicDivision
{
    // uint32_t
    __host__ __device__ static constexpr auto CalculateMagicNumbers(uint32_t divisor)
    {
        // assert(divisior >= 1 && divisior <= INT32_MAX);
        uint32_t shift = 0;
        for(shift = 0; shift < 32; ++shift)
        {
            if((1U << shift) >= divisor)
            {
                break;
            }
        }

        uint64_t one        = 1;
        uint64_t multiplier = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
        // assert(multiplier <= 0xffffffffUL);

        return make_tuple(uint32_t(multiplier), shift);
    }

    __host__ __device__ static constexpr uint32_t CalculateMagicMultiplier(uint32_t divisor)
    {
        auto tmp = CalculateMagicNumbers(divisor);

        return tmp[Number<0>{}];
    }

    __host__ __device__ static constexpr uint32_t CalculateMagicShift(uint32_t divisor)
    {
        auto tmp = CalculateMagicNumbers(divisor);

        return tmp[Number<1>{}];
    }

    // integral_constant<uint32_t, .>
    template <uint32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicNumbers(integral_constant<uint32_t, Divisor>)
    {
        constexpr auto tmp = CalculateMagicNumbers(uint32_t{Divisor});

        constexpr uint32_t multiplier = tmp[Number<0>{}];
        constexpr uint32_t shift      = tmp[Number<1>{}];

        return make_tuple(integral_constant<uint32_t, multiplier>{},
                          integral_constant<uint32_t, shift>{});
    }

    template <uint32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicMultiplier(integral_constant<uint32_t, Divisor>)
    {
        constexpr uint32_t multiplier = CalculateMagicMultiplier(uint32_t{Divisor});

        return integral_constant<uint32_t, multiplier>{};
    }

    template <uint32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicShift(integral_constant<uint32_t, Divisor>)
    {
        constexpr uint32_t shift = CalculateMagicShift(uint32_t{Divisor});

        return integral_constant<uint32_t, shift>{};
    }

    // integral_constant<int32_t, .>
    template <int32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicNumbers(integral_constant<int32_t, Divisor>)
    {
        return CalculateMagicNumbers(integral_constant<uint32_t, Divisor>{});
    }

    template <int32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicMultiplier(integral_constant<int32_t, Divisor>)
    {
        return CalculateMagicMultiplier(integral_constant<uint32_t, Divisor>{});
    }

    template <int32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicShift(integral_constant<int32_t, Divisor>)
    {
        return CalculateMagicShift(integral_constant<uint32_t, Divisor>{});
    }

    // magic division for uint32_t
    __host__ __device__ static constexpr uint32_t
    DoMagicDivision(uint32_t dividend, uint32_t multiplier, uint32_t shift)
    {
        uint32_t tmp = (uint64_t(dividend) * uint64_t(multiplier)) >> 32;
        return (tmp + dividend) >> shift;
    }

#if 1 // debug
    // HACK: magic division for int32_t
    // HACK: use dividend_i32 as if it's uint32_t, dividend_i32 need to be
    // non-negative for result to be correct
    // TODO: figure out how to do magic number divison for int32_t as dividended
    __host__ __device__ static constexpr int32_t
    DoMagicDivision(int32_t dividend_i32, uint32_t multiplier, uint32_t shift)
    {
        uint32_t dividend_u32 = as_type<uint32_t>(dividend_i32);
        uint32_t tmp =
            (static_cast<uint64_t>(dividend_u32) * static_cast<uint64_t>(multiplier)) >> 32;
        return (tmp + dividend_u32) >> shift;
    }
#else
    // the inline ASM is producing wrong result
    __host__ __device__ static int32_t
    DoMagicDivision(int32_t dividend_i32, uint32_t multiplier, uint32_t shift)
    {
        uint32_t r;
        asm volatile("\n \
                v_mul_hi_u32 %0, %1, %2 \n \
                v_add_u32_e32 %0, %1, %0 \n \
                v_lshrrev_b32_e32 %0, %3, %0 \n \
                "
                     : "=v"(r)
                     : "v"(as_type<uint32_t>(dividend_i32)), "s"(multiplier), "s"(shift));

        return as_type<int32_t>(r);
    }
#endif
};

} // namespace ck

#endif
