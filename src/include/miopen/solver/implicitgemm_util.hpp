/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef GUARD_IMPLICITGEMM_UTIL_HPP_
#define GUARD_IMPLICITGEMM_UTIL_HPP_

#include <miopen/conv/problem_description.hpp>
#include <miopen/env.hpp>
#include <miopen/execution_context.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM)

namespace miopen {
namespace solver {

/// \todo move to separate header and use in other solvers.
template <int L, int H>
inline static bool IsTwoPower(const int v)
{
    static_assert(L <= H, "L <= H");
    if(((v - 1) & v) != 0)
        return false;
    return L <= v && v <= H;
}

template <int L, int H>
inline static bool NextTwoPower(int& v)
{
    static_assert((((L - 1) & L) == 0), "L is not power of 2");
    static_assert((((H - 1) & H) == 0), "H is not power of 2");
    assert((IsTwoPower<L, H>(v)));
    if(v == H)
    {
        v = L;
        return true;
    }
    v *= 2;
    return false;
}

template <int L, int H>
inline static bool PreviousTwoPower(int& v)
{
    static_assert((((L - 1) & L) == 0), "L is not power of 2");
    static_assert((((H - 1) & H) == 0), "H is not power of 2");
    assert((IsTwoPower<L, H>(v)));
    if(v == L)
    {
        v = H;
        return true;
    }
    v /= 2;
    return false;
}

template <bool L, bool H>
inline static bool NextFlag(bool& v)
{
    if(v == H)
    {
        v = L;
        return true;
    }
    v = H;
    return false;
}

static inline bool IsXdlopsSupport(const ExecutionContext& ctx)
{
    if(env::enabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE))
        return true;

    // disable xdlops kernels by default due to possible failures:
    // 1) inline asm may crash
    // 2) llvm intrin may has incorrect results
    const bool is_xdlops_supported = ctx.GetStream().GetDeviceName() == "gfx908" ||
                                     ctx.GetStream().GetDeviceName() == "gfx90a" ||
                                     ctx.GetStream().GetDeviceName() == "gfx942" ||
                                     ctx.GetStream().GetDeviceName() == "gfx950";
    return is_xdlops_supported && !env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS);
}

///\todo remove
inline static uint32_t GetReadWriteVectorSize(const int v)
{
    return v % 4 == 0 ? 4 : (v % 2 == 0 ? 2 : 1);
}

///\todo remove
inline static uint32_t GetEPackLength(const ExecutionContext& ctx,
                                      const miopen::conv::ProblemDescription& problem,
                                      bool isXdlopsInvoked)
{
    // Based on data type, Es are packed
    int EPACK = 1;
    if(problem.IsFp16()) // for fp16, either 2 or 4 Es could be packed
    {
        if(IsXdlopsSupport(ctx) && isXdlopsInvoked)
        {
            // in xdlops, 4 fp16s are packed
            EPACK = 4;
        }
        else
        {
            // for fp16, either 2 or 4 Es could be packed in non-xdlops scenarios.
            // EPACK = (C * Y * X % 32) == 0 ? 4 : 2;
            EPACK = 2;
        }
    }
    else if(problem.IsBfp16()) // for bfp16, only 2 Es could be packed
    {
        EPACK = 2;
    }
    return EPACK;
}

///\todo remove
static inline size_t ComputeLDSRequiredSize(const miopen::conv::ProblemDescription& problem,
                                            const int BPerBlock,
                                            const int KPerBlock,
                                            const int EPerBlock,
                                            const unsigned int GemmDataPerReadA,
                                            const unsigned int GemmDataPerReadB,
                                            const unsigned int InBlockCopySubLengths_B,
                                            const unsigned int WeiBlockCopySubLengths_K,
                                            const unsigned int EPACKSize)
{
    // Extend lds size by to take into account alignment
    // See max_algin code inside kernel_aglorithm files
    const std::size_t worst_case_alignment_adjustment =
        (problem.IsBfp16() || problem.IsFp16())
            ? std::max(
                  {GetReadWriteVectorSize(static_cast<int>(InBlockCopySubLengths_B)), EPACKSize})
            : std::max({GetReadWriteVectorSize(static_cast<int>(WeiBlockCopySubLengths_K)),
                        GetReadWriteVectorSize(static_cast<int>(InBlockCopySubLengths_B)),
                        GemmDataPerReadA,
                        GemmDataPerReadB});

    // Multiplied worst_case_alignment_adjustment by 2 as
    // Both A and B matrix LDS size is increased.
    const std::size_t lds_size = (static_cast<std::size_t>(BPerBlock) + KPerBlock) * EPerBlock *
                                     EPACKSize * GetTypeSize(problem.GetInDataType()) * 2 +
                                 2 * static_cast<std::size_t>(worst_case_alignment_adjustment);

    return lds_size;
}

template <typename T>
inline T igemm_get_max_gks(T gemm_k, T gemm_k_per_block, T max_log2_splits)
{
    if(gemm_k % gemm_k_per_block != 0)
        return 0;
    T rem      = gemm_k / gemm_k_per_block;
    T rem_pow2 = rem & (~(rem - 1));
    T gks      = (T)log2(rem_pow2);

    if(gks > max_log2_splits)
        gks = max_log2_splits;
    return gks;
}

// greatest common divisor, aka highest common factor
template <typename T>
T gcd(T x, T y)
{
    assert(!(x == 0 && y == 0));

    if(x < 0 || y < 0)
    {
        return gcd(abs(x), abs(y));
    }
    else if(x == y || x == 0)
    {
        return y;
    }
    else if(y == 0)
    {
        return x;
    }
    else if(x > y)
    {
        return gcd(x % y, y); // NOLINT
    }
    else
    {
        return gcd(x, y % x);
    }
}

template <typename T, typename... Ys>
T gcd(T x, Ys... ys)
{
    return gcd(x, gcd(ys...));
}

// least common multiple
template <typename T>
T lcm(T x, T y)
{
    if(x == 0 || y == 0)
    {
        return 0;
    }
    else
    {
        return (x * y) / gcd(x, y);
    }
}

template <typename T, typename... Ys>
T lcm(T x, Ys... ys)
{
    return lcm(x, lcm(ys...));
}

template <typename T>
T integer_divide_ceil(T x, T y)
{
    if(y == 0)
    {
        MIOPEN_THROW("divisor should not be 0");
    }

    return (x + y - 1) / y;
}

template <typename T>
T integer_least_multiple(T x, T y)
{
    return y * integer_divide_ceil(x, y);
}

} // namespace solver
} // namespace miopen

#endif
