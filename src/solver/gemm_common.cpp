/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <miopen/env.hpp>
#include <miopen/solver/gemm_common.hpp>

#include <algorithm>
#include <limits>

// Vega20 fails to access GPU memory larger than the
// return value of GetMaxMemoryAllocSize() of Vega10.
#define WORKAROUND_MLOPEN_ISSUE_1430 1

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_WORKAROUND_ISSUE_2808)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_WORKAROUND_ISSUE_2809)

namespace miopen {
namespace solver {
namespace conv {
namespace gemm {

std::size_t MaxMemAllocSz(Handle& h, const miopen::conv::ProblemDescription& problem)
{
    constexpr auto wa_2808_16bit_limit    = static_cast<std::size_t>(7287183769);
    constexpr auto wa_2808_other_limit    = wa_2808_16bit_limit * 2;
    constexpr auto wa_2809_fp32_wrw_limit = wa_2808_16bit_limit;

    const auto m = h.GetMaxMemoryAllocSize();
    auto limit   = std::numeric_limits<std::size_t>::max();

#if WORKAROUND_MLOPEN_ISSUE_1430
    const auto device = h.GetTargetProperties().Name();
    if(device == "gfx906" || device == "gfx900")
    {
        // Workaround for MLOpen issue 1430. Vega20 fails to access GPU memory
        // larger than the return value of GetMaxMemoryAllocSize() of Vega10.
        limit = std::min(limit, static_cast<std::size_t>(7287183769));
    }
#endif
    if(!miopen::IsDisabled(ENV(MIOPEN_WORKAROUND_ISSUE_2808)))
    {
        if(problem.IsFp16() || problem.IsBfp16())
            limit = std::min(limit, wa_2808_16bit_limit);
        else
            limit = std::min(limit, wa_2808_other_limit);
    }
    if(!miopen::IsDisabled(ENV(MIOPEN_WORKAROUND_ISSUE_2809)))
    {
        if(problem.IsFp32() && problem.IsDirectionBackwardWrW())
            limit = std::min(limit, wa_2809_fp32_wrw_limit);
    }
    return std::min(m, limit);
}

bool IsAnyBufferBf16(const TensorDescriptor& xDesc,
                     const TensorDescriptor& yDesc,
                     const TensorDescriptor& wDesc)
{
    return xDesc.GetType() == miopenBFloat16    //
           || yDesc.GetType() == miopenBFloat16 //
           || wDesc.GetType() == miopenBFloat16;
}

bool IsAnyBufferFp16(const TensorDescriptor& xDesc,
                     const TensorDescriptor& yDesc,
                     const TensorDescriptor& wDesc)
{
    return xDesc.GetType() == miopenHalf    //
           || yDesc.GetType() == miopenHalf //
           || wDesc.GetType() == miopenHalf;
}

double SlowdownFactor(const int n_oper, const double oper_factor, const double multiple_oper_factor)
{
    if(n_oper > 0)
    {
        auto rv = oper_factor;
        if(n_oper > 1)
            rv *= multiple_oper_factor;
        return rv;
    }
    else
        return 1.0;
}

} // namespace gemm
} // namespace conv
} // namespace solver
} // namespace miopen
