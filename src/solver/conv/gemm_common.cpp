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

// Workaround for MLOpen issue 1430. Vega20 fails to access GPU memory
// larger than the return value of GetMaxMemoryAllocSize() of Vega10.
// Due to historical reasons, this W/A is applied to all targets.
// We are going to keep it as is until the new GEMM backend
// is used instead of rocBLAS. See also issue #2809.
#define WORKAROUND_MLOPEN_ISSUE_1430 1

// Double the limit for FP32, for GemmFwdRest and GemmBwdRest solvers only.
// See issues #2789 and #2808.
//
// IMPORTANT: The limit can only be increased for the "rest-of" kind GEMM solvers,
// since there are dependencies between the applicability of GEMM solvers.
// For example, expanding the applicability of GemmFwd1x1_0_1 will result
// in narrowing the applicability of GemmFwdRest. This side effect will
// lead to errors unless the databases are re-tuned.
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_WORKAROUND_ISSUE_2789)

namespace miopen {
namespace solver {
namespace conv {
namespace gemm {

std::size_t MaxMemAllocSz(Handle& h,
                          const miopen::conv::ProblemDescription& problem,
                          bool double_limit_for_fp32)
{
    const auto m = h.GetMaxMemoryAllocSize();
#if WORKAROUND_MLOPEN_ISSUE_1430
    auto limit = static_cast<std::size_t>(7287183769);
    if(!env::disabled(MIOPEN_WORKAROUND_ISSUE_2789))
    {
        if(problem.IsFp32() && double_limit_for_fp32)
            limit *= 2;
    }
    return std::min(m, limit);
#else
    return m;
#endif
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
