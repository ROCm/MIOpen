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

#include <miopen/config.h>
#include <miopen/solver/gemm_common.hpp>

#include <tuple> // std::ignore

/// This W/A disables all GEMM convolution solvers for xDLOPs
/// targets when MIOpenGEMM is used (OCL BE). More info at
/// https://github.com/ROCmSoftwarePlatform/MIOpen/issues/1315.
///
/// W/A affects ROCm releases starting from 4.5 and also
/// pre-5.0 Mainline HIP builds, e.g. 9148.
#define WORKAROUND_ISSUE_1315 (MIOPEN_USE_MIOPENGEMM && (HIP_PACKAGE_VERSION_FLAT >= 4004000000ULL))

namespace miopen {
namespace conv {
namespace solver {
namespace gemm {

bool IsWorkaroundIssue1315(const miopen::ExecutionContext& ctx)
{
#if WORKAROUND_ISSUE_1315
    const auto device = ctx.GetStream().GetTargetProperties().Name();
    return (device == "gfx908") || (device == "gfx90a");
#else
    std::ignore = ctx;
    return false;
#endif
}

} // namespace gemm
} // namespace solver
} // namespace conv
} // namespace miopen
