/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2026 Advanced Micro Devices, Inc.
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

#include <miopen/solver/conv_direct_naive_conv.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool ConvDirectNaiveConvFwd::IsApplicable(const ExecutionContext& ctx,
                                          const ProblemDescription& problem) const
{
    if(!miopen::debug::AlwaysEnableConvDirectNaive)
    {
        if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD))
            return false;
        if(!ctx.use_hip_kernels)
            return false;
    }

    if(!ConvDirectNaiveConvIsApplicableByKernelType(ctx, problem))
        return false;

    if(!problem.IsLayoutDefault() && !problem.IsLayoutNHWC())
        return false;

    if(!(problem.IsFp32() || problem.IsFp16() || problem.IsBfp16() || problem.IsInt8() ||
         problem.IsFp8() || problem.IsBfp8()))
        return false;

    if(!problem.IsDirectionForward())
        return false;

    if(!problem.AllTensorsLengthsFitIntoInt())
        return false;

    // The naive convolution kernels are launched with a 1-D grid whose global work
    // size is grid_size * block_size, where (forward direction)
    //   grid_size = n * k   (default/NCHW layout)  or  n * ho  (NHWC layout),
    // and block_size = 256 (see conv_internal::GetConv2DFWDSolution). This global work
    // size is passed to hipExtModuleLaunchKernel() as a uint32_t; once it reaches 2^32
    // it is silently truncated, producing an illegal grid (e.g. gridDim.x == 0) and a
    // HIP "invalid configuration argument" at launch time. This is reachable with large
    // batches, e.g. a ViT patch-embedding conv (3->1024, 14x14 kernel, stride 14) with
    // n >= 16384 on gfx950. Declare the solver not applicable for such sizes so that a
    // non-overflowing solver is selected instead.
    if(problem.Is2d())
    {
        const auto n  = static_cast<std::size_t>(ProblemInterpreter::GetBatchN(problem));
        const auto k  = static_cast<std::size_t>(ProblemInterpreter::GetOutputChannelK(problem));
        const auto ho = static_cast<std::size_t>(ProblemInterpreter::GetOutputHeightHo(problem));
        const std::size_t grid_size  = problem.IsLayoutDefault() ? (n * k) : (n * ho);
        const std::size_t block_size = 256;
        if(grid_size * block_size >= (static_cast<std::size_t>(1) << 32))
            return false;
    }

    if(problem.IsTensorsCasted())
    {
        auto test_cast = [&](const TensorDescriptor& desc) {
            if(desc.GetCastType())
            {
                const auto cast_type = *desc.GetCastType();
                if(cast_type == miopenFloat8_fnuz || cast_type == miopenBFloat8_fnuz)
                    return false;
            }
            // all tested tensors must have cast type set
            return true;
        };
        if(test_cast(problem.GetIn()))
            return false;
        if(test_cast(problem.GetWeights()))
            return false;
    }
    return true;
}

ConvSolution ConvDirectNaiveConvFwd::GetSolution(const ExecutionContext& ctx,
                                                 const ProblemDescription& problem) const
{
    ConvSolution result;

    if(problem.Is2d())
    {
        result = conv_internal::GetConv2DFWDSolution(ctx, problem);
    }
    else
    {
        result = conv_internal::GetConv3DFWDSolution(ctx, problem);
    }
    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
