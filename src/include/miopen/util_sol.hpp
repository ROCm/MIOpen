/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 202 Advanced Micro Devices, Inc.
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
#ifndef MIOPEN_UTIL_SOL_HPP_
#define MIOPEN_UTIL_SOL_HPP_

#include <miopen/batched_transpose_sol.hpp>
#include <miopen/miopen.h>
#include <miopen/kernel_info.hpp>
#include <miopen/op_kernel_args.hpp>
#include <miopen/execution_context.hpp>
#include <vector>

namespace miopen {

struct TransposeSolution_NCHW2NHWC
{
    TransposeSolution_NCHW2NHWC(const ExecutionContext& ctx,
                                miopenDataType_t data_type,
                                uint32_t n,
                                uint32_t c,
                                uint32_t h,
                                uint32_t w)
        : batched_transpose_sol(ctx, data_type, n, c, h * w)
    {
    }
    solver::KernelInfo GetKernel() const { return batched_transpose_sol.GetKernel(); }
    std::vector<OpKernelArg> GetKernelArg() const { return batched_transpose_sol.GetKernelArg(); }

    BatchedTransposeSolution batched_transpose_sol;
};

struct TransposeSolution_NHWC2NCHW
{
    TransposeSolution_NHWC2NCHW(const ExecutionContext& ctx,
                                miopenDataType_t data_type,
                                uint32_t n,
                                uint32_t c,
                                uint32_t h,
                                uint32_t w)
        : batched_transpose_sol(ctx, data_type, n, h * w, c)
    {
    }
    solver::KernelInfo GetKernel() const { return batched_transpose_sol.GetKernel(); }
    std::vector<OpKernelArg> GetKernelArg() const { return batched_transpose_sol.GetKernelArg(); }
    BatchedTransposeSolution batched_transpose_sol;
};
} // namespace miopen

#endif // MIOPEN_UTIL_SOL_HPP_
