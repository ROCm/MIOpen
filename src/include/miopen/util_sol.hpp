/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c_) 202 Advanced Micro Devices, Inc.
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

struct TransposeSolutionDefault2Nhwc : public BatchedTransposeSolution
{
    TransposeSolutionDefault2Nhwc(const ExecutionContext& ctx_,
                                  miopenDataType_t data_type_,
                                  uint32_t n_,
                                  uint32_t c_,
                                  uint32_t h_,
                                  uint32_t w_)
        : BatchedTransposeSolution(ctx_, data_type_, n_, c_, h_ * w_)
    {
    }
};

struct TransposeSolutionNhwc2Default : public BatchedTransposeSolution
{
    TransposeSolutionNhwc2Default(const ExecutionContext& ctx_,
                                  miopenDataType_t data_type_,
                                  uint32_t n_,
                                  uint32_t c_,
                                  uint32_t h_,
                                  uint32_t w_)
        : BatchedTransposeSolution(ctx_, data_type_, n_, h_ * w_, c_)
    {
    }
};
} // namespace miopen

#endif // MIOPEN_UTIL_SOL_HPP_
