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
#ifndef MIOPEN_TENSOR_REORDER_UTIL_HPP_
#define MIOPEN_TENSOR_REORDER_UTIL_HPP_

#include <miopen/batched_transpose_sol.hpp>
#include <miopen/general_tensor_reorder_sol.hpp>
#include <miopen/miopen.h>
#include <miopen/kernel_info.hpp>
#include <miopen/op_kernel_args.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/kernels/gpu_tensor_reorder/sequence.hpp>
#include <vector>

namespace miopen {

template<typename dst_order>
struct TensorReorderSolution : public GeneralReorderSolution<dst_order>
{
    TensorReorderSolution(const ExecutionContext& ctx_,
                          miopenDataType_t data_type_,
                          uint32_t dim_0_,
                          uint32_t dim_1_,
                          uint32_t dim_2_,
                          uint32_t dim_3_)
        : GeneralReorderSolution<dst_order>(ctx_, data_type_, dim_0_, dim_1_, dim_2_, dim_3_)
    {
    }
};

template<>
struct TensorReorderSolution<sequence<0, 2, 3, 1>> : public BatchedTransposeSolution
{
    TensorReorderSolution(const ExecutionContext& ctx_,
                          miopenDataType_t data_type_,
                          uint32_t dim_0_,
                          uint32_t dim_1_,
                          uint32_t dim_2_,
                          uint32_t dim_3_)
        : BatchedTransposeSolution(ctx_, data_type_, dim_0_, dim_1_, dim_2_ * dim_3_)
        {
        }
};

template<>
struct TensorReorderSolution<sequence<0, 1, 3, 2>> : public BatchedTransposeSolution
{
    TensorReorderSolution(const ExecutionContext& ctx_,
                          miopenDataType_t data_type_,
                          uint32_t dim_0_,
                          uint32_t dim_1_,
                          uint32_t dim_2_,
                          uint32_t dim_3_)
        : BatchedTransposeSolution(ctx_, data_type_, dim_0_ * dim_1_, dim_2_, dim_3_)
        {
        }
};

template<>
struct TensorReorderSolution<sequence<0, 3, 1, 2>> : public BatchedTransposeSolution
{
    TensorReorderSolution(const ExecutionContext& ctx_,
                          miopenDataType_t data_type_,
                          uint32_t dim_0_,
                          uint32_t dim_1_,
                          uint32_t dim_2_,
                          uint32_t dim_3_)
        : BatchedTransposeSolution(ctx_, data_type_, dim_0_, dim_1_ * dim_2_, dim_3_)
        {
        }
};

template<>
struct TensorReorderSolution<sequence<2, 3, 0, 1>> : public BatchedTransposeSolution
{
    TensorReorderSolution(const ExecutionContext& ctx_,
                          miopenDataType_t data_type_,
                          uint32_t dim_0_,
                          uint32_t dim_1_,
                          uint32_t dim_2_,
                          uint32_t dim_3_)
        : BatchedTransposeSolution(ctx_, data_type_, 1, dim_0 *dim_1_, dim_2_ * dim_3_)
        {
        }
};

template<>
struct TensorReorderSolution<sequence<2, 3, 0, 1>> : public BatchedTransposeSolution
{
    TensorReorderSolution(const ExecutionContext& ctx_,
                          miopenDataType_t data_type_,
                          uint32_t dim_0_,
                          uint32_t dim_1_,
                          uint32_t dim_2_,
                          uint32_t dim_3_)
        : BatchedTransposeSolution(ctx_, data_type_, 1, dim_0 *dim_1_, dim_2_ * dim_3_)
        {
        }
};

template<>
struct TensorReorderSolution<sequence<3, 0, 1, 2>> : public BatchedTransposeSolution
{
    TensorReorderSolution(const ExecutionContext& ctx_,
                          miopenDataType_t data_type_,
                          uint32_t dim_0_,
                          uint32_t dim_1_,
                          uint32_t dim_2_,
                          uint32_t dim_3_)
        : BatchedTransposeSolution(ctx_, data_type_, 1, dim_0 * dim_1_ * dim_2_，dim_3_)
        {
        }
};

} // namespace miopen

#endif // MIOPEN_TENSOR_REORDER_UTIL_HPP_
