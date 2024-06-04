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
#ifndef GUARD_MIOPEN_BATCHED_TRANSPOSE_SOL_HPP
#define GUARD_MIOPEN_BATCHED_TRANSPOSE_SOL_HPP

#include <miopen/miopen.h>
#include <miopen/errors.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/op_kernel_args.hpp>
#include <miopen/execution_context.hpp>
#include <vector>

struct transpose_invoke_param : public miopen::InvokeParams
{
    ConstData_t src = nullptr;
    Data_t dst      = nullptr;

    transpose_invoke_param(ConstData_t src_, Data_t dst_) : src(src_), dst(dst_) {}
    transpose_invoke_param(miopen::InvokeType type_, ConstData_t src_, Data_t dst_)
        : InvokeParams{type_}, src(src_), dst(dst_)
    {
    }

    Data_t GetWorkspace() const { return nullptr; }
    std::size_t GetWorkspaceSize() const { return 0; }
};

namespace miopen {

struct BatchedTransposeParam
{
    int tile_x{0};
    int tile_y{0};
    int pack_x{0};
    int pack_y{0};
    int ediv_x{0};
    int ediv_y{0};
};

struct BatchedTransposeSolution
{
    BatchedTransposeSolution(const ExecutionContext& ctx_,
                             miopenDataType_t data_type_,
                             uint32_t batch_,
                             uint32_t height_,
                             uint32_t width_);
    solver::KernelInfo GetKernelInfo() const;
    std::vector<OpKernelArg> GetKernelArg() const;
    std::string GetKernelName() const;
    bool IsSkippable() const;
    size_t GetOutputTensorSize() const;

    miopenDataType_t data_type;
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    int num_cu;

    BatchedTransposeParam kernel_param_heuristic;

    InvokerFactory MakeBatchedTransposeInvokerFactory();
};

struct BatchedTransposeSolution2D : public BatchedTransposeSolution
{
    BatchedTransposeSolution2D(const ExecutionContext& ctx_,
                               miopenDataType_t data_type_,
                               uint32_t batch_,
                               uint32_t channels_,
                               uint32_t height_,
                               uint32_t width_ = 1)
        : BatchedTransposeSolution(ctx_, data_type_, batch_, channels_, height_ * width_)
    {
        MIOPEN_THROW_IF(size_t(height_ * width_) != (size_t(height_) * size_t(width_)),
                        "integer overflow");
    }
};

struct BatchedTransposeSolution3D : public BatchedTransposeSolution
{
    BatchedTransposeSolution3D(const ExecutionContext& ctx_,
                               miopenDataType_t data_type_,
                               uint32_t batch_,
                               uint32_t channels_,
                               uint32_t depth_,
                               uint32_t height_ = 1,
                               uint32_t width_  = 1)
        : BatchedTransposeSolution(ctx_, data_type_, batch_, channels_, depth_ * height_ * width_)
    {
        MIOPEN_THROW_IF(size_t(depth_ * height_ * width_) !=
                            (size_t(depth_) * size_t(height_) * size_t(width_)),
                        "integer overflow");
    }
};

struct TransposeSolutionDefault2Nhwc : public BatchedTransposeSolution2D
{
    TransposeSolutionDefault2Nhwc(const ExecutionContext& ctx_,
                                  miopenDataType_t data_type_,
                                  uint32_t n_,
                                  uint32_t c_,
                                  uint32_t h_,
                                  uint32_t w_)
        : BatchedTransposeSolution2D(ctx_, data_type_, n_, c_, h_, w_)
    {
        MIOPEN_THROW_IF(size_t(h_ * w_) != (size_t(h_) * size_t(w_)), "integer overflow");
    }
};

struct TransposeSolutionNhwc2Default : public BatchedTransposeSolution2D
{
    TransposeSolutionNhwc2Default(const ExecutionContext& ctx_,
                                  miopenDataType_t data_type_,
                                  uint32_t n_,
                                  uint32_t c_,
                                  uint32_t h_,
                                  uint32_t w_)
        : BatchedTransposeSolution2D(ctx_, data_type_, n_, h_ * w_, c_)
    {
        MIOPEN_THROW_IF(size_t(h_ * w_) != (size_t(h_) * size_t(w_)), "integer overflow");
    }
};

struct TransposeSolutionDefault2Ndhwc : public BatchedTransposeSolution3D
{
    TransposeSolutionDefault2Ndhwc(const ExecutionContext& ctx_,
                                   miopenDataType_t data_type_,
                                   uint32_t n_,
                                   uint32_t c_,
                                   uint32_t d_,
                                   uint32_t h_,
                                   uint32_t w_)
        : BatchedTransposeSolution3D(ctx_, data_type_, n_, c_, d_, h_, w_)
    {
        MIOPEN_THROW_IF(size_t(d_ * h_ * w_) != (size_t(d_) * size_t(h_) * size_t(w_)),
                        "integer overflow");
    }
};

struct TransposeSolutionNdhwc2Default : public BatchedTransposeSolution3D
{
    TransposeSolutionNdhwc2Default(const ExecutionContext& ctx_,
                                   miopenDataType_t data_type_,
                                   uint32_t n_,
                                   uint32_t c_,
                                   uint32_t d_,
                                   uint32_t h_,
                                   uint32_t w_)
        : BatchedTransposeSolution3D(ctx_, data_type_, n_, d_ * h_ * w_, c_)
    {
        MIOPEN_THROW_IF(size_t(d_ * h_ * w_) != (size_t(d_) * size_t(h_) * size_t(w_)),
                        "integer overflow");
    }
};

} // namespace miopen

#endif
