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
#include <miopen/kernel_info.hpp>
#include <miopen/op_kernel_args.hpp>
#include <miopen/execution_context.hpp>
#include <vector>

#define BT_FLAG_DEFAULT 0
#define BT_FLAG_CVT_FP16_FP8_FP16 0x1
#define BT_FLAG_CVT_FP16_BF8_FP16 0x2
#define BT_FLAG_CVT_BF16_FP8_BF16 0x3
#define BT_FLAG_CVT_BF16_BF8_BF16 0x4

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
                             uint32_t width_,
                             uint32_t flag_ = BT_FLAG_DEFAULT,
                             bool stoch_    = true,
                             uint32_t seed_ = 0);
    solver::KernelInfo GetKernel() const;
    solver::KernelInfo GetKernelInfo() const;
    std::vector<OpKernelArg> GetKernelArg() const;
    std::string GetKernelName() const;
    bool IsSkippable() const;
    size_t GetOutputTensorSize() const;
    std::vector<BatchedTransposeParam> GetApplicableParams() const;

    miopenDataType_t data_type;
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t flag;
    int num_cu;
    bool fp8_stochastic_rounding;
    uint32_t fp8_rounding_seed;

    BatchedTransposeParam kernel_param_heuristic;
};

} // namespace miopen

#endif
