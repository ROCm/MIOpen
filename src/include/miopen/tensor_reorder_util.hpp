/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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
#include <miopen/kernel_info.hpp>
#include <miopen/op_kernel_args.hpp>
#include <miopen/execution_context.hpp>
#include <vector>

namespace miopen {
struct TensorReorderAttributesBase
{

    virtual ~TensorReorderAttributesBase()                = default;
    virtual solver::KernelInfo GetKernelInfo() const      = 0;
    virtual std::vector<OpKernelArg> GetKernelArg() const = 0;
    virtual std::string GetKernelName() const             = 0;
    // used in HOST side to check the special cases that either tensor height or width equal = 1.
    // In such cases, we don't need to conduct batched transpose operation,
    // since the transposed tensor layout has exactly same memory layout as before.
    virtual bool IsSkippable() const = 0;
    // workspace (buffer) to hold output tensor of transform Pre/Post convolution
    virtual size_t GetOutputTensorSize() const = 0;
};

struct BatchedTransposeSolution_0132 : TensorReorderAttributesBase
{
    BatchedTransposeSolution impl;
    BatchedTransposeSolution_0132(const ExecutionContext& ctx_,
                                  miopenDataType_t data_type_,
                                  uint32_t dim_0_,
                                  uint32_t dim_1_,
                                  uint32_t dim_2_,
                                  uint32_t dim_3_)
        : impl(ctx_, data_type_, dim_0_ * dim_1_, dim_2_, dim_3_)
    {
    }
    solver::KernelInfo GetKernelInfo() const override { return impl.GetKernelInfo(); }
    std::vector<OpKernelArg> GetKernelArg() const override { return impl.GetKernelArg(); }
    std::string GetKernelName() const override { return impl.GetKernelName(); }
    bool IsSkippable() const override { return impl.IsSkippable(); }
    size_t GetOutputTensorSize() const override { return impl.GetOutputTensorSize(); }
};

struct BatchedTransposeSolution_0231 : TensorReorderAttributesBase
{
    BatchedTransposeSolution impl;
    BatchedTransposeSolution_0231(const ExecutionContext& ctx_,
                                  miopenDataType_t data_type_,
                                  uint32_t dim_0_,
                                  uint32_t dim_1_,
                                  uint32_t dim_2_,
                                  uint32_t dim_3_)
        : impl(ctx_, data_type_, dim_0_, dim_1_, dim_2_ * dim_3_)
    {
    }
    solver::KernelInfo GetKernelInfo() const override { return impl.GetKernelInfo(); }
    std::vector<OpKernelArg> GetKernelArg() const override { return impl.GetKernelArg(); }
    std::string GetKernelName() const override { return impl.GetKernelName(); }
    bool IsSkippable() const override { return impl.IsSkippable(); }
    size_t GetOutputTensorSize() const override { return impl.GetOutputTensorSize(); }
};

struct BatchedTransposeSolution_0312 : TensorReorderAttributesBase
{
    BatchedTransposeSolution impl;
    BatchedTransposeSolution_0312(const ExecutionContext& ctx_,
                                  miopenDataType_t data_type_,
                                  uint32_t dim_0_,
                                  uint32_t dim_1_,
                                  uint32_t dim_2_,
                                  uint32_t dim_3_)
        : impl(ctx_, data_type_, dim_0_, dim_1_ * dim_2_, dim_3_)
    {
    }
    solver::KernelInfo GetKernelInfo() const override { return impl.GetKernelInfo(); }
    std::vector<OpKernelArg> GetKernelArg() const override { return impl.GetKernelArg(); }
    std::string GetKernelName() const override { return impl.GetKernelName(); }
    bool IsSkippable() const override { return impl.IsSkippable(); }
    size_t GetOutputTensorSize() const override { return impl.GetOutputTensorSize(); }
};

struct BatchedTransposeSolution_2301 : TensorReorderAttributesBase
{
    BatchedTransposeSolution impl;
    BatchedTransposeSolution_2301(const ExecutionContext& ctx_,
                                  miopenDataType_t data_type_,
                                  uint32_t dim_0_,
                                  uint32_t dim_1_,
                                  uint32_t dim_2_,
                                  uint32_t dim_3_)
        : impl(ctx_, data_type_, 1, dim_0_ * dim_1_, dim_2_ * dim_3_)
    {
    }
    solver::KernelInfo GetKernelInfo() const override { return impl.GetKernelInfo(); }
    std::vector<OpKernelArg> GetKernelArg() const override { return impl.GetKernelArg(); }
    std::string GetKernelName() const override { return impl.GetKernelName(); }
    bool IsSkippable() const override { return impl.IsSkippable(); }
    size_t GetOutputTensorSize() const override { return impl.GetOutputTensorSize(); }
};

struct BatchedTransposeSolution_3012 : TensorReorderAttributesBase
{
    BatchedTransposeSolution impl;
    BatchedTransposeSolution_3012(const ExecutionContext& ctx_,
                                  miopenDataType_t data_type_,
                                  uint32_t dim_0_,
                                  uint32_t dim_1_,
                                  uint32_t dim_2_,
                                  uint32_t dim_3_)
        : impl(ctx_, data_type_, 1, dim_0_ * dim_1_ * dim_2_, dim_3_)
    {
    }
    solver::KernelInfo GetKernelInfo() const override { return impl.GetKernelInfo(); }
    std::vector<OpKernelArg> GetKernelArg() const override { return impl.GetKernelArg(); }
    std::string GetKernelName() const override { return impl.GetKernelName(); }
    bool IsSkippable() const override { return impl.IsSkippable(); }
    size_t GetOutputTensorSize() const override { return impl.GetOutputTensorSize(); }
};

struct GenericReorderSolution : TensorReorderAttributesBase
{
    GenericReorderSolutionImpl impl;
    GenericReorderSolution(miopenDataType_t data_type_,
                           uint32_t dim_0_,
                           uint32_t dim_1_,
                           uint32_t dim_2_,
                           uint32_t dim_3_,
                           uint32_t order_0_,
                           uint32_t order_1_,
                           uint32_t order_2_,
                           uint32_t order_3_)
        : impl(data_type_, dim_0_, dim_1_, dim_2_, dim_3_, order_0_, order_1_, order_2_, order_3_)
    {
    }
    solver::KernelInfo GetKernelInfo() const override { return impl.GetKernelInfo(); }
    std::vector<OpKernelArg> GetKernelArg() const override { return impl.GetKernelArg(); }
    std::string GetKernelName() const override { return impl.GetKernelName(); }
    bool IsSkippable() const override { return impl.IsSkippable(); }
    size_t GetOutputTensorSize() const override { return impl.GetOutputTensorSize(); }
};

inline std::unique_ptr<TensorReorderAttributesBase>
MakeTensorReorderAttributes(const ExecutionContext& ctx_,
                            miopenDataType_t data_type_,
                            uint64_t dim_0_,
                            uint64_t dim_1_,
                            uint64_t dim_2_,
                            uint64_t dim_3_,
                            uint32_t order_0_,
                            uint32_t order_1_,
                            uint32_t order_2_,
                            uint32_t order_3_)
{
    if(!ctx_.use_hip_kernels)
    {
        return nullptr;
    }
    ///\todo 64-bit tensor dimension
    // Currentlly we have tensor dimension limited to 2^32 - 1
    if((dim_0_ > UINT32_MAX) || (dim_1_ > UINT32_MAX) || (dim_2_ > UINT32_MAX) ||
       (dim_3_ > UINT32_MAX))
    {
        MIOPEN_THROW("Currentlly we have tensor dimension limitation of 2^32 - 1");
    }
    // Default using general reorder
    if(data_type_ == miopenBFloat16)
    {
        MIOPEN_THROW("Unsupported reorder data type");
    }
    // Check required reorder is legal or not
    std::vector<std::vector<uint32_t>> all_possible_order{
        {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3},
        {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0}, {2, 0, 1, 3},
        {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 0, 1, 2},
        {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0}};
    auto found = std::find_if(
        all_possible_order.begin(),
        all_possible_order.end(),
        [order_0_, order_1_, order_2_, order_3_](std::vector<uint32_t> possible_order) {
            return (possible_order[0] == order_0_) && (possible_order[1] == order_1_) &&
                   (possible_order[2] == order_2_) && (possible_order[3] == order_3_);
        });
    if(found == all_possible_order.end())
        MIOPEN_THROW("Unsupported Reorder");

    int which = 0;
    // Special cases that utilize batched transpose kernel
    if(data_type_ != miopenDouble)
    {
        if((order_0_ == 0) && (order_1_ == 1) && (order_2_ == 3) && (order_3_ == 2))
            which = 1;
        else if((order_0_ == 0) && (order_1_ == 2) && (order_2_ == 3) && (order_3_ == 1))
            which = 2;
        else if((order_0_ == 0) && (order_1_ == 3) && (order_2_ == 1) && (order_3_ == 2))
            which = 3;
        else if((order_0_ == 2) && (order_1_ == 3) && (order_2_ == 0) && (order_3_ == 1))
            which = 4;
        else if((order_0_ == 3) && (order_1_ == 0) && (order_2_ == 1) && (order_3_ == 2))
            which = 5;
    }
    // Order [0, 1, 3, 2], [0, 2, 3, 1], [0, 3, 1, 2], [2, 3, 0, 1], [3, 0, 1, 2] are using batched
    // transpose kernel to achieve higher performance. Details as following:
    // reorder to [0, 1, 3, 2] from [0, 1, 2, 3], we can fix layout index [0] and [1], transpose [2,
    // 3] to [3, 2]. reorder to [0, 2, 3, 1] from [0, 1, 2, 3], we can fix layout index [0], see [2,
    // 3] as an entity, then transpose [1, (2, 3)] to [(2, 3), 1]. reorder to [0, 3, 1, 2] from [0,
    // 1, 2, 3], we can fix layout index [0], see [1, 2] as an entity, then transpose [(1, 2), 3)]
    // to [3, (1, 2)]. reorder to [2, 3, 0, 1] from [0, 1, 2, 3], we can add a fixed layout index ,
    // see [0, 1] and [2, 3] as entities, then transpose [(0, 1), (2, 3)] to [(2, 3), (0, 1)].
    // reorder to [3, 0, 1, 2] from [0, 1, 2, 3], we can add a fixed layout index , see [0, 1, 2] as
    // an entity, then transpose [(0, 1, 2), 3] to [3, (0, 1, 2)]. The reason we have different API
    // like BatchedTransposeSolution_0132 is that we choose different fixed index and two
    // dimensions which will be transposed.

    switch(which)
    {
    case 0:
        return std::make_unique<GenericReorderSolution>(
            data_type_, dim_0_, dim_1_, dim_2_, dim_3_, order_0_, order_1_, order_2_, order_3_);
    case 1:
        return std::make_unique<BatchedTransposeSolution_0132>(
            ctx_, data_type_, dim_0_, dim_1_, dim_2_, dim_3_);
    case 2:
        return std::make_unique<BatchedTransposeSolution_0231>(
            ctx_, data_type_, dim_0_, dim_1_, dim_2_, dim_3_);
    case 3:
        return std::make_unique<BatchedTransposeSolution_0312>(
            ctx_, data_type_, dim_0_, dim_1_, dim_2_, dim_3_);
    case 4:
        return std::make_unique<BatchedTransposeSolution_2301>(
            ctx_, data_type_, dim_0_, dim_1_, dim_2_, dim_3_);
    case 5:
        return std::make_unique<BatchedTransposeSolution_3012>(
            ctx_, data_type_, dim_0_, dim_1_, dim_2_, dim_3_);
    default: MIOPEN_THROW("Failed to call tensor reorder solver");
    }
}

} // namespace miopen

#endif // MIOPEN_TENSOR_REORDER_UTIL_HPP_
