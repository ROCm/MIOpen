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
#include <miopen/miopen.h>
#include <miopen/kernel_info.hpp>
#include <miopen/op_kernel_args.hpp>
#include <miopen/execution_context.hpp>
#include <vector>

namespace miopen {
struct TensorReorderSolution{

    virtual ~TensorReorderSolution() = default;
    virtual solver::KernelInfo GetKernel() const = 0;
    virtual std::vector<OpKernelArg> GetKernelArg() const = 0;
    virtual std::string GetKernelName() const = 0;
    virtual bool IsSkippable() const = 0;
    virtual size_t GetSize() const = 0;
};

struct WrapperBatchedTransposeSolution_0132 : TensorReorderSolution
{
    BatchedTransposeSolution m_BatchedTransposeSolution;
    WrapperBatchedTransposeSolution_0132(const ExecutionContext& ctx_,
                                         miopenDataType_t data_type_,
                                         uint32_t dim_0_,
                                         uint32_t dim_1_,
                                         uint32_t dim_2_,
                                         uint32_t dim_3_)
    : m_BatchedTransposeSolution(ctx_, data_type_, dim_0_ * dim_1_, dim_2_, dim_3_)
    {
    }
    solver::KernelInfo GetKernel() const override{ return m_BatchedTransposeSolution.GetKernel();}
    std::vector<OpKernelArg> GetKernelArg() const override{ return m_BatchedTransposeSolution.GetKernelArg();}
    std::string GetKernelName() const override{ return m_BatchedTransposeSolution.GetKernelName();}
    bool IsSkippable() const override{ return m_BatchedTransposeSolution.IsSkippable();}
    size_t GetSize() const override{ return m_BatchedTransposeSolution.GetSize();}
};

struct WrapperBatchedTransposeSolution_0231 : TensorReorderSolution
{
    BatchedTransposeSolution m_BatchedTransposeSolution;
    WrapperBatchedTransposeSolution_0231(const ExecutionContext& ctx_,
                                         miopenDataType_t data_type_,
                                         uint32_t dim_0_,
                                         uint32_t dim_1_,
                                         uint32_t dim_2_,
                                         uint32_t dim_3_)
    : m_BatchedTransposeSolution(ctx_, data_type_, dim_0_, dim_1_, dim_2_ * dim_3_)
    {
    }
    solver::KernelInfo GetKernel() const override{ return m_BatchedTransposeSolution.GetKernel();}
    std::vector<OpKernelArg> GetKernelArg() const override{ return m_BatchedTransposeSolution.GetKernelArg();}
    std::string GetKernelName() const override{ return m_BatchedTransposeSolution.GetKernelName();}
    bool IsSkippable() const override{ return m_BatchedTransposeSolution.IsSkippable();}
    size_t GetSize() const override{ return m_BatchedTransposeSolution.GetSize();}
};

struct WrapperBatchedTransposeSolution_0312 : TensorReorderSolution
{
    BatchedTransposeSolution m_BatchedTransposeSolution;
    WrapperBatchedTransposeSolution_0312(const ExecutionContext& ctx_,
                                         miopenDataType_t data_type_,
                                         uint32_t dim_0_,
                                         uint32_t dim_1_,
                                         uint32_t dim_2_,
                                         uint32_t dim_3_)
    : m_BatchedTransposeSolution(ctx_, data_type_, dim_0_, dim_1_ * dim_2_, dim_3_)
    {
    }
    solver::KernelInfo GetKernel() const override{ return m_BatchedTransposeSolution.GetKernel();}
    std::vector<OpKernelArg> GetKernelArg() const override{ return m_BatchedTransposeSolution.GetKernelArg();}
    std::string GetKernelName() const override{ return m_BatchedTransposeSolution.GetKernelName();}
    bool IsSkippable() const override{ return m_BatchedTransposeSolution.IsSkippable();}
    size_t GetSize() const override{ return m_BatchedTransposeSolution.GetSize();}
};

struct WrapperBatchedTransposeSolution_2301 : TensorReorderSolution
{
    BatchedTransposeSolution m_BatchedTransposeSolution;
    WrapperBatchedTransposeSolution_2301(const ExecutionContext& ctx_,
                                         miopenDataType_t data_type_,
                                         uint32_t dim_0_,
                                         uint32_t dim_1_,
                                         uint32_t dim_2_,
                                         uint32_t dim_3_)
    : m_BatchedTransposeSolution(ctx_, data_type_, 1, dim_0_ *dim_1_, dim_2_ * dim_3_)
    {
    }
    solver::KernelInfo GetKernel() const override{ return m_BatchedTransposeSolution.GetKernel();}
    std::vector<OpKernelArg> GetKernelArg() const override{ return m_BatchedTransposeSolution.GetKernelArg();}
    std::string GetKernelName() const override{ return m_BatchedTransposeSolution.GetKernelName();}
    bool IsSkippable() const override{ return m_BatchedTransposeSolution.IsSkippable();}
    size_t GetSize() const override{ return m_BatchedTransposeSolution.GetSize();}
};


struct WrapperBatchedTransposeSolution_3012 : TensorReorderSolution
{
    BatchedTransposeSolution m_BatchedTransposeSolution;
    WrapperBatchedTransposeSolution_3012(const ExecutionContext& ctx_,
                                         miopenDataType_t data_type_,
                                         uint32_t dim_0_,
                                         uint32_t dim_1_,
                                         uint32_t dim_2_,
                                         uint32_t dim_3_)
    : m_BatchedTransposeSolution(ctx_, data_type_, 1, dim_0_ * dim_1_ * dim_2_, dim_3_)
    {
    }
    solver::KernelInfo GetKernel() const override{ return m_BatchedTransposeSolution.GetKernel();}
    std::vector<OpKernelArg> GetKernelArg() const override{ return m_BatchedTransposeSolution.GetKernelArg();}
    std::string GetKernelName() const override{ return m_BatchedTransposeSolution.GetKernelName();}
    bool IsSkippable() const override{ return m_BatchedTransposeSolution.IsSkippable();}
    size_t GetSize() const override{ return m_BatchedTransposeSolution.GetSize();}
};

struct WrapperGeneralReorderSolution : TensorReorderSolution
{
    GeneralReorderSolution m_GeneralReorderSolution;
    WrapperGeneralReorderSolution(const ExecutionContext& ctx_,
                                    miopenDataType_t data_type_,
                                    uint32_t dim_0_,
                                    uint32_t dim_1_,
                                    uint32_t dim_2_,
                                    uint32_t dim_3_,
                                    uint32_t order_0_,
                                    uint32_t order_1_,
                                    uint32_t order_2_,
                                    uint32_t order_3_)
    : m_GeneralReorderSolution(ctx_, data_type_, dim_0_, dim_1_, dim_2_, dim_3_, order_0_, order_1_, order_2_, order_3_)
    {
    }
    solver::KernelInfo GetKernel() const override{ return m_GeneralReorderSolution.GetKernel();}
    std::vector<OpKernelArg> GetKernelArg() const override{ return m_GeneralReorderSolution.GetKernelArg();}
    std::string GetKernelName() const override{ return m_GeneralReorderSolution.GetKernelName();}
    bool IsSkippable() const override{ return m_GeneralReorderSolution.IsSkippable();}
    size_t GetSize() const override{ return m_GeneralReorderSolution.GetSize();}
};

__inline__ std::unique_ptr<TensorReorderSolution> TensorReorderSolutionConstructor(const ExecutionContext& ctx_,
                                                    miopenDataType_t data_type_,
                                                    uint32_t dim_0_,
                                                    uint32_t dim_1_,
                                                    uint32_t dim_2_,
                                                    uint32_t dim_3_,
                                                    uint32_t order_0_,
                                                    uint32_t order_1_,
                                                    uint32_t order_2_,
                                                    uint32_t order_3_) {
    //Default using general reorder
    int which = 0;
    if( (order_0_ == 0) && (order_1_ == 1) && (order_2_ == 3) && (order_3_ == 2) ) which = 1;
    if( (order_0_ == 0) && (order_1_ == 2) && (order_2_ == 3) && (order_3_ == 1) ) which = 2;
    if( (order_0_ == 0) && (order_1_ == 3) && (order_2_ == 1) && (order_3_ == 2) ) which = 3;
    if( (order_0_ == 2) && (order_1_ == 3) && (order_2_ == 0) && (order_3_ == 1) ) which = 4;
    if( (order_0_ == 3) && (order_1_ == 0) && (order_2_ == 1) && (order_3_ == 2) ) which = 5;

    switch (which) {
        case 0: return std::make_unique<WrapperGeneralReorderSolution>(ctx_, data_type_, dim_0_, dim_1_, dim_2_, dim_3_,
                                                                                        order_0_, order_1_, order_2_, order_3_);
        case 1: return std::make_unique<WrapperBatchedTransposeSolution_0132>(ctx_, data_type_, dim_0_, dim_1_, dim_2_, dim_3_);
        case 2: return std::make_unique<WrapperBatchedTransposeSolution_0231>(ctx_, data_type_, dim_0_, dim_1_, dim_2_, dim_3_);
        case 3: return std::make_unique<WrapperBatchedTransposeSolution_0312>(ctx_, data_type_, dim_0_, dim_1_, dim_2_, dim_3_);
        case 4: return std::make_unique<WrapperBatchedTransposeSolution_2301>(ctx_, data_type_, dim_0_, dim_1_, dim_2_, dim_3_);
        case 5: return std::make_unique<WrapperBatchedTransposeSolution_3012>(ctx_, data_type_, dim_0_, dim_1_, dim_2_, dim_3_);
        default : return nullptr;
    }
    return nullptr;
}

} // namespace miopen

#endif // MIOPEN_TENSOR_REORDER_UTIL_HPP_
