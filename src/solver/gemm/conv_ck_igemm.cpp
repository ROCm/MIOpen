
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <vector>
#include <cstdint>

#include <miopen/check_numerics.hpp>
#include <miopen/solver.hpp>
#include <miopen/conv/fused_data_invoke_params.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/gemm/problem_description.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#endif
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CK_IGEMM)

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::PassThrough;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using F16   = ck::half_t;

using ADataType = F16;
using BDataType = F16;
using CDataType = F16;

const auto a_element_op = AElementOp{};
const auto b_element_op = BElementOp{};
const auto c_element_op = CElementOp{};
#endif 

namespace miopen {
namespace solver {
namespace fusion {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

using DeviceOp = ck::tensor_operation::device::DeviceGemm<ALayout,
                                                          BLayout,
                                                          CLayout,
                                                          ADataType,
                                                          BDataType,
                                                          CDataType,
                                                          AElementOp,
                                                          BElementOp,
                                                          CElementOp>;


struct CKArgs
{
    CKArgs(const miopen::gemm::ProblemDescription& problem)
    {
        // convert problem's NHWC to MNK .. here we do the conversion
        M = problem.GetGemmDescriptor().GetM();
        N = problem.GetGemmDescriptor().GetM();
        K = problem.GetGemmDescriptor().GetM();

        StrideA = problem.GetGemmDescriptor().GetStrideA();
        StrideB = problem.GetGemmDescriptor().GetStrideB();
        StrideC = problem.GetGemmDescriptor().GetStrideC();
    }

    int M;
    int N;
    int K;

    int StrideA;
    int StrideB;
    int StrideC;
 
};

template <typename DataType>
void PerformanceConfigCKIgemm::Init(const miopen::gemm::ProblemDescription& problem)
{
    const auto& args = CKArgs{problem};
    const auto gemm_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();
    assert(!gemm_ptrs.empty());
    // we need to add unique_id since ck's GetTypeString() does not give unique name of the kernel.
    int unique_id = 0;
    for(const auto& it : gemm_ptrs)
    {
        auto argument_ptr = it->MakeArgumentPointer(nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    args.M,
                                                    args.N,
                                                    args.K,
                                                    args.StrideA,
                                                    args.StrideB,
                                                    args.StrideC,
                                                    a_element_op,
                                                    b_element_op,
                                                    c_element_op);
        if(it->IsSupportedArgument(argument_ptr.get()))
        {
            valid_kernels.push_back(it->GetTypeString() + "_" + std::to_string(unique_id));
        }
        ++unique_id;
    }

    assert(!valid_kernels.empty());
    this->index     = 0;
    this->kernel_id = valid_kernels[0];
}

template <typename DataType>
bool PerformanceConfigCKIgemm::CheckIsSupportCKArgs(
    const miopen::gemm::ProblemDescription& problem) const
{
    const auto& args = CKArgs{problem};
    const auto gemm_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();
    
    int i = 0;
    for(; i < gemm_ptrs.size(); i++)
    {
        if(gemm_ptrs[i]->GetTypeString() == this->kernel_id)
        {
            break;
        }
    }
    if(i == valid_kernels.size())
    {
        return false;
    }
    auto argument_ptr = gemm_ptrs[i]->MakeArgumentPointer(nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    args.M,
                                                    args.N,
                                                    args.K,
                                                    args.StrideA,
                                                    args.StrideB,
                                                    args.StrideC,
                                                    a_element_op,
                                                    b_element_op,
                                                    c_element_op);
    return gemm_ptrs[i]->IsSupportedArgument(argument_ptr.get());
}

template <typename DataType>
bool CKIgemm::CheckCKApplicability(const miopen::gemm::ProblemDescription& problem) const
{
    const auto& args = CKArgs{problem};
    const auto gemm_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();
    for(const auto& it : gemm_ptrs)
    {
        auto argument_ptr = it->MakeArgumentPointer(nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    args.M,
                                                    args.N,
                                                    args.K,
                                                    args.StrideA,
                                                    args.StrideB,
                                                    args.StrideC,
                                                    a_element_op,
                                                    b_element_op,
                                                    c_element_op);
        if(it->IsSupportedArgument(argument_ptr.get()))
            return true;
    }
    return false;
}

template <typename DataType>
void RunCKSolution(const Handle& handle,
                   const AnyInvokeParams& primitive_parameters,
                   const miopen::gemm::ProblemDescription& problem,
                   const PerformanceConfigCKIgemm& config)
{
    const auto& args = CKArgs{problem};
    const auto gemm_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    // we need to add unique_id since ck's GetTypeString() does not give unique name of the kernel.
    int unique_id = 0;
    for(; unique_id < gemm_ptrs.size(); unique_id++)
    {
        if(gemm_ptrs[unique_id]->GetTypeString() + "_" + std::to_string(unique_id) ==
           config.kernel_id)
        {
            break;
        }
    }
    assert(unique_id < gemm_ptrs.size());
    auto& gemm_ck          = gemm_ptrs.at(unique_id);
    const auto& invoke_ctx = primitive_parameters.CastTo<miopen::fusion::FusionInvokeParams>();
    const auto& b_buf =
        dynamic_cast<miopen::fusion::GemmOpInvokeParam&>(*invoke_ctx.op_args.params[0]).B_data;

    auto argument_ptr = gemm_ck->MakeArgumentPointer(
        const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
            static_cast<const void*>(invoke_ctx.in)), // 
        const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
            static_cast<const void*>(b_buf)), // b
        invoke_ctx.out, // c
        args.M,
        args.N,
        args.K,
        args.StrideA,
        args.StrideB,
        args.StrideC,
        a_element_op,
        b_element_op,
        c_element_op);
                                                    
    auto invoker_ptr            = gemm_ck->MakeInvokerPointer();
    const auto enable_profiling = handle.IsProfilingEnabled();

    float elapsed_time =
        invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), enable_profiling});
    if(enable_profiling)
    {
        handle.ResetKernelTime();
        handle.AccumKernelTime(elapsed_time);
    }
}
#endif

void PerformanceConfigCKIgemm::HeuristicInit(const FusionContext& ctx)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
#else
    const auto& gemm_prob = ctx.problem.GetGemmProblem(0);
    switch(gemm_prob.GetADataType())
    {
    case miopenHalf: Init<ck::half_t>(gemm_prob); break;
    case miopenInt8:
    case miopenFloat:
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble: break;
    }
#endif
}

bool PerformanceConfigCKIgemm::SetNextValue(const FusionContext& ctx)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    return false;
#else
    if(this->valid_kernels.empty())
    {
        this->HeuristicInit(ctx);
        assert(!valid_kernels.empty());
        return true;
    }
    if((this->index + 1) < valid_kernels.size())
    {
        ++this->index;
        this->kernel_id = this->valid_kernels[index];
        return true;
    }
    else
        return false;
#endif
}

bool PerformanceConfigCKIgemm::IsValidValue() const
{
    return this->index >= 0 && this->index < valid_kernels.size();
}

bool PerformanceConfigCKIgemm::IsValid(const FusionContext& ctx) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    return false;
#else
    // Extract convolution problem from the fusion context.
    const auto& problem = ctx.problem.GetGemmProblem(0);
    switch(problem.GetADataType())
    {
    case miopenHalf: return CheckIsSupportCKArgs<ck::half_t>(problem);
    case miopenInt8:
    case miopenFloat:
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble: break;
    }
    return false;
#endif
}

bool PerformanceConfigCKIgemm::operator==(
    const PerformanceConfigCKIgemm& other) const
{
    return this->kernel_id == other.kernel_id;
}
PerformanceConfigCKIgemm
CKIgemm::GetDefaultPerformanceConfig(const FusionContext& ctx) const
{
    PerformanceConfigCKIgemm pp;
    pp.HeuristicInit(ctx);
    return pp;
}

bool CKIgemm::IsValidPerformanceConfig(
    const FusionContext& ctx, const PerformanceConfigCKIgemm& config) const
{
    return config.IsValid(ctx);
}

PerformanceConfigCKIgemm
CKIgemm::Search(const FusionContext& ctx, const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, invoke_ctx);
}

bool CKIgemm::IsApplicable(const FusionContext& ctx) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    return false;
#else
    const auto& problem = ctx.problem.GetGemmProblem(0);
    if(miopen::IsDisabled(MIOPEN_DEBUG_CK_IGEMM{}))
        return false;
    if(problem.GetADataType() != problem.GetBDataType() ||
       problem.GetADataType() != problem.GetCDataType())
        return false;
    const std::string arch = ctx.GetStream().GetDeviceName();
    if(arch != "gfx908" && arch != "gfx90a")
        return false;
    //if(!problem.IsLayoutNHWC())
    //    return false;

    switch(problem.GetADataType())
    {
    case miopenHalf: return CheckCKApplicability<ck::half_t>(problem);
    case miopenInt8:
    case miopenFloat:
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble: break;
    }
    return false;
#endif
}

ConvSolution
CKIgemm::GetSolution(const FusionContext& ctx,
                                     const PerformanceConfigCKIgemm& config) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    std::ignore = config;
    return {};
#else
    const auto& problem = ctx.problem.GetGemmProblem(0);
    ConvSolution result;
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        std::ignore = kernels;
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            switch(problem.GetADataType())
            {
            case miopenHalf:
                RunCKSolution<ck::half_t>(handle, primitive_parameters, problem, config);
                break;
            case miopenInt8:
            case miopenFloat:
            case miopenInt32:
            case miopenInt8x4:
            case miopenBFloat16:
            case miopenDouble: break;
            }
        };
    };
    return result;
#endif
}

} // namespace fusion
} // namespace solver
} // namespace miopen
