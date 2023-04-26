
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
#include "ck/library/tensor_operation_instance/gpu/gemm_fastgelu.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#endif
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CK_IGEMM)

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using FastGelu   = ck::tensor_operation::element_wise::FastGelu;

using CDEElementOp = FastGelu;

using Row     = ck::tensor_layout::gemm::RowMajor;
using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

const auto a_element_op   = AElementOp{};
const auto b_element_op   = BElementOp{};
const auto cde_element_op = CDEElementOp{};
#endif

namespace miopen {
namespace solver {
namespace fusion {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

template <typename DataType>
using DeviceOp = ck::tensor_operation::device::DeviceGemmMultipleD<
    ALayout,
    BLayout,
    ck::Tuple<>,
    CLayout,
    DataType,
    DataType,
    ck::Tuple<>,
    DataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::FastGelu>;

template <typename DataType>
using DeviceOpGEMMActivPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceOp<DataType>>;

struct CKArgs
{
    CKArgs(const miopen::gemm::ProblemDescription& problem)
    {
        M = problem.GetM();
        N = problem.GetN();
        K = problem.GetK();

        ldA = problem.GetGemmDescriptor().GetldA();
        ldB = problem.GetGemmDescriptor().GetldB();
        ldC = problem.GetGemmDescriptor().GetldC();
    }

    int M;
    int N;
    int K;

    int ldA;
    int ldB;
    int ldC;
};

template <typename DataType>
void PerformanceConfigCKGEMActiv::Init(const miopen::gemm::ProblemDescription& problem)
{
    const auto& args     = CKArgs{problem};
    const auto gemm_ptrs = DeviceOpGEMMActivPtrs<DataType>::GetInstances();
    assert(!gemm_ptrs.empty());
    for(const auto& it : gemm_ptrs)
    {
        auto argument_ptr = it->MakeArgumentPointer(nullptr,
                                                    nullptr,
                                                    std::array<const void*, 0>{},
                                                    nullptr,
                                                    args.M,
                                                    args.N,
                                                    args.K,
                                                    args.ldA,
                                                    args.ldB,
                                                    std::array<ck::index_t, 0>{},
                                                    args.ldC,
                                                    a_element_op,
                                                    b_element_op,
                                                    cde_element_op);
        if(it->IsSupportedArgument(argument_ptr.get()))
        {
            valid_kernels.push_back(it->GetTypeString());
        }
    }

    assert(!valid_kernels.empty());
    this->index     = 0;
    this->kernel_id = valid_kernels[0];
}

template <typename DataType>
bool PerformanceConfigCKGEMActiv::CheckIsSupportCKArgs(
    const miopen::gemm::ProblemDescription& problem) const
{
    const auto& args     = CKArgs{problem};
    const auto gemm_ptrs = DeviceOpGEMMActivPtrs<DataType>::GetInstances();
    int i                = 0;
    for(; i < gemm_ptrs.size(); i++)
    {
        if(gemm_ptrs[i]->GetTypeString() + "_" + std::to_string(i) == this->kernel_id)
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
                                                          std::array<const void*, 0>{},
                                                          nullptr,
                                                          args.M,
                                                          args.N,
                                                          args.K,
                                                          args.ldA,
                                                          args.ldB,
                                                          std::array<ck::index_t, 0>{},
                                                          args.ldC,
                                                          a_element_op,
                                                          b_element_op,
                                                          cde_element_op);
    return gemm_ptrs[i]->IsSupportedArgument(argument_ptr.get());
}

template <typename DataType>
bool CKGEMMActiv::CheckCKApplicability(const miopen::gemm::ProblemDescription& problem) const
{
    const auto& args     = CKArgs{problem};
    const auto gemm_ptrs = DeviceOpGEMMActivPtrs<DataType>::GetInstances();
    for(const auto& it : gemm_ptrs)
    {
        auto argument_ptr = it->MakeArgumentPointer(nullptr,
                                                    nullptr,
                                                    std::array<const void*, 0>{},
                                                    nullptr,
                                                    args.M,
                                                    args.N,
                                                    args.K,
                                                    args.ldA,
                                                    args.ldB,
                                                    std::array<ck::index_t, 0>{},
                                                    args.ldC,
                                                    a_element_op,
                                                    b_element_op,
                                                    cde_element_op);
        if(it->IsSupportedArgument(argument_ptr.get()))
        {
            return true;
        }
    }
    return false;
}

template <typename DataType>
void RunCKSolution(const Handle& handle,
                   const AnyInvokeParams& primitive_parameters,
                   const miopen::gemm::ProblemDescription& problem,
                   const PerformanceConfigCKGEMActiv& config)
{
    const auto& args     = CKArgs{problem};
    const auto gemm_ptrs = DeviceOpGEMMActivPtrs<DataType>::GetInstances();

    int id = 0;
    for(; id < gemm_ptrs.size(); id++)
    {
        if(gemm_ptrs[id]->GetTypeString() == config.kernel_id)
        {
            break;
        }
    }

    assert(id < gemm_ptrs.size());
    auto& gemm_ck          = gemm_ptrs.at(id);
    const auto& invoke_ctx = primitive_parameters.CastTo<miopen::fusion::FusionInvokeParams>();
    const auto& b_buf =
        dynamic_cast<miopen::fusion::GemmOpInvokeParam&>(*invoke_ctx.op_args.params[0]).B_data;

    auto argument_ptr = gemm_ck->MakeArgumentPointer(
        const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
            static_cast<const void*>(invoke_ctx.in)), // a
        const_cast<void*>(                    // NOLINT (cppcoreguidelines-pro-type-const-cast)
            static_cast<const void*>(b_buf)), // b
        std::array<const void*, 0>{},
        invoke_ctx.out, // c
        args.M,
        args.N,
        args.K,
        args.ldA,
        args.ldB,
        std::array<ck::index_t, 0>{},
        args.ldC,
        a_element_op,
        b_element_op,
        cde_element_op);

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

void PerformanceConfigCKGEMActiv::HeuristicInit(const FusionDescription& fdesc_problem)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
#else
    const auto& gemm_prob = fdesc_problem.GetGemmProblem(0);
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

bool PerformanceConfigCKGEMActiv::SetNextValue(const FusionDescription& fdesc_problem)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
    return false;
#else
    if(this->valid_kernels.empty())
    {
        this->HeuristicInit(fdesc_problem);
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

bool PerformanceConfigCKGEMActiv::IsValidValue() const
{
    return this->index >= 0 && this->index < valid_kernels.size();
}

bool PerformanceConfigCKGEMActiv::IsValid(const FusionContext&,
                                          const FusionDescription& fdesc_problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
    return false;
#else
    // Extract convolution problem from the fusion context.
    const auto& problem = fdesc_problem.GetGemmProblem(0);
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

bool PerformanceConfigCKGEMActiv::operator==(const PerformanceConfigCKGEMActiv& other) const
{
    return this->kernel_id == other.kernel_id;
}
PerformanceConfigCKGEMActiv
CKGEMMActiv::GetDefaultPerformanceConfig(const FusionContext&,
                                         const FusionDescription& fdesc_problem) const
{
    PerformanceConfigCKGEMActiv pp;
    pp.HeuristicInit(fdesc_problem);
    return pp;
}

bool CKGEMMActiv::IsValidPerformanceConfig(const FusionContext& ctx,
                                           const FusionDescription& fdesc_problem,
                                           const PerformanceConfigCKGEMActiv& config) const
{
    return config.IsValid(ctx, fdesc_problem);
}

PerformanceConfigCKGEMActiv CKGEMMActiv::Search(const FusionContext& ctx,
                                                const FusionDescription& fdesc_problem,
                                                const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, fdesc_problem, invoke_ctx);
}

bool CKGEMMActiv::IsApplicable(const FusionContext& ctx,
                               const FusionDescription& fdesc_problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    std::ignore = fdesc_problem;
    return false;
#else
    const auto& fp_desc = *fdesc_problem.fusion_plan_desc;
    if(fp_desc.op_map.empty())
    {
        MIOPEN_THROW(miopenStatusInternalError, "desc.op_map.empty()");
    }
    if(miopen::IsDisabled(MIOPEN_DEBUG_CK_IGEMM{}))
    {
        return false;
    }
    // check the sequence of prims
    if(fp_desc.op_map.size() != 2)
        return false;
    if(fp_desc.op_map[0]->kind() != miopenFusionOpGEMM)
        return false;
    if(fp_desc.op_map[1]->kind() != miopenFusionOpActivForward)
        return false;
    const auto& activ_op = dynamic_cast<ActivFwdFusionOpDescriptor&>(*fp_desc.op_map[1]);
    if(activ_op.activMode != miopenActivationFGELU)
        return false;
    const auto& problem = fdesc_problem.GetGemmProblem(0);
    if(problem.GetADataType() != problem.GetBDataType() ||
       problem.GetADataType() != problem.GetCDataType())
    {
        return false;
    }
    const std::string arch = ctx.GetStream().GetDeviceName();
    if(arch != "gfx908" && arch != "gfx90a")
    {
        return false;
    }
    const auto& args = CKArgs{problem};
    // check dimensional mismatch.
    if(args.ldA != args.K || args.ldB != args.N || args.ldC != args.N)
    {
        return false;
    }

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

ConvSolution CKGEMMActiv::GetSolution(const FusionContext&,
                                      const FusionDescription& fdesc_problem,
                                      const PerformanceConfigCKGEMActiv& config) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
    std::ignore = config;
    return {};
#else
    const auto& problem = fdesc_problem.GetGemmProblem(0);
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
