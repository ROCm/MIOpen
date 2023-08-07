
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
#include <miopen/fusion/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <ck/library/tensor_operation_instance/gpu/batchnorm_backward.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_CK_BN_BACK)

namespace miopen {
namespace solver {
namespace fusion {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

using PassThroughOp = ck::tensor_operation::element_wise::PassThrough;
using index_t     = int32_t;

constexpr index_t Rank                  = 4;
constexpr index_t NumBatchNormReduceDim = 3;

template <typename DataType>
using DeviceOp = ck::tensor_operation::device::DeviceBatchNormBwd<DataType,
                                                                      DataType,
                                                                      DataType,
                                                                      DataType,
                                                                      DataType,
                                                                      DataType,
                                                                      DataType,
                                                                      PassThroughOp,
                                                                      Rank,
                                                                      NumBatchNormReduceDim>;
struct CKArgsBNormFwd
{
    CKArgsBNormFwd(const miopen::batchnorm::ProblemDescription& problem)
    {
        std::copy(problem.GetXDesc().GetLengths().begin(),
                  problem.GetXDesc().GetLengths().end(),
                  xyLengths.begin());

        std::copy(problem.GetXDesc().GetStrides().begin(),
                  problem.GetXDesc().GetStrides().end(),
                  xyStrides.begin());

        auto scaleBiasMeanVarStrides = problem.GetScaleBiasDiffDesc().GetStrides();
        std::copy(scaleBiasMeanVarStrides.begin(),
              scaleBiasMeanVarStrides.end(),
              arrScaleBiasMeanVarStrides.begin());

        std::vector<size_t> scaleBiasMeanVarLengths;
        for(int dim = 0; dim < Rank; dim++)
        {
            if(std::none_of(reduceDims.begin(), reduceDims.end(), [&](int d) { return dim == d; }))
            {
                scaleBiasMeanVarLengths.push_back(xyLengths[dim]);
            };
        }
        std::copy(scaleBiasMeanVarLengths.begin(),
              scaleBiasMeanVarLengths.end(),
              arrScaleBiasMeanVarLengths.begin());
    }

    std::array<ck::index_t, Rank> xyLengths; // inOutLengths
    std::array<ck::index_t, Rank> xyStrides; // inOutStrides
    std::vector<int> invariantDims;

    std::array<index_t, Rank - NumBatchNormReduceDim> arrScaleBiasMeanVarLengths;
    std::array<index_t, Rank - NumBatchNormReduceDim> arrScaleBiasMeanVarStrides;

    double epsilon = 0.0001;
    std::array<int, NumBatchNormReduceDim> reduceDims{0, 1, 2};
};

template <typename DataType>
void PerformanceConfigCKBnBwdTraining::Init(const miopen::batchnorm::ProblemDescription& problem)
{
    const auto& args       = CKArgsBNormFwd{problem};
    const auto bn_bwd_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp<DataType>>::GetInstances();
    assert(!bn_bwd_ptrs.empty());
    for(const auto& it : bn_bwd_ptrs)
    {
        auto argument_ptr = it->MakeArgumentPointer(args.xyLengths,
                                                    args.xyStrides,
                                                    args.xyStrides,
                                                    args.xyStrides,
                                                    args.reduceDims,
                                                    args.arrScaleBiasMeanVarLengths,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    args.epsilon,
                                                    PassThroughOp{},
                                                    nullptr,
                                                    nullptr,
                                                    nullptr);
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
bool PerformanceConfigCKBnBwdTraining::CheckIsSupportCKArgs(
    const miopen::batchnorm::ProblemDescription& problem) const
{
    const auto& args       = CKArgsBNormFwd{problem};
    const auto bn_bwd_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp<DataType>>::GetInstances();

    int i = 0;
    for(; i < bn_bwd_ptrs.size(); i++)
    {
        if(bn_bwd_ptrs[i]->GetTypeString() == this->kernel_id)
        {
            break;
        }
    }
    if(i == valid_kernels.size())
    {
        return false;
    }
    auto argument_ptr =
        bn_bwd_ptrs[i]->MakeArgumentPointer(args.xyLengths,
                                                    args.xyStrides,
                                                    args.xyStrides,
                                                    args.xyStrides,
                                                    args.reduceDims,
                                                    args.arrScaleBiasMeanVarLengths,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    args.epsilon,
                                                    PassThroughOp{},
                                                    nullptr,
                                                    nullptr,
                                                    nullptr);
    return bn_bwd_ptrs[i]->IsSupportedArgument(argument_ptr.get());
}

template <typename DataType>
bool CKBnBwdTraining::CheckCKApplicability(
    const miopen::batchnorm::ProblemDescription& problem) const
{
    const auto& args       = CKArgsBNormFwd{problem};
    const auto bn_bwd_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp<DataType>>::GetInstances();
    assert(!bn_bwd_ptrs.empty());

    for(const auto& it : bn_bwd_ptrs)
    {
        auto argument_ptr = it->MakeArgumentPointer(args.xyLengths,
                                                    args.xyStrides,
                                                    args.xyStrides,
                                                    args.xyStrides,
                                                    args.reduceDims,
                                                    args.arrScaleBiasMeanVarLengths,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    args.epsilon,
                                                    PassThroughOp{},
                                                    nullptr,
                                                    nullptr,
                                                    nullptr);
        if(it->IsSupportedArgument(argument_ptr.get()))
            return true;
    }
    return false;
}

template <typename DataType>
void RunCKSolution(const Handle& handle,
                   const AnyInvokeParams& primitive_parameters,
                   const miopen::batchnorm::ProblemDescription& problem,
                   const PerformanceConfigCKBnBwdTraining& config)
{
    const auto& args = CKArgsBNormFwd{problem};

    const auto bn_bwd_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp<DataType>>::GetInstances();

    int index = 0;
    for(; index < bn_bwd_ptrs.size(); index++)
    {
        if(bn_bwd_ptrs[index]->GetTypeString() == config.kernel_id)
        {
            break;
        }
    }
    assert(index < bn_bwd_ptrs.size());
    auto& bn_ptr           = bn_bwd_ptrs.at(index);
    const auto& invoke_ctx = primitive_parameters.CastTo<miopen::fusion::FusionInvokeParams>();
    assert(invoke_ctx.op_args.params[0] != nullptr);
    const auto& params = dynamic_cast<miopen::fusion::BatchNormBwdTrainingOpInvokeParam&>(
        *invoke_ctx.op_args.params[0]);

    const auto& dy_buf = invoke_ctx.in;
    const auto& dx_buf = invoke_ctx.out;

    auto argument_ptr = bn_ptr->MakeArgumentPointer(args.xyLengths,
                                                    args.xyStrides,
                                                    args.xyStrides,
                                                    args.xyStrides,
                                                    args.reduceDims,
                                                    args.arrScaleBiasMeanVarLengths,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    params.x,
                                                    dy_buf,
                                                    params.bnScale,
                                                    params.savedMean,
                                                    params.savedInvVariance,
                                                    args.epsilon,
                                                    PassThroughOp{},
                                                    dx_buf,
                                                    params.resBnScaleDiff,
                                                    params.resBnBiasDiff);

    auto invoker_ptr            = bn_ptr->MakeInvokerPointer();
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

void PerformanceConfigCKBnBwdTraining::HeuristicInit(const FusionDescription& fdesc_problem)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
#else
    const auto& bn_problem =
        fdesc_problem.GetBnProblem(0, miopen::batchnorm::Direction::Backward);
    switch(bn_problem.GetXDesc().GetType())
    {
    case miopenHalf: Init<ck::half_t>(bn_problem); break;
    case miopenInt8:
    case miopenFloat: Init<float>(bn_problem); break;
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }

#endif
}

bool PerformanceConfigCKBnBwdTraining::SetNextValue(const FusionDescription& fdesc_problem)
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

bool PerformanceConfigCKBnBwdTraining::IsValidValue() const
{
    return this->index >= 0 && this->index < valid_kernels.size();
}

bool PerformanceConfigCKBnBwdTraining::IsValid(const FusionContext&,
                                                const FusionDescription& fdesc_problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
    return false;
#else
    // Extract convolution problem from the fusion context.
    const auto& bn_problem =
        fdesc_problem.GetBnProblem(0, miopen::batchnorm::Direction::Backward);
    switch(bn_problem.GetDXDesc().GetType())
    {
    case miopenHalf: return CheckIsSupportCKArgs<ck::half_t>(bn_problem);
    case miopenInt8:
    case miopenFloat: return CheckIsSupportCKArgs<float>(bn_problem);
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return false;
#endif
}

bool PerformanceConfigCKBnBwdTraining::operator==(
    const PerformanceConfigCKBnBwdTraining& other) const
{
    return this->kernel_id == other.kernel_id;
}
PerformanceConfigCKBnBwdTraining
CKBnBwdTraining::GetDefaultPerformanceConfig(const FusionContext&,
                                              const FusionDescription& fdesc_problem) const
{
    PerformanceConfigCKBnBwdTraining pp;
    pp.HeuristicInit(fdesc_problem);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool CKBnBwdTraining::IsValidPerformanceConfig(
    const FusionContext& ctx,
    const FusionDescription& fdesc_problem,
    const PerformanceConfigCKBnBwdTraining& config) const
{
    return config.IsValid(ctx, fdesc_problem);
}

PerformanceConfigCKBnBwdTraining CKBnBwdTraining::Search(const FusionContext& ctx,
                                                           const FusionDescription& fdesc_problem,
                                                           const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, fdesc_problem, invoke_ctx);
}

bool CKBnBwdTraining::IsApplicable(const FusionContext& ctx,
                                    const FusionDescription& fdesc_problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    std::ignore = fdesc_problem;
    return false;
#else
    const auto& desc = *fdesc_problem.fusion_plan_desc;
    if(desc.op_map.empty())
        MIOPEN_THROW(miopenStatusInternalError, "desc.op_map.empty()");
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_CK_BN_BACK{}))
        return false;
    if(desc.op_map.size() != 1)
        return false;
    const auto& bn_op = dynamic_cast<BatchNormBwdTrainFusionOpDescriptor&>(*desc.op_map[0]);
    if(bn_op.kind() != miopenFusionOpBatchNormBwdTrain)
        return false;
    const auto& bn_problem =
        fdesc_problem.GetBnProblem(0, miopen::batchnorm::Direction::Backward);
    if(!bn_problem.IsLayoutNHWC())
        return false;

    const std::string arch = ctx.GetStream().GetDeviceName();
    if(arch != "gfx908" && arch != "gfx90a" && arch != "gfx1030") // add proper function for check
        return false;

    switch(bn_problem.GetXDesc().GetType())
    {
    case miopenHalf: return CheckCKApplicability<ck::half_t>(bn_problem);
    case miopenInt8:
    case miopenFloat: return CheckCKApplicability<float>(bn_problem);
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return false;
#endif
}

ConvSolution CKBnBwdTraining::GetSolution(const FusionContext&,
                                           const FusionDescription& fdesc_problem,
                                           const PerformanceConfigCKBnBwdTraining& config) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
    std::ignore = config;
    return {};
#else
    const auto& bn_problem =
        fdesc_problem.GetBnProblem(0, miopen::batchnorm::Direction::Backward);

    ConvSolution result;
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        std::ignore = kernels;
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            switch(bn_problem.GetXDesc().GetType()) // add api GetInDataType in bn_problem
            {
            case miopenHalf:
                RunCKSolution<ck::half_t>(handle, primitive_parameters, bn_problem, config);
                break;
            case miopenInt8:
            case miopenFloat:
                RunCKSolution<float>(handle, primitive_parameters, bn_problem, config);
                break;
            case miopenInt32:
            case miopenInt8x4:
            case miopenBFloat16:
            case miopenDouble:
            default: MIOPEN_THROW("Unsupported datatype");
            }
        };
    };
    return result;
#endif
}

} // namespace fusion
} // namespace solver
} // namespace miopen
