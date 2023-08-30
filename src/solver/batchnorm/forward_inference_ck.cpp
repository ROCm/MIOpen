
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
#include <miopen/solver/ck_utility_common.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <ck/library/tensor_operation_instance/gpu/batchnorm_infer.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_CK_BN_INFER)

namespace miopen {
namespace solver {
namespace fusion {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using index_t     = int32_t;
using Normalize   = ck::tensor_operation::element_wise::NormalizeInInfer;

constexpr index_t Rank                  = 4;
constexpr index_t NumBatchNormReduceDim = 3;

template <typename DataType>
using DeviceOp = ck::tensor_operation::device::DeviceElementwise<
    ck::Tuple<DataType, DataType, DataType, DataType, DataType>,
    ck::Tuple<DataType>,
    Normalize,
    Rank>;

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
        // prep for CK
        std::sort(xyStrides.begin(), xyStrides.end(), std::greater<>());
        std::rotate(xyLengths.begin() + 1, xyLengths.begin() + 2, xyLengths.end());

        aligned_scaleBiasMeanVarStrides[0] = 0;
        aligned_scaleBiasMeanVarStrides[1] = 0;
        aligned_scaleBiasMeanVarStrides[2] = 0;
        aligned_scaleBiasMeanVarStrides[3] = 1;
    }

    std::array<ck::index_t, Rank> xyLengths;
    std::array<ck::index_t, Rank> xyStrides;
    std::vector<int> invariantDims;

    std::array<index_t, Rank> aligned_scaleBiasMeanVarStrides{3};

    double epsilon = 0.0001;
    std::array<int, NumBatchNormReduceDim> reduceDims{0, 1, 2};
};

template <typename DataType>
void PerformanceConfigCKBnFwdInference::Init(const miopen::batchnorm::ProblemDescription& problem)
{
    const auto& args       = CKArgsBNormFwd{problem};
    const auto bn_fwd_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp<DataType>>::GetInstances();
    assert(!bn_fwd_ptrs.empty());
    for(const auto& it : bn_fwd_ptrs)
    {
        auto argument_ptr = it->MakeArgumentPointer(args.xyLengths,
                                                    {args.xyStrides,
                                                     args.aligned_scaleBiasMeanVarStrides,
                                                     args.aligned_scaleBiasMeanVarStrides,
                                                     args.aligned_scaleBiasMeanVarStrides,
                                                     args.aligned_scaleBiasMeanVarStrides},
                                                    {args.xyStrides},
                                                    {nullptr, nullptr, nullptr, nullptr, nullptr},
                                                    {nullptr},
                                                    Normalize{0.0});
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
bool PerformanceConfigCKBnFwdInference::CheckIsSupportCKArgs(
    const miopen::batchnorm::ProblemDescription& problem) const
{
    const auto& args       = CKArgsBNormFwd{problem};
    const auto bn_fwd_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp<DataType>>::GetInstances();

    int i = 0;
    for(; i < bn_fwd_ptrs.size(); i++)
    {
        if(bn_fwd_ptrs[i]->GetTypeString() == this->kernel_id)
        {
            break;
        }
    }
    if(i == valid_kernels.size())
    {
        return false;
    }
    auto argument_ptr =
        bn_fwd_ptrs[i]->MakeArgumentPointer(args.xyLengths,
                                            {args.xyStrides,
                                             args.aligned_scaleBiasMeanVarStrides,
                                             args.aligned_scaleBiasMeanVarStrides,
                                             args.aligned_scaleBiasMeanVarStrides,
                                             args.aligned_scaleBiasMeanVarStrides},
                                            {args.xyStrides},
                                            {nullptr, nullptr, nullptr, nullptr, nullptr},
                                            {nullptr},
                                            Normalize{0.0});
    return bn_fwd_ptrs[i]->IsSupportedArgument(argument_ptr.get());
}

template <typename DataType>
bool CKBnFwdInference::CheckCKApplicability(
    const miopen::batchnorm::ProblemDescription& problem) const
{
    const auto& args       = CKArgsBNormFwd{problem};
    const auto bn_fwd_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp<DataType>>::GetInstances();
    assert(!bn_fwd_ptrs.empty());

    for(const auto& it : bn_fwd_ptrs)
    {
        auto argument_ptr = it->MakeArgumentPointer(args.xyLengths,
                                                    {args.xyStrides,
                                                     args.aligned_scaleBiasMeanVarStrides,
                                                     args.aligned_scaleBiasMeanVarStrides,
                                                     args.aligned_scaleBiasMeanVarStrides,
                                                     args.aligned_scaleBiasMeanVarStrides},
                                                    {args.xyStrides},
                                                    {nullptr, nullptr, nullptr, nullptr, nullptr},
                                                    {nullptr},
                                                    Normalize{0.0});
        if(it->IsSupportedArgument(argument_ptr.get()))
            return true;
    }
    return false;
}

template <typename DataType>
void RunCKSolution(const Handle& handle,
                   const AnyInvokeParams& primitive_parameters,
                   const miopen::batchnorm::ProblemDescription& problem,
                   const PerformanceConfigCKBnFwdInference& config)
{
    const auto& args = CKArgsBNormFwd{problem};

    const auto bn_fwd_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp<DataType>>::GetInstances();

    int index = 0;
    for(; index < bn_fwd_ptrs.size(); index++)
    {
        if(bn_fwd_ptrs[index]->GetTypeString() == config.kernel_id)
        {
            break;
        }
    }
    assert(index < bn_fwd_ptrs.size());
    auto& bn_ptr           = bn_fwd_ptrs.at(index);
    const auto& invoke_ctx = primitive_parameters.CastTo<miopen::fusion::FusionInvokeParams>();
    assert(invoke_ctx.op_args.params[0] != nullptr);
    const auto& params = dynamic_cast<miopen::fusion::BatchNormInferenceOpInvokeParam&>(
        *invoke_ctx.op_args.params[0]);

    auto argument_ptr = bn_ptr->MakeArgumentPointer(args.xyLengths,
                                                    {args.xyStrides,
                                                     args.aligned_scaleBiasMeanVarStrides,
                                                     args.aligned_scaleBiasMeanVarStrides,
                                                     args.aligned_scaleBiasMeanVarStrides,
                                                     args.aligned_scaleBiasMeanVarStrides},
                                                    {args.xyStrides},
                                                    {invoke_ctx.in,
                                                     params.estimatedMean,
                                                     params.estimatedVariance,
                                                     params.bnScale,
                                                     params.bnBias},
                                                    {invoke_ctx.out},
                                                    Normalize{params.epsilon});

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

void PerformanceConfigCKBnFwdInference::HeuristicInit(const FusionDescription& fdesc_problem)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
#else
    const auto& bn_problem =
        fdesc_problem.GetBnProblem(0, miopen::batchnorm::Direction::ForwardInference);
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

bool PerformanceConfigCKBnFwdInference::SetNextValue(const FusionDescription& fdesc_problem)
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

bool PerformanceConfigCKBnFwdInference::IsValidValue() const
{
    return this->index >= 0 && this->index < valid_kernels.size();
}

bool PerformanceConfigCKBnFwdInference::IsValid(const FusionContext&,
                                                const FusionDescription& fdesc_problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
    return false;
#else
    // Extract convolution problem from the fusion context.
    const auto& bn_problem =
        fdesc_problem.GetBnProblem(0, miopen::batchnorm::Direction::ForwardInference);
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

bool PerformanceConfigCKBnFwdInference::operator==(
    const PerformanceConfigCKBnFwdInference& other) const
{
    return this->kernel_id == other.kernel_id;
}
PerformanceConfigCKBnFwdInference
CKBnFwdInference::GetDefaultPerformanceConfig(const FusionContext&,
                                              const FusionDescription& fdesc_problem) const
{
    PerformanceConfigCKBnFwdInference pp;
    pp.HeuristicInit(fdesc_problem);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool CKBnFwdInference::IsValidPerformanceConfig(
    const FusionContext& ctx,
    const FusionDescription& fdesc_problem,
    const PerformanceConfigCKBnFwdInference& config) const
{
    return config.IsValid(ctx, fdesc_problem);
}

PerformanceConfigCKBnFwdInference CKBnFwdInference::Search(const FusionContext& ctx,
                                                           const FusionDescription& fdesc_problem,
                                                           const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, fdesc_problem, invoke_ctx);
}

bool CKBnFwdInference::IsApplicable(const FusionContext& ctx,
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
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_CK_BN_INFER{}))
        return false;
    if(desc.op_map.size() != 1)
        return false;
    const auto& bn_op = dynamic_cast<BatchNormInferenceFusionOpDescriptor&>(*desc.op_map[0]);
    if(bn_op.kind() != miopenFusionOpBatchNormInference)
        return false;
    const auto& bn_problem =
        fdesc_problem.GetBnProblem(0, miopen::batchnorm::Direction::ForwardInference);
    if(!bn_problem.IsLayoutNHWC())
        return false;
    if(!ck_utility::is_ck_supported_hardware(ctx.GetStream()))
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

ConvSolution CKBnFwdInference::GetSolution(const FusionContext&,
                                           const FusionDescription& fdesc_problem,
                                           const PerformanceConfigCKBnFwdInference& config) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
    std::ignore = config;
    return {};
#else
    const auto& bn_problem =
        fdesc_problem.GetBnProblem(0, miopen::batchnorm::Direction::ForwardInference);

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
