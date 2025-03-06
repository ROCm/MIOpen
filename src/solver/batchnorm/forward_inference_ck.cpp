
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

#include <miopen/batchnorm/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batch_norm.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <ck/library/tensor_operation_instance/gpu/batchnorm_infer.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CK_BN_INFER)

namespace miopen {
namespace solver {
namespace batchnorm {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using index_t     = int32_t;
using Normalize   = ck::tensor_operation::element_wise::NormalizeInInfer;

constexpr index_t Rank                  = 4;
constexpr index_t NumBatchNormReduceDim = 3;

using F16  = ck::half_t;
using F32  = float;
using F64  = double;
using BF16 = ushort;

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
using DeviceOpBnFwdInfPtrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceElementwise<
        ck::Tuple<XDataType, MeanVarDataType, MeanVarDataType, ScaleDataType, BiasDataType>,
        ck::Tuple<YDataType>,
        Normalize,
        Rank>>;

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

        if(problem.IsLayoutNHWC())
        {
            std::rotate(xyLengths.begin() + 1, xyLengths.begin() + 2, xyLengths.end());
            reduceDims                         = {0, 1, 2};
            aligned_scaleBiasMeanVarStrides[0] = 0;
            aligned_scaleBiasMeanVarStrides[1] = 0;
            aligned_scaleBiasMeanVarStrides[2] = 0;
            aligned_scaleBiasMeanVarStrides[3] = 1;
        }
        else if(problem.IsLayoutNCHW())
        {
            reduceDims                         = {0, 2, 3};
            aligned_scaleBiasMeanVarStrides[0] = 0;
            aligned_scaleBiasMeanVarStrides[1] = 1;
            aligned_scaleBiasMeanVarStrides[2] = 0;
            aligned_scaleBiasMeanVarStrides[3] = 0;
        }
        else
        {
            MIOPEN_THROW(miopenStatusInternalError,
                         "BnCKFwdInference operation does not support this data layout");
        }
    }

    std::array<ck::index_t, Rank> xyLengths;
    std::array<ck::index_t, Rank> xyStrides;
    std::vector<int> invariantDims;

    std::array<index_t, Rank> aligned_scaleBiasMeanVarStrides{3};
    std::array<index_t, Rank - NumBatchNormReduceDim> arrScaleBiasMeanVarStrides;

    std::array<int, NumBatchNormReduceDim> reduceDims;

    template <typename InvokerPtr, typename InvokerParams>
    auto MakeArgPtr(const InvokerPtr& invoker_ptr, const InvokerParams& data_ctx) const
    {
        return invoker_ptr->MakeArgumentPointer(xyLengths,
                                                {xyStrides,
                                                 aligned_scaleBiasMeanVarStrides,
                                                 aligned_scaleBiasMeanVarStrides,
                                                 aligned_scaleBiasMeanVarStrides,
                                                 aligned_scaleBiasMeanVarStrides},
                                                {xyStrides},
                                                {data_ctx.x,
                                                 data_ctx.estimatedMean,
                                                 data_ctx.estimatedVariance,
                                                 data_ctx.bnScale,
                                                 data_ctx.bnBias},
                                                {data_ctx.y},
                                                Normalize{data_ctx.epsilon});
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& invoker_ptr) const
    {
        auto arg_ptr = MakeArgPtr(invoker_ptr, miopen::batchnorm::InfInvokeParams{});
        return invoker_ptr->IsSupportedArgument(arg_ptr.get());
    }
};

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
void PerformanceConfigBnCKFwdInference::Init(
    const miopen::batchnorm::ProblemDescription& problem_desc)
{
    const auto& args = CKArgsBNormFwd{problem_desc};
    const auto bn_fwd_ptrs =
        DeviceOpBnFwdInfPtrs<XDataType, YDataType, ScaleDataType, BiasDataType, MeanVarDataType>::
            GetInstances();
    if(bn_fwd_ptrs.empty())
        MIOPEN_THROW(miopenStatusInternalError, "BnCKFwdInference bn_fwd_ptrs empty");

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

    if(valid_kernels.empty())
        MIOPEN_THROW(miopenStatusInternalError, "BnCKFwdInference valid_kernels empty");
    this->index     = 0;
    this->kernel_id = valid_kernels[0];
}

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
bool PerformanceConfigBnCKFwdInference::CheckIsSupportCKArgs(
    const miopen::batchnorm::ProblemDescription& problem) const
{
    return IsCKArgsSupported<
        DeviceOpBnFwdInfPtrs<XDataType, YDataType, ScaleDataType, BiasDataType, MeanVarDataType>,
        CKArgsBNormFwd>(problem, this->kernel_id);
}

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
static bool CheckCKApplicability(const miopen::batchnorm::ProblemDescription& problem)
{
    return IsCKApplicable<
        DeviceOpBnFwdInfPtrs<XDataType, YDataType, ScaleDataType, BiasDataType, MeanVarDataType>,
        CKArgsBNormFwd>(problem);
}
#endif

void PerformanceConfigBnCKFwdInference::HeuristicInit(
    const miopen::batchnorm::ProblemDescription& problem_desc)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem_desc;
#else
    switch(problem_desc.GetXDesc().GetType())
    {
    case miopenHalf: Init<F16, F16, F32, F16, F16, F32>(problem_desc); break;
    case miopenBFloat16: Init<BF16, BF16, F32, BF16, BF16, F32>(problem_desc); break;
    case miopenFloat: Init<F32, F32, F32, F32, F32, F32>(problem_desc); break;
    case miopenDouble: Init<F64, F64, F64, F64, F64, F64>(problem_desc); break;
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64:
    default: MIOPEN_THROW("Unsupported datatype");
    }

#endif
}

bool PerformanceConfigBnCKFwdInference::SetNextValue(
    const miopen::batchnorm::ProblemDescription& problem_desc)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem_desc;
    return false;
#else
    if(this->valid_kernels.empty())
    {
        this->HeuristicInit(problem_desc);
        if(valid_kernels.empty())
            MIOPEN_THROW(miopenStatusInternalError, "BnCKFwdInference valid_kernels empty");
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

bool PerformanceConfigBnCKFwdInference::IsValidValue() const
{
    return this->index >= 0 && this->index < valid_kernels.size();
}

bool PerformanceConfigBnCKFwdInference::IsValid(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& problem_desc) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem_desc;
    return false;
#else
    switch(problem_desc.GetXDesc().GetType())
    {
    case miopenHalf: return CheckIsSupportCKArgs<F16, F16, F32, F16, F16, F32>(problem_desc);
    case miopenBFloat16:
        return CheckIsSupportCKArgs<BF16, BF16, F32, BF16, BF16, F32>(problem_desc);
    case miopenFloat: return CheckIsSupportCKArgs<F32, F32, F32, F32, F32, F32>(problem_desc);
    case miopenDouble: return CheckIsSupportCKArgs<F64, F64, F64, F64, F64, F64>(problem_desc);
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return false;
#endif
}

bool PerformanceConfigBnCKFwdInference::operator==(
    const PerformanceConfigBnCKFwdInference& other) const
{
    return this->kernel_id == other.kernel_id;
}

PerformanceConfigBnCKFwdInference BnCKFwdInference::GetDefaultPerformanceConfig(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& problem_desc) const
{
    PerformanceConfigBnCKFwdInference pp;
    pp.HeuristicInit(problem_desc);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool BnCKFwdInference::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const miopen::batchnorm::ProblemDescription& problem_desc,
    const PerformanceConfigBnCKFwdInference& config) const
{
    return config.IsValid(ctx, problem_desc);
}

PerformanceConfigBnCKFwdInference
BnCKFwdInference::Search(const ExecutionContext& ctx,
                         const miopen::batchnorm::ProblemDescription& problem_desc,
                         const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem_desc, invoke_ctx);
}

bool BnCKFwdInference::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::batchnorm::ProblemDescription& bn_problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_CK_BN_INFER))
        return false;
    if(!bn_problem.IsLayoutNHWC() && !bn_problem.IsLayoutNCHW())
        return false;
    if(!ck_utility::is_ck_supported_hardware(context.GetStream()))
        return false;
    if(!bn_problem.Is2D())
        return false;
    if(bn_problem.GetDirection() != miopen::batchnorm::Direction::ForwardInference)
        return false;
    if(bn_problem.GetMode() != miopenBNSpatial)
        return false;
    if(!IsCKInferTypeValid(bn_problem))
        return false;

    switch(bn_problem.GetXDesc().GetType())
    {
    case miopenHalf: return CheckCKApplicability<F16, F16, F32, F16, F16, F32>(bn_problem);
    case miopenBFloat16: return CheckCKApplicability<BF16, BF16, F32, BF16, BF16, F32>(bn_problem);
    case miopenFloat: return CheckCKApplicability<F32, F32, F32, F32, F32, F32>(bn_problem);
    case miopenDouble: return CheckCKApplicability<F64, F64, F64, F64, F64, F64>(bn_problem);
    case miopenInt64:
    case miopenInt32:
    case miopenInt8:
    case miopenFloat8:
    case miopenBFloat8: break;
    }
#endif
    return false;
}

template <typename InvokerFactoryMaker>
ConvSolution MakeAnyInvokerFactory(const miopen::batchnorm::ProblemDescription& problem,
                                   InvokerFactoryMaker&& invoker_factory_maker)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    switch(problem.GetXDesc().GetType())
    {
    case miopenFloat: return invoker_factory_maker(F32{});
    case miopenDouble: return invoker_factory_maker(F64{});
    case miopenHalf: return invoker_factory_maker(F16{});
    case miopenBFloat16: return invoker_factory_maker(BF16{});
    default:
        MIOPEN_THROW(miopenStatusInternalError,
                     "BnCKFwdInference operation does not support this data type");
    }
#else
    return {};
#endif
}

ConvSolution BnCKFwdInference::GetSolution(
    [[maybe_unused]] const ExecutionContext&,
    [[maybe_unused]] const miopen::batchnorm::ProblemDescription& bn_problem,
    [[maybe_unused]] const PerformanceConfigBnCKFwdInference& config) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    return MakeAnyInvokerFactory(
        bn_problem,
        [&](auto data_type_val) {
            using T = decltype(data_type_val);

            using AccTy = std::conditional_t<std::is_same_v<T, F64>,
                                             T,    // T==F64
                                             F32>; // T==F32
            return InitAnyInvokerFactory<DeviceOpBnFwdInfPtrs<T, T, T, T, AccTy>,
                                         CKArgsBNormFwd,
                                         miopen::batchnorm::InfInvokeParams,
                                         miopen::batchnorm::ProblemDescription>(bn_problem,
                                                                                config.kernel_id);
        }
        // Todo: InvokerFactoryMakerNCHW
    );
#else
    std::ignore = bn_problem;
    std::ignore = config;
    return {};
#endif
}

} // namespace batchnorm
} // namespace solver
} // namespace miopen
