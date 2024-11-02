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
#include <miopen/bfloat16.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <ck/library/tensor_operation_instance/gpu/batchnorm_backward.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CK_BN_BACK)

namespace miopen {
namespace solver {
namespace batchnorm {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

using PassThroughOp = ck::tensor_operation::element_wise::PassThrough;
using index_t       = int32_t;

constexpr index_t Rank                  = 4;
constexpr index_t NumBatchNormReduceDim = 3;

using F16  = ck::half_t;
using F32  = float;
using F64  = double;
using BF16 = ushort;

template <typename XDataType,
          typename DxDataType,
          typename DyDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType>
using DeviceOpBNBwdPtrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceBatchNormBwd<XDataType,
                                                     DxDataType,
                                                     DyDataType,
                                                     AccDataType,
                                                     ScaleDataType,
                                                     DscaleDbiasDataType,
                                                     MeanVarDataType,
                                                     PassThroughOp,
                                                     Rank,
                                                     NumBatchNormReduceDim>>;

struct CKArgsBNormBwd
{
    CKArgsBNormBwd(const miopen::batchnorm::ProblemDescription& problem)
    {
        std::copy(problem.GetXDesc().GetLengths().begin(),
                  problem.GetXDesc().GetLengths().end(),
                  lens.begin());

        std::copy(problem.GetXDesc().GetStrides().begin(),
                  problem.GetXDesc().GetStrides().end(),
                  in_strides.begin());
        arrScaleBiasMeanVarLengths[0] = lens[1]; // get channel
        arrScaleBiasMeanVarStrides[0] = 1;

        // prep for CK
        std::sort(in_strides.begin(), in_strides.end(), std::greater<>());

        if(problem.IsLayoutNHWC())
        {
            std::rotate(lens.begin() + 1, lens.begin() + 2, lens.end());
            reduceDims = {0, 1, 2};
        }
        else if(problem.IsLayoutNCHW())
        {
            reduceDims = {0, 2, 3};
        }
        else
        {
            MIOPEN_THROW(miopenStatusInternalError,
                         "BnCKBwd operation does not support this data layout");
        }
    }

    CKArgsBNormBwd(const CKArgsBNormBwd&) = default;
    CKArgsBNormBwd(CKArgsBNormBwd&&)      = default;
    CKArgsBNormBwd& operator=(const CKArgsBNormBwd&) = default;

    template <typename InvokerPtr, typename InvokerParams>
    auto MakeArgPtr(const InvokerPtr& invoker_ptr, const InvokerParams& data_ctx) const
    {
        return invoker_ptr->MakeArgumentPointer(lens,
                                                in_strides,
                                                in_strides,
                                                in_strides,
                                                reduceDims,
                                                arrScaleBiasMeanVarLengths,
                                                arrScaleBiasMeanVarStrides,
                                                arrScaleBiasMeanVarStrides,
                                                arrScaleBiasMeanVarStrides,
                                                data_ctx.x,
                                                data_ctx.dy,
                                                data_ctx.bnScale,
                                                data_ctx.savedMean,
                                                data_ctx.savedInvVariance,
                                                epsilon,
                                                PassThroughOp{},
                                                data_ctx.dx,
                                                data_ctx.resultBnScaleDiff,
                                                data_ctx.resultBnBiasDiff);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& invoker_ptr) const
    {
        auto arg_ptr = MakeArgPtr(invoker_ptr, miopen::batchnorm::BwdInvokeParams{});
        return invoker_ptr->IsSupportedArgument(arg_ptr.get());
    }

    std::array<ck::index_t, Rank> lens;
    std::array<ck::index_t, Rank> in_strides;
    std::vector<int> invariantDims;

    std::array<index_t, Rank - NumBatchNormReduceDim> arrScaleBiasMeanVarLengths;
    std::array<index_t, Rank - NumBatchNormReduceDim> arrScaleBiasMeanVarStrides;

    double epsilon = 1e-5;
    std::array<int, NumBatchNormReduceDim> reduceDims;
};

template <typename XDataType,
          typename DxDataType,
          typename DyDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType>
void PerformanceConfigBnCKBwdBackward::Init(
    const miopen::batchnorm::ProblemDescription& problem_desc)
{
    const auto& args       = CKArgsBNormBwd{problem_desc};
    const auto bn_bwd_ptrs = DeviceOpBNBwdPtrs<XDataType,
                                               DxDataType,
                                               DyDataType,
                                               AccDataType,
                                               ScaleDataType,
                                               DscaleDbiasDataType,
                                               MeanVarDataType>::GetInstances();
    if(bn_bwd_ptrs.empty())
        MIOPEN_THROW(miopenStatusInternalError, "BnCKBwdBackward bn_bwd_ptrs empty");

    for(const auto& it : bn_bwd_ptrs)
    {
        auto argument_ptr = it->MakeArgumentPointer(args.lens,
                                                    args.in_strides,
                                                    args.in_strides,
                                                    args.in_strides,
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
                                                    0.0,
                                                    PassThroughOp{},
                                                    nullptr,
                                                    nullptr,
                                                    nullptr);
        if(it->IsSupportedArgument(argument_ptr.get()))
        {
            valid_kernels.push_back(it->GetTypeString());
        }
    }

    if(valid_kernels.empty())
        MIOPEN_THROW(miopenStatusInternalError, "BnCKBwdBackward valid_kernels empty");

    this->index     = 0;
    this->kernel_id = valid_kernels[0];
}

template <typename XDataType,
          typename DxDataType,
          typename DyDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType>
bool PerformanceConfigBnCKBwdBackward::CheckIsSupportCKArgs(
    const miopen::batchnorm::ProblemDescription& problem) const
{
    return IsCKArgsSupported<DeviceOpBNBwdPtrs<XDataType,
                                               DxDataType,
                                               DyDataType,
                                               AccDataType,
                                               ScaleDataType,
                                               DscaleDbiasDataType,
                                               MeanVarDataType>,
                             CKArgsBNormBwd>(problem, this->kernel_id);
}

template <typename XDataType,
          typename DxDataType,
          typename DyDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType>
static bool CheckCKApplicability(const miopen::batchnorm::ProblemDescription& problem)
{
    return IsCKApplicable<DeviceOpBNBwdPtrs<XDataType,
                                            DxDataType,
                                            DyDataType,
                                            AccDataType,
                                            ScaleDataType,
                                            DscaleDbiasDataType,
                                            MeanVarDataType>,
                          CKArgsBNormBwd>(problem);
}
#endif

void PerformanceConfigBnCKBwdBackward::HeuristicInit(
    const miopen::batchnorm::ProblemDescription& problem_desc)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem_desc;
#else
    switch(problem_desc.GetXDesc().GetType())
    {
    case miopenHalf: Init<F16, F32, F32, F32, F16, F32, F32>(problem_desc); break;
    case miopenBFloat16: Init<BF16, F32, F32, F32, BF16, F32, F32>(problem_desc); break;
    case miopenFloat: Init<F32, F32, F32, F32, F32, F32, F32>(problem_desc); break;
    case miopenDouble: Init<F64, F64, F64, F64, F64, F64, F64>(problem_desc); break;
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64:
    default: MIOPEN_THROW("Unsupported datatype");
    }

#endif
}

bool PerformanceConfigBnCKBwdBackward::SetNextValue(
    const miopen::batchnorm::ProblemDescription& problem_desc)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem_desc;
    return false;
#else
    if(this->valid_kernels.empty())
    {
        this->HeuristicInit(problem_desc);
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

bool PerformanceConfigBnCKBwdBackward::IsValidValue() const
{
    return this->index >= 0 && this->index < valid_kernels.size();
}

bool PerformanceConfigBnCKBwdBackward::IsValid(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& problem_desc) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem_desc;
    return false;
#else
    switch(problem_desc.GetXDesc().GetType())
    {
    case miopenHalf: return CheckIsSupportCKArgs<F16, F32, F32, F32, F16, F32, F32>(problem_desc);
    case miopenBFloat16:
        return CheckIsSupportCKArgs<BF16, F32, F32, F32, BF16, F32, F32>(problem_desc);
    case miopenFloat: return CheckIsSupportCKArgs<F32, F32, F32, F32, F32, F32, F32>(problem_desc);
    case miopenDouble: return CheckIsSupportCKArgs<F64, F64, F64, F64, F64, F64, F64>(problem_desc);
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

bool PerformanceConfigBnCKBwdBackward::operator==(
    const PerformanceConfigBnCKBwdBackward& other) const
{
    return this->kernel_id == other.kernel_id;
}

PerformanceConfigBnCKBwdBackward BnCKBwdBackward::GetDefaultPerformanceConfig(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& problem_desc) const
{
    PerformanceConfigBnCKBwdBackward pp;
    pp.HeuristicInit(problem_desc);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool BnCKBwdBackward::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const miopen::batchnorm::ProblemDescription& problem_desc,
    const PerformanceConfigBnCKBwdBackward& config) const
{
    return config.IsValid(ctx, problem_desc);
}

PerformanceConfigBnCKBwdBackward
BnCKBwdBackward::Search(const ExecutionContext& ctx,
                        const miopen::batchnorm::ProblemDescription& problem_desc,
                        const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem_desc, invoke_ctx);
}

bool BnCKBwdBackward::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::batchnorm::ProblemDescription& bn_problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_CK_BN_BACK))
        return false;
    if(!bn_problem.IsLayoutNHWC() && !bn_problem.IsLayoutNCHW())
        return false;
    if(!ck_utility::is_ck_supported_hardware(context.GetStream()))
        return false;
    if(!bn_problem.Is2D())
        return false;
    if(bn_problem.GetDirection() != miopen::batchnorm::Direction::Backward)
        return false;
    if(bn_problem.GetMode() != miopenBNSpatial)
        return false;
    if(!bn_problem.Is2D())
        return false;
    if(!IsCKBwdTypeValid(bn_problem))
        return false;

    switch(bn_problem.GetXDesc().GetType())
    {
    case miopenHalf: return CheckCKApplicability<F16, F32, F32, F32, F16, F32, F32>(bn_problem);
    case miopenBFloat16:
        return CheckCKApplicability<BF16, F32, F32, F32, BF16, F32, F32>(bn_problem);
    case miopenFloat: return CheckCKApplicability<F32, F32, F32, F32, F32, F32, F32>(bn_problem);
    case miopenDouble: return CheckCKApplicability<F64, F64, F64, F64, F64, F64, F64>(bn_problem);
    case miopenInt64:
    case miopenInt32:
    case miopenInt8:
    case miopenFloat8:
    case miopenBFloat8: break;
    }
#endif
    return false;
}

template <typename InvokerFactoryMakerNHWC>
ConvSolution MakeAnyInvokerFactory(const miopen::batchnorm::ProblemDescription& problem,
                                   InvokerFactoryMakerNHWC&& invoker_factory_maker_nhwc)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    switch(problem.GetXDesc().GetType())
    {
    case miopenFloat: return invoker_factory_maker_nhwc(F32{});
    case miopenDouble: return invoker_factory_maker_nhwc(F64{});
    case miopenHalf: return invoker_factory_maker_nhwc(F16{});
    case miopenBFloat16: return invoker_factory_maker_nhwc(BF16{});
    default:
        MIOPEN_THROW(miopenStatusInternalError,
                     "BnCKBwdBackward operation does not support this data type");
    }
#else
    return {};
#endif
}

ConvSolution BnCKBwdBackward::GetSolution(
    [[maybe_unused]] const ExecutionContext&,
    [[maybe_unused]] const miopen::batchnorm::ProblemDescription& bn_problem,
    [[maybe_unused]] const PerformanceConfigBnCKBwdBackward& config) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    return MakeAnyInvokerFactory(
        bn_problem,
        [&](auto data_type_val) {
            using T = decltype(data_type_val);

            using AccTy = std::conditional_t<std::is_same_v<T, F64>,
                                             T,    // T==F64
                                             F32>; // T==F32
            return InitAnyInvokerFactory<DeviceOpBNBwdPtrs<T, AccTy, AccTy, AccTy, T, AccTy, AccTy>,
                                         CKArgsBNormBwd,
                                         miopen::batchnorm::BwdInvokeParams,
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
