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
#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/bfloat16.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/batchnorm_backward.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_CK_BN_BACK)

namespace miopen {
namespace solver {
namespace batchnorm {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using index_t     = int32_t;

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
                                                     PassThrough,
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
                  strides.begin());
        arrScaleBiasMeanVarLengths[0] = lens[1]; // get channel
        arrScaleBiasMeanVarStrides[0] = 1;

        // prep for CK
        std::sort(strides.begin(), strides.end(), std::greater<>());
        std::rotate(lens.begin() + 1, lens.begin() + 2, lens.end());
    }

    CKArgsBNormBwd(const CKArgsBNormBwd&) = default;
    CKArgsBNormBwd(CKArgsBNormBwd&&)      = default;
    CKArgsBNormBwd& operator=(const CKArgsBNormBwd&) = default;

    template <typename InvokerPtr, typename InvokerParams>
    auto MakeArgPtr(const InvokerPtr& invoker_ptr, const InvokerParams& data_ctx) const
    {
        return invoker_ptr->MakeArgumentPointer(lens,
                                                strides,
                                                strides,
                                                strides,
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
                                                PassThrough{},
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

    std::array<ck::index_t, Rank> lens;    // inOutLengths
    std::array<ck::index_t, Rank> strides; // inOutStrides
    std::vector<int> invariantDims;

    std::array<index_t, Rank - NumBatchNormReduceDim> arrScaleBiasMeanVarLengths;
    std::array<index_t, Rank - NumBatchNormReduceDim> arrScaleBiasMeanVarStrides;

    double epsilon = 1e-5;
    std::array<int, NumBatchNormReduceDim> reduceDims{0, 1, 2};
};

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

bool BnCKBwdBackward::IsApplicable(const ExecutionContext& context,
                                   const miopen::batchnorm::ProblemDescription& bn_problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = context;
    std::ignore = bn_problem;
    return false;
#else
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_CK_BN_BACK{}))
        return false;
    if(!bn_problem.IsLayoutNHWC())
        return false;
    if(!ck_utility::is_ck_supported_hardware(context.GetStream()))
        return false;
    if(bn_problem.GetXDesc().GetType() != bn_problem.GetScaleBiasDiffDesc().GetType())
        return false;

    switch(bn_problem.GetXDesc().GetType())
    {
    case miopenFloat: return CheckCKApplicability<F32, F32, F32, F32, F32, F32, F32>(bn_problem);
    case miopenDouble: return CheckCKApplicability<F64, F64, F64, F64, F64, F64, F64>(bn_problem);
    case miopenHalf: return CheckCKApplicability<F16, F32, F32, F32, F16, F32, F32>(bn_problem);
    case miopenBFloat16:
        return CheckCKApplicability<BF16, F32, F32, F32, BF16, F32, F32>(bn_problem);
    case miopenInt32:
    case miopenInt8:
    case miopenInt8x4:
    case miopenBFloat8:
    case miopenFloat8:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return false;
#endif
}

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
template <typename XDataType,
          typename DxDataType,
          typename DyDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType>
ConvSolution MakeAnyInvokerFactory(const miopen::batchnorm::ProblemDescription& bn_problem)
{
    const auto& valid_kernel_ids = FillValidKernelsIDs<DeviceOpBNBwdPtrs<XDataType,
                                                                         DxDataType,
                                                                         DyDataType,
                                                                         AccDataType,
                                                                         ScaleDataType,
                                                                         DscaleDbiasDataType,
                                                                         MeanVarDataType>,
                                                       CKArgsBNormBwd>(bn_problem);
    assert(!valid_kernel_ids.empty());
    const auto& kernel_id = valid_kernel_ids[0];
    return InitAnyInvokerFactory<DeviceOpBNBwdPtrs<XDataType,
                                                   DxDataType,
                                                   DyDataType,
                                                   AccDataType,
                                                   ScaleDataType,
                                                   DscaleDbiasDataType,
                                                   MeanVarDataType>,
                                 CKArgsBNormBwd,
                                 miopen::batchnorm::BwdInvokeParams>(bn_problem, kernel_id);
}
#endif

ConvSolution BnCKBwdBackward::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::batchnorm::ProblemDescription& bn_problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    switch(bn_problem.GetXDesc().GetType())
    {

    case miopenFloat: return MakeAnyInvokerFactory<F32, F32, F32, F32, F32, F32, F32>(bn_problem);
    case miopenDouble: return MakeAnyInvokerFactory<F64, F64, F64, F64, F64, F64, F64>(bn_problem);
    case miopenHalf: return MakeAnyInvokerFactory<F16, F32, F32, F32, F16, F32, F32>(bn_problem);
    case miopenBFloat16:
        return MakeAnyInvokerFactory<BF16, F32, F32, F32, BF16, F32, F32>(bn_problem);
    case miopenInt8:
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat8:
    case miopenFloat8:
    default:
        MIOPEN_THROW(miopenStatusInternalError, "BnCKBwdBackward operation not for this data type");
    }
#endif
    return {};
}

} // namespace batchnorm
} // namespace solver
} // namespace miopen
