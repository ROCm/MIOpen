
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
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/batchnorm_forward.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_CK_BN_FWD_TRAINING)

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
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
using DeviceOpBNFwdTrainingPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        ck::tensor_operation::device::DeviceBatchNormFwd<XDataType,
                                                         YDataType,
                                                         AccDataType,
                                                         ScaleDataType,
                                                         BiasDataType,
                                                         MeanVarDataType,
                                                         PassThroughOp,
                                                         Rank,
                                                         NumBatchNormReduceDim>>;

struct CKArgsBNormFwdTraining
{
    CKArgsBNormFwdTraining(const miopen::batchnorm::ProblemDescription& problem)
    {
        std::copy(problem.GetXDesc().GetLengths().begin(),
                  problem.GetXDesc().GetLengths().end(),
                  xyLengths.begin());

        std::copy(problem.GetXDesc().GetStrides().begin(),
                  problem.GetXDesc().GetStrides().end(),
                  xyStrides.begin());
        arrScaleBiasMeanVarLengths[0] = xyLengths[1]; // get channel
        arrScaleBiasMeanVarStrides[0] = 1;

        // prep for CK
        std::sort(xyStrides.begin(), xyStrides.end(), std::greater<>());
        std::rotate(xyLengths.begin() + 1, xyLengths.begin() + 2, xyLengths.end());
    }

    CKArgsBNormFwdTraining(const CKArgsBNormFwdTraining&) = default;
    CKArgsBNormFwdTraining(CKArgsBNormFwdTraining&&)      = default;
    CKArgsBNormFwdTraining& operator=(const CKArgsBNormFwdTraining&) = default;

    template <typename InvokerPtr, typename InvokerParams>
    auto MakeArgPtr(const InvokerPtr& invoker_ptr, const InvokerParams& data_ctx) const
    {
        return invoker_ptr->MakeArgumentPointer(xyLengths,
                                                xyStrides,
                                                xyStrides,
                                                reduceDims,
                                                arrScaleBiasMeanVarLengths,
                                                arrScaleBiasMeanVarStrides,
                                                arrScaleBiasMeanVarStrides,
                                                arrScaleBiasMeanVarStrides,
                                                data_ctx.x,
                                                data_ctx.bnScale,
                                                data_ctx.bnBias,
                                                data_ctx.epsilon,
                                                PassThroughOp{},
                                                data_ctx.y,
                                                data_ctx.resultSaveMean,
                                                data_ctx.resultSaveInvVariance,
                                                data_ctx.expAvgFactor,
                                                data_ctx.resultRunningMean,
                                                data_ctx.resultRunningVariance);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& invoker_ptr) const
    {
        auto arg_ptr = MakeArgPtr(invoker_ptr, miopen::batchnorm::InvokeParams{});
        return invoker_ptr->IsSupportedArgument(arg_ptr.get());
    }

    std::array<ck::index_t, Rank> xyLengths;
    std::array<ck::index_t, Rank> xyStrides;
    std::vector<int> invariantDims;

    std::array<index_t, Rank - NumBatchNormReduceDim> arrScaleBiasMeanVarLengths;
    std::array<index_t, Rank - NumBatchNormReduceDim> arrScaleBiasMeanVarStrides;

    std::array<int, NumBatchNormReduceDim> reduceDims{0, 1, 2};
};

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
static bool CheckCKApplicability(const miopen::batchnorm::ProblemDescription& problem)
{
    return IsCKApplicable<DeviceOpBNFwdTrainingPtrs<XDataType,
                                                    YDataType,
                                                    AccDataType,
                                                    ScaleDataType,
                                                    BiasDataType,
                                                    MeanVarDataType>,
                          CKArgsBNormFwdTraining>(problem);
}
#endif

bool BnCKFwdTraining::IsApplicable(const ExecutionContext& context,
                                   const miopen::batchnorm::ProblemDescription& bn_problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = context;
    std::ignore = bn_problem;
    return false;
#else
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_CK_BN_FWD_TRAINING{}))
        return false;
    if(!bn_problem.IsLayoutNHWC())
        return false;
    if(!ck_utility::is_ck_supported_hardware(context.GetStream()))
        return false;

    switch(bn_problem.GetXDesc().GetType())
    {
    case miopenHalf: return CheckCKApplicability<F16, F16, F32, F16, F16, F32>(bn_problem);
    case miopenFloat: return CheckCKApplicability<F32, F32, F32, F32, F32, F32>(bn_problem);
    case miopenDouble: return CheckCKApplicability<F64, F64, F64, F64, F64, F64>(bn_problem);
    case miopenBFloat16: return CheckCKApplicability<BF16, BF16, F32, BF16, BF16, F32>(bn_problem);
    case miopenInt32:
    case miopenInt8:
    case miopenInt8x4:
    case miopenBFloat8:
    case miopenFloat8:
    default: MIOPEN_THROW("BnCKFwdTraining operation does not supprot this data type");
    }
    return false;
#endif
}

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
ConvSolution MakeAnyInvokerFactory(const miopen::batchnorm::ProblemDescription& bn_problem)
{
    const auto& valid_kernel_ids = FillValidKernelsIDs<DeviceOpBNFwdTrainingPtrs<XDataType,
                                                                                 YDataType,
                                                                                 AccDataType,
                                                                                 ScaleDataType,
                                                                                 BiasDataType,
                                                                                 MeanVarDataType>,
                                                       CKArgsBNormFwdTraining>(bn_problem);
    assert(!valid_kernel_ids.empty());
    const auto& kernel_id = valid_kernel_ids[0];
    return InitAnyInvokerFactory<DeviceOpBNFwdTrainingPtrs<XDataType,
                                                           YDataType,
                                                           AccDataType,
                                                           ScaleDataType,
                                                           BiasDataType,
                                                           MeanVarDataType>,
                                 CKArgsBNormFwdTraining,
                                 miopen::batchnorm::InvokeParams>(bn_problem, kernel_id);
}
#endif

ConvSolution BnCKFwdTraining::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::batchnorm::ProblemDescription& bn_problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    switch(bn_problem.GetXDesc().GetType())
    {

    case miopenFloat: return MakeAnyInvokerFactory<F32, F32, F32, F32, F32, F32>(bn_problem);
    case miopenDouble: return MakeAnyInvokerFactory<F64, F64, F64, F64, F64, F64>(bn_problem);
    case miopenHalf: return MakeAnyInvokerFactory<F16, F16, F32, F16, F16, F32>(bn_problem);
    case miopenBFloat16: return MakeAnyInvokerFactory<BF16, BF16, F32, BF16, BF16, F32>(bn_problem);
    case miopenInt8:
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat8:
    case miopenFloat8:
    default:
        MIOPEN_THROW(miopenStatusInternalError, "BnCKFwdTraining operation not for this data type");
    }
#endif
    return {};
}

} // namespace batchnorm
} // namespace solver
} // namespace miopen
