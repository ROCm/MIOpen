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

#include <miopen/norm/solvers.hpp>
#include <miopen/norm/invoke_params.hpp>
#include <miopen/layernorm.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/normalization.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_CK_LN)

namespace miopen {
namespace solver {
namespace norm {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

using F16  = ck::half_t;
using F32  = float;
using F64  = double;
using BF16 = ushort;

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename ComputeDataType,
          typename YDataType>
using DeviceOp = ck::tensor_operation::device::DeviceNormalization<
    XDataType,
    GammaDataType,
    BetaDataType,
    ComputeDataType,
    YDataType,
    ck::tensor_operation::element_wise::PassThrough,
    2,
    1>;
template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename ComputeDataType,
          typename YDataType>
using DeviceOpLnFwdPtrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    DeviceOp<XDataType, GammaDataType, BetaDataType, ComputeDataType, YDataType>>;

namespace {
struct CKArgs
{
    CKArgs(const miopen::norm::ProblemDescription& problem)
    {
        auto length = problem.GetXDesc().GetLengths();

        N = length[0];
        W = length[1];

        N_stride = W;
        W_stride = 1;

        xyLengths    = {N, W};
        xyStrides    = {N_stride, W_stride};
        gammaStrides = {0, W_stride};
        betaStrides  = {0, W_stride};
        epsilon      = problem.GetEpsilon();
        ;
    }

    CKArgs(const CKArgs&) = default;
    CKArgs(CKArgs&&)      = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename LNPtr>
    auto MakeArgPtr(
        const LNPtr& ln_ptr, ConstData_t x, ConstData_t weight, ConstData_t bias, Data_t y) const
    {
        return ln_ptr->MakeArgumentPointer(xyLengths,
                                           xyStrides,
                                           gammaStrides,
                                           betaStrides,
                                           xyStrides,
                                           {1},
                                           epsilon,
                                           x,
                                           weight,
                                           bias,
                                           y,
                                           nullptr,
                                           nullptr,
                                           ck::tensor_operation::element_wise::PassThrough{});
    }

    template <typename LNPtr>
    bool IsSupportedBy(const LNPtr& ln_ptr) const
    {
        auto arg_ptr = MakeArgPtr(ln_ptr, nullptr, nullptr, nullptr, nullptr);
        return ln_ptr->IsSupportedArgument(arg_ptr.get());
    }

    int32_t N;
    int32_t W;
    int32_t N_stride;
    int32_t W_stride;
    std::vector<int32_t> xyLengths;
    std::vector<int32_t> xyStrides;
    std::vector<int32_t> gammaStrides;
    std::vector<int32_t> betaStrides;
    float epsilon;
};
} // namespace

template <typename DeviceOpType>
bool CheckCKApplicability(const miopen::norm::ProblemDescription& problem)
{
    const auto ln_args = CKArgs{problem};
    const auto ln_ptrs = DeviceOpType::GetInstances();

    return std::any_of(ln_ptrs.begin(), ln_ptrs.end(), [&ln_args](auto& ln_ptrs) {
        return ln_args.IsSupportedBy(ln_ptrs);
    });
}

template <typename LnPtrsType>
typename LnPtrsType::iterator FindLnPtr(LnPtrsType& ln_ptrs,
                                        const miopen::norm::ProblemDescription& problem)
{
    const auto ln_args = CKArgs{problem};
    return std::find_if(ln_ptrs.begin(), ln_ptrs.end(), [&ln_args](auto& ln_ptrs) {
        return ln_args.IsSupportedBy(ln_ptrs);
    });
}

template <typename DeviceOpType,
          typename CKArgsType,
          typename CastType,
          typename ProblemDescriptionType = ProblemDescription>
ConvSolution MakeInvokerFactory([[maybe_unused]] const ExecutionContext& context,
                                const miopen::norm::ProblemDescription& problem)
{
    auto ln_ptr      = DeviceOpType::GetInstances();
    auto ln_ptr_iter = FindLnPtr(ln_ptr, problem);

    if(ln_ptr_iter == ln_ptr.end())
    {
        MIOPEN_LOG_E("Layernorm kernel does not exist.");
        return {miopenStatusInvalidValue};
    }

    ConvSolution result;
    result.invoker_factory =
        [ck_args   = CKArgsType{problem},
         sh_ln_ptr = std::shared_ptr{std::move(*ln_ptr_iter)}](const std::vector<Kernel>&) mutable {
            return [ck_args = std::move(ck_args), sh_ln_ptr = std::move(sh_ln_ptr)](
                       const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                const auto& data_ctx = primitive_parameters.CastTo<CastType>();
                auto argument_ptr    = ck_args.MakeArgPtr(
                    sh_ln_ptr, data_ctx.x, data_ctx.weight, data_ctx.bias, data_ctx.y);
                auto invoker_ptr = sh_ln_ptr->MakeInvokerPointer();

                const auto enable_profiling = handle.IsProfilingEnabled();
                float elapsed_time =
                    invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), enable_profiling});
                if(enable_profiling)
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed_time);
                }
            };
        };
    return result;
}
#endif

bool Layernorm2DCKForward::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::norm::ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP || MIOPEN_USE_COMPOSABLEKERNEL
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_CK_LN{}))
        return false;
    if(!problem.IsRank2Dim1())
        return false;
    if(!problem.IsLargeSize())
        return false;
    if(!ck_utility::is_ck_supported_hardware(context.GetStream()))
        return false;

    switch(problem.GetXDesc().GetType())
    {
    case miopenHalf:
        return CheckCKApplicability<DeviceOpLnFwdPtrs<F16, F16, F16, F32, F16>>(problem);
    case miopenFloat:
        return CheckCKApplicability<DeviceOpLnFwdPtrs<F32, F32, F32, F32, F32>>(problem);
    case miopenDouble:
    case miopenBFloat16:
    case miopenInt32:
    case miopenInt8:
    case miopenFloat8:
    case miopenBFloat8:
    default: MIOPEN_THROW("Unsupported datatype");
    }
#endif
    return false;
}

ConvSolution Layernorm2DCKForward::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::norm::ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP || MIOPEN_USE_COMPOSABLEKERNEL
    switch(problem.GetXDesc().GetType())
    {
    case miopenHalf:
        return MakeInvokerFactory<DeviceOpLnFwdPtrs<F16, F16, F16, F32, F16>,
                                  CKArgs,
                                  miopen::norm::InvokeParams>(context, problem);
    case miopenFloat:
        return MakeInvokerFactory<DeviceOpLnFwdPtrs<F32, F32, F32, F32, F32>,
                                  CKArgs,
                                  miopen::norm::InvokeParams>(context, problem);
    case miopenDouble:
    case miopenBFloat16:
    case miopenInt8:
    case miopenInt32:
    case miopenFloat8:
    case miopenBFloat8:
    default:
        MIOPEN_THROW(miopenStatusInternalError,
                     "ConvHipImplicitGemmFwdXdlops operation not implemented for this data type");
    }
#endif
    return {};
}

} // namespace norm
} // namespace solver
} // namespace miopen
