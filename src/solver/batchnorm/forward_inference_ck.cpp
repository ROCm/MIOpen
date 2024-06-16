
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
#include <ck/library/tensor_operation_instance/gpu/batchnorm_infer.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_CK_BN_INFER)

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

    std::array<int, NumBatchNormReduceDim> reduceDims{0, 1, 2};
};

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
static int CheckCKApplicability(const miopen::batchnorm::ProblemDescription& problem)
{
    const auto& args = CKArgsBNormFwd{problem};
    using DeviceOp   = ck::tensor_operation::device::DeviceElementwise<
        ck::Tuple<XDataType, MeanVarDataType, MeanVarDataType, ScaleDataType, BiasDataType>,
        ck::Tuple<YDataType>,
        Normalize,
        Rank>;
    const auto bn_fwd_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();
    assert(!bn_fwd_ptrs.empty());
    int count = 0;
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
            return count;
        }
        count++;
    }
    return -1;
}

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
static void RunCKSolution(const Handle& handle,
                          const AnyInvokeParams& primitive_parameters,
                          const miopen::batchnorm::ProblemDescription& problem)
{
    const auto& args = CKArgsBNormFwd{problem};

    using DeviceOp = ck::tensor_operation::device::DeviceElementwise<
        ck::Tuple<XDataType, MeanVarDataType, MeanVarDataType, ScaleDataType, BiasDataType>,
        ck::Tuple<YDataType>,
        Normalize,
        Rank>;
    const auto bn_fwd_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    int kernel_index = CheckCKApplicability<XDataType,
                                            YDataType,
                                            AccDataType,
                                            ScaleDataType,
                                            BiasDataType,
                                            MeanVarDataType>(problem);
    assert(kernel_index >= 0 && kernel_index < bn_fwd_ptrs.size());
    auto& bn_ptr       = bn_fwd_ptrs.at(kernel_index);
    const auto& params = primitive_parameters.CastTo<miopen::batchnorm::InfInvokeParams>();

    auto argument_ptr = bn_ptr->MakeArgumentPointer(
        args.xyLengths,
        {args.xyStrides,
         args.aligned_scaleBiasMeanVarStrides,
         args.aligned_scaleBiasMeanVarStrides,
         args.aligned_scaleBiasMeanVarStrides,
         args.aligned_scaleBiasMeanVarStrides},
        {args.xyStrides},
        {params.x, params.estimatedMean, params.estimatedVariance, params.bnScale, params.bnBias},
        {params.y},
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

bool BnCKFwdInference::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::batchnorm::ProblemDescription& bn_problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_CONV_CK_BN_INFER))
        return false;
    if(!bn_problem.IsLayoutNHWC())
        return false;
    if(!ck_utility::is_ck_supported_hardware(context.GetStream()))
        return false;
    if(!bn_problem.Is2D())
        return false;
    if(bn_problem.GetDirection() != miopen::batchnorm::Direction::ForwardInference)
        return false;

    switch(bn_problem.GetXDesc().GetType())
    {
    case miopenHalf: return (CheckCKApplicability<F16, F16, F32, F16, F16, F32>(bn_problem) != -1);
    case miopenFloat: return (CheckCKApplicability<F32, F32, F32, F32, F32, F32>(bn_problem) != -1);
    case miopenDouble:
        return (CheckCKApplicability<F64, F64, F64, F64, F64, F64>(bn_problem) != -1);
    case miopenBFloat16:
        return (CheckCKApplicability<BF16, BF16, F32, BF16, BF16, F32>(bn_problem) != -1);
    case miopenInt64:
    case miopenInt32:
    case miopenInt8:
    case miopenFloat8:
    case miopenBFloat8: break;
    }
#endif
    return false;
}

ConvSolution BnCKFwdInference::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::batchnorm::ProblemDescription& bn_problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    ConvSolution result;
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        std::ignore = kernels;
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            switch(bn_problem.GetXDesc().GetType())
            {
            case miopenHalf:
                RunCKSolution<F16, F16, F32, F16, F16, F32>(
                    handle, primitive_parameters, bn_problem);
                break;
            case miopenFloat:
                RunCKSolution<F32, F32, F32, F32, F32, F32>(
                    handle, primitive_parameters, bn_problem);
                break;
            case miopenDouble:
                RunCKSolution<F64, F64, F64, F64, F64, F64>(
                    handle, primitive_parameters, bn_problem);
                break;
            case miopenBFloat16:
                RunCKSolution<BF16, BF16, F32, BF16, BF16, F32>(
                    handle, primitive_parameters, bn_problem);
                break;
            case miopenInt8:
            case miopenInt32:
            case miopenInt64:
            case miopenFloat8:
            case miopenBFloat8:
            default: MIOPEN_THROW("Unsupported datatype");
            }
        };
    };
    return result;
#else
    return {};
#endif
}

} // namespace batchnorm
} // namespace solver
} // namespace miopen
