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

#include <miopen/normalization/solvers.hpp>
#include <miopen/normalization/invoke_params.hpp>
#include <miopen/batch_norm.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/normalization.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_CK_LN)

namespace miopen {
namespace solver {
namespace normalization {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

constexpr index_t Rank         = 2;
constexpr index_t NumReduceDim = 1;

using F16  = ck::half_t;
using F32  = float;
using F64  = double;
using BF16 = ushort;

struct CKArgsLNormFwd
{
    CKArgsLNormFwd(const miopen::normalization::ProblemDescription& problem)
    {
        std::copy(problem.GetXDesc().GetLengths().begin(),
                  problem.GetXDesc().GetLengths().end(),
                  xyLengths.begin());

        std::copy(problem.GetXDesc().GetStrides().begin(),
                  problem.GetXDesc().GetStrides().end(),
                  xyStrides.begin());

        std::copy(problem.GetWeightDesc().GetStrides().begin(),
                  problem.GetWeightDesc().GetStrides().end(),
                  gammaStrides.begin());

        std::copy(problem.GetBiasDesc().GetStrides().begin(),
                  problem.GetBiasDesc().GetStrides().end(),
                  betaStrides.begin());

        epsilon = problem.GetEpsilon();
    }

    std::array<ck::index_t, Rank> xyLengths;
    std::array<ck::index_t, Rank> xyStrides;
    std::array<ck::index_t, Rank> gammaStrides;
    std::array<ck::index_t, Rank> betaStrides;
    float epsilon;
};

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename ComputeDataType,
          typename YDataType>
static int CheckCKApplicability(const miopen::normalization::ProblemDescription& problem)
{
    const auto& args       = CKArgsLNormFwd{problem};
    using DeviceOp         = ck::tensor_operation::device::DeviceNormalization<XDataType,
                                                                       GammaDataType,
                                                                       BetaDataType,
                                                                       ComputeDataType,
                                                                       YDataType,
                                                                       PassThrough,
                                                                       Rank,
                                                                       NumReduceDim>;
    const auto ln_fwd_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();
    assert(!ln_fwd_ptrs.empty());
    int count = 0;
    for(const auto& it : ln_fwd_ptrs)
    {
        auto argument_ptr = it->MakeArgumentPointer(args.xyLengths,
                                                    {args.xyStrides},
                                                    {args.gammaStrides},
                                                    {args.betaStrides},
                                                    {args.xyStrides},
                                                    {NumReduceDim},
                                                    args.epsilon,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    PassThrough{});
        if(it->IsSupportedArgument(argument_ptr.get()))
        {
            return count;
        }
        count++;
    }
    return -1;
}

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename ComputeDataType,
          typename YDataType>
static void RunCKSolution(const Handle& handle,
                          const AnyInvokeParams& primitive_parameters,
                          const miopen::normalization::ProblemDescription& problem)
{
    const auto& args = CKArgsLNormFwd{problem};

    using DeviceOp         = ck::tensor_operation::device::DeviceNormalization<XDataType,
                                                                       GammaDataType,
                                                                       BetaDataType,
                                                                       ComputeDataType,
                                                                       YDataType,
                                                                       PassThrough,
                                                                       Rank,
                                                                       NumReduceDim>;
    const auto ln_fwd_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    int kernel_index =
        CheckCKApplicability<XDataType, GammaDataType, BetaDataType, ComputeDataType, YDataType>(
            problem);
    assert(kernel_index >= 0 && kernel_index < ln_fwd_ptrs.size());
    auto& ln_ptr       = ln_fwd_ptrs.at(kernel_index);
    const auto& params = primitive_parameters.CastTo<miopen::normalization::InfInvokeParams>();

    auto argument_ptr = ln_ptr->MakeArgumentPointer(args.xyLengths,
                                                    {args.xyStrides},
                                                    {args.gammaStrides},
                                                    {args.betaStrides},
                                                    {args.xyStrides},
                                                    {NumReduceDim},
                                                    args.epsilon,
                                                    {params.x},
                                                    {params.weight},
                                                    {params.bias},
                                                    {params.y},
                                                    nullptr,
                                                    nullptr,
                                                    PassThrough{});

    auto invoker_ptr            = ln_ptr->MakeInvokerPointer();
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

bool Layernorm2DCKForward::IsApplicable(
    const ExecutionContext& context, const miopen::normalization::ProblemDescription& problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = context;
    std::ignore = problem;
    return false;
#else
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
    case miopenHalf: return (CheckCKApplicability<F16, F16, F16, F32, F16>(problem) != -1);
    case miopenFloat: return (CheckCKApplicability<F32, F32, F32, F32, F32>(problem) != -1);
    case miopenDouble:
    case miopenBFloat16:
    case miopenInt32:
    case miopenInt8:
    case miopenInt8x4: // Support discontinued.
    case miopenFloat8:
    case miopenBFloat8:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return false;
#endif
}

ConvSolution
Layernorm2DCKForward::GetSolution(const ExecutionContext& context,
                                  const miopen::normalization::ProblemDescription& problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = context;
    std::ignore = problem;
    return {};
#else
    std::ignore = context;

    ConvSolution result;
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        std::ignore = kernels;
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            switch(problem.GetXDesc().GetType())
            {
            case miopenHalf:
                RunCKSolution<F16, F16, F16, F32, F16>(handle, primitive_parameters, problem);
                break;
            case miopenFloat:
                RunCKSolution<F32, F32, F32, F32, F32>(handle, primitive_parameters, problem);
                break;
            case miopenDouble:
            case miopenInt8:
            case miopenInt32:
            case miopenInt8x4: // Support discontinued.
            case miopenFloat8:
            case miopenBFloat8:
            default: MIOPEN_THROW("Unsupported datatype");
            }
        };
    };
    return result;
#endif
}

} // namespace normalization
} // namespace solver
} // namespace miopen
