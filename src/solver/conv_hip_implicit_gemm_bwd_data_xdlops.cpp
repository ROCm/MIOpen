/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <ck/library/tensor_operation_instance/gpu/convolution_backward_data.hpp>
#endif
#include <miopen/solver/implicitgemm_util.hpp>
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS)

namespace miopen {
namespace solver {

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
template <typename DataType>
using DeviceOpBwd = ck::tensor_operation::device::DeviceConvBwdData<
    2,
    ck::tensor_layout::convolution::NHWC,
    ck::tensor_layout::convolution::KYXC,
    ck::tensor_layout::convolution::NHWK,
    DataType,
    DataType,
    DataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

template <typename DataType>
using DeviceOpBwdPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceOpBwd<DataType>>;

namespace {
struct CKArgsBwd
{
    CKArgsBwd(const ProblemDescription& problem)
    {
        N        = ProblemInterpreter::GetBatchN(problem);
        K        = ProblemInterpreter::GetOutputChannelK(problem);
        C        = ProblemInterpreter::GetInputChannelC(problem);
        input    = {ProblemInterpreter::GetInputHeightHi(problem),
                 ProblemInterpreter::GetInputWidthWi(problem)};
        output   = {ProblemInterpreter::GetOutputHeightHo(problem),
                  ProblemInterpreter::GetOutputWidthWo(problem)};
        filter   = {ProblemInterpreter::GetFilterHeightY(problem),
                  ProblemInterpreter::GetFilterWidthX(problem)};
        strides  = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                   ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding = {ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgsBwd(const CKArgsBwd&) = default;
    CKArgsBwd(CKArgsBwd&&)      = default;
    CKArgsBwd& operator=(const CKArgsBwd&) = default;
    ~CKArgsBwd()                      = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr, Data_t out, ConstData_t w, ConstData_t in) const
    {
        return conv_ptr->MakeArgumentPointer(out,
                                             w,
                                             in,
                                             N,
                                             K,
                                             C,
                                             input,
                                             filter,
                                             output,
                                             strides,
                                             dilation,
                                             lPadding,
                                             rPadding,
                                             {},
                                             {},
                                             {});
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr, const ConvDataTensors& tensors) const
    {
        return MakeArgPtr(conv_ptr, tensors.out, tensors.w, tensors.in);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    int N;
    int K;
    int C;
    std::vector<int> input;
    std::vector<int> output;
    std::vector<int> filter;
    std::vector<int> strides;
    std::vector<int> dilation;
    std::vector<int> lPadding;
    std::vector<int> rPadding;
};

template <typename DataType, typename ConvPtrsType>
typename ConvPtrsType::iterator FindConvPtrByID(ConvPtrsType& conv_ptrs,
                                                const std::string& kernel_id)
{
    return std::find_if(conv_ptrs.begin(), conv_ptrs.end(), [&kernel_id](const auto& ptr) {
        return ptr->GetTypeString() == kernel_id;
    });
}
} // namespace

template <typename DataType>
void PerformanceConfigHipImplicitGemmBwdXdlops::Init(const ProblemDescription& problem)
{
    const auto args      = CKArgsBwd{problem};
    const auto conv_ptrs = DeviceOpBwdPtrs<DataType>::GetInstances();
    assert(!conv_ptrs.empty());
    valid_kernels.reserve(conv_ptrs.size());
    for(size_t idx = 0; idx < conv_ptrs.size(); ++idx)
    {
        if(args.IsSupportedBy(conv_ptrs[idx]))
            valid_kernels.emplace_back(std::move(conv_ptrs[idx]->GetTypeString()));
    }
    assert(!valid_kernels.empty());
    index     = 0;
    kernel_id = valid_kernels[0];
}

template <typename DataType>
bool PerformanceConfigHipImplicitGemmBwdXdlops::CheckIsSupportCKArgs(
    const ProblemDescription& problem) const
{
    auto conv_ptrs = DeviceOpBwdPtrs<DataType>::GetInstances();
    auto ptr_iter  = FindConvPtrByID<DataType>(conv_ptrs, kernel_id);

    if(ptr_iter == conv_ptrs.end())
        return false;

    return CKArgsBwd{problem}.IsSupportedBy(*ptr_iter);
}

template <typename DataType>
bool ConvHipImplicitGemmBwdXdlops::CheckCKApplicability(const ProblemDescription& problem) const
{
    const auto args = CKArgsBwd{problem};
    if(!std::all_of(args.strides.begin(), args.strides.end(), [](auto x) { return x == 1; }))
        return false;

    const auto ptrs = DeviceOpBwdPtrs<DataType>::GetInstances();
    return std::any_of(
        ptrs.begin(), ptrs.end(), [&args](auto& ptr) { return args.IsSupportedBy(ptr); });
}

namespace {

template <typename DataType, typename CKConvPtr>
void RunCKSolution(const Handle& handle,
                   const AnyInvokeParams& primitive_parameters,
                   const CKArgsBwd& ck_args,
                   const CKConvPtr& conv_ptr)
{
    const auto& data_ctx        = primitive_parameters.CastTo<conv::DataInvokeParams>();
    auto argument_ptr           = ck_args.MakeArgPtr(conv_ptr, data_ctx.tensors);
    auto invoker_ptr            = conv_ptr->MakeInvokerPointer();
    const auto enable_profiling = handle.IsProfilingEnabled();

    float elapsed_time =
        invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), enable_profiling});
    if(enable_profiling)
    {
        handle.ResetKernelTime();
        handle.AccumKernelTime(elapsed_time);
    }
}

} // namespace
#endif

void PerformanceConfigHipImplicitGemmBwdXdlops::HeuristicInit(const ProblemDescription& problem)
{
    index = 0;
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem;
#else
    kernel_id = "";

    switch(problem.GetInDataType())
    {
    case miopenHalf: Init<ck::half_t>(problem); break;
    case miopenFloat: Init<float>(problem); break;
    case miopenInt8:
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble: break;
    }
#endif
}

bool PerformanceConfigHipImplicitGemmBwdXdlops::SetNextValue(const ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        HeuristicInit(problem);
        assert(!valid_kernels.empty());
        return true;
    }
    if((index + 1) < valid_kernels.size())
    {
        ++index;
        kernel_id = valid_kernels[index];
        return true;
    }
    else
        return false;
}

bool PerformanceConfigHipImplicitGemmBwdXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemmBwdXdlops::IsValid(const ProblemDescription& problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem;
    return false;
#else
    switch(problem.GetInDataType())
    {
    case miopenHalf: return CheckIsSupportCKArgs<ck::half_t>(problem);
    case miopenFloat: return CheckIsSupportCKArgs<float>(problem);
    case miopenInt8:
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble: break;
    }
    return false;
#endif
}

bool PerformanceConfigHipImplicitGemmBwdXdlops::operator==(
    const PerformanceConfigHipImplicitGemmBwdXdlops& other) const
{
    return kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemmBwdXdlops
ConvHipImplicitGemmBwdXdlops::GetDefaultPerformanceConfig(const ConvolutionContext&,
                                                          const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemmBwdXdlops pp;
    pp.HeuristicInit(problem);
    return pp;
}

bool ConvHipImplicitGemmBwdXdlops::IsValidPerformanceConfig(
    const ConvolutionContext&,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmBwdXdlops& config) const
{
    return config.IsValid(problem);
}

PerformanceConfigHipImplicitGemmBwdXdlops
ConvHipImplicitGemmBwdXdlops::Search(const ConvolutionContext& ctx,
                                     const ProblemDescription& problem,
                                     const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemmBwdXdlops::IsApplicable(const ConvolutionContext& ctx,
                                                const ProblemDescription& problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    std::ignore = problem;
    return false;
#else
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS{}))
        return false;
    if(problem.GetConv().attribute.deterministic)
        return false;
    if(problem.GetInDataType() != problem.GetWeightsDataType() ||
       problem.GetWeightsDataType() != problem.GetOutDataType() ||
       problem.GetInDataType() != problem.GetOutDataType())
        return false;
    if(!problem.direction.IsBackwardData())
        return false;
    if(!problem.Is2d())
        return false;
    if(!problem.IsLayoutNHWC())
        return false;
    if(!IsXdlopsSupport(ctx))
        return false;
    if(!IsComposableKernelSupportedHardware(ctx))
        return false;
    const std::string& arch = ctx.GetStream().GetDeviceName();
    if(arch == "gfx90a" && problem.IsGfx90aFp16altRequired())
        return false;
    if(!IsIndexRangeLargeEnough(problem))
        return false;
    if(problem.GetGroupCount() > 1)
        return false;
    switch(problem.GetInDataType())
    {
    case miopenHalf: return CheckCKApplicability<ck::half_t>(problem);
    case miopenFloat: return CheckCKApplicability<float>(problem);
    case miopenInt8:
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble: break;
    }
    return false;
#endif
}

ConvSolution ConvHipImplicitGemmBwdXdlops::GetSolution(
    const ConvolutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmBwdXdlops& config) const
{
    std::ignore = ctx;
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem;
    std::ignore = config;
    return {};
#else

    auto InitInvokerFactory = [&problem, &kernel_id = config.kernel_id](auto DataTypeVal) {
        using DataType = decltype(DataTypeVal);

        auto conv_ptrs = DeviceOpBwdPtrs<DataType>::GetInstances();
        auto ptr_iter  = FindConvPtrByID<DataType>(conv_ptrs, kernel_id);

        if(ptr_iter == conv_ptrs.end())
            MIOPEN_THROW("PerformanceConfig kernel '" + kernel_id + "' does not exist");

        ConvSolution result;
        result.invoker_factory = [ck_args     = CKArgsBwd{problem},
                                  sh_conv_ptr = std::shared_ptr{std::move(*ptr_iter)}](
                                     const std::vector<Kernel>&) mutable {
            return [ck_args = std::move(ck_args), sh_conv_ptr = std::move(sh_conv_ptr)](
                       const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                RunCKSolution<DataType>(handle, primitive_parameters, ck_args, sh_conv_ptr);
            };
        };
        return result;
    };

    switch(problem.GetInDataType())
    {
    case miopenHalf: return InitInvokerFactory(ck::half_t{});
    case miopenFloat: return InitInvokerFactory(float{});
    case miopenInt8:
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble:
    default:
        MIOPEN_THROW(miopenStatusInternalError,
                     "ConvHipImplicitGemmFwdXdlops operation not implemented for this data type");
    }

    return {};
#endif
}

} // namespace solver
} // namespace miopen
