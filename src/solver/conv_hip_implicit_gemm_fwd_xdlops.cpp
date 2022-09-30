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
#include <ck/library/tensor_operation_instance/gpu/convolution_forward.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS)

namespace miopen {
namespace solver {

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
template <typename DataType>
using DeviceOp =
    ck::tensor_operation::device::DeviceConvFwd<2,
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
using DeviceOpPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceOp<DataType>>;

struct CKArgs
{
    CKArgs(const ConvolutionContext& ctx)
    {
        N        = ProblemInterpreter::GetBatchN(ctx.problem);
        K        = ProblemInterpreter::GetOutputChannelK(ctx.problem);
        C        = ProblemInterpreter::GetInputChannelC(ctx.problem);
        input    = {ProblemInterpreter::GetInputHeightHi(ctx.problem),
                 ProblemInterpreter::GetInputWidthWi(ctx.problem)};
        output   = {ProblemInterpreter::GetOutputHeightHo(ctx.problem),
                  ProblemInterpreter::GetOutputWidthWo(ctx.problem)};
        filter   = {ProblemInterpreter::GetFilterHeightY(ctx.problem),
                  ProblemInterpreter::GetFilterWidthX(ctx.problem)};
        strides  = {ProblemInterpreter::GetAdjustedConvolutionStrideH(ctx.problem),
                   ProblemInterpreter::GetAdjustedConvolutionStrideW(ctx.problem)};
        dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(ctx.problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(ctx.problem)};
        lPadding = {ProblemInterpreter::GetInputLeftPadH(ctx.problem),
                    ProblemInterpreter::GetInputLeftPadW(ctx.problem)};
        rPadding = {ProblemInterpreter::GetAdjustedInputRightPadH(ctx.problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(ctx.problem)};
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

template <typename DataType>
auto ArgumentPointer(const ConvolutionContext& ctx, std::vector<std::unique_ptr<DeviceOp<DataType>>>& conv_ptrs, int i)
{
    const auto args  = CKArgs{ctx};
    auto argument_ptr = conv_ptrs[i]->MakeArgumentPointer(nullptr,
                                                          nullptr,
                                                          nullptr,
                                                          args.N,
                                                          args.K,
                                                          args.C,
                                                          args.input,
                                                          args.filter,
                                                          args.input,
                                                          args.strides,
                                                          args.dilation,
                                                          args.lPadding,
                                                          args.rPadding,
                                                          {},
                                                          {},
                                                          {});
    return argument_ptr;
}

template <typename DataType>
void InitHelper(int& idx, int& t_size, std::string& k_id){
    const auto conv_ptrs = DeviceOpPtrs<DatType>::GetInstances();
    assert(!conv_ptrs.empty());
    t_size = conv_ptrs.size();
    for(int i=0;i<conv_ptrs.size();i++)
    {
        auto argument_ptr = ArgumentPointer<DataType>(ctx, conv_ptrs, i);
        if(conv_ptrs[i]->IsSupportedArgument(argument_ptr.get()))
        {
            k_id = conv_ptrs[i]->GetTypeString();
            break;
        }
        ++idx;
    }
}

template <typename DataType>
auto ValidHelper(const ConvolutionContext& ctx, int& idx){
    const auto conv_ptrs = DeviceOpPtrs<DatType>::GetInstances();
    auto argument_ptr    = ArgumentPointer<DataType>(ctx, conv_ptrs, idx);;
    return conv_ptrs[idx]->IsSupportedArgument(argument_ptr.get());
}

template <typename DataType>
bool ApplicableHelper(const ConvolutionContext& ctx){
    const auto conv_ptrs = DeviceOpPtrs<DatType>::GetInstances();
    assert(!conv_ptrs.empty());
    const auto args = CKArgs{ctx};
    if(!std::all_of(args.strides.begin(), args.strides.end(), [&](auto x) { return x == 1; }))
        return false;    
    for(int i=0;i<conv_ptrs.size();i++)
    {
        auto argument_ptr = ArgumentPointer<DataType>(ctx, conv_ptrs, i);
        if(conv_ptrs[i]->IsSupportedArgument(argument_ptr.get()))
            return true;
    }
    return false;
}

template <typename DataType>
void SolutionHelper(const ConvolutionContext& ctx, const PerformanceConfigHipImplicitGemmFwdXdlops& config){
    const auto conv_ptrs = DeviceOpPtrs<DatType>::GetInstances();
    auto& conv_ptr       = conv_ptrs.at(config.index);
    const auto& data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
    const auto& tensors  = data_ctx.tensors;
    auto argument_ptr    = conv_ptr->MakeArgumentPointer(
        const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
            static_cast<const void*>(tensors.in)),
        const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
            static_cast<const void*>(tensors.w)),
        static_cast<void*>(tensors.out),
        args.N,
        args.K,
        args.C,
        args.input,
        args.filter,
        args.input,
        args.strides,
        args.dilation,
        args.lPadding,
        args.rPadding,
        {},
        {},
        {});
    auto invoker_ptr = conv_ptr->MakeInvokerPointer();
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

void PerformanceConfigHipImplicitGemmFwdXdlops::HeuristicInit(const ConvolutionContext& ctx)
{
    this->index = 0;
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
#else
    this->index      = 0;
    this->total_size = 0;
    this->kernel_id  = "";
    switch(ctx.problem.conv_problem.GetInDataType()){
        case miopenInt8:
            InitHelper<int8_t>(index, total_size, kernel_id);
            break;
        case miopenHalf:
            InitHelper<half_t>(index, total_size, kernel_id);
            break;
        case miopenFloat:
            InitHelper<float>(index, total_size, kernel_id);
            break;
        default:
            break;
    }
#endif
}

bool PerformanceConfigHipImplicitGemmFwdXdlops::SetNextValue(const ConvolutionContext& ctx)
{
    if(total_size == -1)
        this->HeuristicInit(ctx);
    assert(total_size != -1);
    if((index + 1) < total_size)
    {
        ++index;
        return true;
    }
    else
        return false;
}

bool PerformanceConfigHipImplicitGemmFwdXdlops::IsValidValue() const { return index < total_size; }

bool PerformanceConfigHipImplicitGemmFwdXdlops::IsValid(const ConvolutionContext& ctx) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    return false;
#else
    switch(ctx.problem.conv_problem.GetInDataType()){
        case miopenInt8:
            ValidHelper<int8_t>(ctx, this->index);
            break;
        case miopenHalf:
            ValidHelper<half_t>(ctx, this->index);
            break;
        case miopenFloat:
            ValidHelper<float>(ctx, this->index);
            break;
        default:
            break;
    }
#endif
}

bool PerformanceConfigHipImplicitGemmFwdXdlops::operator==(
    const PerformanceConfigHipImplicitGemmFwdXdlops& other) const
{
    return this->index == other.index;
}

PerformanceConfigHipImplicitGemmFwdXdlops
ConvHipImplicitGemmFwdXdlops::GetDefaultPerformanceConfig(const ConvolutionContext& ctx) const
{
    PerformanceConfigHipImplicitGemmFwdXdlops pp;
    pp.HeuristicInit(ctx);
    return pp;
}

bool ConvHipImplicitGemmFwdXdlops::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceConfigHipImplicitGemmFwdXdlops& config) const
{
    return config.IsValid(ctx);
}

PerformanceConfigHipImplicitGemmFwdXdlops
ConvHipImplicitGemmFwdXdlops::Search(const ConvolutionContext& ctx,
                                     const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, invoke_ctx);
}

size_t ConvHipImplicitGemmFwdXdlops::GetWorkspaceSize(const ConvolutionContext& ctx) const
{
    std::ignore = ctx;
    return 0;
}

bool ConvHipImplicitGemmFwdXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    return false;
#else
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS{}))
        return false;
    if(miopen::IsEnabled(MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC{}))
        return false;
    if(ctx.problem.conv_problem.GetInDataType() != ctx.problem.conv_problem.GetWeightsDataType() ||
       ctx.problem.conv_problem.GetWeightsDataType() != ctx.problem.conv_problem.GetOutDataType() ||
       ctx.problem.conv_problem.GetInDataType() != ctx.problem.conv_problem.GetOutDataType())
        return false;
    if(!ctx.problem.direction.IsForward())
        return false;
    if(!ctx.problem.Is2d())
        return false;
    if(ctx.GetStream().GetDeviceName() != "gfx908")
        return false;
    if(!ctx.problem.IsLayoutNHWC())
        return false;
    switch(ctx.problem.conv_problem.GetInDataType()){
        case miopenInt8:
            return ApplicableHelper<int8_t>(ctx);
        case miopenHalf:
            return ApplicableHelper<half_t>(ctx);
        case miopenFloat:
            return ApplicableHelper<float>(ctx);
        default:
            break;
    }
    return false;
#endif
}

ConvSolution ConvHipImplicitGemmFwdXdlops::GetSolution(
    const ConvolutionContext& ctx, const PerformanceConfigHipImplicitGemmFwdXdlops& config) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    std::ignore = config;
    return {};
#else
    ConvSolution result;
    const auto args        = CKArgs{ctx};
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        std::ignore = kernels;
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            auto invoker_ptr = SolutionHelper(ctx, config);
            switch(ctx.problem.conv_problem.GetInDataType()){
                case miopenInt8:
                    SolutionHelper<int8_t>(ctx, config);
                    break;
                case miopenHalf:
                    SolutionHelper<half_t>(ctx, config);
                    break;
                case miopenFloat:
                    SolutionHelper<float>(ctx, config);
                    break;
                default:
                    break;
            }
        };
    };
    return result;
#endif
}

} // namespace solver
} // namespace miopen
