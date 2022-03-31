#include <vector>

#include <ck/library/host/host_interface.hpp>

#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/convolution_context_interpreter.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS)

namespace miopen {
namespace solver {

void PerformanceConfigHipImplicitGemmFwdXdlops::HeuristicInit(const ConvolutionContext& ctx)
{
    std::ignore = ctx;
    this->index = 0;
    if(total_size == -1)
    {
        // Get the length of the vector
        // total_size = conv_instances.size();
    }
}

bool PerformanceConfigHipImplicitGemmFwdXdlops::SetNextValue(const ConvolutionContext& ctx)
{
    std::ignore = ctx;
    assert(total_size != -1);
    if(index < total_size)
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
    std::ignore = ctx;
    return true;
}

PerformanceConfigHipImplicitGemmFwdXdlops
ConvHipImplicitGemmFwdXdlops::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    std::ignore = ctx;
    return {};
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
    std::ignore = ctx;
    std::ignore = invoke_ctx;
    return {};
}

size_t ConvHipImplicitGemmFwdXdlops::GetWorkspaceSize(const ConvolutionContext& ctx) const
{
    std::ignore = ctx;
    return 0;
}

bool ConvHipImplicitGemmFwdXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
    std::ignore = ctx;
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS{}))
        return false;
    if(miopen::IsEnabled(MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC{}))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!ctx.IsInt8())
        return false;
    if(!ctx.direction.IsForward())
        return false;
    if(!ctx.Is2d())
        return false;
    if(ctx.GetStream().GetDeviceName() != "gfx908")
        return false;
    if(!ctx.IsLayoutNHWC())
        return false;

    std::vector<DeviceConvFwdPtr_t> conv_ptrs;
    add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances_t(conv_ptrs);
    assert(!conv_ptrs.empty());
    // TODO: Move to convolution_context_interpreter.hpp
    // side-effect:: need to include ck's host_interface.hpp there
    const auto N   = ConvolutionContextInterpreter::GetBatchN(ctx);
    const auto K   = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
    const auto C   = ConvolutionContextInterpreter::GetInputChannelC(ctx);
    const auto Hi  = ConvolutionContextInterpreter::GetInputHeightHi(ctx);
    const auto Wi  = ConvolutionContextInterpreter::GetInputWidthWi(ctx);
    const auto Ho  = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    const auto Wo  = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    const auto Y   = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    const auto X   = ConvolutionContextInterpreter::GetFilterWidthX(ctx);
    const auto Sy  = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx);
    const auto Sx  = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx);
    const auto Dy  = ConvolutionContextInterpreter::GetAdjustedConvolutionDilationH(ctx);
    const auto Dx  = ConvolutionContextInterpreter::GetAdjustedConvolutionDilationW(ctx);
    const auto lPy = ConvolutionContextInterpreter::GetInputLeftPadH(ctx);
    const auto lPx = ConvolutionContextInterpreter::GetInputLeftPadW(ctx);
    const auto rPy = ConvolutionContextInterpreter::GetAdjustedInputRightPadH(ctx);
    const auto rPx = ConvolutionContextInterpreter::GetAdjustedInputRightPadW(ctx);

    for(auto& conv_ptr : conv_ptrs)
    {
        auto argument_ptr = conv_ptr.MakeArgumentPointer(nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         N,
                                                         K,
                                                         C,
                                                         {Hi, Wi},
                                                         {Y, X},
                                                         {Ho, Wo},
                                                         {Sx, Sy},
                                                         {Dy, Dx},
                                                         {lPy, lPx},
                                                         {rPy, rPx});
        if(conv_ptr.IsSupportedArgument(argument_ptr.get()))
            return true;
    }
    return false;
}

ConvSolution ConvHipImplicitGemmFwdXdlops::GetSolution(
    const ConvolutionContext& ctx, const PerformanceConfigHipImplicitGemmFwdXdlops& config) const
{
    ConvSolution result;
    // TODO: Move to convolution_context_interpreter.hpp
    // side-effect:: need to include ck's host_interface.hpp there

    const auto N   = ConvolutionContextInterpreter::GetBatchN(ctx);
    const auto K   = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
    const auto C   = ConvolutionContextInterpreter::GetInputChannelC(ctx);
    const auto Hi  = ConvolutionContextInterpreter::GetInputHeightHi(ctx);
    const auto Wi  = ConvolutionContextInterpreter::GetInputWidthWi(ctx);
    const auto Ho  = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    const auto Wo  = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    const auto Y   = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    const auto X   = ConvolutionContextInterpreter::GetFilterWidthX(ctx);
    const auto Sy  = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx);
    const auto Sx  = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx);
    const auto Dy  = ConvolutionContextInterpreter::GetAdjustedConvolutionDilationH(ctx);
    const auto Dx  = ConvolutionContextInterpreter::GetAdjustedConvolutionDilationW(ctx);
    const auto lPy = ConvolutionContextInterpreter::GetInputLeftPadH(ctx);
    const auto lPx = ConvolutionContextInterpreter::GetInputLeftPadW(ctx);
    const auto rPy = ConvolutionContextInterpreter::GetAdjustedInputRightPadH(ctx);
    const auto rPx = ConvolutionContextInterpreter::GetAdjustedInputRightPadW(ctx);
    std::vector<DeviceConvFwdPtr_t> conv_ptrs;
    add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances_t(conv_ptrs);
    assert(!conv_ptrs.empty());
    auto& conv_ptr = conv_ptrs.at(config.index);
    result.invoker_factory = [&] (const std::vector<Kernel>& kernels)
        {
            std::ignore = kernels;
            return [&](const Handle& handle, const AnyInvokeParams& primitive_parameters){
                const auto& data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
                const auto& tensors  = data_ctx.tensors;
                auto argument_ptr = conv_ptr.MakeArgumentPointer(const_cast<void*>(static_cast<const void*>(tensors.in)),
                                                                const_cast<void*>(static_cast<const void*>(tensors.w)),
                                                                static_cast<void*>(tensors.out),
                                                                N,
                                                                K,
                                                                C,
                                                                {Hi, Wi},
                                                                {Y, X},
                                                                {Ho, Wo},
                                                                {Sx, Sy},
                                                                {Dy, Dx},
                                                                {lPy, lPx},
                                                                {rPy, rPx});
                auto invoker_ptr = conv_ptr.MakeInvokerPointer();

                invoker_ptr->Run(argument_ptr.get(), 1, handle.GetStream());
            };
        };
    return result;
}

} // namespace solver
} // namespace miopen
