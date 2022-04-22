#include <vector>

#include <ck/library/host/host_interface.hpp>

#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/convolution_context_interpreter.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS)

namespace miopen {
namespace solver {

struct CKArgs
{
    CKArgs(const ConvolutionContext& ctx)
    {
        N        = ConvolutionContextInterpreter::GetBatchN(ctx);
        K        = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
        C        = ConvolutionContextInterpreter::GetInputChannelC(ctx);
        input    = {ConvolutionContextInterpreter::GetInputHeightHi(ctx),
                 ConvolutionContextInterpreter::GetInputWidthWi(ctx)};
        output   = {ConvolutionContextInterpreter::GetOutputHeightHo(ctx),
                  ConvolutionContextInterpreter::GetOutputWidthWo(ctx)};
        filter   = {ConvolutionContextInterpreter::GetFilterHeightY(ctx),
                  ConvolutionContextInterpreter::GetFilterWidthX(ctx)};
        strides  = {ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx),
                   ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx)};
        dilation = {ConvolutionContextInterpreter::GetAdjustedConvolutionDilationH(ctx),
                    ConvolutionContextInterpreter::GetAdjustedConvolutionDilationW(ctx)};
        lPadding = {ConvolutionContextInterpreter::GetInputLeftPadH(ctx),
                    ConvolutionContextInterpreter::GetInputLeftPadW(ctx)};
        rPadding = {ConvolutionContextInterpreter::GetAdjustedInputRightPadH(ctx),
                    ConvolutionContextInterpreter::GetAdjustedInputRightPadW(ctx)};
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

void PerformanceConfigHipImplicitGemmFwdXdlops::HeuristicInit(const ConvolutionContext& ctx)
{
    this->index = 0;
    std::vector<DeviceConvFwdPtr_t> conv_ptrs;
    add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances_t(conv_ptrs);
    assert(!conv_ptrs.empty());
    this->total_size = conv_ptrs.size();
    const auto args  = CKArgs{ctx};
    for(auto& conv_ptr : conv_ptrs)
    {
        auto argument_ptr = conv_ptr.MakeArgumentPointer(nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         args.N,
                                                         args.K,
                                                         args.C,
                                                         args.input,
                                                         args.filter,
                                                         args.output,
                                                         args.strides,
                                                         args.dilation,
                                                         args.lPadding,
                                                         args.rPadding);
        if(conv_ptr.IsSupportedArgument(argument_ptr.get()))
        {
            this->kernel_id = conv_ptr.GetTypeString();
            break;
        }
        ++this->index;
    }
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
    std::vector<DeviceConvFwdPtr_t> conv_ptrs;
    add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances_t(conv_ptrs);
    const auto args   = CKArgs{ctx};
    auto argument_ptr = conv_ptrs[this->index].MakeArgumentPointer(nullptr,
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
                                                                   args.rPadding);
    return conv_ptrs[this->index].IsSupportedArgument(argument_ptr.get());
}

bool PerformanceConfigHipImplicitGemmFwdXdlops::operator==(
    const PerformanceConfigHipImplicitGemmFwdXdlops& other) const
{
    return this->index == other.index;
}

PerformanceConfigHipImplicitGemmFwdXdlops
ConvHipImplicitGemmFwdXdlops::GetPerformanceConfig(const ConvolutionContext& ctx) const
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
#if !MIOPEN_BACKEND_HIP
    std::ignore = ctx;
    return false;
#else
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
    const auto args = CKArgs{ctx};

    for(auto& conv_ptr : conv_ptrs)
    {
        auto argument_ptr = conv_ptr.MakeArgumentPointer(nullptr,
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
                                                         args.rPadding);
        if(conv_ptr.IsSupportedArgument(argument_ptr.get()))
            return true;
    }
    return false;
#endif
}

ConvSolution
ConvHipImplicitGemmFwdXdlops::GetSolution(const ConvolutionContext& ctx,
                                          const PerformanceConfigHipImplicitGemmFwdXdlops& config,
                                          bool disableConfigOverridefromEnv) const
{
    std::ignore = disableConfigOverridefromEnv;
#if !MIOPEN_BACKEND_HIP
    std::ignore = ctx;
    std::ignore = config;
    return {};
#else
    ConvSolution result;
    const auto args        = CKArgs{ctx};
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        std::ignore = kernels;
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            std::vector<DeviceConvFwdPtr_t> conv_ptrs;
            add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances_t(conv_ptrs);
            auto& conv_ptr       = conv_ptrs.at(config.index);
            const auto& data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors  = data_ctx.tensors;
            auto argument_ptr    = conv_ptr.MakeArgumentPointer(
                const_cast<void*>(static_cast<const void*>(
                    tensors.in)), // NOLINT (cppcoreguidelines-pro-type-const-cast)
                const_cast<void*>(static_cast<const void*>(
                    tensors.w)), // NOLINT (cppcoreguidelines-pro-type-const-cast)
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
                args.rPadding);
            auto invoker_ptr            = conv_ptr.MakeInvokerPointer();
            const auto enable_profiling = handle.IsProfilingEnabled();

            float elapsed_time =
                invoker_ptr->Run(argument_ptr.get(), 1, handle.GetStream(), enable_profiling);
            if(enable_profiling)
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed_time);
            }
        };
    };
    return result;
#endif
}

} // namespace solver
} // namespace miopen
