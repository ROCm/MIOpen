#include <miopen/conv/invokers/impl_gemm_dynamic.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/algorithm.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>

#include <boost/any.hpp>

namespace miopen {
namespace conv {

float CallImplicitGemmDynamic(const miopen::Handle& handle,
                              const ConvolutionContext& ctx,
                              ConstData_t src,
                              Data_t dst,
                              ConstData_t wei,
                              const std::vector<KernelInvoke>& kernels)
{
    float elapsed = 0.0f;

    auto kernel = kernels[0];
    MIOPEN_LOG_I(kernel.GetName());
    bool kernel_is_1x1 = (kernel.GetName().find("igemm_v4r1_1x1_dynamic") == 0);
    // clang-format off
    int hi          = ctx.in_height;
    int wi          = ctx.in_width;
    int n           = ctx.batch_sz;
    int k           = ctx.n_outputs;
    int c           = ctx.n_inputs;
    int ho          = ctx.out_height;
    int wo          = ctx.out_width;
    int stride_h    = ctx.kernel_stride_h;
    int stride_w    = ctx.kernel_stride_w;
    int dilation_h  = ctx.kernel_dilation_h;
    int dilation_w  = ctx.kernel_dilation_w;
    int pad_h       = ctx.pad_h;
    int pad_w       = ctx.pad_w;
    int y           = ctx.kernel_size_h;
    int x           = ctx.kernel_size_w;
    int __pack0     = 0;
    // clang-format on
    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(src);
    opArgs.emplace_back(wei);
    opArgs.emplace_back(dst);
    opArgs.emplace_back(hi);
    opArgs.emplace_back(wi);
    opArgs.emplace_back(n);
    opArgs.emplace_back(k);
    opArgs.emplace_back(c);
    opArgs.emplace_back(ho);
    opArgs.emplace_back(wo);
    opArgs.emplace_back(stride_h);
    opArgs.emplace_back(stride_w);
    opArgs.emplace_back(dilation_h);
    opArgs.emplace_back(dilation_w);
    opArgs.emplace_back(pad_h);
    opArgs.emplace_back(pad_w);
    if(kernel_is_1x1)
    {
        opArgs.emplace_back(__pack0);
    }
    else
    {
        opArgs.emplace_back(y);
        opArgs.emplace_back(x);
        opArgs.emplace_back(__pack0);
    }
    kernel(opArgs);

    if(handle.IsProfilingEnabled())
        elapsed += handle.GetKernelTime();
    return elapsed;
}

int GetImplicitGemmWrwV4R1DynamicGemmkGroups(const ConvolutionContext& ctx,
                                             const int& GemmKPerBlock)
{
    int gemmk        = ctx.batch_sz * ctx.in_height * ctx.in_width;
    int gemmk_groups = 1;
    for(int i = 0; i < 6; i++)
    {
        if(0 == (gemmk % ((1 << i) * GemmKPerBlock)))
            gemmk_groups = i;
        else
            break;
    }
    // gemmk_groups = 0;
    return gemmk_groups;
}

float CallImplicitGemmWrwDynamic(const miopen::Handle& handle,
                                 const ConvolutionContext& ctx,
                                 ConstData_t src,
                                 ConstData_t dst,
                                 Data_t wei,
                                 Data_t wei_workspace,
                                 const std::vector<KernelInvoke>& kernels)
{
    float elapsed = 0.0f;

    auto kernel = kernels[0];
    // clang-format off
    int hi             = ctx.out_height;
    int wi             = ctx.out_width;
    int n              = ctx.n_outputs;
    int k              = ctx.n_inputs;
    int c              = ctx.batch_sz;
    int ho             = ctx.kernel_size_h;
    int wo             = ctx.kernel_size_w;
    int stride_h       = ctx.kernel_dilation_h;
    int stride_w       = ctx.kernel_dilation_w;
    int dilation_h     = ctx.kernel_stride_h;
    int dilation_w     = ctx.kernel_stride_w;
    int pad_h          = ctx.pad_h;
    int pad_w          = ctx.pad_w;
    int y              = ctx.in_height;
    int x              = ctx.in_width;
    int gemmk_groups   = 0;
    int GemmKPerBlock;

    if (kernel.GetName().find(std::string("igemm_v4r1_dynamic_wrw_128x128x16")) != std::string::npos)
        GemmKPerBlock = 16;
    else 
        GemmKPerBlock = 4;

    gemmk_groups = GetImplicitGemmWrwV4R1DynamicGemmkGroups(ctx, GemmKPerBlock);

    MIOPEN_LOG_I(kernel.GetName() << " with groups for reduction: " << (1 << gemmk_groups) << " GemmKPerBlock: " << GemmKPerBlock);

    // clang-format on
    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(src);
    opArgs.emplace_back(dst);
    if(gemmk_groups > 0)
        opArgs.emplace_back(wei_workspace);
    else
        opArgs.emplace_back(wei);
    opArgs.emplace_back(hi);
    opArgs.emplace_back(wi);
    opArgs.emplace_back(n);
    opArgs.emplace_back(k);
    opArgs.emplace_back(c);
    opArgs.emplace_back(ho);
    opArgs.emplace_back(wo);
    opArgs.emplace_back(stride_h);
    opArgs.emplace_back(stride_w);
    opArgs.emplace_back(dilation_h);
    opArgs.emplace_back(dilation_w);
    opArgs.emplace_back(pad_h);
    opArgs.emplace_back(pad_w);
    opArgs.emplace_back(y);
    opArgs.emplace_back(x);
    opArgs.emplace_back(gemmk_groups);
    kernel(opArgs);

    if(handle.IsProfilingEnabled())
        elapsed += handle.GetKernelTime();

    // reduction section
    if(gemmk_groups > 0)
    {
        auto kernel_reduction = kernels[1];
        int reduction_groups  = 1 << gemmk_groups;
        MIOPEN_LOG_I(kernel_reduction.GetName() << " with groups: " << reduction_groups);
        std::vector<OpKernelArg> opArgs_reduction;
        int reduction_per_thread = 8;
        int in_stride            = n * k * ho * wo;
        opArgs_reduction.emplace_back(wei);
        opArgs_reduction.emplace_back(wei_workspace);
        opArgs_reduction.emplace_back(reduction_per_thread);
        opArgs_reduction.emplace_back(in_stride);
        opArgs_reduction.emplace_back(reduction_groups);
        kernel_reduction(opArgs_reduction);
        if(handle.IsProfilingEnabled())
            elapsed += handle.GetKernelTime();
    }

    return elapsed;
}

InvokerFactory MakeImplGemmDynamicDataInvokerFactory(const ConvolutionContext& ctx)
{
    if(ctx.direction.IsForward())
    {
        return [ctx](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle, const boost::any& primitive_parameters) {
                const auto data_ctx = boost::any_cast<conv::DataInvokeParams>(primitive_parameters);
                const auto& tensors = data_ctx.tensors;
                auto kernel         = handle.Run(kernels[0]);
                if(kernel.GetName().find("igemm_v4r1_dynamic") == 0 ||
                   kernel.GetName().find("igemm_v4r1_1x1_dynamic") == 0)
                {
                    std::vector<KernelInvoke> ks;
                    std::transform(kernels.begin(),
                                   kernels.end(),
                                   std::back_inserter(ks),
                                   [&](const Kernel& k) { return handle.Run(k); });
                    float elapsed = 0;
                    elapsed       = CallImplicitGemmDynamic(
                        handle, ctx, tensors.in, tensors.out, tensors.w, ks);
                    if(handle.IsProfilingEnabled())
                    {
                        handle.ResetKernelTime();
                        handle.AccumKernelTime(elapsed);
                    }
                }
                else
                {
                    MIOPEN_THROW(
                        "Error running dynamic implicit GEMM convolution (invalid kernel name?)");
                }
            };
        };
    }
    else if(ctx.direction.IsBackwardWrW())
    {
        return [ctx](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle, const boost::any& primitive_parameters) {
                const auto data_ctx = boost::any_cast<conv::WrWInvokeParams>(primitive_parameters);
                const auto& tensors = data_ctx.tensors;

                std::vector<KernelInvoke> ks;
                std::transform(kernels.begin(),
                               kernels.end(),
                               std::back_inserter(ks),
                               [&](const Kernel& k) { return handle.Run(k); });
                float elapsed = 0;
                elapsed       = CallImplicitGemmWrwDynamic(
                    handle, ctx, tensors.x, tensors.dy, tensors.dw, data_ctx.workSpace, ks);
                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };
    }
    else
    {
        MIOPEN_THROW(
            "Error running dynamic implicit GEMM convolution (currently only support forward)");
    }
}

} // namespace conv
} // namespace miopen
