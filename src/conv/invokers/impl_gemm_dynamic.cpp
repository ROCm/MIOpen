#include <miopen/conv/invokers/impl_gemm_dynamic.hpp>
#include <miopen/conv/data_invoke_params.hpp>
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
    else
    {
        MIOPEN_THROW(
            "Error running dynamic implicit GEMM convolution (currently only support forward)");
    }
}

} // namespace conv
} // namespace miopen
