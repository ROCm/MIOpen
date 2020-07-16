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
    if(ctx.direction.IsForward())
    {
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
        // clang-format off
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
    }else if (ctx.direction.IsBackwardData()){
        if(kernel.GetName().find("igemm_bwd_gtc") == 0)
        {
            // clang-format off
            int hi          = ctx.out_height;
            int wi          = ctx.out_width;
            int n           = ctx.batch_sz;
            int k           = ctx.n_inputs;
            int c           = ctx.n_outputs;
            int ho          = ctx.in_height;
            int wo          = ctx.in_width;
            int stride_h    = ctx.in_height > 1 ? ctx.kernel_stride_h : 1;
            int stride_w    = ctx.in_width > 1 ? ctx.kernel_stride_w : 1;
            int dilation_h  = ctx.kernel_size_h > 1? ctx.kernel_dilation_h : 1;
            int dilation_w  = ctx.kernel_size_w > 1? ctx.kernel_dilation_w : 1;
            int pad_h       = ctx.pad_h;
            int pad_w       = ctx.pad_w;
            int y           = ctx.kernel_size_h;
            int x           = ctx.kernel_size_w;

            int gcd_stride_dilation_h = igemm_dynamic::gcd(stride_h, dilation_h);
            int gcd_stride_dilation_w = igemm_dynamic::gcd(stride_w, dilation_w);
            int y_tilda     = stride_h / gcd_stride_dilation_h;
            int x_tilda     = stride_w / gcd_stride_dilation_w;

            int y_dot = (y +  y_tilda - 1) / y_tilda;
            int x_dot = (x +  x_tilda - 1) / x_tilda;

            int h_tilda     = ho + (dilation_h * (y - 1) + stride_h - 1) / stride_h;
            int w_tilda     = wo + (dilation_w * (x - 1) + stride_w - 1) / stride_w;

            int h_tilda_left = igemm_dynamic::max(0, pad_h - dilation_h * (y_tilda - 1)) / stride_h;
            int w_tilda_left = igemm_dynamic::max(0, pad_w - dilation_w * (x_tilda - 1)) / stride_w;

            int h_tilda_right = igemm_dynamic::min(h_tilda, (pad_h + hi - 1 + stride_h - 1) / stride_h + 1);
            int w_tilda_right = igemm_dynamic::min(w_tilda, (pad_w + wi - 1 + stride_w - 1) / stride_w + 1);

            int h_tilda_slice = h_tilda_right - h_tilda_left;
            int w_tilda_slice = w_tilda_right - w_tilda_left;

            int num_of_gemms  = x_tilda * y_tilda;

            int dtile_iy    = 0;
            int dtile_ix    = 0;
            int dtile_dy    = dilation_h / gcd_stride_dilation_h;
            int dtile_dx    = dilation_w / gcd_stride_dilation_w;
            int dtile_y     = y_tilda;
            int dtile_x     = x_tilda;
            int dtile_h     = h_tilda;
            int dtile_w     = w_tilda;
            int dslice_y    = 0;
            int dslice_x    = 0;
            int dslice_h    = h_tilda_slice;
            int dslice_w    = w_tilda_slice;
            int dslice_h_left   = h_tilda_left;
            int dslice_w_left   = w_tilda_left;
            int __pack0     = 0;
// clang-format on
#if 0
            MIOPEN_LOG_I2("hi:" << hi << ", wi:" << wi << ", n:" << n << ", k:" << k << ", c:" << c
                                << ", ho:"
                                << ho
                                << ", wo:"
                                << wo
                                << ", stride_h:"
                                << stride_h
                                << ", stride_w:"
                                << stride_w
                                << ", dilation_h:"
                                << dilation_h
                                << ", dilation_w:"
                                << dilation_w
                                << ", pad_h:"
                                << pad_h
                                << ", pad_w:"
                                << pad_w
                                << ", y:"
                                << y
                                << ", x:"
                                << x
                                << ", gcd_stride_dilation_h:"
                                << gcd_stride_dilation_h
                                << ", gcd_stride_dilation_w:"
                                << gcd_stride_dilation_w
                                << ", y_tilda:"
                                << y_tilda
                                << ", x_tilda:"
                                << x_tilda
                                << ", h_tilda:"
                                << h_tilda
                                << ", w_tilda:"
                                << w_tilda
                                << ", h_tilda_left:"
                                << h_tilda_left
                                << ", w_tilda_left:"
                                << w_tilda_left
                                << ", h_tilda_slice:"
                                << h_tilda_slice
                                << ", w_tilda_slice:"
                                << w_tilda_slice);
#endif

            std::vector<OpKernelArg> opArgs;
            opArgs.emplace_back(dst);
            opArgs.emplace_back(wei);
            opArgs.emplace_back(src);
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
            opArgs.emplace_back(dtile_iy);
            opArgs.emplace_back(dtile_ix);
            opArgs.emplace_back(dtile_dy);
            opArgs.emplace_back(dtile_dx);
            opArgs.emplace_back(dtile_y);
            opArgs.emplace_back(dtile_x);
            opArgs.emplace_back(dtile_h);
            opArgs.emplace_back(dtile_w);
            opArgs.emplace_back(dslice_y);
            opArgs.emplace_back(dslice_x);
            opArgs.emplace_back(dslice_h);
            opArgs.emplace_back(dslice_w);
            opArgs.emplace_back(dslice_h_left);
            opArgs.emplace_back(dslice_w_left);
            opArgs.emplace_back(__pack0);

            for(int gemm_id = 0; gemm_id < num_of_gemms; gemm_id++)
            {
                int _dtile_iy    = gemm_id / x_tilda;
                int _dtile_ix    = gemm_id % x_tilda;
                int _y_dot_slice = (_dtile_iy + 1) * y_dot <= y ? y_dot : y % y_dot;
                int _x_dot_slice = (_dtile_ix + 1) * x_dot <= x ? x_dot : x % x_dot;
                opArgs[18]       = OpKernelArg(_dtile_iy);
                opArgs[19]       = OpKernelArg(_dtile_ix);
                opArgs[26]       = OpKernelArg(_y_dot_slice);
                opArgs[27]       = OpKernelArg(_x_dot_slice);
                kernels[gemm_id](opArgs);
            }
        }
        else
        {
            MIOPEN_THROW("Error running dynamic implicit GEMM convolution for bwd, no such kernel");
        }
    }
    else if(ctx.direction.IsBackwardWrW())
    {
        MIOPEN_THROW("Error running dynamic implicit GEMM convolution for wrw");
    }

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
    else if(ctx.direction.IsBackwardData())
    {
        return [ctx](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle, const boost::any& primitive_parameters) {
                const auto data_ctx = boost::any_cast<conv::DataInvokeParams>(primitive_parameters);
                const auto& tensors = data_ctx.tensors;
                auto kernel         = handle.Run(kernels[0]);
                if(kernel.GetName().find("igemm_bwd_gtc") == 0)
                {
                    std::vector<KernelInvoke> ks;
                    std::transform(kernels.begin(),
                                   kernels.end(),
                                   std::back_inserter(ks),
                                   [&](const Kernel& k) { return handle.Run(k); });
                    float elapsed = 0;

                    elapsed = CallImplicitGemmDynamic(
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
