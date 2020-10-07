#include <miopen/conv/invokers/impl_gemm_dynamic.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/algorithm.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/numeric.hpp>
#include <boost/any.hpp>

namespace miopen {
namespace conv {

float CallImplGemmDynamicForward(const miopen::Handle& handle,
                                 const ProblemDescription& conv_problem,
                                 ConstData_t src,
                                 Data_t dst,
                                 ConstData_t wei,
                                 const std::vector<KernelInvoke>& kernels)
{
    float elapsed = 0.0f;

    auto kernel = kernels[0];
    MIOPEN_LOG_I(kernel.GetName());

    // clang-format off
    int hi          = conv_problem.GetInHeight();
    int wi          = conv_problem.GetInWidth();
    int n           = conv_problem.GetInBatchSize();
    int k           = conv_problem.GetOutChannels();
    int c           = conv_problem.GetInChannels();
    int ho          = conv_problem.GetOutHeight();
    int wo          = conv_problem.GetOutWidth();
    int stride_h    = conv_problem.GetKernelStrideH();
    int stride_w    = conv_problem.GetKernelStrideW();
    int dilation_h  = conv_problem.GetDilationH();
    int dilation_w  = conv_problem.GetDilationW();
    int pad_h       = conv_problem.GetPadH();
    int pad_w       = conv_problem.GetPadW();
    int y           = conv_problem.GetWeightsHeight();
    int x           = conv_problem.GetWeightsWidth();
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
    opArgs.emplace_back(y);
    opArgs.emplace_back(x);
    opArgs.emplace_back(__pack0);

    kernel(opArgs);

    if(handle.IsProfilingEnabled())
        elapsed += handle.GetKernelTime();
    return elapsed;
}

float CallImplGemmDynamicForward1x1(const miopen::Handle& handle,
                                    const ProblemDescription& conv_problem,
                                    ConstData_t src,
                                    Data_t dst,
                                    ConstData_t wei,
                                    const std::vector<KernelInvoke>& kernels)
{
    float elapsed = 0.0f;

    auto kernel = kernels[0];
    MIOPEN_LOG_I(kernel.GetName());

    // clang-format off
    int hi          = conv_problem.GetInHeight();
    int wi          = conv_problem.GetInWidth();
    int n           = conv_problem.GetInBatchSize();
    int k           = conv_problem.GetOutChannels();
    int c           = conv_problem.GetInChannels();
    int ho          = conv_problem.GetOutHeight();
    int wo          = conv_problem.GetOutWidth();
    int stride_h    = conv_problem.GetKernelStrideH();
    int stride_w    = conv_problem.GetKernelStrideW();
    int dilation_h  = conv_problem.GetDilationH();
    int dilation_w  = conv_problem.GetDilationW();
    int pad_h       = conv_problem.GetPadH();
    int pad_w       = conv_problem.GetPadW();
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
    opArgs.emplace_back(__pack0);

    kernel(opArgs);

    if(handle.IsProfilingEnabled())
        elapsed += handle.GetKernelTime();
    return elapsed;
}

float CallImplGemmDynamicBackwardData(const miopen::Handle& handle,
                                      const ProblemDescription& conv_problem,
                                      ConstData_t src,
                                      Data_t dst,
                                      ConstData_t wei,
                                      const std::vector<KernelInvoke>& kernels)
{
    float elapsed = 0.0f;

    auto kernel = kernels[0];
    MIOPEN_LOG_I(kernel.GetName());

    // clang-format off
    int hi          = conv_problem.GetOutHeight();
    int wi          = conv_problem.GetOutWidth();
    int n           = conv_problem.GetInBatchSize();
    int k           = conv_problem.GetInChannels();
    int c           = conv_problem.GetOutChannels();
    int ho          = conv_problem.GetInHeight();
    int wo          = conv_problem.GetInWidth();
    int stride_h    = conv_problem.GetInHeight() > 1 ? conv_problem.GetKernelStrideH() : 1;
    int stride_w    = conv_problem.GetInWidth() > 1 ? conv_problem.GetKernelStrideW() : 1;
    int dilation_h  = conv_problem.GetWeightsHeight() > 1? conv_problem.GetDilationH() : 1;
    int dilation_w  = conv_problem.GetWeightsWidth() > 1? conv_problem.GetDilationW() : 1;
    int pad_h       = conv_problem.GetPadH();
    int pad_w       = conv_problem.GetPadW();
    int y           = conv_problem.GetWeightsHeight();
    int x           = conv_problem.GetWeightsWidth();

    int gcd_stride_dilation_h = gcd(stride_h, dilation_h);
    int gcd_stride_dilation_w = gcd(stride_w, dilation_w);
    int y_tilda     = stride_h / gcd_stride_dilation_h;
    int x_tilda     = stride_w / gcd_stride_dilation_w;

    int y_dot = (y +  y_tilda - 1) / y_tilda;
    int x_dot = (x +  x_tilda - 1) / x_tilda;

    int h_tilda     = ho + (dilation_h * (y - 1) + stride_h - 1) / stride_h;
    int w_tilda     = wo + (dilation_w * (x - 1) + stride_w - 1) / stride_w;

    int h_tilda_left = std::max(0, pad_h - dilation_h * (y_tilda - 1)) / stride_h;
    int w_tilda_left = std::max(0, pad_w - dilation_w * (x_tilda - 1)) / stride_w;

    int h_tilda_right = std::min(h_tilda, (pad_h + hi - 1 + stride_h - 1) / stride_h + 1);
    int w_tilda_right = std::min(w_tilda, (pad_w + wi - 1 + stride_w - 1) / stride_w + 1);

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
        int _dtile_iy          = gemm_id / x_tilda;
        int _dtile_ix          = gemm_id % x_tilda;
        int _y_dot_slice       = (_dtile_iy + 1) * y_dot <= y ? y_dot : y % y_dot;
        int _x_dot_slice       = (_dtile_ix + 1) * x_dot <= x ? x_dot : x % x_dot;
        int _gemm_k            = k * _y_dot_slice * _x_dot_slice;
        bool is_gemm_not_empty = _gemm_k > 0;
        opArgs[18]             = OpKernelArg(_dtile_iy);
        opArgs[19]             = OpKernelArg(_dtile_ix);
        opArgs[26]             = OpKernelArg(_y_dot_slice);
        opArgs[27]             = OpKernelArg(_x_dot_slice);
        if(is_gemm_not_empty)
        {
            kernel(opArgs);
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();
        }
    }

    return elapsed;
}

InvokerFactory MakeImplGemmDynamicForwardInvokerFactory(const ConvolutionContext& ctx)
{
    const auto& conv_problem = ctx.conv_problem;
    return [conv_problem](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            auto kernel             = handle.Run(kernels[0]);

            std::vector<KernelInvoke> ks;
            std::transform(kernels.begin(),
                           kernels.end(),
                           std::back_inserter(ks),
                           [&](const Kernel& k) { return handle.Run(k); });
            float elapsed = 0;
            elapsed       = CallImplGemmDynamicForward(
                handle, conv_problem, tensors.in, tensors.out, tensors.w, ks);
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

InvokerFactory MakeImplGemmDynamicForward1x1InvokerFactory(const ConvolutionContext& ctx)
{
    const auto& conv_problem = ctx.conv_problem;
    return [conv_problem](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            auto kernel             = handle.Run(kernels[0]);

            std::vector<KernelInvoke> ks;
            std::transform(kernels.begin(),
                           kernels.end(),
                           std::back_inserter(ks),
                           [&](const Kernel& k) { return handle.Run(k); });
            float elapsed = 0;
            elapsed       = CallImplGemmDynamicForward1x1(
                handle, conv_problem, tensors.in, tensors.out, tensors.w, ks);
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

InvokerFactory MakeImplGemmDynamicBackwardDataInvokerFactory(const ConvolutionContext& ctx)
{
    const auto& conv_problem = ctx.conv_problem;
    return [conv_problem](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            auto kernel             = handle.Run(kernels[0]);

            std::vector<KernelInvoke> ks;
            std::transform(kernels.begin(),
                           kernels.end(),
                           std::back_inserter(ks),
                           [&](const Kernel& k) { return handle.Run(k); });
            float elapsed = 0;

            elapsed = CallImplGemmDynamicBackwardData(
                handle, conv_problem, tensors.in, tensors.out, tensors.w, ks);

            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

} // namespace conv
} // namespace miopen
