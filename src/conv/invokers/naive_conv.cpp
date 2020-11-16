#include <miopen/conv/invokers/impl_gemm_dynamic.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/algorithm.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/numeric.hpp>
#include <boost/any.hpp>

namespace miopen {
namespace conv {

InvokerFactory MakeNaiveConvFwdInvokerFactory(const ConvolutionContext& ctx)
{
    // const auto& conv_param = ctx.conv_problem;
    // clang-format on
    int di          = ctx.in_depth;
    int hi          = ctx.in_height;
    int wi          = ctx.in_width;
    int n           = ctx.batch_sz;
    int k           = ctx.n_outputs;
    int c           = ctx.n_inputs;
    int do_         = ctx.out_depth;
    int ho          = ctx.out_height;
    int wo          = ctx.out_width;
    int sz          = ctx.kernel_stride_d;
    int sy          = ctx.kernel_stride_h;
    int sx          = ctx.kernel_stride_w;
    int dz          = ctx.kernel_dilation_d;
    int dy          = ctx.kernel_dilation_h;
    int dx          = ctx.kernel_dilation_w;
    int pz          = ctx.pad_d;
    int py          = ctx.pad_h;
    int px          = ctx.pad_w;
    int fz          = ctx.kernel_size_d;
    int fy          = ctx.kernel_size_h;
    int fx          = ctx.kernel_size_w;
    int group       = ctx.group_counts;
    int c_per_group = c / group;
    int k_per_group = k / group;
    // clang-format off

    if(ctx.Is2d())
        return [=](const std::vector<Kernel>& kernels) {
            const auto kernel = kernels[0];
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
                const auto& tensors     = data_ctx.tensors;
                float elapsed           = 0;

                handle.Run(kernel)(tensors.in,
                                tensors.w,
                                tensors.out,
                                hi,
                                wi,
                                n,
                                k_per_group,
                                c_per_group,
                                ho,
                                wo,
                                sy,
                                sx,
                                dy,
                                dx,
                                py,
                                px,
                                fy,
                                fx,
                                group);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };
    else
        return [=](const std::vector<Kernel>& kernels) {
            const auto kernel = kernels[0];
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
                const auto& tensors     = data_ctx.tensors;
                float elapsed           = 0;

                handle.Run(kernel)(tensors.in,
                                tensors.w,
                                tensors.out,
                                di,
                                hi,
                                wi,
                                n,
                                k_per_group,
                                c_per_group,
                                do_,
                                ho,
                                wo,
                                sz,
                                sy,
                                sx,
                                dz,
                                dy,
                                dx,
                                pz,
                                py,
                                px,
                                fz,
                                fy,
                                fx,
                                group);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };
}


InvokerFactory MakeNaiveConvBwdInvokerFactory(const ConvolutionContext& ctx)
{
    // const auto& conv_param = ctx.conv_problem;
    // clang-format on
    int di          = ctx.out_depth;
    int hi          = ctx.out_height;
    int wi          = ctx.out_width;
    int n           = ctx.batch_sz;
    int k           = ctx.n_inputs;
    int c           = ctx.n_outputs;
    int do_         = ctx.in_depth;
    int ho          = ctx.in_height;
    int wo          = ctx.in_width;
    int sz          = ctx.in_depth > 1 ? ctx.kernel_stride_d : 1;
    int sy          = ctx.in_height > 1 ? ctx.kernel_stride_h : 1;
    int sx          = ctx.in_width > 1 ? ctx.kernel_stride_w : 1;
    int dz          = ctx.kernel_size_d > 1 ? ctx.kernel_dilation_d : 1;
    int dy          = ctx.kernel_size_h > 1 ? ctx.kernel_dilation_h : 1;
    int dx          = ctx.kernel_size_w > 1 ? ctx.kernel_dilation_w : 1;
    int pz          = ctx.pad_d;
    int py          = ctx.pad_h;
    int px          = ctx.pad_w;
    int fz          = ctx.kernel_size_d;
    int fy          = ctx.kernel_size_h;
    int fx          = ctx.kernel_size_w;
    int group       = ctx.group_counts;
    int c_per_group = c / group;
    int k_per_group = k / group;
    // clang-format off

    if(ctx.Is2d())
        return [=](const std::vector<Kernel>& kernels) {
            const auto kernel = kernels[0];
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
                const auto& tensors     = data_ctx.tensors;
                float elapsed           = 0;

                handle.Run(kernel)(tensors.out,
                                tensors.w,
                                tensors.in,
                                hi,
                                wi,
                                n,
                                k_per_group,
                                c_per_group,
                                ho,
                                wo,
                                sy,
                                sx,
                                dy,
                                dx,
                                py,
                                px,
                                fy,
                                fx,
                                group);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };
    else
        return [=](const std::vector<Kernel>& kernels) {
            const auto kernel = kernels[0];
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
                const auto& tensors     = data_ctx.tensors;
                float elapsed           = 0;

                handle.Run(kernel)(tensors.out,
                                tensors.w,
                                tensors.in,
                                di,
                                hi,
                                wi,
                                n,
                                k_per_group,
                                c_per_group,
                                do_,
                                ho,
                                wo,
                                sz,
                                sy,
                                sx,
                                dz,
                                dy,
                                dx,
                                pz,
                                py,
                                px,
                                fz,
                                fy,
                                fx,
                                group);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };
}

InvokerFactory MakeNaiveConvWrwInvokerFactory(const ConvolutionContext& ctx)
{
    // const auto& conv_param = ctx.conv_problem;
    // clang-format on
    int di          = ctx.out_depth;
    int hi          = ctx.out_height;
    int wi          = ctx.out_width;
    int n           = ctx.batch_sz;
    int k           = ctx.n_inputs;
    int c           = ctx.n_outputs;
    int do_         = ctx.in_depth;
    int ho          = ctx.in_height;
    int wo          = ctx.in_width;
    int sz          = ctx.in_depth > 1 ? ctx.kernel_stride_d : 1;
    int sy          = ctx.in_height > 1 ? ctx.kernel_stride_h : 1;
    int sx          = ctx.in_width > 1 ? ctx.kernel_stride_w : 1;
    int dz          = ctx.kernel_size_d > 1 ? ctx.kernel_dilation_d : 1;
    int dy          = ctx.kernel_size_h > 1 ? ctx.kernel_dilation_h : 1;
    int dx          = ctx.kernel_size_w > 1 ? ctx.kernel_dilation_w : 1;
    int pz          = ctx.pad_d;
    int py          = ctx.pad_h;
    int px          = ctx.pad_w;
    int fz          = ctx.kernel_size_d;
    int fy          = ctx.kernel_size_h;
    int fx          = ctx.kernel_size_w;
    int group       = ctx.group_counts;
    int c_per_group = c / group;
    int k_per_group = k / group;
    // clang-format off

    if(ctx.Is2d())
        return [=](const std::vector<Kernel>& kernels) {
            const auto kernel = kernels[0];
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                decltype(auto) data_ctx = primitive_parameters.CastTo<conv::WrWInvokeParams>();
                const auto& tensors     = data_ctx.tensors;
                float elapsed           = 0;

                handle.Run(kernel)(tensors.x,
                                tensors.dw,
                                tensors.dy,
                                hi,
                                wi,
                                n,
                                k_per_group,
                                c_per_group,
                                ho,
                                wo,
                                sy,
                                sx,
                                dy,
                                dx,
                                py,
                                px,
                                fy,
                                fx,
                                group);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };
    else
        return [=](const std::vector<Kernel>& kernels) {
            const auto kernel = kernels[0];
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                decltype(auto) data_ctx = primitive_parameters.CastTo<conv::WrWInvokeParams>();
                const auto& tensors     = data_ctx.tensors;
                float elapsed           = 0;

                handle.Run(kernel)(tensors.x,
                                tensors.dw,
                                tensors.dy,
                                di,
                                hi,
                                wi,
                                n,
                                k_per_group,
                                c_per_group,
                                do_,
                                ho,
                                wo,
                                sz,
                                sy,
                                sx,
                                dz,
                                dy,
                                dx,
                                pz,
                                py,
                                px,
                                fz,
                                fy,
                                fx,
                                group);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

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
