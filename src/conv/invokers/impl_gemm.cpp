#include <miopen/conv/invokers/impl_gemm.hpp>

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/algorithm.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>

#include <boost/any.hpp>

namespace miopen {
namespace conv {

InvokerFactory MakeImplGemmDataInvokerFactory(const ConvolutionContext& ctx)
{
    if(ctx.direction.IsForward())
    {
        return [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle, const boost::any& primitive_parameters) {
                const auto data_ctx = boost::any_cast<conv::DataInvokeParams>(primitive_parameters);
                const auto& tensors = data_ctx.tensors;
                handle.Run(kernels[0])(tensors.in, tensors.w, tensors.out);
            };
        };
    }
    else
    {
        if(ctx.direction.IsBackwardWrW())
            MIOPEN_THROW("MakeImplGemmDataInvokerFactory shouldn't be used for WrW invokers.");

        const auto& conv       = ctx.conv_problem.GetConv();
        const auto& lowp_quant = conv.lowp_quant;

        return [conv, lowp_quant](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle, const boost::any& primitive_parameters) {
                const auto data_ctx = boost::any_cast<conv::DataInvokeParams>(primitive_parameters);
                const auto& tensors = data_ctx.tensors;
                const auto& workSpace = data_ctx.workSpace;

                // Miminum checks. Only check what is required to select
                // proper invocation procedure & workspace sanity.
                auto kernel = handle.Run(kernels[0]);

                float elapsed = 0;
                // clang-format off
                if((tensors.outDesc.GetType() == miopenHalf ||
                    tensors.outDesc.GetType() == miopenBFloat16) &&
                   (kernel.GetName() == "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_nchw_kcyx_nkhw" ||
                    kernel.GetName() == "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_gnchw_gkcyx_gnkhw" ||
                    kernel.GetName() == "gridwise_convolution_backward_data_implicit_gemm_v1r1_nchw_kcyx_nkhw" ||
                    kernel.GetName() == "gridwise_convolution_backward_data_implicit_gemm_v1r1_ncdhw_kczyx_nkdhw"))
                // clang-format on
                {
                    const auto y          = tensors.wDesc.GetLengths()[2];
                    const auto x          = tensors.wDesc.GetLengths()[3];
                    const auto stride_h   = conv.GetConvStrides()[0];
                    const auto stride_w   = conv.GetConvStrides()[1];
                    const auto dilation_h = conv.GetConvDilations()[0];
                    const auto dilation_w = conv.GetConvDilations()[1];

                    bool need_atomic_add = (stride_h < dilation_h * (y - 1) + 1) ||
                                           (stride_w < dilation_w * (x - 1) + 1);
                    bool every_pixel_is_written =
                        (dilation_h == 1 && stride_h <= y) && (dilation_w == 1 && stride_w <= x);
                    bool need_set_zero = need_atomic_add || !(every_pixel_is_written);
                    bool need_cast     = need_atomic_add;

                    if(need_set_zero && !need_cast)
                    {
                        float zero = 0.f;
                        SetTensor(handle, tensors.outDesc, tensors.out, &zero);
                        if(handle.IsProfilingEnabled())
                            elapsed += handle.GetKernelTime();

                        kernel(tensors.in, tensors.w, tensors.out);
                        if(handle.IsProfilingEnabled())
                            elapsed += handle.GetKernelTime();
                    }
                    else if(!need_set_zero && !need_cast)
                    {
                        kernel(tensors.in, tensors.w, tensors.out);
                        if(handle.IsProfilingEnabled())
                            elapsed += handle.GetKernelTime();
                    }
                    else
                    {
                        float zero = 0.f;
                        TensorDescriptor workspaceDesc(miopenFloat,
                                                       tensors.outDesc.GetLengths(),
                                                       tensors.outDesc.GetStrides());
                        SetTensor(handle, workspaceDesc, workSpace, &zero);
                        if(handle.IsProfilingEnabled())
                            elapsed += handle.GetKernelTime();

                        kernel(tensors.in, tensors.w, workSpace);
                        if(handle.IsProfilingEnabled())
                            elapsed += handle.GetKernelTime();

                        CastTensor(handle,
                                   &lowp_quant,
                                   workspaceDesc,
                                   workSpace,
                                   tensors.outDesc,
                                   tensors.out,
                                   0,
                                   0);
                        if(handle.IsProfilingEnabled())
                            elapsed += handle.GetKernelTime();
                    }
                }
                // clang-format off
                else if(kernel.GetName() == "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_nchw_kcyx_nkhw" ||
                        kernel.GetName() == "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_gnchw_gkcyx_gnkhw")
                // clang-format on
                {
                    float zero = 0.f;
                    SetTensor(handle, tensors.outDesc, tensors.out, &zero);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();

                    kernel(tensors.in, tensors.w, tensors.out);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
                // clang-format off
                else if(
                    kernel.GetName() == "gridwise_convolution_backward_data_implicit_gemm_v1r1_nchw_kcyx_nkhw" ||
                    kernel.GetName() == "gridwise_convolution_backward_data_implicit_gemm_v1r1_ncdhw_kczyx_nkdhw")
                // clang-format on
                {
                    // this kernel accumulate results into input tensor, therefore need to set zero
                    bool is_1x1_s1 = false;
                    if(miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
                       miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; }))
                    {
                        if(tensors.wDesc.GetLengths()[2] == 1 && tensors.wDesc.GetLengths()[3] == 1)
                        { // filter = 1
                            if(tensors.wDesc.GetSize() == 4 ||
                               (tensors.wDesc.GetSize() == 5 && tensors.wDesc.GetLengths()[4] == 1))
                            {
                                is_1x1_s1 = true;
                            }
                        }
                    }

                    if(!is_1x1_s1)
                    {
                        float zero = 0.f;
                        SetTensor(handle, tensors.outDesc, tensors.out, &zero);
                        if(handle.IsProfilingEnabled())
                            elapsed += handle.GetKernelTime();
                    }

                    kernel(tensors.in, tensors.w, tensors.out);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
                // clang-format off
                else if(
                    kernel.GetName() == "gridwise_convolution_backward_data_implicit_gemm_v4r1_nchw_kcyx_nkhw" ||
                    kernel.GetName() == "gridwise_convolution_backward_data_implicit_gemm_v4r1_xdlops_nchw_kcyx_nkhw" ||
                    kernel.GetName() == "gridwise_convolution_backward_data_implicit_gemm_v4r1_xdlops_gnchw_gkcyx_gnkhw" ||
                    kernel.GetName() == "gridwise_convolution_backward_data_implicit_gemm_v4r1_ncdhw_kczyx_nkdhw")
                // clang-format on
                {
                    // \todo this kernel doesn't always need to set-zero
                    bool filterGeStride = false;
                    if(miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }))
                    {
                        if(tensors.wDesc.GetSize() == 4)
                        { // 2d
                            if(tensors.wDesc.GetLengths()[2] >= conv.GetConvStrides()[0] &&
                               tensors.wDesc.GetLengths()[3] >= conv.GetConvStrides()[1])
                            {
                                filterGeStride = true;
                            }
                        }
                        else
                        { // 3d
                            if(tensors.wDesc.GetLengths()[2] >= conv.GetConvStrides()[0] &&
                               tensors.wDesc.GetLengths()[3] >= conv.GetConvStrides()[1] &&
                               tensors.wDesc.GetLengths()[4] >= conv.GetConvStrides()[2])
                            {
                                filterGeStride = true;
                            }
                        }
                    }

                    if(!filterGeStride)
                    {
                        float zero = 0.f;
                        SetTensor(handle, tensors.outDesc, tensors.out, &zero);

                        if(handle.IsProfilingEnabled())
                            elapsed += handle.GetKernelTime();
                    }

                    // a group kernels (compiled from same source code) will be launched
                    for(const auto& k : kernels)
                    {
                        handle.Run(k)(tensors.in, tensors.w, tensors.out);
                        elapsed += handle.GetKernelTime();
                    }
                }
                else
                {
                    MIOPEN_THROW(
                        "Error running implicit GEMM backward data convolution (none workspace?)");
                }

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };
    }
}

} // namespace conv
} // namespace miopen
