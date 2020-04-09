#include <miopen/conv/invokers/impl_gemm.hpp>

#include <miopen/conv/data_invoke_params.hpp>
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
            return [=](Handle& handle, const boost::any& primitive_parameters) {
                const auto data_ctx = boost::any_cast<conv::DataInvokeParams>(primitive_parameters);
                const auto& tensors = data_ctx.tensors;
                handle.Run(kernels[0])(tensors.in, tensors.w, tensors.out);
            };
        };
    }
    else
    {
        const auto lowp_quant = ctx.conv_problem.GetConv().lowp_quant;

        return [lowp_quant](const std::vector<Kernel>& kernels) {
            return [=](Handle& handle, const boost::any& primitive_parameters) {
                const auto data_ctx = boost::any_cast<conv::DataInvokeParams>(primitive_parameters);
                const auto& tensors = data_ctx.tensors;
                const auto& workSpace = data_ctx.workSpace;

                // Miminum checks. Only check what is required to select
                // proper invocation procedure & workspace sanity.
                const auto kernel = handle.Run(kernels[0]);

                float elapsed = 0;
                if((tensors.outDesc.GetType() == miopenHalf || tensors.outDesc.GetType() == miopenBFloat16) &&
                   (kernel.GetName() ==
                        "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_nchw_kcyx_nkhw" ||
                    kernel.GetName() ==
                        "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_gnchw_gkcyx_gnkhw" ||
                    kernel.GetName() ==
                        "gridwise_convolution_backward_data_implicit_gemm_v1r1_nchw_kcyx_nkhw"))
                {
                    float zero = 0.f;
                    TensorDescriptor workspaceDesc(
                        miopenFloat, tensors.outDesc.GetLengths(), tensors.outDesc.GetStrides());
                    SetTensor(handle, workspaceDesc, workSpace, &zero);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();

                    kernel(tensors.in, tensors.w, workSpace);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();

                    CastTensor(
                        handle, &lowp_quant, workspaceDesc, workSpace, tensors.outDesc, tensors.out, 0, 0);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
                else if((kernel.GetName() ==
                         "gridwise_convolution_implicit_gemm_v4_nchw_kc1x1_nkhw_lds_double_buffer") ||
                        (kernel.GetName() == "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kc1x1_"
                                             "nkhw_lds_double_buffer"))
                {
                    bool hasStride = (tensors.inDesc.GetLengths()[2] != tensors.outDesc.GetLengths()[2]) ||
                                     (tensors.inDesc.GetLengths()[3] != tensors.outDesc.GetLengths()[3]);
                    /// \todo set zero within implicitGEMM kernel
                    if(hasStride)
                    {
                        MIOPEN_LOG_I2("hasStride, call SetTensor with zero");
                        float zero = 0.f;
                        SetTensor(handle, tensors.outDesc, tensors.out, &zero);

                        if(handle.IsProfilingEnabled())
                            elapsed += handle.GetKernelTime();
                    }

                    kernel(tensors.in, tensors.w, tensors.out);

                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
                else if(kernel.GetName() ==
                            "gridwise_convolution_backward_data_implicit_gemm_v1r1_nchw_kcyx_nkhw" ||
                        kernel.GetName() ==
                            "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_nchw_kcyx_nkhw" ||
                        kernel.GetName() == "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_"
                                            "gnchw_gkcyx_gnkhw")
                {
                    // this kernel accumulate results into input tensor, therefore need to set zero
                    float zero = 0.f;
                    SetTensor(handle, tensors.outDesc, tensors.out, &zero);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();

                    kernel(tensors.in, tensors.w, tensors.out);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
                else if(kernel.GetName() ==
                        "gridwise_convolution_backward_data_implicit_gemm_v4r1_nchw_kcyx_nkhw")
                {
                    // \todo this kernel doesn't always need to set-zero
                    float zero = 0.f;
                    SetTensor(handle, tensors.outDesc, tensors.out, &zero);

                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();

                    // a group kernels (compiled from same source code) will be launched
                    for(const auto& k : kernels)
                    {
                        handle.Run(k)(tensors.in, tensors.w, tensors.out);
                        elapsed += handle.GetKernelTime();
                    }
                }
                else
                {
                    MIOPEN_THROW("Error running implicit GEMM backward data convolution (none workspace?)");
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
