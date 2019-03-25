/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <miopen/lrn.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/visit_float.hpp>

namespace miopen {

miopenStatus_t LRNDescriptor::Forward(Handle& handle,
                                      const void* /*alpha*/,
                                      const TensorDescriptor& xDesc,
                                      ConstData_t x,
                                      const void* /*beta*/,
                                      const TensorDescriptor& yDesc,
                                      Data_t y,
                                      bool do_backward,
                                      Data_t workSpace) const
{

    miopenStatus_t status = miopenStatusSuccess;
    mlo_construct_norm construct_params(1); // forward

    construct_params.setStream(&handle);

    int nOut;
    int cOut;
    int hOut;
    int wOut;
    int nOutStride;
    int cOutStride;
    int hOutStride;
    int wOutStride;

    std::tie(nOut, cOut, hOut, wOut)                         = tien<4>(yDesc.GetLengths());
    std::tie(nOutStride, cOutStride, hOutStride, wOutStride) = tien<4>(yDesc.GetStrides());

    construct_params.setTopDescFromMLDesc(yDesc);

    int nIn;
    int cIn;
    int hIn;
    int wIn;
    int nInStride;
    int cInStride;
    int hInStride;
    int wInStride;

    std::tie(nIn, cIn, hIn, wIn)                         = tien<4>(xDesc.GetLengths());
    std::tie(nInStride, cInStride, hInStride, wInStride) = tien<4>(xDesc.GetStrides());

    construct_params.setBotDescFromMLDesc(xDesc);

    int norm_reg     = GetMode();
    int local_area   = static_cast<int>(GetN());
    double lrn_alpha = GetAlpha();
    double lrn_beta  = GetBeta();
    double lrn_K     = GetK();

    construct_params.doBackward(do_backward);
    construct_params.setNormDescr(norm_reg, local_area, lrn_alpha, lrn_beta, lrn_K);

    mloConstruct(construct_params);
    int norm_region;
    int local_ar;
    // whithin channel alphaoverarea is going to be culculate based on actual areal size (cut by
    // borders).
    double norm_alpha;
    double norm_beta;
    double norm_K;
    double norm_alphaoverarea;

    construct_params.getNormDescr(
        norm_region, local_ar, norm_alpha, norm_beta, norm_K, norm_alphaoverarea);
    auto f_norm_alpha         = static_cast<float>(norm_alpha);
    auto f_norm_beta          = static_cast<float>(norm_beta);
    auto f_norm_K             = static_cast<float>(norm_K);
    auto f_norm_alphaoverarea = static_cast<float>(norm_alphaoverarea);

    if(float_equal(f_norm_K, 0.0))
        MIOPEN_THROW("Expect non-zero bias/K");

    std::string algo_name = "miopenLRNForward";
    std::string network_config =
        std::to_string(f_norm_alpha) + std::to_string(f_norm_beta) + std::to_string(f_norm_K) +
        std::to_string(f_norm_alphaoverarea) + std::to_string(local_ar) +
        std::to_string(norm_region) + std::to_string(static_cast<int>(do_backward)) +
        std::to_string(xDesc.GetType()) + std::to_string(nInStride) + std::to_string(nOutStride) +
        std::to_string(nIn) + std::to_string(nOut) + std::to_string(nInStride) +
        std::to_string(nOutStride) + std::to_string(cIn) + std::to_string(cOut) +
        std::to_string(cInStride) + std::to_string(cOutStride) + std::to_string(hIn) +
        std::to_string(hOut);

    auto&& kernels = handle.GetKernels(algo_name, network_config);
    if(!kernels.empty())
    {
        visit_float(xDesc.GetType(), [&](auto as_float) {
            if(do_backward)
            {
                kernels.front()(x,
                                y,
                                workSpace,
                                as_float(f_norm_alphaoverarea),
                                as_float(f_norm_alpha),
                                as_float(f_norm_beta),
                                as_float(f_norm_K));
            }
            else
            {
                kernels.front()(x,
                                y,
                                as_float(f_norm_alphaoverarea),
                                as_float(f_norm_alpha),
                                as_float(f_norm_beta),
                                as_float(f_norm_K));
            }
        });
    }
    else
    {
        const std::string program_name = construct_params.getKernelFile(); // CL kernel filename
        const std::string kernel_name  = construct_params.getKernelName(); // kernel name
        const std::string& compiler_parms =
            construct_params.getCompilerOptions(); // kernel parameters
        const std::vector<size_t>& vld = construct_params.getLocalWkSize();
        const std::vector<size_t>& vgd = construct_params.getGlobalWkSize();

        KernelInvoke obj = handle.AddKernel(
            algo_name, network_config, program_name, kernel_name, vld, vgd, compiler_parms);
        visit_float(xDesc.GetType(), [&](auto as_float) {
            if(do_backward)
            {
                obj(x,
                    y,
                    workSpace,
                    as_float(f_norm_alphaoverarea),
                    as_float(f_norm_alpha),
                    as_float(f_norm_beta),
                    as_float(f_norm_K));
            }
            else
            {
                obj(x,
                    y,
                    as_float(f_norm_alphaoverarea),
                    as_float(f_norm_alpha),
                    as_float(f_norm_beta),
                    as_float(f_norm_K));
            }
        });
    }
    return (status);
}

miopenStatus_t LRNDescriptor::Backward(Handle& handle,
                                       const void* /*alpha*/,
                                       const TensorDescriptor& yDesc,
                                       ConstData_t y,
                                       const TensorDescriptor& dyDesc,
                                       ConstData_t dy,
                                       const TensorDescriptor& xDesc,
                                       ConstData_t x,
                                       const void* /*beta*/,
                                       const TensorDescriptor& dxDesc,
                                       Data_t dx,
                                       ConstData_t workSpace) const
{
    miopenStatus_t status = miopenStatusSuccess;
    mlo_construct_norm construct_params(0); // backward

    construct_params.setStream(&handle);
    int ndOut;
    int cdOut;
    int hdOut;
    int wdOut;
    int ndOutStride;
    int cdOutStride;
    int hdOutStride;
    int wdOutStride;

    std::tie(ndOut, cdOut, hdOut, wdOut)                         = tien<4>(dyDesc.GetLengths());
    std::tie(ndOutStride, cdOutStride, hdOutStride, wdOutStride) = tien<4>(dyDesc.GetStrides());

    construct_params.setTopDfDescFromMLDesc(dyDesc);

    int nOut;
    int cOut;
    int hOut;
    int wOut;
    int nOutStride;
    int cOutStride;
    int hOutStride;
    int wOutStride;

    std::tie(nOut, cOut, hOut, wOut)                         = tien<4>(yDesc.GetLengths());
    std::tie(nOutStride, cOutStride, hOutStride, wOutStride) = tien<4>(yDesc.GetStrides());

    construct_params.setTopDescFromMLDesc(yDesc);

    int ndIn;
    int cdIn;
    int hdIn;
    int wdIn;
    int ndInStride;
    int cdInStride;
    int hdInStride;
    int wdInStride;

    std::tie(ndIn, cdIn, hdIn, wdIn)                         = tien<4>(dxDesc.GetLengths());
    std::tie(ndInStride, cdInStride, hdInStride, wdInStride) = tien<4>(dxDesc.GetStrides());

    construct_params.setBotDfDescFromMLDesc(dxDesc);

    int nIn;
    int cIn;
    int hIn;
    int wIn;
    int nInStride;
    int cInStride;
    int hInStride;
    int wInStride;

    std::tie(nIn, cIn, hIn, wIn)                         = tien<4>(xDesc.GetLengths());
    std::tie(nInStride, cInStride, hInStride, wInStride) = tien<4>(xDesc.GetStrides());

    construct_params.setBotDescFromMLDesc(xDesc);

    int norm_reg = GetMode();

    int local_area = static_cast<int>(GetN());

    double lrn_alpha = GetAlpha();
    double lrn_beta  = GetBeta();
    double lrn_K     = GetK();

    construct_params.setNormDescr(norm_reg, local_area, lrn_alpha, lrn_beta, lrn_K);

    mloConstruct(construct_params);

    int norm_region;
    int local_ar;
    // whithin channel alphaoverarea is going to be culculate based on actual areal size (cut by
    // borders).
    double norm_alpha;
    double norm_beta;
    double norm_K;
    double norm_alphaoverarea;

    construct_params.getNormDescr(
        norm_region, local_ar, norm_alpha, norm_beta, norm_K, norm_alphaoverarea);
    auto f_norm_alpha = static_cast<float>(norm_alpha);
    auto f_norm_beta  = static_cast<float>(norm_beta);
    auto f_norm_ratio =
        static_cast<float>(2. * norm_alpha * norm_beta / static_cast<double>(local_ar));

    if(float_equal(norm_K, 0.0))
        MIOPEN_THROW("Expect non-zero bias/K");

    std::string algo_name = "miopenLRNBackward";
    std::string network_config =
        std::to_string(f_norm_alpha) + std::to_string(f_norm_beta) + std::to_string(norm_K) +
        std::to_string(norm_alphaoverarea) + std::to_string(local_ar) +
        std::to_string(norm_region) + std::to_string(f_norm_ratio) +
        std::to_string(xDesc.GetType()) + std::to_string(nInStride) + std::to_string(nOutStride) +
        std::to_string(nIn) + std::to_string(nOut) + std::to_string(nInStride) +
        std::to_string(nOutStride) + std::to_string(cIn) + std::to_string(cOut) +
        std::to_string(cInStride) + std::to_string(cOutStride) + std::to_string(hIn) +
        std::to_string(hOut);

    auto&& kernels = handle.GetKernels(algo_name, network_config);
    if(!kernels.empty())
    {
        visit_float(xDesc.GetType(), [&](auto as_float) {
            kernels.front()(y,
                            x,
                            dy,
                            workSpace,
                            dx,
                            as_float(f_norm_ratio),
                            as_float(f_norm_alpha),
                            as_float(f_norm_beta));
        });
    }
    else
    {
        const std::string program_name = construct_params.getKernelFile(); // CL kernel filename
        const std::string kernel_name  = construct_params.getKernelName(); // kernel name
        const std::string& compiler_parms =
            construct_params.getCompilerOptions(); // kernel parameters

        const std::vector<size_t>& vld = construct_params.getLocalWkSize();
        const std::vector<size_t>& vgd = construct_params.getGlobalWkSize();

        visit_float(xDesc.GetType(), [&](auto as_float) {
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_name, vld, vgd, compiler_parms)(
                y,
                x,
                dy,
                workSpace,
                dx,
                as_float(f_norm_ratio),
                as_float(f_norm_alpha),
                as_float(f_norm_beta));
        });
    }
    return (status);
}
} // namespace miopen
