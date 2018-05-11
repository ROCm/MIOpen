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
#include <miopen/batch_norm_activ.hpp>
#include <miopen/util.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/logger.hpp>
#include <chrono>

namespace miopen {

//============ BEGIN FORWARD INFERENCE ===============
void BatchNormActivForwardInference(Handle& handle,
                                    miopenBatchNormMode_t bn_mode,
                                    const void* alpha,
                                    const void* beta,
                                    const TensorDescriptor& xDesc,
                                    ConstData_t x,
                                    const TensorDescriptor& yDesc,
                                    Data_t y,
                                    const TensorDescriptor& bnScaleBiasMeanVarDesc,
                                    ConstData_t bnScale,
                                    ConstData_t bnBias,
                                    ConstData_t estimatedMean,
                                    ConstData_t estimatedVariance,
                                    double epsilon,
                                    miopenActivationMode_t activ_mode,
                                    double activ_alpha,
                                    double activ_beta,
                                    double activ_gama)
{

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, bnScale);
        miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, bnBias);
        miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, estimatedMean);
        miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, estimatedVariance);
    }

    assert(estimatedMean != nullptr && estimatedVariance != nullptr);

    if(x == nullptr || y == nullptr || bnScale == nullptr || bnBias == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetSize() != yDesc.GetSize() || xDesc.GetSize() != bnScaleBiasMeanVarDesc.GetSize())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetType() != yDesc.GetType() || xDesc.GetType() != bnScaleBiasMeanVarDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetSize() < 3)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_LOG_E("Only alpha=1 and beta=0 is supported");
        MIOPEN_THROW(miopenStatusBadParm);
    }

    bool bfp16parm = false;
    bool bfp32parm = true;
    if(xDesc.GetType() == miopenHalf)
    {
        bfp16parm = true;
        bfp32parm = false;
    }

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(xDesc.GetLengths());

    unsigned int in_nstride = c * h * w;
    unsigned int in_cstride = h * w;

    // size_t read_len  = n * h * w;
    size_t read_len  = h * w;
    size_t read_unit = (in_cstride % 4 == 0) ? 4 : (in_cstride % 2 == 0) ? 2 : 1;
    // size_t read_unit = 1;
    size_t MAP_RD = read_len / read_unit;
    const std::string READ_TYPE =
        (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(read_unit);

    size_t xlocalsize = 256;
    size_t ylocalsize = 1;
    size_t zlocalsize = 1;

    std::vector<size_t> vld;
    std::vector<size_t> vgd;

    size_t xgridsize = size_t(MAP_RD);
    size_t ygridsize = size_t(c);
    size_t zgridsize = 1;

    std::string algo_name = "miopenBatchNormalizationActiveForwardInference";
    std::string network_config =
        std::to_string(n) + std::to_string(in_cstride) + std::to_string(in_nstride) + "dims" +
        std::to_string(xgridsize) + std::to_string(ygridsize) + std::to_string(xlocalsize) +
        std::to_string(ylocalsize) + +"type" + std::to_string(static_cast<int>(bfp16parm)) +
        std::to_string(static_cast<int>(bfp32parm)) + "mode" + std::to_string(bn_mode) +
        std::to_string(activ_mode);

    auto&& kernels = handle.GetKernels(algo_name, network_config);
    if(!kernels.empty())
    {
        auto kernel = kernels.front();
        kernel(x, y, estimatedMean, estimatedVariance, bnScale, bnBias, epsilon);
    }
    else
    {

        std::string program_name = "MIOpenBatchNormActivFwdInfer"; // build this up
        std::string kernel_name  = "MIOpenBatchNormActivFwdInfer";
        if(bn_mode == miopenBNSpatial)
        { // SPATIAL kernels
            program_name += "Spatial.cl";
            kernel_name += "SpatialEst";
        }
        else
        { // PER ACTIVATION
            program_name += "PerAct.cl";
            kernel_name += "PerActEst";
        }

        std::string parms =
            " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int>(bfp16parm)) +
            " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int>(bfp32parm)) + " -DMIO_BN_N=" +
            std::to_string(n) + " -DMIO_BN_HW=" + std::to_string(in_cstride) + " -DMIO_BN_CHW=" +
            std::to_string(in_nstride) + " -DMIO_BN_GRP0=" + std::to_string(xlocalsize) +
            " -DMIOPEN_READ_UNIT=" + std::to_string(read_unit) + " -DMIO_BN_HW_RD=" +
            std::to_string(in_cstride / read_unit) + " -DMIOPEN_READ_TYPE=" + READ_TYPE +
            " -DMIO_BN_GRP1=" + std::to_string(ylocalsize) + " -DMIO_BN_GRP2=" +
            std::to_string(zlocalsize) + " -DMIOPEN_NRN_OP_ID=" + std::to_string(activ_mode) +
            " -DMIO_BN_N=" + std::to_string(n);

        vld.push_back(xlocalsize);
        vld.push_back(ylocalsize);
        vld.push_back(zlocalsize);
        vgd.push_back(xgridsize);
        vgd.push_back(ygridsize);
        vgd.push_back(zgridsize);

        visit_float(xDesc.GetType(), [&](auto as_float) {

            auto f_activ_alpha = as_float(activ_alpha);
            auto f_activ_beta  = as_float(activ_beta);
            auto f_activ_gama  = as_float(activ_gama);

            MIOPEN_LOG_I2(kernel_name << ":: " << parms);

            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                x,
                y,
                estimatedMean,
                estimatedVariance,
                bnScale,
                bnBias,
                epsilon,
                f_activ_gama,
                f_activ_alpha,
                f_activ_beta);
        });
    }

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsOutput(handle, yDesc, y);
    }
}
//================= END FORWARD INFERENCE ====================

} // namespace miopen
