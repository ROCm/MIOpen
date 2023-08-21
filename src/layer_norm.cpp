/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/layernorm.hpp>
#ifdef MIOPEN_BETA_API
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/tensor.hpp>

#define LOCAL_SIZE 256

namespace miopen {

miopenStatus_t LayerNormForward(const Handle& handle,
                                const TensorDescriptor& xDesc,
                                ConstData_t x,
                                const TensorDescriptor& weightDesc,
                                ConstData_t weight,
                                const TensorDescriptor& biasDesc,
                                ConstData_t bias,
                                const TensorDescriptor& yDesc,
                                Data_t y,
                                const TensorDescriptor& meanDesc,
                                Data_t mean,
                                const TensorDescriptor& rstdDesc,
                                Data_t rstd,
                                miopenLayerNormMode_t mode,
                                const float epsilon,
                                const int normalized_dim)
{
    if(x == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    if(xDesc.GetType() != yDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
    }

    if(xDesc.GetLengths() != yDesc.GetLengths())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
    }

    bool is_all_packed = xDesc.IsPacked() && weightDesc.IsPacked() && biasDesc.IsPacked() &&
                         yDesc.IsPacked() && meanDesc.IsPacked() && rstdDesc.IsPacked();

    auto dims         = xDesc.GetLengths();
    size_t grid_size  = 1;
    size_t outer_size = 1;
    size_t inner_size = 1;
    size_t i          = 0;
    for(; i < normalized_dim; i++)
    {
        outer_size *= dims[i];
        grid_size *= dims[i];
    }

    for(; i < dims.size(); i++)
    {
        inner_size *= dims[i];
        grid_size *= dims[i];
    }

    auto dtype = xDesc.GetType();

    const std::vector<size_t> vld{LOCAL_SIZE, 1, 1};
    const std::vector<size_t> vgd{outer_size * vld[0], 1, 1};

    std::string algo_name = "LayerNormForward";
    std::string network_config =
        "lnfwd-dtype" + std::to_string(static_cast<int>(dtype)) + "g" + std::to_string(vgd[0]) +
        "l" + std::to_string(vld[0]) + "normalized_dim" + std::to_string(normalized_dim) + "grid" +
        std::to_string(grid_size) + "outer_size" + std::to_string(outer_size) + "inner_size" +
        std::to_string(inner_size) + "xpk" + std::to_string(static_cast<int>(xDesc.IsPacked()));
    if(weight)
        network_config += "weightpk" + std::to_string(static_cast<int>(weightDesc.IsPacked()));
    if(bias)
        network_config += "biaspk" + std::to_string(static_cast<int>(biasDesc.IsPacked()));
    network_config += "ypk" + std::to_string(static_cast<int>(yDesc.IsPacked()));
    if(mean)
        network_config += "meanpk" + std::to_string(static_cast<int>(meanDesc.IsPacked()));
    if(rstd)
        network_config += "rstdpk" + std::to_string(static_cast<int>(rstdDesc.IsPacked()));
    network_config += "mode" + std::to_string(static_cast<int>(mode)) + "eps" +
                      std::to_string(static_cast<float>(epsilon));

    std::string program_name = "MIOpenLayerNorm.cpp";
    std::string kernel_name  = "LayernormFwdContiguous";

    // compile parameters
    std::string parms =
        " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int>(dtype == miopenHalf)) +
        " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int>(dtype == miopenFloat)) +
        " -DMIOPEN_USE_FP64=" + std::to_string(static_cast<int>(dtype == miopenDouble)) +
        " -DMIOPEN_USE_BF16=" + std::to_string(static_cast<int>(dtype == miopenBFloat16));

    if(mode == MIOPEN_ELEMENTWISE_AFFINE)
        parms += " -DUSE_MIOPEN_ELEMENTWISE_AFFINE=1";
    else
        parms += " -DUSE_MIOPEN_WEIGHT_BIAS=1";

    parms += " -DRUN_FORWARD=1";
    parms +=
        " -DIS_INPUT_PACKED=" +
        std::to_string(static_cast<int>(xDesc.IsPacked() && (!weight || weightDesc.IsPacked()) &&
                                        (!bias || biasDesc.IsPacked()))) +
        " -DIS_OUTPUT_PACKED=" +
        std::to_string(static_cast<int>(yDesc.IsPacked() && (!mean || meanDesc.IsPacked()) &&
                                        (!rstd || rstdDesc.IsPacked())));

    auto&& kernels = handle.GetKernels(algo_name, network_config);
    if(!kernels.empty())
    {
        kernels.front()(
            x, y, weight, bias, mean, rstd, epsilon, inner_size, mode);
    }
    else
    {
        handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
            x, y, weight, bias, mean, rstd, epsilon, inner_size, mode);
    }

    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, yDesc, y);
    }
    return miopenStatusSuccess;
}

miopenStatus_t LayerNormBackward(const Handle& handle,
                                 const TensorDescriptor& xDesc,
                                 ConstData_t x,
                                 const TensorDescriptor& dyDesc,
                                 ConstData_t dy,
                                 const TensorDescriptor& weightDesc,
                                 ConstData_t weight,
                                 const TensorDescriptor& meanDesc,
                                 ConstData_t mean,
                                 const TensorDescriptor& rstdDesc,
                                 ConstData_t rstd,
                                 const TensorDescriptor& dxDesc,
                                 Data_t dx,
                                 const TensorDescriptor& dwDesc,
                                 Data_t dw,
                                 const TensorDescriptor& dbDesc,
                                 Data_t db,
                                 miopenLayerNormMode_t mode,
                                 const int normalized_dim)
{
    if(dx == nullptr || x == nullptr || mean == nullptr || rstd == nullptr || dy == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    if(dxDesc.GetType() != dyDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
    }

    if(dxDesc.GetLengths() != dyDesc.GetLengths())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
    }

    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsInput(handle, xDesc, x);
    }

    bool is_all_packed = xDesc.IsPacked() && dyDesc.IsPacked() && weightDesc.IsPacked() &&
                         meanDesc.IsPacked() && rstdDesc.IsPacked() && dxDesc.IsPacked() &&
                         dwDesc.IsPacked() && dbDesc.IsPacked();

    auto dims         = xDesc.GetLengths();
    size_t grid_size  = 1;
    size_t outer_size = 1;
    size_t inner_size = 1;
    size_t i          = 0;
    for(; i < normalized_dim; i++)
    {
        outer_size *= dims[i];
        grid_size *= dims[i];
    }

    for(; i < dims.size(); i++)
    {
        inner_size *= dims[i];
        grid_size *= dims[i];
    }

    auto dtype = xDesc.GetType();

    const std::vector<size_t> vld{LOCAL_SIZE, 1, 1};
    const std::vector<size_t> vgd{outer_size * vld[0], 1, 1};

    std::string algo_name = "LayerNormBackward";
    std::string network_config =
        "lnbwd-dtype" + std::to_string(static_cast<int>(dtype)) + "g" + std::to_string(vgd[0]) +
        "l" + std::to_string(vld[0]) + "normalized_dim" + std::to_string(normalized_dim) + "grid" +
        std::to_string(grid_size) + "outer_size" + std::to_string(outer_size) + "inner_size" +
        std::to_string(inner_size) + "xpk" + std::to_string(static_cast<int>(xDesc.IsPacked())) +
        "dypk" + std::to_string(static_cast<int>(dyDesc.IsPacked()));
    if(weight)
        network_config += "weightpk" + std::to_string(static_cast<int>(weightDesc.IsPacked()));
    if(mean)
        network_config += "meanpk" + std::to_string(static_cast<int>(meanDesc.IsPacked()));
    if(rstd)
        network_config += "rstdpk" + std::to_string(static_cast<int>(rstdDesc.IsPacked()));
    network_config += "dxpk" + std::to_string(static_cast<int>(dxDesc.IsPacked())) + "mode" +
                      std::to_string(static_cast<int>(mode));

    std::string program_name = "MIOpenLayerNorm.cpp";
    std::string kernel_name  = "LayerNormBwdContiguous";

    // compile parameters
    std::string parms =
        " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int>(dtype == miopenHalf)) +
        " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int>(dtype == miopenFloat)) +
        " -DMIOPEN_USE_FP64=" + std::to_string(static_cast<int>(dtype == miopenDouble)) +
        " -DMIOPEN_USE_BF16=" + std::to_string(static_cast<int>(dtype == miopenBFloat16));

    if(mode == MIOPEN_ELEMENTWISE_AFFINE)
        parms += " -DUSE_MIOPEN_ELEMENTWISE_AFFINE=1";
    else
        parms += " -DUSE_MIOPEN_WEIGHT_BIAS=1";

    parms += " -DRUN_FORWARD=0";
    parms += " -DIS_INPUT_PACKED=" +
             std::to_string(static_cast<int>(xDesc.IsPacked() && dyDesc.IsPacked() &&
                                             (!weight || weightDesc.IsPacked()) &&
                                             meanDesc.IsPacked() && rstdDesc.IsPacked())) +
             " -DIS_OUTPUT_PACKED=" + std::to_string(static_cast<int>(dxDesc.IsPacked()));

    auto&& kernels = handle.GetKernels(algo_name, network_config);
    if(!kernels.empty())
    {
        kernels.front()(x, dy, weight, mean, rstd, dx, inner_size, mode);
    }
    else
    {
        handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
            x, dy, weight, mean, rstd, dx, inner_size, mode);
    }

    if(dw && db)
    {
        const std::vector<size_t> vld2{LOCAL_SIZE, 1, 1};
        const std::vector<size_t> vgd2{inner_size * vld[0], 1, 1};

        std::string algo_name2 = "LayerNormWeightBiasBackward";
        std::string network_config2 =
            "lnbwd-dtype" + std::to_string(static_cast<int>(dtype)) + "g" +
            std::to_string(vgd2[0]) + "l" + std::to_string(vld2[0]) + "normalized_dim" +
            std::to_string(normalized_dim) + "grid" + std::to_string(grid_size) + "outer_size" +
            std::to_string(outer_size) + "inner_size" + std::to_string(inner_size) + "xpk" +
            std::to_string(static_cast<int>(xDesc.IsPacked())) + "dypk" +
            std::to_string(static_cast<int>(dyDesc.IsPacked()));
        if(mean)
            network_config2 += "meanpk" + std::to_string(static_cast<int>(meanDesc.IsPacked()));
        if(rstd)
            network_config2 += "rstdpk" + std::to_string(static_cast<int>(rstdDesc.IsPacked()));
        network_config2 += "dwpk" + std::to_string(static_cast<int>(dwDesc.IsPacked())) + "dbpk" +
                           std::to_string(static_cast<int>(dbDesc.IsPacked())) + "mode" +
                           std::to_string(static_cast<int>(mode));

        std::string program_name2 = "MIOpenLayerNorm.cpp";
        std::string kernel_name2  = "LayernormBwdWeightBiasContiguous";

        // compile parameters
        std::string parms2 =
            " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int>(dtype == miopenHalf)) +
            " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int>(dtype == miopenFloat)) +
            " -DMIOPEN_USE_FP64=" + std::to_string(static_cast<int>(dtype == miopenDouble)) +
            " -DMIOPEN_USE_BF16=" + std::to_string(static_cast<int>(dtype == miopenBFloat16));

        if(mode == MIOPEN_ELEMENTWISE_AFFINE)
            parms2 += " -DUSE_MIOPEN_ELEMENTWISE_AFFINE=1";
        else
            parms2 += " -DUSE_MIOPEN_WEIGHT_BIAS=1";

        parms2 += " -DRUN_FORWARD=0";
        parms2 += " -DIS_INPUT_PACKED=" +
                  std::to_string(static_cast<int>(xDesc.IsPacked() && meanDesc.IsPacked() &&
                                                  rstdDesc.IsPacked())) +
                  " -DIS_OUTPUT_PACKED=" +
                  std::to_string(static_cast<int>(dwDesc.IsPacked() && dbDesc.IsPacked()));
        auto&& kernels2 = handle.GetKernels(algo_name2, network_config2);
        if(!kernels2.empty())
        {
            kernels2.front()(x, dy, mean, rstd, dw, db, outer_size, inner_size);
        }
        else
        {
            handle.AddKernel(
                algo_name2, network_config2, program_name2, kernel_name2, vld2, vgd2, parms2)(
                x, dy, mean, rstd, dw, db, outer_size, inner_size);
        }
    }

    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, dxDesc, dx);
    }

    return miopenStatusSuccess;
}

} // namespace miopen
#endif
