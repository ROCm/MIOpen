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
                                float epsilon,
                                int32_t normalized_dim)
{
    if(x == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "LayerNormForward: Null pointer for tensor.");
    }

    if(xDesc.GetType() != yDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "LayerNormForward: Tensor types do not match.");
    }

    if(xDesc.GetLengths() != yDesc.GetLengths())
    {
        MIOPEN_THROW(miopenStatusBadParm, "LayerNormForward: Tensor dimension lengths do not match.");
    }

    bool is_all_packed = xDesc.IsPacked() && weightDesc.IsPacked() && biasDesc.IsPacked() &&
                         yDesc.IsPacked() && meanDesc.IsPacked() && rstdDesc.IsPacked();

    if(!is_all_packed)
    {
        MIOPEN_THROW(miopenStatusBadParm, "LayerNormForward: Unpacked tensors not supported.");
    }

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
        "lnfwd-dtype" + std::to_string(static_cast<int32_t>(dtype)) + "g" + std::to_string(vgd[0]) +
        "l" + std::to_string(vld[0]) + "normalized_dim" + std::to_string(normalized_dim) + "grid" +
        std::to_string(grid_size) + "outer_size" + std::to_string(outer_size) + "inner_size" +
        std::to_string(inner_size) + "mode" + std::to_string(static_cast<int32_t>(mode)) + "eps" +
        std::to_string(static_cast<float>(epsilon));

    std::string program_name = "MIOpenLayerNorm.cpp";
    std::string kernel_name  = "LayernormFwdContiguous";

    // compile parameters
    std::string parms =
        " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int32_t>(dtype == miopenHalf)) +
        " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int32_t>(dtype == miopenFloat)) +
        " -DMIOPEN_USE_FP64=" + std::to_string(static_cast<int32_t>(dtype == miopenDouble)) +
        " -DMIOPEN_USE_BFP16=" + std::to_string(static_cast<int32_t>(dtype == miopenBFloat16));

    parms += " -DLOCAL_SIZE=" + std::to_string(LOCAL_SIZE);

    auto&& kernels = handle.GetKernels(algo_name, network_config);
    if(!kernels.empty())
    {
        kernels.front()(x, y, weight, bias, mean, rstd, epsilon, inner_size, mode);
    }
    else
    {
        handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
            x, y, weight, bias, mean, rstd, epsilon, inner_size, mode);
    }

    return miopenStatusSuccess;
}

} // namespace miopen
#endif
