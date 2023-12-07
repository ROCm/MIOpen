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
#include <miopen/groupnorm.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/norm/invoke_params.hpp>
#include <miopen/norm/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

miopenStatus_t GroupNormForward(Handle& handle,
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
                                int32_t num_groups,
                                float epsilon)
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

    if(xDesc.GetLengths()[1] % num_groups != 0)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor channel size must be divided by num_groups.");
    }

    bool is_all_packed = xDesc.IsPacked() && weightDesc.IsPacked() && biasDesc.IsPacked() &&
                         yDesc.IsPacked() && meanDesc.IsPacked() && rstdDesc.IsPacked();

    if(!is_all_packed)
    {
        MIOPEN_THROW(miopenStatusBadParm, "All tensors must be packed.");
    }

    auto dims         = xDesc.GetLengths();
    size_t numel      = xDesc.GetElementSize();
    size_t numel_per_channel = numel / dims[0] / dims[1];
    size_t num_channels      = dims[1];

    size_t outer_size = dims[0] * num_groups;
    // size_t inner_size = numel / outer_size;

    auto dtype = xDesc.GetType();

    const std::vector<size_t> vld{256, 1, 1};
    const std::vector<size_t> vgd{outer_size * vld[0], 1, 1};

    std::string algo_name = "GroupNormForward";
    std::string network_config = "gnfwd-dtype" + std::to_string(static_cast<int32_t>(dtype)) +
                                 "-g" + std::to_string(vgd[0]) + "-l" + std::to_string(vld[0]) +
                                 "-num_groups" + std::to_string(num_groups) + "-num_channels" +
                                 std::to_string(num_channels) + "-numel_per_channel" +
                                 std::to_string(numel_per_channel) + "-mode" +
                                 std::to_string(static_cast<int32_t>(mode)) + "-eps" +
                                 std::to_string(static_cast<float>(epsilon));

    std::string program_name = "MIOpenGroupNorm.cpp";
    std::string kernel_name  = "GroupNormFwdContiguous";

    // compile parameters
    std::string parms =
        " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int32_t>(dtype == miopenHalf)) +
        " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int32_t>(dtype == miopenFloat)) +
        " -DMIOPEN_USE_FP64=" + std::to_string(static_cast<int32_t>(dtype == miopenDouble)) +
        " -DMIOPEN_USE_BFP16=" + std::to_string(static_cast<int32_t>(dtype == miopenBFloat16));

    parms += " -DMIOPEN_BETA_API=1";
    parms += " -DLOCAL_SIZE=" + std::to_string(256);

    auto&& kernels = handle.GetKernels(algo_name, network_config);
    if(!kernels.empty())
    {
        kernels.front()(x,
                        y,
                        weight,
                        bias,
                        mean,
                        rstd,
                        epsilon,
                        static_cast<size_t>(num_groups),
                        num_channels,
                        numel_per_channel,
                        mode);
    }
    else
    {
        handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
            x,
            y,
            weight,
            bias,
            mean,
            rstd,
            epsilon,
            static_cast<size_t>(num_groups),
            num_channels,
            numel_per_channel,
            mode);
    }

    return miopenStatusSuccess;
}

} // namespace miopen
