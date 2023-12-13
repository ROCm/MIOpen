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
#include <miopen/cat.hpp>
#include <algorithm>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/tensor.hpp>

#define LOCAL_SIZE 256
#define MAX_TENSOR_X_COUNT 8

namespace miopen {

inline size_t AlignUp(size_t num, size_t align) { return (num + align - 1) / align * align; }

miopenStatus_t CatForward(const Handle& handle,
                          const int32_t xCount,
                          const TensorDescriptor* const* xDescs,
                          const ConstData_t* xs,
                          const TensorDescriptor& yDesc,
                          Data_t y,
                          int32_t dim)
{
    if(xCount > MAX_TENSOR_X_COUNT)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Exceeded the number of tensors.");
    }

    if(y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    const auto dtype   = yDesc.GetType();
    auto ydims         = yDesc.GetLengths();
    ydims[dim]         = 0;
    bool is_all_packed = yDesc.IsPacked();
    std::vector<size_t> x_dim_sizes;
    size_t x_dim_size_max = 0;

    for(int i = 0; i < xCount; i++)
    {
        auto x      = xs[i];
        auto xDesc  = xDescs[i];
        auto& xdims = xDesc->GetLengths();

        if(x == nullptr)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
        }

        if(xDesc->GetType() != dtype)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
        }

        if(ydims.size() != xdims.size())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
        }

        for(int j = 0; j < ydims.size(); j++)
        {
            if((j != dim) && (ydims[j] != xdims[j]))
            {
                MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
            }
        }

        x_dim_size_max = std::max(x_dim_size_max, xdims[dim]);
        x_dim_sizes.push_back(xdims[dim]);
        ydims[dim] += xdims[dim];
        is_all_packed &= xDesc->IsPacked();
    }

    if(ydims[dim] != yDesc.GetLengths()[dim])
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
    }

    if(!is_all_packed)
    {
        MIOPEN_THROW(miopenStatusBadParm, "All tensor is not packed.");
    }

    if(yDesc.GetElementSize() == 0)
    {
        return miopenStatusSuccess;
    }

    size_t outer_size = 1;

    for(int i = 0; i < dim; i++)
    {
        outer_size *= ydims[i];
    }

    auto stride         = yDesc.GetStrides()[dim];
    auto y_dim_size     = ydims[dim];
    int32_t fusion_size = 2;

    while(fusion_size < xCount)
    {
        fusion_size *= 2;
    }

    for(int i = xCount; i < fusion_size; i++)
    {
        x_dim_sizes.push_back(0);
    }

    auto numCu = handle.GetMaxComputeUnits();

    std::vector<size_t> vld{1, 1, 1};
    std::vector<size_t> vgd{1, 1, 1};

    vld[0] = std::min(static_cast<int>(x_dim_size_max * stride), LOCAL_SIZE);
    vld[1] = std::max(static_cast<int>(LOCAL_SIZE / vld[0]), 1);

    vgd[1] = AlignUp(outer_size, vld[1]);
    vgd[0] = std::max(static_cast<int>(numCu * 8 / (vgd[1] / vld[1])), 1) * vld[0];
    vgd[0] = std::min(vgd[0], AlignUp(x_dim_size_max * stride, vld[0]));

    std::string algo_name      = "CatForward";
    std::string network_config = "cat" + std::to_string(static_cast<int32_t>(fusion_size)) +
                                 "fwd-dtype" + std::to_string(static_cast<int32_t>(dtype)) + "g" +
                                 std::to_string(vgd[0]) + "," + std::to_string(vgd[1]) + "l" +
                                 std::to_string(vld[0]) + "," + std::to_string(vld[1]) + "dim" +
                                 std::to_string(dim) + "outer_size" + std::to_string(outer_size);

    std::string parms =
        " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int32_t>(dtype == miopenHalf)) +
        " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int32_t>(dtype == miopenFloat)) +
        " -DMIOPEN_USE_FP64=" + std::to_string(static_cast<int32_t>(dtype == miopenDouble)) +
        " -DMIOPEN_USE_BFP16=" + std::to_string(static_cast<int32_t>(dtype == miopenBFloat16));

    parms += " -DLOCAL_SIZE=" + std::to_string(LOCAL_SIZE);

    auto&& kernels = handle.GetKernels(algo_name, network_config);
    if(!kernels.empty())
    {
        switch(fusion_size)
        {
        case 2:
            kernels.front()(xs[0],
                            xs[1],
                            y,
                            x_dim_sizes[0],
                            x_dim_sizes[1],
                            dim,
                            outer_size,
                            stride,
                            y_dim_size);
            break;
        case 4:
            kernels.front()(xs[0],
                            xs[1],
                            xs[2],
                            xs[3],
                            y,
                            x_dim_sizes[0],
                            x_dim_sizes[1],
                            x_dim_sizes[2],
                            x_dim_sizes[3],
                            dim,
                            outer_size,
                            stride,
                            y_dim_size);
            break;
        case 8:
            kernels.front()(xs[0],
                            xs[1],
                            xs[2],
                            xs[3],
                            xs[4],
                            xs[5],
                            xs[6],
                            xs[7],
                            y,
                            x_dim_sizes[0],
                            x_dim_sizes[1],
                            x_dim_sizes[2],
                            x_dim_sizes[3],
                            x_dim_sizes[4],
                            x_dim_sizes[5],
                            x_dim_sizes[6],
                            x_dim_sizes[7],
                            dim,
                            outer_size,
                            stride,
                            y_dim_size);
            break;
        default: break;
        }
    }
    else
    {
        std::string program_name = "MIOpenCat.cpp";
        std::string kernel_name;
        switch(fusion_size)
        {
        case 2:
            kernel_name = "Cat2FwdPacked";
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                xs[0],
                xs[1],
                y,
                x_dim_sizes[0],
                x_dim_sizes[1],
                dim,
                outer_size,
                stride,
                y_dim_size);
            break;
        case 4:
            kernel_name = "Cat4FwdPacked";
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                xs[0],
                xs[1],
                xs[2],
                xs[3],
                y,
                x_dim_sizes[0],
                x_dim_sizes[1],
                x_dim_sizes[2],
                x_dim_sizes[3],
                dim,
                outer_size,
                stride,
                y_dim_size);
            break;
        case 8:
            kernel_name = "Cat8FwdPacked";
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                xs[0],
                xs[1],
                xs[2],
                xs[3],
                xs[4],
                xs[5],
                xs[6],
                xs[7],
                y,
                x_dim_sizes[0],
                x_dim_sizes[1],
                x_dim_sizes[2],
                x_dim_sizes[3],
                x_dim_sizes[4],
                x_dim_sizes[5],
                x_dim_sizes[6],
                x_dim_sizes[7],
                dim,
                outer_size,
                stride,
                y_dim_size);
            break;
        default: break;
        }
    }

    return miopenStatusSuccess;
}

} // namespace miopen
