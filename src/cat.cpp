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
#ifdef MIOPEN_BETA_API
#include <algorithm>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/tensor.hpp>

#define LOCAL_SIZE 256

namespace miopen {

inline size_t DivCeil(size_t dividend, size_t divisor)
{
    return (dividend + divisor - 1) / divisor;
}

inline size_t AlignUp(size_t num, size_t align) { return DivCeil(num, align) * align; }

miopenStatus_t CatForward(const Handle& handle,
                          const std::vector<TensorDescriptor>& inputDescs,
                          std::vector<ConstData_t> inputs,
                          const TensorDescriptor& outputDesc,
                          Data_t output,
                          int32_t dim)
{
    if(inputs.size() > 8)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Exceeded the number of input tensors.");
    }

    if(inputs.size() != inputDescs.size())
    {
        MIOPEN_THROW(miopenStatusBadParm, "The number of input tensors does not match.");
    }

    if(output == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    for(auto input : inputs)
    {
        if(input == nullptr)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
        }
    }

    const auto dtype   = outputDesc.GetType();
    auto dims          = outputDesc.GetLengths();
    dims[dim]          = 0;
    bool is_all_packed = outputDesc.IsPacked();
    auto it1           = inputDescs.begin();
    auto it2           = inputs.begin();
    std::vector<size_t> input_dim_sizes;
    size_t input_dim_size_max = 0;

    while(it1 != inputDescs.end())
    {
        auto& in_dims = it1->GetLengths();

        if(it1->GetType() != dtype)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
        }

        if(dims.size() != in_dims.size())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
        }

        for(int i = 0; i < dims.size(); i++)
        {
            if((i != dim) && (dims[i] != in_dims[i]))
            {
                MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
            }
        }

        if(in_dims[dim] == 0)
        {
            it2 = inputs.erase(it2);
        }
        else
        {
            input_dim_size_max = std::max(input_dim_size_max, in_dims[dim]);
            input_dim_sizes.push_back(in_dims[dim]);
            dims[dim] += in_dims[dim];
            is_all_packed &= it1->IsPacked();
            it2++;
        }
        it1++;
    }

    if(dims[dim] != outputDesc.GetLengths()[dim])
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
    }

    if(!is_all_packed)
    {
        MIOPEN_THROW(miopenStatusBadParm, "All tensor is not packed.");
    }

    if(outputDesc.GetElementSize() == 0)
    {
        return miopenStatusSuccess;
    }

    size_t outer_size = 1;

    for(int i = 0; i < dim; i++)
    {
        outer_size *= dims[i];
    }

    auto ninputs         = inputs.size();
    auto stride          = outputDesc.GetStrides()[dim];
    auto output_dim_size = dims[dim];
    int32_t fusion_size  = 2;

    while(fusion_size < ninputs)
    {
        fusion_size *= 2;
    }

    for(int i = ninputs; i < fusion_size; i++)
    {
        input_dim_sizes.push_back(0);
    }

    // 120 for mi120, 104 for mi210,
    auto numCu = handle.GetMaxComputeUnits();

    std::vector<size_t> vld{1, 1, 1};
    std::vector<size_t> vgd{1, 1, 1};

    // don't need thread more than input dim size max
    vld[0] = std::min(static_cast<int>(input_dim_size_max * stride), LOCAL_SIZE);
    vld[1] = std::max(static_cast<int>(LOCAL_SIZE / vld[0]), 1);

    vgd[1] = AlignUp(outer_size, vld[1]); // outer_size
    // set workgroup num as number of compute unit * 8
    // 8 is hueristic number
    vgd[0] = std::max(static_cast<int>(numCu * 8 / (vgd[1] / vld[1])), 1) * vld[0];
    vgd[0] = std::min(vgd[0], AlignUp(input_dim_size_max * stride, vld[0]));

    std::string algo_name      = "CatForward";
    std::string network_config = "cat" + std::to_string(static_cast<int32_t>(fusion_size)) +
                                 "fwd-dtype" + std::to_string(static_cast<int32_t>(dtype)) + "g" +
                                 std::to_string(vgd[0]) + "," + std::to_string(vgd[1]) + "l" +
                                 std::to_string(vld[0]) + "," + std::to_string(vld[1]) + "dim" +
                                 std::to_string(dim) + "outer_size" + std::to_string(outer_size);

    // compile parameters
    std::string parms =
        " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int32_t>(dtype == miopenHalf)) +
        " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int32_t>(dtype == miopenFloat)) +
        " -DMIOPEN_USE_FP64=" + std::to_string(static_cast<int32_t>(dtype == miopenDouble)) +
        " -DMIOPEN_USE_BFP16=" + std::to_string(static_cast<int32_t>(dtype == miopenBFloat16));

    parms += " -DMIOPEN_BETA_API=1";
    parms += " -DLOCAL_SIZE=" + std::to_string(LOCAL_SIZE);

    auto&& kernels = handle.GetKernels(algo_name, network_config);
    if(!kernels.empty())
    {
        switch(fusion_size)
        {
        case 2:
            kernels.front()(inputs[0],
                            inputs[1],
                            output,
                            input_dim_sizes[0],
                            input_dim_sizes[1],
                            dim,
                            outer_size,
                            stride,
                            output_dim_size);
            break;
        case 4:
            kernels.front()(inputs[0],
                            inputs[1],
                            inputs[2],
                            inputs[3],
                            output,
                            input_dim_sizes[0],
                            input_dim_sizes[1],
                            input_dim_sizes[2],
                            input_dim_sizes[3],
                            dim,
                            outer_size,
                            stride,
                            output_dim_size);
            break;
        case 8:
            kernels.front()(inputs[0],
                            inputs[1],
                            inputs[2],
                            inputs[3],
                            inputs[4],
                            inputs[5],
                            inputs[6],
                            inputs[7],
                            output,
                            input_dim_sizes[0],
                            input_dim_sizes[1],
                            input_dim_sizes[2],
                            input_dim_sizes[3],
                            input_dim_sizes[4],
                            input_dim_sizes[5],
                            input_dim_sizes[6],
                            input_dim_sizes[7],
                            dim,
                            outer_size,
                            stride,
                            output_dim_size);
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
                inputs[0],
                inputs[1],
                output,
                input_dim_sizes[0],
                input_dim_sizes[1],
                dim,
                outer_size,
                stride,
                output_dim_size);
            break;
        case 4:
            kernel_name = "Cat4FwdPacked";
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                output,
                input_dim_sizes[0],
                input_dim_sizes[1],
                input_dim_sizes[2],
                input_dim_sizes[3],
                dim,
                outer_size,
                stride,
                output_dim_size);
            break;
        case 8:
            kernel_name = "Cat8FwdPacked";
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                inputs[7],
                output,
                input_dim_sizes[0],
                input_dim_sizes[1],
                input_dim_sizes[2],
                input_dim_sizes[3],
                input_dim_sizes[4],
                input_dim_sizes[5],
                input_dim_sizes[6],
                input_dim_sizes[7],
                dim,
                outer_size,
                stride,
                output_dim_size);
            break;
        default: break;
        }
    }

    return miopenStatusSuccess;
}

} // namespace miopen
#endif
