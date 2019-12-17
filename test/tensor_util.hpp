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

#ifndef GUARD_TENSOR_UTIL_HPP
#define GUARD_TENSOR_UTIL_HPP

#include <iostream>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <utility>
#include <cstdlib>
#include "tensor_holder.hpp"

// loop over sub-tensor, and operate on each data
template <typename T, template <typename> class data_operator_t>
void operate_over_subtensor(const data_operator_t<T>& r_data_operator,
                            tensor<T>& rSuperTensor,
                            const miopen::TensorDescriptor& rSubDesc,
                            const int offset)
{
    operate_over_subtensor_impl(r_data_operator, rSuperTensor, rSubDesc, 0, offset);
}

// loop over part of sub-tensor (dimensions lower than "current_dim"), and operate on
// each data
template <typename T, template <typename> class data_operator_t>
void operate_over_subtensor_impl(const data_operator_t<T>& r_data_operator,
                                 tensor<T>& rSuperTensor,
                                 const miopen::TensorDescriptor& rSubDesc,
                                 const uint current_dim,
                                 const int offset)
{
    auto max_dim        = static_cast<int>(rSubDesc.GetLengths().size() - 1);
    auto current_stride = static_cast<int>(rSubDesc.GetStrides()[current_dim]);

    int index = offset;

    for(int i = 0; i < rSubDesc.GetLengths()[current_dim]; ++i)
    {
        if(current_dim == max_dim)
            r_data_operator(rSuperTensor[index]);
        else
            operate_over_subtensor_impl<T, data_operator_t>(
                r_data_operator, rSuperTensor, rSubDesc, current_dim + 1, index);

        index += current_stride;
    }
}

template <typename T>
void output_tensor_to_csv(const tensor<T>& x, std::string filename)
{
    int dim = x.desc.GetSize();
    std::vector<int> index(dim);

    std::ofstream file;

    file.open(filename);

    for(int j = 0; j < dim; ++j)
        file << "d" << j << ", ";
    file << "x" << std::endl;

    for(int i = 0; i < x.data.size(); ++i)
    {
        int is = i;
        for(int j = 0; j < dim; ++j)
        {
            index[j] = is / x.desc.GetStrides()[j];
            is -= index[j] * x.desc.GetStrides()[j];
        }

        for(int j = 0; j < dim; ++j)
        {
            file << index[j] << ", ";
        }
        file << x[i] << std::endl;
    }

    file.close();
}

template <typename T>
void output_tensor_to_bin(const char* fileName, T* data, size_t dataNumItems)
{
    std::ofstream outFile(fileName, std::ios::binary);
    if(outFile)
    {
        outFile.write(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
        outFile.close();
    }
    else
    {
        std::cerr << "Could not open file " << fileName << " for writing" << std::endl;
    }
}

#endif
