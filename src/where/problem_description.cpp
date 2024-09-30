/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "miopen/datatype.hpp"
#include "miopen/tensor.hpp"
#include <miopen/where/problem_description.hpp>
#include <miopen/where/solvers.hpp>
#include <miopen/names.hpp>

namespace miopen {

namespace where {

bool isSameShape(const TensorDescriptor& x, const TensorDescriptor& y)
{
    return x.GetNumDims() == y.GetNumDims();
}

// Strides of broadcasted but contiguous tensors are like [0, x * y, y, 1]
template <int N>
int64_t checkBroadcastedContiguous(const tensor_view_t<N>& tensorView)
{
    int64_t num_elems = 1;

    for(int i = N - 1; i >= 0; i--)
    {
        if(tensorView.stride[i] != 0 && tensorView.stride[i] != num_elems)
            return 0;
        if(tensorView.stride[i] == 0)
        {
            for(int j = i; j >= 0; j--)
                if(tensorView.stride[j] != 0)
                    return 0;
            return num_elems;
        }
        num_elems *= tensorView.size[i];
    }

    return num_elems;
}

template int64_t checkBroadcastedContiguous(const tensor_view_t<5>&);

NetworkConfig BackwardProblemDescription::MakeNetworkConfig() const
{
    auto cond_contig_size       = checkBroadcastedContiguous(condition_tv);
    auto input_grad_contig_size = checkBroadcastedContiguous(inputGrad_tv);
    auto other_grad_contig_size = checkBroadcastedContiguous(otherGrad_tv);

    bool is_all_contiguous             = this->isAllContiguous();
    bool is_all_broadcasted_contiguous = this->isAllBroadcastedContiguous();
    bool is_condition_broadcasted      = this->isConditionBroadcasted();

    auto output_grad_numel = outputGradDesc.GetElementSize();
    auto outputGrad_dtype  = miopen::GetDataType(outputGradDesc.GetType());
    auto inputGrad_dtype   = miopen::GetDataType(inputGradDesc.GetType());

    std::ostringstream ss;

    ss << "is_all_contiguous" << is_all_contiguous;
    ss << "is_all_broadcasted_contiguous" << is_all_broadcasted_contiguous;
    ss << "is_condition_broadcasted" << is_condition_broadcasted;
    ss << "inputGrad_dtype" << inputGrad_dtype;
    ss << "outputGrad_dtype" << outputGrad_dtype;
    ss << "outputGrad_numel" << output_grad_numel;
    ss << "cond_contig_size" << cond_contig_size;
    ss << "input_contig_size" << input_grad_contig_size;
    ss << "other_contig_size" << other_grad_contig_size;
    ss << isAllContiguous();

    return NetworkConfig{ss.str()};
}

} // namespace where

} // namespace miopen
