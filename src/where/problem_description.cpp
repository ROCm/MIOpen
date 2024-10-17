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

#include <miopen/datatype.hpp>
#include <miopen/names.hpp>
#include "miopen/tensor.hpp"
#include <miopen/where/problem_description.hpp>
#include <miopen/where/solvers.hpp>

namespace miopen {

namespace where {

bool isSameShape(const TensorDescriptor& x, const TensorDescriptor& y)
{
    if(x.GetNumDims() != y.GetNumDims())
    {
        return false;
    }

    for(int i = 0; i < x.GetNumDims(); i++)
    {
        if(x.GetLengths()[i] != y.GetLengths()[i])
        {
            return false;
        }
    }
    return true;
}

NetworkConfig BackwardProblemDescription::MakeNetworkConfig() const
{
    auto output_grad_numel = outputGradDesc.GetElementSize();
    auto io_dtype          = miopen::GetDataType(inputGradDesc.GetType());

    std::ostringstream ss;

    ss << "io_dtype" << io_dtype;
    ss << "outputGrad_numel" << output_grad_numel;

    return NetworkConfig{ss.str()};
}

} // namespace where

} // namespace miopen
