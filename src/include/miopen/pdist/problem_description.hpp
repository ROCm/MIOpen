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
#pragma once

#include <cstddef>

#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/problem_description_base.hpp>
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace pdist {

struct BackwardProblemDescription : public ProblemDescriptionBase
{
    BackwardProblemDescription(const TensorDescriptor& inputDesc_,
                               const TensorDescriptor& outputDesc_,
                               const TensorDescriptor& doutputDesc_,
                               const TensorDescriptor& dinputDesc_,
                               const double p_)
        : inputDesc(inputDesc_),
          outputDesc(outputDesc_),
          doutputDesc(doutputDesc_),
          dinputDesc(dinputDesc_),
          p(p_)
    {
        if(p < 0)
        {
            MIOPEN_THROW(miopenStatusBadParm, "PdistBackward: p must be non-negative.");
        }

        IsSameType();
        IsRightLength();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetdOutputDesc() const { return doutputDesc; }
    const TensorDescriptor& GetdInputDesc() const { return dinputDesc; }
    double GetPValue() const { return p; }

    bool IsSameType() const
    {
        if((inputDesc.GetType() != outputDesc.GetType()) ||
           (outputDesc.GetType() != doutputDesc.GetType()) ||
           (dinputDesc.GetType() != inputDesc.GetType()))
        {
            MIOPEN_THROW(miopenStatusBadParm, "PdistBackward: Tensor types do not match.");
            return false;
        }

        return true;
    }

    bool IsRightLength() const
    {
        if(inputDesc.GetNumDims() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm, "PdistBackward: Input tensor must be 2D.");
            return false;
        }

        auto N           = inputDesc.GetLengths()[0];
        auto output_size = N * (N - 1) / 2;

        if(!(outputDesc.GetNumDims() == 1 && outputDesc.GetLengths()[0] == output_size))
        {
            std::size_t expected_num_dims = 1;
            std::size_t expected_size     = output_size;
            std::size_t actual_num_dims   = outputDesc.GetNumDims();
            std::size_t actual_size       = outputDesc.GetLengths()[0];

            MIOPEN_THROW(miopenStatusBadParm,
                         "PdistBackward: Output tensor shape is incorrect. Expected: " +
                             std::to_string(expected_num_dims) + "D tensor of size " +
                             std::to_string(expected_size) +
                             ". Got: " + std::to_string(actual_num_dims) + "D tensor of size " +
                             std::to_string(actual_size) + ".");
            return false;
        }

        return true;
    }

    bool IsAllContiguous() const
    {
        if(!(inputDesc.IsContiguous() && outputDesc.IsContiguous() && doutputDesc.IsContiguous() &&
             dinputDesc.IsContiguous()))
        {
            MIOPEN_THROW(miopenStatusBadParm, "PdistBackward: Uncontiguous tensors not supported.");
            return false;
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(inputDesc.IsPacked() && outputDesc.IsPacked() && doutputDesc.IsPacked() &&
             dinputDesc.IsPacked()))
        {
            MIOPEN_THROW(miopenStatusBadParm, "PdistBackward: Unpacked tensors not supported.");
            return false;
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor doutputDesc;
    TensorDescriptor dinputDesc;

    double p;
};

} // namespace pdist

} // namespace miopen
