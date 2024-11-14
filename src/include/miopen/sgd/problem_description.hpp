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

#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace SGD {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& paramInDesc_,
                       const TensorDescriptor& paramOutDesc_,
                       const TensorDescriptor& gradDesc_,
                       const TensorDescriptor& momentumBufferInDesc_,
                       const TensorDescriptor& momentumBufferOutDesc_)
        : paramInDesc(paramInDesc_),
          paramOutDesc(paramOutDesc_),
          gradDesc(gradDesc_),
          momentumBufferInDesc(momentumBufferInDesc_),
          momentumBufferOutDesc(momentumBufferOutDesc_)
    {
        IsSameType();
        IsSameLength();
    }

    const TensorDescriptor& GetParamInDesc() const { return paramInDesc; }
    const TensorDescriptor& GetParamOutDesc() const { return paramOutDesc; }
    const TensorDescriptor& GetGradDesc() const { return gradDesc; }
    const TensorDescriptor& GetMomentumBufferInDesc() const { return momentumBufferInDesc; }
    const TensorDescriptor& GetMomentumBufferOutDesc() const { return momentumBufferOutDesc; }

    bool IsSameType() const
    {
        if(paramInDesc.GetType() != paramOutDesc.GetType() ||
           paramOutDesc.GetType() != gradDesc.GetType() ||
           gradDesc.GetType() != momentumBufferInDesc.GetType() ||
           momentumBufferInDesc.GetType() != momentumBufferOutDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "SGDForward: Tensor types do not match.");
        }
        return true;
    }

    bool IsSameLength() const
    {
        for(int32_t i = 0; i < paramInDesc.GetLengths().size(); ++i)
        {
            size_t len = paramInDesc.GetLengths()[i];
            if(paramOutDesc.GetLengths()[i] != len || gradDesc.GetLengths()[i] != len ||
               momentumBufferInDesc.GetLengths()[i] != len ||
               momentumBufferOutDesc.GetLengths()[i] != len)
            {
                MIOPEN_THROW(miopenStatusBadParm, "SGDForward: Tensor lengths do not match.");
            }
        }
        return true;
    }

    bool IsValidLength() const
    {
        auto input_dims = paramInDesc.GetLengths().size();
        if(input_dims > 4)
        {
            return false;
        }
        return true;
    }

    bool IsAllContiguous() const
    {
        return paramInDesc.IsContiguous() && paramOutDesc.IsContiguous() &&
               gradDesc.IsContiguous() && momentumBufferInDesc.IsContiguous() &&
               momentumBufferOutDesc.IsContiguous();
    }
    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor paramInDesc;
    TensorDescriptor paramOutDesc;
    TensorDescriptor gradDesc;
    TensorDescriptor momentumBufferInDesc;
    TensorDescriptor momentumBufferOutDesc;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace SGD
} // namespace miopen
