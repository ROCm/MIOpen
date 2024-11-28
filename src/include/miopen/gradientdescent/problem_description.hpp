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

namespace GradientDescent {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& varInDesc_,
                       const TensorDescriptor& varOutDesc_,
                       const TensorDescriptor& alphaInDesc_,
                       const TensorDescriptor& deltaInDesc_)
        : varInDesc(varInDesc_),
          varOutDesc(varOutDesc_),
          alphaInDesc(alphaInDesc_),
          deltaInDesc(deltaInDesc_)
    {
        IsSameType();
        IsSameDims();
    }

    const TensorDescriptor& GetvarInDesc() const { return varInDesc; }
    const TensorDescriptor& GetvarOutDesc() const { return varOutDesc; }
    const TensorDescriptor& GetalphaInDesc() const { return alphaInDesc; }

    bool IsSameType() const
    {
        if(varInDesc.GetType() != varOutDesc.GetType() ||
           varInDesc.GetType() != alphaInDesc.GetType() ||
           varInDesc.GetType() != deltaInDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "GradientDescent: Tensor types do not match.");
        }
        return true;
    }

    bool IsSameDims() const
    {
        if(varInDesc.GetLengths() != varOutDesc.GetLengths() ||
           varInDesc.GetLengths() != deltaInDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm, "GradientDescent: Tensor dimensions do not match.");
        }
        return true;
    }

    bool IsAllContiguous() const
    {
        return varInDesc.IsContiguous() && varOutDesc.IsContiguous() && deltaInDesc.IsContiguous();
    }
    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor varInDesc;
    TensorDescriptor varOutDesc;
    TensorDescriptor alphaInDesc;
    TensorDescriptor deltaInDesc;
};

} // namespace GradientDescent

} // namespace miopen
