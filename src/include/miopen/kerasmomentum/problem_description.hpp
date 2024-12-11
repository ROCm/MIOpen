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

namespace KerasMomentum {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& varInDesc_,
                       const TensorDescriptor& varOutDesc_,
                       const TensorDescriptor& accumInDesc_,
                       const TensorDescriptor& accumOutDesc_,
                       const TensorDescriptor& lrInDesc_,
                       const TensorDescriptor& gradInDesc_,
                       const TensorDescriptor& momentumInDesc_)
        : varInDesc(varInDesc_),
          varOutDesc(varOutDesc_),
          accumInDesc(accumInDesc_),
          accumOutDesc(accumOutDesc_),
          lrInDesc(lrInDesc_),
          gradInDesc(gradInDesc_),
          momentumInDesc(momentumInDesc_)
    {
        IsSameType();
        IsSameDims();
    }

    const TensorDescriptor& GetvarInDesc() const { return varInDesc; }
    const TensorDescriptor& GetvarOutDesc() const { return varOutDesc; }
    const TensorDescriptor& GetaccumInDesc() const { return accumInDesc; }
    const TensorDescriptor& GetaccumOutDesc() const { return accumOutDesc; }
    const TensorDescriptor& GetlrInDesc() const { return lrInDesc; }
    const TensorDescriptor& GetgradInDesc() const { return gradInDesc; }
    const TensorDescriptor& GetmomentumInDesc() const { return momentumInDesc; }

    bool IsSameType() const
    {
        if(varInDesc.GetType() != varOutDesc.GetType() ||
           varInDesc.GetType() != accumInDesc.GetType() ||
           varInDesc.GetType() != accumOutDesc.GetType() ||
           varInDesc.GetType() != lrInDesc.GetType() ||
           varInDesc.GetType() != gradInDesc.GetType() ||
           varInDesc.GetType() != momentumInDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "KerasMomentum: Tensor types do not match.");
        }
        return true;
    }

    bool IsSameDims() const
    {
        if(varInDesc.GetLengths() != varOutDesc.GetLengths() ||
           varInDesc.GetLengths() != accumInDesc.GetLengths() ||
           varInDesc.GetLengths() != accumOutDesc.GetLengths() ||
           varInDesc.GetLengths() != gradInDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm, "KerasMomentum: Tensor dimensions do not match.");
        }
        return true;
    }

    bool IsAllPackedSameStride() const
    {
        return varInDesc.IsPacked() && varOutDesc.IsPacked() && accumInDesc.IsPacked() &&
               accumOutDesc.IsPacked() && gradInDesc.IsPacked() &&
               varInDesc.GetStrides() == varOutDesc.GetStrides() &&
               varInDesc.GetStrides() == accumInDesc.GetStrides() &&
               varInDesc.GetStrides() == accumOutDesc.GetStrides() &&
               varInDesc.GetStrides() == gradInDesc.GetStrides();
    }
    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor varInDesc;
    TensorDescriptor varOutDesc;
    TensorDescriptor accumInDesc;
    TensorDescriptor accumOutDesc;
    TensorDescriptor lrInDesc;
    TensorDescriptor gradInDesc;
    TensorDescriptor momentumInDesc;
};

} // namespace KerasMomentum

} // namespace miopen
