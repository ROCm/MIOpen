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

#pragma once

#include <miopen/problem_description_base.hpp>
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

enum gradType
{
    NONE,
    ONE,
    TWO
};

namespace outer {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const bool is_fwd_,
                       gradType grad_,
                       const TensorDescriptor& x1Desc_,
                       const TensorDescriptor& x2Desc_,
                       const TensorDescriptor& yDesc_)
        : is_fwd(is_fwd_), grad(grad_), x1Desc(x1Desc_), x2Desc(x2Desc_), yDesc(yDesc_)
    {
        const auto dtype = yDesc.GetType();
        if(x1Desc.GetType() != dtype)
        {
            MIOPEN_THROW(miopenStatusBadParm, "OuterForward: Tensor types do not match.");
        }
        if(x2Desc.GetType() != dtype)
        {
            MIOPEN_THROW(miopenStatusBadParm, "OuterForward: Tensor types do not match.");
        }
    }

    const TensorDescriptor& GetX1Desc() const { return x1Desc; }
    const TensorDescriptor& GetX2Desc() const { return x2Desc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }

    bool IsAllPacked() const
    {
        if(!x1Desc.IsPacked())
            return false;
        if(!x2Desc.IsPacked())
            return false;
        if(!yDesc.IsPacked())
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    bool is_fwd;
    gradType grad;
    TensorDescriptor x1Desc;
    TensorDescriptor x2Desc;
    TensorDescriptor yDesc;
};

} // namespace outer
} // namespace miopen
