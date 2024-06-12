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
#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace rope {

struct ProblemDescriptionFwd : ProblemDescriptionBase
{
    ProblemDescriptionFwd(const TensorDescriptor& xDesc_,
                          const TensorDescriptor& cosDesc_,
                          const TensorDescriptor& sinDesc_,
                          const TensorDescriptor& yDesc_)
        : xDesc(xDesc_), cosDesc(cosDesc_), sinDesc(sinDesc_), yDesc(yDesc_)
    {
        IsValidLength();
    }

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetCosDesc() const { return cosDesc; }
    const TensorDescriptor& GetSinDesc() const { return sinDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }

    bool IsValidLength() const
    {
        if(xDesc.GetLengths() != yDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RoPEForward: Tensor x and y dimension lengths do not match.");
        }

        if(cosDesc.GetLengths() != sinDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RoPEForward: Tensor cos and sin dimension lengths do not match.");
        }
        return true;
    }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
        {
            return false;
        }
        return true;
    }

    bool IsAllContiguous() const
    {
        if(!(xDesc.IsContiguous() && cosDesc.IsContiguous() && sinDesc.IsContiguous() &&
             yDesc.IsContiguous()))
        {
            return false;
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor xDesc;
    TensorDescriptor cosDesc;
    TensorDescriptor sinDesc;
    TensorDescriptor yDesc;

    NetworkConfig MakeForwardNetworkConfig() const;
};

struct ProblemDescriptionBwd : ProblemDescriptionBase
{
    ProblemDescriptionBwd(const TensorDescriptor& dyDesc_,
                          const TensorDescriptor& cosDesc_,
                          const TensorDescriptor& sinDesc_,
                          const TensorDescriptor& dxDesc_)
        : dyDesc(dyDesc_), cosDesc(cosDesc_), sinDesc(sinDesc_), dxDesc(dxDesc_)
    {
        IsValidLength();
    }

    const TensorDescriptor& GetDYDesc() const { return dyDesc; }
    const TensorDescriptor& GetCosDesc() const { return cosDesc; }
    const TensorDescriptor& GetSinDesc() const { return sinDesc; }
    const TensorDescriptor& GetDXDesc() const { return dxDesc; }

    bool IsValidLength() const
    {
        if(dyDesc.GetLengths() != dxDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RoPEForward: Tensor dy and dx dimension lengths do not match.");
        }

        if(cosDesc.GetLengths() != sinDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RoPEForward: Tensor cos and sin dimension lengths do not match.");
        }
        return true;
    }

    bool IsSameType() const
    {
        if(dyDesc.GetType() != dxDesc.GetType())
        {
            return false;
        }
        return true;
    }

    bool IsAllContiguous() const
    {
        if(!(dyDesc.IsContiguous() && cosDesc.IsContiguous() && sinDesc.IsContiguous() &&
             dxDesc.IsContiguous()))
        {
            return false;
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor dyDesc;
    TensorDescriptor cosDesc;
    TensorDescriptor sinDesc;
    TensorDescriptor dxDesc;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace rope

} // namespace miopen
