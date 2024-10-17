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
#include <miopen/problem_description_base.hpp>
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

namespace miopen {
struct NetworkConfig;
namespace mseloss {
namespace forward {
struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& xDesc_, const TensorDescriptor& yDesc_)
        : xDesc(xDesc_), yDesc(yDesc_)
    {
        if(!DoesTensorsMatch())
        {
            MIOPEN_THROW("Target and Input does not match");
        }
        if(!IsSameType())
        {
            MIOPEN_THROW("Target and Input does not match");
        }
    };

    NetworkConfig MakeNetworkConfig() const override;

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }

    bool IsSameType() const { return xDesc.GetType() == yDesc.GetType(); }

    bool IsContiguous() const { return xDesc.IsContiguous() && yDesc.IsContiguous(); }

    bool DoesTensorsMatch() const
    {
        if(xDesc.GetLengths().size() != yDesc.GetLengths().size())
            return false;

        for(auto i = 0; i < xDesc.GetLengths().size(); ++i)
        {
            if(xDesc.GetLengths()[i] != yDesc.GetLengths()[i])
                return false;
        }
        return true;
    }

    bool IsImprovementOverROCm() const
    {
        // Mostly thanks to parallel reduction, since we lose pretty much everywhere else
        return true;
    }

private:
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
};
} // namespace forward

namespace forward_unreduced {
struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& zDesc_)
        : xDesc(xDesc_), yDesc(yDesc_), zDesc(zDesc_)
    {
        if(!DoesTensorsMatch())
        {
            MIOPEN_THROW("Target and Input does not match");
        }
        if(!IsSameType())
        {
            MIOPEN_THROW("Target and Input does not match");
        }
    };

    NetworkConfig MakeNetworkConfig() const override;

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    const TensorDescriptor& GetZDesc() const { return zDesc; }

    bool IsSameType() const
    {
        return xDesc.GetType() == yDesc.GetType() && xDesc.GetType() == zDesc.GetType();
    }

    bool IsContiguous() const
    {
        return xDesc.IsContiguous() && yDesc.IsContiguous() && zDesc.IsContiguous();
    }

    bool DoesTensorsMatch() const
    {
        if(xDesc.GetLengths().size() != yDesc.GetLengths().size() ||
           xDesc.GetLengths().size() != zDesc.GetLengths().size())
            return false;

        for(auto i = 0; i < xDesc.GetLengths().size(); ++i)
        {
            if(xDesc.GetLengths()[i] != yDesc.GetLengths()[i])
                return false;

            if(xDesc.GetLengths()[i] != zDesc.GetLengths()[i])
                return false;
        }
        return true;
    }

    bool IsImprovementOverROCm() const
    {
        // not faster in any tested cases
        return false;
    }

private:
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
    const TensorDescriptor& zDesc;
};
} // namespace forward_unreduced

namespace backward {
struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& zDesc_,
                       const TensorDescriptor& dxDesc_,
                       const TensorDescriptor& dyDesc_,
                       const float divisor_ = 1.0f)
        : xDesc(xDesc_),
          yDesc(yDesc_),
          zDesc(zDesc_),
          dxDesc(dxDesc_),
          dyDesc(dyDesc_),
          divisor(divisor_)
    {
        if(!DoesTensorsMatch())
        {
            MIOPEN_THROW("Target and Input does not match");
        }
        if(!IsSameType())
        {
            MIOPEN_THROW("Target and Input does not match");
        }
    };

    NetworkConfig MakeNetworkConfig() const override;

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    const TensorDescriptor& GetZDesc() const { return zDesc; }

    const TensorDescriptor& GetDXDesc() const { return dxDesc; }
    const TensorDescriptor& GetDYDesc() const { return dyDesc; }

    float GetDivisor() const { return divisor; }

    bool IsSameType() const { return xDesc.GetType() == yDesc.GetType(); }

    bool IsContiguous() const { return xDesc.IsContiguous() && yDesc.IsContiguous(); }

    bool DoesTensorsMatch() const
    {
        if(xDesc.GetLengths().size() != yDesc.GetLengths().size())
            return false;

        for(auto i = 0; i < xDesc.GetLengths().size(); ++i)
        {
            if(xDesc.GetLengths()[i] != yDesc.GetLengths()[i])
                return false;
        }
        return true;
    }

    bool IsImprovementOverROCm() const
    {
        // Backward, reduced is seems only faster on 2d, non-contiguous tensors
        if(xDesc.GetLengths().size() == 2 && !IsContiguous())
            return true;
        return false;
    }

private:
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
    const TensorDescriptor& zDesc;
    const TensorDescriptor& dxDesc;
    const TensorDescriptor& dyDesc;
    const float divisor = 1.0f;
};
} // namespace backward

namespace backward_unreduced {
struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& zDesc_,
                       const TensorDescriptor& dxDesc_,
                       const TensorDescriptor& dyDesc_)
        : xDesc(xDesc_), yDesc(yDesc_), zDesc(zDesc_), dxDesc(dxDesc_), dyDesc(dyDesc_)
    {
        if(!DoesTensorsMatch())
        {
            MIOPEN_THROW("Target and Input does not match");
        }
        if(!IsSameType())
        {
            MIOPEN_THROW("Target and Input does not match");
        }
    };

    NetworkConfig MakeNetworkConfig() const override;

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    const TensorDescriptor& GetZDesc() const { return zDesc; }

    const TensorDescriptor& GetDXDesc() const { return dxDesc; }
    const TensorDescriptor& GetDYDesc() const { return dyDesc; }

    bool IsSameType() const { return xDesc.GetType() == yDesc.GetType(); }

    bool IsContiguous() const { return xDesc.IsContiguous() && yDesc.IsContiguous(); }

    bool DoesTensorsMatch() const
    {
        if(xDesc.GetLengths().size() != yDesc.GetLengths().size())
            return false;

        for(auto i = 0; i < xDesc.GetLengths().size(); ++i)
        {
            if(xDesc.GetLengths()[i] != yDesc.GetLengths()[i])
                return false;
        }
        return true;
    }

    bool IsImprovementOverROCm() const
    {
        // only a few % tested is at least 20% faster
        return false;
    }

private:
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
    const TensorDescriptor& zDesc;
    const TensorDescriptor& dxDesc;
    const TensorDescriptor& dyDesc;
};
} // namespace backward_unreduced
} // namespace mseloss
} // namespace miopen
