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

#include "miopen/miopen.h"
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace multilabel_margin_loss {

struct MultilabelMarginLossFwdProblemDescriptionBase : ProblemDescriptionBase
{
    MultilabelMarginLossFwdProblemDescriptionBase(const TensorDescriptor& iDesc_,
                                                const TensorDescriptor& tDesc_,
                                                const TensorDescriptor& oDesc_)
        : iDesc(iDesc_), tDesc(tDesc_), oDesc(oDesc_)
    {
        if(!IsSameLength())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Multilabel Margin Loss: Tensor sizes do not match");
        }
        if(!IsRightDim())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Multilabel Margin Loss: Only accept 2d tensor (N, C)");
        }
    }
    
    bool IsSameLength() const
    {
        if(iDesc.GetSize() != tDesc.GetSize())
            return false;
        for(int32_t i = 0; i < iDesc.GetSize(); ++i)
        {
            if(iDesc.GetLengths()[i] != tDesc.GetLengths()[i])
                return false;
        }
        return true;
    }

    bool IsRightDim() const
    {
        if(!(iDesc.GetSize() == 2 && tDesc.GetSize() == 2))
        {
            return false;
        }
        return true;
    }

protected:
    const TensorDescriptor& iDesc;
    const TensorDescriptor& tDesc;
    const TensorDescriptor& oDesc;
};

struct MultilabelMarginLossFwdProblemDescription : MultilabelMarginLossFwdProblemDescriptionBase
{
    MultilabelMarginLossFwdProblemDescription(const TensorDescriptor& iDesc_,
                                                const TensorDescriptor& tDesc_,
                                                const TensorDescriptor& oDesc_)
        : MultilabelMarginLossFwdProblemDescriptionBase(iDesc_, tDesc_, oDesc_)
    {
    }

    const TensorDescriptor& GetIDesc() const { return iDesc; }
    const TensorDescriptor& GetTDesc() const { return tDesc; }
    const TensorDescriptor& GetODesc() const { return oDesc; }

    NetworkConfig MakeNetworkConfig() const override;
};

struct MultilabelMarginLossUnreducedFwdProblemDescription : MultilabelMarginLossFwdProblemDescriptionBase
{
    MultilabelMarginLossUnreducedFwdProblemDescription(const TensorDescriptor& iDesc_,
                                                const TensorDescriptor& tDesc_,
                                                const TensorDescriptor& oDesc_)
        : MultilabelMarginLossFwdProblemDescriptionBase(iDesc_, tDesc_, oDesc_)
    {
    }

    const TensorDescriptor& GetIDesc() const { return iDesc; }
    const TensorDescriptor& GetTDesc() const { return tDesc; }
    const TensorDescriptor& GetODesc() const { return oDesc; }

    NetworkConfig MakeNetworkConfig() const override;
};

struct MultilabelMarginLossBwdProblemDescriptionBase : ProblemDescriptionBase
{
    MultilabelMarginLossBwdProblemDescriptionBase(const TensorDescriptor& iDesc_,
                                                const TensorDescriptor& tDesc_,
                                                const TensorDescriptor& dODesc_,
                                                const TensorDescriptor& dIDesc_)
        : iDesc(iDesc_), tDesc(tDesc_), dODesc(dODesc_), dIDesc(dIDesc_)
    {
        if(!IsSameLength())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Multilabel Margin Loss: Tensor sizes do not match");
        }
        if(!IsRightDim())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Multilabel Margin Loss: Only accept 2d tensor (N, C)");
        }
    }
    
    bool IsSameLength() const
    {
        if(iDesc.GetSize() != tDesc.GetSize())
            return false;
        for(int32_t i = 0; i < iDesc.GetSize(); ++i)
        {
            if(iDesc.GetLengths()[i] != tDesc.GetLengths()[i])
                return false;
        }
        return true;
    }

    bool IsRightDim() const
    {
        if(!(iDesc.GetSize() == 2 && tDesc.GetSize() == 2))
        {
            return false;
        }
        return true;
    }

protected:
    const TensorDescriptor& iDesc;
    const TensorDescriptor& tDesc;
    const TensorDescriptor& dODesc;
    const TensorDescriptor& dIDesc;
};

struct MultilabelMarginLossBwdProblemDescription : MultilabelMarginLossBwdProblemDescriptionBase
{
    MultilabelMarginLossBwdProblemDescription(const TensorDescriptor& iDesc_,
                                                const TensorDescriptor& tDesc_,
                                                const TensorDescriptor& dODesc_,
                                                const TensorDescriptor& dIDesc_)
        : MultilabelMarginLossBwdProblemDescriptionBase(iDesc_, tDesc_, dODesc_, dIDesc_)
    {
    }

    const TensorDescriptor& GetIDesc() const { return iDesc; }
    const TensorDescriptor& GetTDesc() const { return tDesc; }
    const TensorDescriptor& GetdODesc() const { return dODesc; }
    const TensorDescriptor& GetdIDesc() const { return dIDesc; }


    NetworkConfig MakeNetworkConfig() const override;
};

struct MultilabelMarginLossUnreducedBwdProblemDescription : MultilabelMarginLossBwdProblemDescriptionBase
{
    MultilabelMarginLossUnreducedBwdProblemDescription(const TensorDescriptor& iDesc_,
                                                const TensorDescriptor& tDesc_,
                                                const TensorDescriptor& dODesc_,
                                                const TensorDescriptor& dIDesc_)
        : MultilabelMarginLossBwdProblemDescriptionBase(iDesc_, tDesc_, dODesc_, dIDesc_)
    {
    }

    const TensorDescriptor& GetIDesc() const { return iDesc; }
    const TensorDescriptor& GetTDesc() const { return tDesc; }
    const TensorDescriptor& GetdODesc() const { return dODesc; }
    const TensorDescriptor& GetdIDesc() const { return dIDesc; }


    NetworkConfig MakeNetworkConfig() const override;
};

} // namespace multilabel_margin_loss

} // namespace miopen
