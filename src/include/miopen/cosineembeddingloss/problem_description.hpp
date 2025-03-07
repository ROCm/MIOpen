/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <miopen/miopen.h>
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace cosineembeddingloss {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& input1Desc_,
                       const TensorDescriptor& input2Desc_,
                       const TensorDescriptor& targetDesc_,
                       const TensorDescriptor& outputDesc_,
                       const float margin_,
                       bool is_fwd_)
        : input1Desc(input1Desc_),
          input2Desc(input2Desc_),
          targetDesc(targetDesc_),
          outputDesc(outputDesc_),
          margin(margin_),
          is_fwd(is_fwd_)
    {
        IsSameType();
    }

    const TensorDescriptor& GetInput1Desc() const { return input1Desc; }
    const TensorDescriptor& GetInput2Desc() const { return input2Desc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    size_t GetNtotal() const { return targetDesc.GetElementSize(); }
    size_t GetInputTotal() const { return input1Desc.GetElementSize(); }

    bool IsValidLength() const
    {
        if(targetDesc.GetLengths()[0] != input1Desc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm, "CosineEmbeddingLoss: Tensor sizes do not match.");
        }
        for(int i = 0; i < input1Desc.GetNumDims(); ++i)
        {
            if(input1Desc.GetLengths()[i] != input2Desc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "CosineEmbeddingLoss: Tensor sizes do not match.");
            }
        }
        if(input1Desc.GetNumDims() > 2 || input2Desc.GetNumDims() > 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "CosineEmbeddingLoss: Input tensor size > 2 is not valid.");
        }

        if(targetDesc.GetNumDims() > 1)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "CosineEmbeddingLoss: Target tensor size > 1 is not valid.");
        }

        if(outputDesc.GetNumDims() > 1)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "CosineEmbeddingLoss: Output tensor size > 1 is not valid.");
        }
        return true;
    }

    bool IsSameType() const
    {
        if(input1Desc.GetType() != input2Desc.GetType() ||
           input1Desc.GetType() != outputDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "CosineEmbeddingLoss: Tensor types do not match.");
        }
        return true;
    }

protected:
    TensorDescriptor input1Desc;
    TensorDescriptor input2Desc;
    TensorDescriptor targetDesc;
    TensorDescriptor outputDesc;

    float margin;
    bool is_fwd;
};

struct FwdUnreducedProblemDescription : ProblemDescription
{
    FwdUnreducedProblemDescription(const TensorDescriptor& input1Desc_,
                                   const TensorDescriptor& input2Desc_,
                                   const TensorDescriptor& targetDesc_,
                                   const TensorDescriptor& outputDesc_,
                                   const float margin_)
        : ProblemDescription(input1Desc_, input2Desc_, targetDesc_, outputDesc_, margin_, true)
    {
        IsValidLength();
    }

    NetworkConfig MakeNetworkConfig() const override;
};

struct FwdReducedProblemDescription : ProblemDescription
{
    FwdReducedProblemDescription(const TensorDescriptor& input1Desc_,
                                 const TensorDescriptor& input2Desc_,
                                 const TensorDescriptor& targetDesc_,
                                 const TensorDescriptor& outputDesc_,
                                 const float margin_)
        : ProblemDescription(input1Desc_, input2Desc_, targetDesc_, outputDesc_, margin_, true)
    {
        IsValidLength();
    }

    bool IsValidLength() const
    {
        if(outputDesc.GetLengths()[0] != 1)
            MIOPEN_THROW(miopenStatusBadParm,
                         "CosineEmbeddingLoss: Output Tensor length must be (1).");
        if(!ProblemDescription::IsValidLength())
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;
};

struct BwdUnreducedProblemDescription : ProblemDescription
{
    BwdUnreducedProblemDescription(const TensorDescriptor& input1Desc_,
                                   const TensorDescriptor& input2Desc_,
                                   const TensorDescriptor& targetDesc_,
                                   const TensorDescriptor& outputGradDesc_,
                                   const TensorDescriptor& input1GradDesc_,
                                   const TensorDescriptor& input2GradDesc_,
                                   const float margin_)
        : ProblemDescription(
              input1Desc_, input2Desc_, targetDesc_, outputGradDesc_, margin_, false),
          input1GradDesc(input1GradDesc_),
          input2GradDesc(input2GradDesc_)
    {
        IsValidLength();
        IsSameType();
    }
    const TensorDescriptor& GetInput1GradDesc() const { return input1GradDesc; }

    bool IsValidLength() const
    {
        if(input1GradDesc.GetNumDims() > 2 || input2GradDesc.GetNumDims() > 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "CosineEmbeddingLoss: Input grad tensors size > 2 are not valid.");
        }

        for(int i = 0; i < input1Desc.GetNumDims(); ++i)
        {
            if(input1GradDesc.GetLengths()[i] != input2GradDesc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "CosineEmbeddingLoss: Tensor sizes do not match.");
            }
        }

        if(!ProblemDescription::IsValidLength())
            return false;
        return true;
    }

    bool IsSameType() const
    {
        if(input1GradDesc.GetType() != input2GradDesc.GetType() ||
           input1GradDesc.GetType() != outputDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "CosineEmbeddingLoss: Tensor types do not match.");
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor input1GradDesc;
    TensorDescriptor input2GradDesc;
};

struct BwdReducedProblemDescription : ProblemDescription
{
    BwdReducedProblemDescription(const TensorDescriptor& input1Desc_,
                                 const TensorDescriptor& input2Desc_,
                                 const TensorDescriptor& targetDesc_,
                                 const TensorDescriptor& outputGradDesc_,
                                 const TensorDescriptor& input1GradDesc_,
                                 const TensorDescriptor& input2GradDesc_,
                                 const float margin_)
        : ProblemDescription(
              input1Desc_, input2Desc_, targetDesc_, outputGradDesc_, margin_, false),
          input1GradDesc(input1GradDesc_),
          input2GradDesc(input2GradDesc_)
    {
        IsValidLength();
        IsSameType();
    }
    const TensorDescriptor& GetInput1GradDesc() const { return input1GradDesc; }

    bool IsValidLength() const
    {
        if(outputDesc.GetLengths()[0] != 1)
            MIOPEN_THROW(miopenStatusBadParm,
                         "CosineEmbeddingLoss: Output Tensor length must be (1).");

        if(input1GradDesc.GetNumDims() > 2 || input2GradDesc.GetNumDims() > 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "CosineEmbeddingLoss: Input grad tensors size > 2 are not valid.");
        }

        for(int i = 0; i < input1Desc.GetNumDims(); ++i)
        {
            if(input1GradDesc.GetLengths()[i] != input2GradDesc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "CosineEmbeddingLoss: Tensor sizes do not match.");
            }
        }

        if(!ProblemDescription::IsValidLength())
            return false;
        return true;
    }

    bool IsSameType() const
    {
        if(input1GradDesc.GetType() != input2GradDesc.GetType() ||
           input1GradDesc.GetType() != outputDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "CosineEmbeddingLoss: Tensor types do not match.");
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor input1GradDesc;
    TensorDescriptor input2GradDesc;
};

} // namespace cosineembeddingloss

} // namespace miopen
