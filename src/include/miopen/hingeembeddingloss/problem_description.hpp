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

#include "miopen/errors.hpp"
#include "miopen/tensor.hpp"
#include <miopen/problem_description_base.hpp>

namespace miopen {

struct NetworkConfig;

namespace hingeembeddingloss {

struct ForwardProblemDescription : ProblemDescriptionBase
{
    ForwardProblemDescription(const TensorDescriptor& inputDesc_,
                              const TensorDescriptor& targetDesc_,
                              const TensorDescriptor& outputDesc_,
                              const miopenLossReductionMode_t reduction_)
        : inputDesc(inputDesc_),
          targetDesc(targetDesc_),
          outputDesc(outputDesc_),
          reduction(reduction_)
    {
        if(inputDesc.GetType() != outputDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "HingeEmbeddingLossForward: Input and output tensors need to be "
                         "same data type.");
        }
        if(targetDesc.GetType() != miopenInt8)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "HingeEmbeddingLossForward: Target tensor type need to be miopenInt8.");
        }
        if(inputDesc.GetLengths() != targetDesc.GetLengths())
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "HingeEmbeddingLossForward: Input and target tensors need to be same shape.");
        }
        // Check output tensor dimension
        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            // non-reduction case
            if(inputDesc.GetLengths() != outputDesc.GetLengths())
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "HingeEmbeddingLossForward: When doing forward non-reduction, output "
                             "tensor need to be same shape as input tensor.");
            }
        }
        else
        {
            // reduction case
            if(outputDesc.GetNumDims() != 1 || outputDesc.GetLengths()[0] != 1)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "HingeEmbeddingLossForward: When doing forward reduction, output "
                             "tensor need to be a scalar.");
            }
        }
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    miopenLossReductionMode_t GetReduction() const { return reduction; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor targetDesc;
    TensorDescriptor outputDesc;
    miopenLossReductionMode_t reduction;
};

struct BackwardProblemDescription : ProblemDescriptionBase
{
    BackwardProblemDescription(const TensorDescriptor& inputDesc_,
                               const TensorDescriptor& targetDesc_,
                               const TensorDescriptor& doutputDesc_,
                               const TensorDescriptor& dinputDesc_,
                               const miopenLossReductionMode_t reduction_)
        : inputDesc(inputDesc_),
          targetDesc(targetDesc_),
          doutputDesc(doutputDesc_),
          dinputDesc(dinputDesc_),
          reduction(reduction_)
    {
        if(inputDesc.GetType() != doutputDesc.GetType() ||
           inputDesc.GetType() != dinputDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "HingeEmbeddingLossBackward: Input, output gradient, and input "
                         "gradient tensors need to "
                         "be same data type.");
        }
        if(targetDesc.GetType() != miopenInt8)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "HingeEmbeddingLossForward: Target tensor type need to be miopenInt8.");
        }
        if(inputDesc.GetLengths() != targetDesc.GetLengths() ||
           inputDesc.GetLengths() != dinputDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "HingeEmbeddingLossBackward: Input, target and input gradient tensors "
                         "need to be same shape.");
        }
        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            if(inputDesc.GetLengths() != doutputDesc.GetLengths())
            {
                MIOPEN_THROW(
                    miopenStatusBadParm,
                    "HingeEmbeddingLossBackward: When doing backward non-reduction, output gradient"
                    "tensor need to be same shape as input tensor.");
            }
        }
        else
        {
            if(doutputDesc.GetNumDims() != 1 || doutputDesc.GetLengths()[0] != 1)
            {
                MIOPEN_THROW(
                    miopenStatusBadParm,
                    "HingeEmbeddingLossBackward: When doing backward reduction, output gradient"
                    "tensor need to be a scalar.");
            }
        }
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }
    const TensorDescriptor& GetDOutputDesc() const { return doutputDesc; }
    const TensorDescriptor& GetDInputDesc() const { return dinputDesc; }
    miopenLossReductionMode_t GetReduction() const { return reduction; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor targetDesc;
    TensorDescriptor doutputDesc;
    TensorDescriptor dinputDesc;
    miopenLossReductionMode_t reduction;
};

} // namespace hingeembeddingloss

} // namespace miopen
