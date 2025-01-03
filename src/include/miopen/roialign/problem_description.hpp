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

#include "miopen/problem_description_base.hpp"
#include "miopen/tensor.hpp"

namespace miopen {

struct NetworkConfig;

namespace roialign {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& roisDesc_,
                          const TensorDescriptor& outputDesc_,
                          const uint64_t alignedHeight_,
                          const uint64_t alignedWidth_)
        : inputDesc(inputDesc_),
          roisDesc(roisDesc_),
          outputDesc(outputDesc_),
          alignedHeight(alignedHeight_),
          alignedWidth(alignedWidth_)
    {
        IsSameType();
        IsRightLength();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetRoisDesc() const { return roisDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    uint64_t GetAlignedHeight() const { return alignedHeight; }
    uint64_t GetAlignedWidth() const { return alignedWidth; }

    bool IsRightDim() const
    {
        if(inputDesc.GetNumDims() != 4)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RoIAlignForward: input tensor should be 4-dimensions");
        }

        if(roisDesc.GetNumDims() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RoIAlignForward: rois tensor should be 2-dimensions");
        }

        if(roisDesc.GetLengths()[1] != 5)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RoIAlignForward: rois tensor should have shape in format [K, 5]");
        }

        return true;
    }

    bool IsRightLength() const
    {
        const auto input_grad_lengths = inputDesc.GetLengths();
        const auto rois_lengths       = roisDesc.GetLengths();

        const auto C = input_grad_lengths[1];
        const auto K = rois_lengths[0];

        if(outputDesc.GetLengths() != std::vector<std::size_t>{K, C, alignedHeight, alignedWidth})
        {
            MIOPEN_THROW(miopenStatusBadParm, "RoIAlignForward: Invalid output tensor dimensions");
        }

        return true;
    }

    bool IsSameType() const
    {
        if(inputDesc.GetType() != roisDesc.GetType() || inputDesc.GetType() != outputDesc.GetType())
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "RoIAlignForward: input, output and rois tensors should have the same type");
        }

        return true;
    }

    bool IsAllContiguous() const
    {
        return inputDesc.IsContiguous() && roisDesc.IsContiguous() && outputDesc.IsContiguous();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor& inputDesc;
    const TensorDescriptor& roisDesc;
    const TensorDescriptor& outputDesc;

    const uint64_t alignedHeight;
    const uint64_t alignedWidth;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& roisDesc_,
                          const TensorDescriptor& inputGradDesc_,
                          const uint64_t alignedHeight_,
                          const uint64_t alignedWidth_)
        : outputGradDesc(outputGradDesc_),
          roisDesc(roisDesc_),
          inputGradDesc(inputGradDesc_),
          alignedHeight(alignedHeight_),
          alignedWidth(alignedWidth_)
    {
        IsRightDim();
        IsRightLength();
        IsSameType();
    }

    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetRoisDesc() const { return roisDesc; }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }

    uint64_t GetAlignedHeight() const { return alignedHeight; }
    uint64_t GetAlignedWidth() const { return alignedWidth; }

    bool IsRightDim() const
    {
        if(inputGradDesc.GetNumDims() != 4)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RoIAlignBackward: input grad tensor should be 4-dimensions");
        }

        if(roisDesc.GetNumDims() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RoIAlignBackward: rois tensor should be 2-dimensions");
        }

        if(roisDesc.GetLengths()[1] != 5)
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "RoIAlignBackward: rois tensor should have 5 elements in the second dimension");
        }

        return true;
    }

    bool IsRightLength() const
    {
        const auto input_grad_lengths = inputGradDesc.GetLengths();
        const auto rois_lengths       = roisDesc.GetLengths();

        const auto C = input_grad_lengths[1];

        const auto K = rois_lengths[0];

        if(outputGradDesc.GetLengths() !=
           std::vector<std::size_t>{K, C, alignedHeight, alignedWidth})
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RoIAlignBackward: Invalid output grad tensor dimensions");
        }

        return true;
    }

    bool IsSameType() const
    {
        if(outputGradDesc.GetType() != roisDesc.GetType() ||
           outputGradDesc.GetType() != inputGradDesc.GetType())
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "RoIAlignBackward: input, output and rois tensors should have the same type");
        }

        return true;
    }

    bool IsAllContiguous() const
    {
        return inputGradDesc.IsContiguous() && roisDesc.IsContiguous() &&
               outputGradDesc.IsContiguous();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor& outputGradDesc;
    const TensorDescriptor& roisDesc;
    const TensorDescriptor& inputGradDesc;

    const uint64_t alignedHeight;
    const uint64_t alignedWidth;
};

} // namespace roialign
} // namespace miopen
