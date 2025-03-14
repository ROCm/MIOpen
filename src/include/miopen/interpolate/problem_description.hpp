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

namespace interpolate {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& scaleFactorsDesc_,
                       const miopenInterpolateMode_t mode_,
                       const bool align_corners_)
        : scaleFactorsDesc(scaleFactorsDesc_), mode(mode_), align_corners(align_corners_)
    {
        IsValidMode();
        IsValidType();
    }

    const TensorDescriptor& GetScaleFactorsDesc() const { return scaleFactorsDesc; }
    miopenInterpolateMode_t GetMode() const { return mode; }
    bool GetAlignCorners() const { return align_corners; }

    bool IsValidMode() const
    {
        if(mode != MIOPEN_INTERPOLATE_MODE_NEAREST && mode != MIOPEN_INTERPOLATE_MODE_LINEAR &&
           mode != MIOPEN_INTERPOLATE_MODE_BILINEAR && mode != MIOPEN_INTERPOLATE_MODE_TRILINEAR &&
           mode != MIOPEN_INTERPOLATE_MODE_BICUBIC)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Interpolate: Invalid mode.");
        }
        return true;
    }

    bool IsValidType() const
    {
        if(scaleFactorsDesc.GetType() != miopenFloat)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Interpolate: Scale factor type should be miopenFloat.");
        }

        return true;
    }

protected:
    TensorDescriptor scaleFactorsDesc;
    miopenInterpolateMode_t mode;
    bool align_corners = false;
};

struct FwdProblemDescription : ProblemDescription
{
    FwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& outputDesc_,
                          const TensorDescriptor& scaleFactorsDesc_,
                          const miopenInterpolateMode_t mode_,
                          const bool align_corners_)
        : ProblemDescription(scaleFactorsDesc_, mode_, align_corners_),
          inputDesc(inputDesc_),
          outputDesc(outputDesc_)
    {
        IsValidDims();
        IsValidLength();
        IsSameType();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    bool IsValidLength() const
    {
        if(inputDesc.GetNumDims() < 3 || inputDesc.GetNumDims() > 5)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Interpolate: Input or output tensor size < 3 or > 5 is not valid.");
        }

        if(outputDesc.GetNumDims() != inputDesc.GetNumDims())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Interpolate: Input and output tensor size do not match.");
        }

        if((outputDesc.GetNumDims() - 2) != scaleFactorsDesc.GetElementSize())
        {
            if(mode != MIOPEN_INTERPOLATE_MODE_NEAREST)
            {
                MIOPEN_THROW(
                    miopenStatusBadParm,
                    "Interpolate: Output tensor size and scale factors length do not match.");
            }
        }
        return true;
    }

    bool IsValidDims() const
    {
        if(mode == MIOPEN_INTERPOLATE_MODE_LINEAR)
        {
            if(inputDesc.GetNumDims() != 3)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Interpolate: Linear mode requires 3D tensors.");
            }
        }
        if(mode == MIOPEN_INTERPOLATE_MODE_BILINEAR)
        {
            if(inputDesc.GetNumDims() != 4)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Interpolate: Bilinear mode requires 4D tensors.");
            }
        }
        if(mode == MIOPEN_INTERPOLATE_MODE_BICUBIC)
        {
            if(inputDesc.GetNumDims() != 4)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Interpolate: Bicubic mode requires 4D tensors.");
            }
        }
        if(mode == MIOPEN_INTERPOLATE_MODE_TRILINEAR)
        {
            if(inputDesc.GetNumDims() != 5)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Interpolate: Trilinear mode requires 5D tensors.");
            }
        }
        return true;
    }

    bool IsSameType() const
    {
        if(inputDesc.GetType() != outputDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Interpolate: Input and output tensor type do not match.");
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
};

struct BwdProblemDescription : ProblemDescription
{
    BwdProblemDescription(const TensorDescriptor& inputGradDesc_,
                          const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& scaleFactorsDesc_,
                          const miopenInterpolateMode_t mode_,
                          const bool align_corners_)
        : ProblemDescription(scaleFactorsDesc_, mode_, align_corners_),
          inputGradDesc(inputGradDesc_),
          outputGradDesc(outputGradDesc_)
    {
        IsValidDims();
        IsValidLength();
        IsSameType();
    }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }

    bool IsValidLength() const
    {
        if(inputGradDesc.GetNumDims() < 3 || inputGradDesc.GetNumDims() > 5)
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "Interpolate: Input grad or output grad tensor size < 3 or > 5 is not valid.");
        }

        if(outputGradDesc.GetNumDims() != inputGradDesc.GetNumDims())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Interpolate: Input grad and output grad tensor size do not match.");
        }

        if((outputGradDesc.GetNumDims() - 2) != scaleFactorsDesc.GetElementSize())
        {
            if(mode != MIOPEN_INTERPOLATE_MODE_NEAREST)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Interpolate: Tensor size and scale factors length do not match.");
            }
        }
        return true;
    }

    bool IsValidDims() const
    {
        if(mode == MIOPEN_INTERPOLATE_MODE_LINEAR)
        {
            if(inputGradDesc.GetNumDims() != 3)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Interpolate: Linear mode requires 3D tensors.");
            }
        }
        if(mode == MIOPEN_INTERPOLATE_MODE_BILINEAR)
        {
            if(inputGradDesc.GetNumDims() != 4)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Interpolate: Bilinear mode requires 4D tensors.");
            }
        }
        if(mode == MIOPEN_INTERPOLATE_MODE_BICUBIC)
        {
            if(inputGradDesc.GetNumDims() != 4)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Interpolate: Bicubic mode requires 4D tensors.");
            }
        }
        if(mode == MIOPEN_INTERPOLATE_MODE_TRILINEAR)
        {
            if(inputGradDesc.GetNumDims() != 5)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Interpolate: Trilinear mode requires 5D tensors.");
            }
        }
        return true;
    }

    bool IsSameType() const
    {
        if(inputGradDesc.GetType() != outputGradDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Interpolate: Input grad and output grad tensor type do not match.");
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputGradDesc;
    TensorDescriptor outputGradDesc;
};

} // namespace interpolate

} // namespace miopen
