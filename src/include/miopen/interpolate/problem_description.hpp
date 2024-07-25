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
#include <miopen/problem_description_base.hpp>
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>
#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace interpolate {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& scaleFactorsDesc_,
                       const miopenInterpolateMode_t mode_,
                       const bool align_corners_,
                       bool is_fwd_)
        : scaleFactorsDesc(scaleFactorsDesc_),
          mode(mode_),
          align_corners(align_corners_),
          is_fwd(is_fwd_)
    {
        IsValidMode();
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
            std::cout << "MODE: " << mode << std::endl;
            MIOPEN_THROW(miopenStatusBadParm, "Interpolate: Invalid mode.");
        }
        return true;
    }

protected:
    TensorDescriptor scaleFactorsDesc;
    miopenInterpolateMode_t mode;
    bool align_corners = false;
    bool is_fwd;

    NetworkConfig MakeForwardNetworkConfig() const;
};

struct FwdProblemDescription : ProblemDescription
{
    FwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& outputDesc_,
                          const TensorDescriptor& scaleFactorsDesc_,
                          const miopenInterpolateMode_t mode_,
                          const bool align_corners_)
        : ProblemDescription(scaleFactorsDesc_, mode_, align_corners_, true)
    {
        inputDesc  = inputDesc_;
        outputDesc = outputDesc_;
        IsValidDims();
        IsValidLength();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    bool IsValidLength() const
    {
        if(inputDesc.GetSize() < 3 || inputDesc.GetSize() > 5)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Interpolate: Input tensor size < 3 or > 5 is not valid.");
        }

        if(outputDesc.GetSize() < 3 || outputDesc.GetSize() > 5)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Interpolate: Output tensor size < 3 or > 5 is not valid.");
        }

        if((outputDesc.GetSize() - 2) != scaleFactorsDesc.GetElementSize())
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
            if(inputDesc.GetSize() != 3 || outputDesc.GetSize() != 3)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Interpolate: Linear mode requires 3D tensors.");
            }
        }
        if(mode == MIOPEN_INTERPOLATE_MODE_BILINEAR)
        {
            if(inputDesc.GetSize() != 4 || outputDesc.GetSize() != 4)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Interpolate: Bilinear mode requires 4D tensors.");
            }
        }
        if(mode == MIOPEN_INTERPOLATE_MODE_BICUBIC)
        {
            if(inputDesc.GetSize() != 4 || outputDesc.GetSize() != 4)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Interpolate: Bicubic mode requires 4D tensors.");
            }
        }
        if(mode == MIOPEN_INTERPOLATE_MODE_TRILINEAR)
        {
            if(inputDesc.GetSize() != 5 || outputDesc.GetSize() != 5)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Interpolate: Trilinear mode requires 5D tensors.");
            }
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    NetworkConfig MakeForwardNetworkConfig() const;
};

struct BwdProblemDescription : ProblemDescription
{
    BwdProblemDescription(const TensorDescriptor& inputGradDesc_,
                          const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& scaleFactorsDesc_,
                          const miopenInterpolateMode_t mode_,
                          const bool align_corners_)
        : ProblemDescription(scaleFactorsDesc_, mode_, align_corners_, false)
    {
        inputGradDesc  = inputGradDesc_;
        outputGradDesc = outputGradDesc_;
        IsValidDims();
        IsValidLength();
    }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }

    bool IsValidLength() const
    {
        if(inputGradDesc.GetSize() < 3 || inputGradDesc.GetSize() > 5)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Interpolate: Input grad tensor size < 3 or > 5 is not valid.");
        }

        if(outputGradDesc.GetSize() < 3 || outputGradDesc.GetSize() > 5)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Interpolate: Output grad tensor size < 3 or > 5 is not valid.");
        }

        if((outputGradDesc.GetSize() - 2) != scaleFactorsDesc.GetElementSize())
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
            if(inputGradDesc.GetSize() != 3 || outputGradDesc.GetSize() != 3)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Interpolate: Linear mode requires 3D tensors.");
            }
        }
        if(mode == MIOPEN_INTERPOLATE_MODE_BILINEAR)
        {
            if(inputGradDesc.GetSize() != 4 || outputGradDesc.GetSize() != 4)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Interpolate: Bilinear mode requires 4D tensors.");
            }
        }
        if(mode == MIOPEN_INTERPOLATE_MODE_BICUBIC)
        {
            if(inputGradDesc.GetSize() != 4 || outputGradDesc.GetSize() != 4)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Interpolate: Bicubic mode requires 4D tensors.");
            }
        }
        if(mode == MIOPEN_INTERPOLATE_MODE_TRILINEAR)
        {
            if(inputGradDesc.GetSize() != 5 || outputGradDesc.GetSize() != 5)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Interpolate: Trilinear mode requires 5D tensors.");
            }
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputGradDesc;
    TensorDescriptor outputGradDesc;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace interpolate

} // namespace miopen
