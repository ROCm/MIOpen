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
        if(mode > 5)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Interpolate: Invalid mode.");
        }
        return true;
    }

    bool IsValidStride(TensorDescriptor td) const
    {
        auto strides = td.GetStrides();
        auto lengths = td.GetLengths();
        std::vector<std::pair<size_t, size_t>> p;
        p.reserve(td.GetSize());
        std::transform(strides.begin(),
                       strides.end(),
                       lengths.begin(),
                       std::back_inserter(p),
                       [](size_t a, size_t b) { return std::make_pair(a, b); });
        std::sort(p.begin(), p.end());
        for(int i = 1; i < p.size(); ++i)
        {
            if(p[i].first != p[i - 1].first * p[i - 1].second)
                MIOPEN_THROW(miopenStatusBadParm, "Interpolate: Tensor strides do not valid.");
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
        IsValidLength();
        IsAllValidStride();
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
                         "Interpolate: Output tensor size < 1 or > 3 is not valid.");
        }

        if(outputDesc.GetSize() != scaleFactorsDesc.GetElementSize())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Interpolate: Output tensor size and scale factors length do not match.");
        }
        return true;
    }

    bool IsAllValidStride() const { return IsValidStride(inputDesc) && IsValidStride(outputDesc); }

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
        IsValidLength();
        IsAllValidStride();
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
            MIOPEN_THROW(
                miopenStatusBadParm,
                "Interpolate: Output grad tensor size and scale factors length do not match.");
        }
        return true;
    }

    bool IsAllValidStride() const
    {
        return IsValidStride(inputGradDesc) && IsValidStride(outputGradDesc);
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputGradDesc;
    TensorDescriptor outputGradDesc;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace interpolate

} // namespace miopen
