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

#include <cstdint>
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace UnsortedSegmentSum {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& InputDesc_,
                          const TensorDescriptor& OutputDesc_,
                          const TensorDescriptor& SegmentIdsDesc_,
                          const uint64_t num_segments_)
        : InputDesc(InputDesc_),
          OutputDesc(OutputDesc_),
          SegmentIdsDesc(SegmentIdsDesc_),
          num_segments(num_segments_)
    {
        IsSameType();
    }

    const TensorDescriptor& GetInputDesc() const { return InputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return OutputDesc; }
    const TensorDescriptor& GetSegmentIdsDesc() const { return SegmentIdsDesc; }
    uint64_t GetNumSegments() const { return num_segments; }

    bool IsSameType() const
    {
        if(InputDesc.GetType() != OutputDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "UnsortedSegmentSumForward: Tensor types do not match.");
        }
        return true;
    }

    bool IsAllContiguous() const
    {
        return InputDesc.IsContiguous() && OutputDesc.IsContiguous() &&
               SegmentIdsDesc.IsContiguous();
    }
    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor InputDesc;
    TensorDescriptor OutputDesc;
    TensorDescriptor SegmentIdsDesc;
    uint64_t num_segments;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& OutputGradDesc_,
                          const TensorDescriptor& InputGradDesc_,
                          const TensorDescriptor& SegmentIdsDesc_,
                          const uint64_t num_segments_)
        : OutputGradDesc(OutputGradDesc_),
          InputGradDesc(InputGradDesc_),
          SegmentIdsDesc(SegmentIdsDesc_),
          num_segments(num_segments_)
    {
        IsSameType();
    }

    const TensorDescriptor& GetInputGradDesc() const { return InputGradDesc; }
    const TensorDescriptor& GetOutputGradDesc() const { return OutputGradDesc; }
    const TensorDescriptor& GetSegmentIdsDesc() const { return SegmentIdsDesc; }
    uint64_t GetNumSegments() const { return num_segments; }

    bool IsSameType() const
    {
        if(InputGradDesc.GetType() != OutputGradDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "UnsortedSegmentSumForward: Tensor types do not match.");
        }
        return true;
    }

    bool IsAllContiguous() const
    {
        return InputGradDesc.IsContiguous() && OutputGradDesc.IsContiguous() &&
               SegmentIdsDesc.IsContiguous();
    }
    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor OutputGradDesc;
    TensorDescriptor InputGradDesc;
    TensorDescriptor SegmentIdsDesc;
    uint64_t num_segments;
};

} // namespace UnsortedSegmentSum

} // namespace miopen
