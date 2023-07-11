/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#ifndef PROBLEM_DESCRIPTION_INTERPRETER_HPP_
#define PROBLEM_DESCRIPTION_INTERPRETER_HPP_

#include <miopen/env.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/rocm_features.hpp>
#include <algorithm>

namespace miopen {
namespace solver {

// 1. get the original dimension of conv problem
//    (undo the dimension swapping and tensor swapping happened inside ProblemDescription)
// 2. adjust right padding size so that filter will not move out-of-bound
// 3. adjust stride to 1 if output image size is 1
// 4. adjust dilation to 1 if filter size is 1
struct ProblemInterpreter
{
    static auto GetGroupCountG(const ProblemDescription& problem)
    {
        return problem.GetGroupCount();
    }

    static auto GetBatchN(const ProblemDescription& problem) { return problem.GetBatchSize2(); }

    static auto GetOutputLayout(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.GetOutLayout();
        else
            return problem.GetInLayout();
    }

    static auto GetOutputChannelK(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.GetOutChannels2();
        else
            return problem.GetInChannels2();
    }

    static auto GetInputLayout(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.GetInLayout();
        else
            return problem.GetOutLayout();
    }

    static auto GetInputChannelC(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.GetInChannels2();
        else
            return problem.GetOutChannels2();
    }

    static auto GetInputDepthDi(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.GetInDepth2();
        else
            return problem.GetOutDepth2();
    }

    static auto GetInputHeightHi(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.GetInHeight2();
        else
            return problem.GetOutHeight2();
    }

    static auto GetInputWidthWi(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.GetInWidth2();
        else
            return problem.GetOutWidth2();
    }

    static auto GetOutputDepthDo(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.GetOutDepth2();
        else
            return problem.GetInDepth2();
    }

    static auto GetOutputHeightHo(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.GetOutHeight2();
        else
            return problem.GetInHeight2();
    }

    static auto GetOutputWidthWo(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.GetOutWidth2();
        else
            return problem.GetInWidth2();
    }

    static auto GetOutputDataType(const ProblemDescription& problem)
    {
        return problem.direction.IsForward() ? problem.GetOutDataType() : problem.GetInDataType();
    }

    static auto GetInputDataType(const ProblemDescription& problem)
    {
        return problem.direction.IsForward() ? problem.GetInDataType() : problem.GetOutDataType();
    }

    static auto GetFilterDepthZ(const ProblemDescription& problem)
    {
        return problem.GetWeightsDepth2();
    }

    static auto GetFilterLayout(const ProblemDescription& problem)
    {
        return problem.GetWeightsLayout();
    }

    static auto GetFilterHeightY(const ProblemDescription& problem)
    {
        return problem.GetWeightsHeight2();
    }

    static auto GetFilterWidthX(const ProblemDescription& problem)
    {
        return problem.GetWeightsWidth2();
    }

    // adjust conv_stride_d to 1 if Do is 1
    static auto GetAdjustedConvolutionStrideD(const ProblemDescription& problem)
    {
        return GetOutputDepthDo(problem) > 1 ? problem.GetKernelStrideD() : 1;
    }

    // adjust conv_stride_h to 1 if Ho is 1
    static auto GetAdjustedConvolutionStrideH(const ProblemDescription& problem)
    {
        return GetOutputHeightHo(problem) > 1 ? problem.GetKernelStrideH() : 1;
    }

    // adjust conv_stride_w to 1 if Wo is 1
    static auto GetAdjustedConvolutionStrideW(const ProblemDescription& problem)
    {
        return GetOutputWidthWo(problem) > 1 ? problem.GetKernelStrideW() : 1;
    }

    // adjust conv_dilation_d to 1 if Z is 1
    static auto GetAdjustedConvolutionDilationD(const ProblemDescription& problem)
    {
        return GetFilterDepthZ(problem) > 1 ? problem.GetDilationD() : 1;
    }

    // adjust conv_dilation_h to 1 if Y is 1
    static auto GetAdjustedConvolutionDilationH(const ProblemDescription& problem)
    {
        return GetFilterHeightY(problem) > 1 ? problem.GetDilationH() : 1;
    }

    // adjust conv_dilation_w to 1 if X is 1
    static auto GetAdjustedConvolutionDilationW(const ProblemDescription& problem)
    {
        return GetFilterWidthX(problem) > 1 ? problem.GetDilationW() : 1;
    }

    static auto GetInputLeftPadD(const ProblemDescription& problem) { return problem.GetPadD(); }

    static auto GetInputLeftPadH(const ProblemDescription& problem) { return problem.GetPadH(); }

    static auto GetInputLeftPadW(const ProblemDescription& problem) { return problem.GetPadW(); }

    // adjust right padding size so that filter will not move out-of-bound
    static auto GetAdjustedInputRightPadD(const ProblemDescription& problem)
    {
        const int di              = GetInputDepthDi(problem);
        const int dout            = GetOutputDepthDo(problem);
        const int z               = GetFilterDepthZ(problem);
        const int conv_stride_d   = GetAdjustedConvolutionStrideD(problem);
        const int conv_dilation_d = GetAdjustedConvolutionDilationD(problem);
        const int in_left_pad_d   = GetInputLeftPadD(problem);

        const int di_padded = 1 + (z - 1) * conv_dilation_d + (dout - 1) * conv_stride_d;

        const int in_right_pad_d =
            di_padded > (in_left_pad_d + di) ? di_padded - (in_left_pad_d + di) : 0;

        return in_right_pad_d;
    }

    // adjust right padding size so that filter will not move out-of-bound
    static auto GetAdjustedInputRightPadH(const ProblemDescription& problem)
    {
        const int hi              = GetInputHeightHi(problem);
        const int ho              = GetOutputHeightHo(problem);
        const int y               = GetFilterHeightY(problem);
        const int conv_stride_h   = GetAdjustedConvolutionStrideH(problem);
        const int conv_dilation_h = GetAdjustedConvolutionDilationH(problem);
        const int in_left_pad_h   = GetInputLeftPadH(problem);

        const int hi_padded = 1 + (y - 1) * conv_dilation_h + (ho - 1) * conv_stride_h;

        const int in_right_pad_h =
            hi_padded > (in_left_pad_h + hi) ? hi_padded - (in_left_pad_h + hi) : 0;

        return in_right_pad_h;
    }

    // adjust right padding size so that filter will not move out-of-bound
    static auto GetAdjustedInputRightPadW(const ProblemDescription& problem)
    {
        const int wi              = GetInputWidthWi(problem);
        const int wo              = GetOutputWidthWo(problem);
        const int x               = GetFilterWidthX(problem);
        const int conv_stride_w   = GetAdjustedConvolutionStrideW(problem);
        const int conv_dilation_w = GetAdjustedConvolutionDilationW(problem);
        const int in_left_pad_w   = GetInputLeftPadW(problem);

        const int wi_padded = 1 + (x - 1) * conv_dilation_w + (wo - 1) * conv_stride_w;

        const int in_right_pad_w =
            wi_padded > (in_left_pad_w + wi) ? wi_padded - (in_left_pad_w + wi) : 0;

        return in_right_pad_w;
    }
};

} // namespace solver
} // namespace miopen

#endif
