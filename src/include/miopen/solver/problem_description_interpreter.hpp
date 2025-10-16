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

#include <miopen/conv/problem_description.hpp>

namespace miopen {
namespace solver {

// 1. get the original dimension of conv problem
//    (undo the dimension swapping and tensor swapping happened inside ProblemDescription)
// 2. adjust right padding size so that filter will not move out-of-bound
// 3. adjust stride to 1 if output image size is 1
// 4. adjust dilation to 1 if filter size is 1
struct ProblemInterpreter
{
    static auto GetGroupCountG(const miopen::conv::ProblemDescription& problem)
    {
        return problem.GetGroupCount();
    }

    static int GetBatchN(const miopen::conv::ProblemDescription& problem)
    {
        return problem.GetBatchSize();
    }

    static auto GetOutputLayout(const miopen::conv::ProblemDescription& problem)
    {
        if(problem.IsDirectionForward())
            return problem.GetOutLayout();
        else
            return problem.GetInLayout();
    }

    static int GetOutputChannelK(const miopen::conv::ProblemDescription& problem)
    {
        if(problem.IsDirectionForward())
            return problem.GetOutChannels();
        else
            return problem.GetInChannels();
    }

    static auto GetInputLayout(const miopen::conv::ProblemDescription& problem)
    {
        if(problem.IsDirectionForward())
            return problem.GetInLayout();
        else
            return problem.GetOutLayout();
    }

    static int GetInputChannelC(const miopen::conv::ProblemDescription& problem)
    {
        if(problem.IsDirectionForward())
            return problem.GetInChannels();
        else
            return problem.GetOutChannels();
    }

    static int GetInputDepthDi(const miopen::conv::ProblemDescription& problem)
    {
        if(problem.IsDirectionForward())
            return problem.GetInDepth();
        else
            return problem.GetOutDepth();
    }

    static int GetInputHeightHi(const miopen::conv::ProblemDescription& problem)
    {
        if(problem.IsDirectionForward())
            return problem.GetInHeight();
        else
            return problem.GetOutHeight();
    }

    static int GetInputWidthWi(const miopen::conv::ProblemDescription& problem)
    {
        if(problem.IsDirectionForward())
            return problem.GetInWidth();
        else
            return problem.GetOutWidth();
    }

    static auto GetInputCastType(const miopen::conv::ProblemDescription& problem)
    {
        if(problem.IsDirectionForward())
            return problem.GetInCastType();
        else
            return problem.GetOutCastType();
    }

    static int GetOutputDepthDo(const miopen::conv::ProblemDescription& problem)
    {
        if(problem.IsDirectionForward())
            return problem.GetOutDepth();
        else
            return problem.GetInDepth();
    }

    static int GetOutputHeightHo(const miopen::conv::ProblemDescription& problem)
    {
        if(problem.IsDirectionForward())
            return problem.GetOutHeight();
        else
            return problem.GetInHeight();
    }

    static int GetOutputWidthWo(const miopen::conv::ProblemDescription& problem)
    {
        if(problem.IsDirectionForward())
            return problem.GetOutWidth();
        else
            return problem.GetInWidth();
    }

    static auto GetOutputCastType(const miopen::conv::ProblemDescription& problem)
    {
        if(problem.IsDirectionForward())
            return problem.GetOutCastType();
        else
            return problem.GetInCastType();
    }

    static auto GetOutputDataType(const miopen::conv::ProblemDescription& problem)
    {
        return problem.IsDirectionForward() ? problem.GetOutDataType() : problem.GetInDataType();
    }

    static auto GetInputDataType(const miopen::conv::ProblemDescription& problem)
    {
        return problem.IsDirectionForward() ? problem.GetInDataType() : problem.GetOutDataType();
    }

    static auto GetWeightsDataType(const miopen::conv::ProblemDescription& problem)
    {
        return problem.GetWeightsDataType();
    }

    static int GetFilterDepthZ(const miopen::conv::ProblemDescription& problem)
    {
        return problem.GetWeightsDepth();
    }

    static auto GetFilterLayout(const miopen::conv::ProblemDescription& problem)
    {
        return problem.GetWeightsLayout();
    }

    static int GetFilterHeightY(const miopen::conv::ProblemDescription& problem)
    {
        return problem.GetWeightsHeight();
    }

    static int GetFilterWidthX(const miopen::conv::ProblemDescription& problem)
    {
        return problem.GetWeightsWidth();
    }

    // adjust conv_stride_d to 1 if Do is 1
    static auto GetAdjustedConvolutionStrideD(const miopen::conv::ProblemDescription& problem)
    {
        return GetOutputDepthDo(problem) > 1 ? problem.GetKernelStrideD() : 1;
    }

    // adjust conv_stride_h to 1 if Ho is 1
    static auto GetAdjustedConvolutionStrideH(const miopen::conv::ProblemDescription& problem)
    {
        return GetOutputHeightHo(problem) > 1 ? problem.GetKernelStrideH() : 1;
    }

    // adjust conv_stride_w to 1 if Wo is 1
    static auto GetAdjustedConvolutionStrideW(const miopen::conv::ProblemDescription& problem)
    {
        return GetOutputWidthWo(problem) > 1 ? problem.GetKernelStrideW() : 1;
    }

    // adjust conv_dilation_d to 1 if Z is 1
    static auto GetAdjustedConvolutionDilationD(const miopen::conv::ProblemDescription& problem)
    {
        return GetFilterDepthZ(problem) > 1 ? problem.GetDilationD() : 1;
    }

    // adjust conv_dilation_h to 1 if Y is 1
    static auto GetAdjustedConvolutionDilationH(const miopen::conv::ProblemDescription& problem)
    {
        return GetFilterHeightY(problem) > 1 ? problem.GetDilationH() : 1;
    }

    // adjust conv_dilation_w to 1 if X is 1
    static auto GetAdjustedConvolutionDilationW(const miopen::conv::ProblemDescription& problem)
    {
        return GetFilterWidthX(problem) > 1 ? problem.GetDilationW() : 1;
    }

    static auto GetInputLeftPadD(const miopen::conv::ProblemDescription& problem)
    {
        return problem.GetPadD();
    }

    static auto GetInputLeftPadH(const miopen::conv::ProblemDescription& problem)
    {
        return problem.GetPadH();
    }

    static auto GetInputLeftPadW(const miopen::conv::ProblemDescription& problem)
    {
        return problem.GetPadW();
    }

    static const Scalar& GetAlpha(const miopen::conv::ProblemDescription& problem)
    {
        return problem.GetAlpha();
    }

    static const Scalar& GetBeta(const miopen::conv::ProblemDescription& problem)
    {
        return problem.GetBeta();
    }
    static miopenAlphaBetaCase_t GetAlphaBetaCase(const miopen::conv::ProblemDescription& problem)
    {
        return problem.GetAlphaBetaCase();
    }

    // adjust right padding size so that filter will not move out-of-bound
    static auto GetAdjustedInputRightPadD(const miopen::conv::ProblemDescription& problem)
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
    static auto GetAdjustedInputRightPadH(const miopen::conv::ProblemDescription& problem)
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
    static auto GetAdjustedInputRightPadW(const miopen::conv::ProblemDescription& problem)
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

    /// @brief Get adjusted stride for ASM
    /// @param problem Convolution problem description
    /// @return Adjusted stride value
    /// @note If the output dim is greater than 1, return the original kernel stride. Otherwise,
    /// calculate the total padding and right padding, and return the adjusted stride if the
    /// right padding is non-negative (valid).
    static auto GetAdjustedAsmInputStrideH(const miopen::conv::ProblemDescription& problem)
    {
        const int ho = GetOutputHeightHo(problem);

        if(ho > 1)
            return problem.GetKernelStrideH();

        const int hi              = GetInputHeightHi(problem);
        const int y               = GetFilterHeightY(problem);
        const int conv_stride_h   = GetAdjustedConvolutionStrideH(problem);
        const int conv_dilation_h = GetAdjustedConvolutionDilationH(problem);

        // For ho == 1: total padding = conv_dilation_h * (y - 1) - hi + 1;
        // (General formula: (ho - 1) * conv_stride_h - hi + conv_dilation_h * (y - 1) + 1)
        const int hi_tot_pad   = conv_dilation_h * (y - 1) - hi + 1;
        const int hi_right_pad = hi_tot_pad - problem.GetPadH();

        return (hi_right_pad >= 0) ? conv_stride_h : problem.GetKernelStrideH();
    }

    /// @brief Get adjusted stride for ASM
    /// @param problem Convolution problem description
    /// @return Adjusted stride value
    /// @note If the output dim is greater than 1, return the original kernel stride. Otherwise,
    /// calculate the total padding and right padding, and return the adjusted stride if the
    /// right padding is non-negative (valid).
    static auto GetAdjustedAsmInputStrideW(const miopen::conv::ProblemDescription& problem)
    {
        const int wo = GetOutputWidthWo(problem);

        if(wo > 1)
            return problem.GetKernelStrideW();

        const int wi              = GetInputWidthWi(problem);
        const int x               = GetFilterWidthX(problem);
        const int conv_stride_w   = GetAdjustedConvolutionStrideW(problem);
        const int conv_dilation_w = GetAdjustedConvolutionDilationW(problem);

        // For wo == 1: total padding = conv_dilation_w * (x - 1) - wi + 1;
        // (General formula: (wo - 1) * conv_stride_w - wi + conv_dilation_w * (x - 1) + 1)
        const int wi_tot_pad   = conv_dilation_w * (x - 1) - wi + 1;
        const int wi_right_pad = wi_tot_pad - problem.GetPadW();

        return (wi_right_pad >= 0) ? conv_stride_w : problem.GetKernelStrideW();
    }
};

} // namespace solver
} // namespace miopen

#endif
