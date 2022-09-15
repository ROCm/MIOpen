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
    static auto GetGroupCountG(const ProblemDescription& problem) { return problem.group_counts; }

    static auto GetBatchN(const ProblemDescription& problem) { return problem.batch_sz; }

    static auto GetOutputLayout(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.out_layout;
        else
            return problem.in_layout;
    }

    static auto GetOutputChannelK(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.n_outputs;
        else
            return problem.n_inputs;
    }

    static auto GetInputLayout(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.in_layout;
        else
            return problem.out_layout;
    }

    static auto GetInputChannelC(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.n_inputs;
        else
            return problem.n_outputs;
    }

    static auto GetInputDepthDi(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.in_depth;
        else
            return problem.out_depth;
    }

    static auto GetInputHeightHi(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.in_height;
        else
            return problem.out_height;
    }

    static auto GetInputWidthWi(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.in_width;
        else
            return problem.out_width;
    }

    static auto GetOutputDepthDo(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.out_depth;
        else
            return problem.in_depth;
    }

    static auto GetOutputHeightHo(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.out_height;
        else
            return problem.in_height;
    }

    static auto GetOutputWidthWo(const ProblemDescription& problem)
    {
        if(problem.direction.IsForward())
            return problem.out_width;
        else
            return problem.in_width;
    }

    static auto GetOutputDataType(const ProblemDescription& problem)
    {
        return problem.direction.IsForward() ? problem.out_data_type : problem.in_data_type;
    }

    static auto GetInputDataType(const ProblemDescription& problem)
    {
        return problem.direction.IsForward() ? problem.in_data_type : problem.out_data_type;
    }

    static auto GetFilterDepthZ(const ProblemDescription& problem) { return problem.kernel_size_d; }

    static auto GetFilterLayout(const ProblemDescription& problem)
    {
        return problem.weights_layout;
    }

    static auto GetFilterHeightY(const ProblemDescription& problem)
    {
        return problem.kernel_size_h;
    }

    static auto GetFilterWidthX(const ProblemDescription& problem) { return problem.kernel_size_w; }

    // adjust conv_stride_d to 1 if Do is 1
    static auto GetAdjustedConvolutionStrideD(const ProblemDescription& problem)
    {
        return GetOutputDepthDo(problem) > 1 ? problem.kernel_stride_d : 1;
    }

    // adjust conv_stride_h to 1 if Ho is 1
    static auto GetAdjustedConvolutionStrideH(const ProblemDescription& problem)
    {
        return GetOutputHeightHo(problem) > 1 ? problem.kernel_stride_h : 1;
    }

    // adjust conv_stride_w to 1 if Wo is 1
    static auto GetAdjustedConvolutionStrideW(const ProblemDescription& problem)
    {
        return GetOutputWidthWo(problem) > 1 ? problem.kernel_stride_w : 1;
    }

    // adjust conv_dilation_d to 1 if Z is 1
    static auto GetAdjustedConvolutionDilationD(const ProblemDescription& problem)
    {
        return GetFilterDepthZ(problem) > 1 ? problem.kernel_dilation_d : 1;
    }

    // adjust conv_dilation_h to 1 if Y is 1
    static auto GetAdjustedConvolutionDilationH(const ProblemDescription& problem)
    {
        return GetFilterHeightY(problem) > 1 ? problem.kernel_dilation_h : 1;
    }

    // adjust conv_dilation_w to 1 if X is 1
    static auto GetAdjustedConvolutionDilationW(const ProblemDescription& problem)
    {
        return GetFilterWidthX(problem) > 1 ? problem.kernel_dilation_w : 1;
    }

    static auto GetInputLeftPadD(const ProblemDescription& problem) { return problem.pad_d; }

    static auto GetInputLeftPadH(const ProblemDescription& problem) { return problem.pad_h; }

    static auto GetInputLeftPadW(const ProblemDescription& problem) { return problem.pad_w; }

    // adjust right padding size so that filter will not move out-of-bound
    static auto GetAdjustedInputRightPadD(const ProblemDescription& problem)
    {
        int di              = GetInputDepthDi(problem);
        int dout            = GetOutputDepthDo(problem);
        int z               = GetFilterDepthZ(problem);
        int conv_stride_d   = GetAdjustedConvolutionStrideD(problem);
        int conv_dilation_d = GetAdjustedConvolutionDilationD(problem);
        int in_left_pad_d   = GetInputLeftPadD(problem);

        int di_padded = 1 + (z - 1) * conv_dilation_d + (dout - 1) * conv_stride_d;

        int in_right_pad_d =
            di_padded > (in_left_pad_d + di) ? di_padded - (in_left_pad_d + di) : 0;

        return in_right_pad_d;
    }

    // adjust right padding size so that filter will not move out-of-bound
    static auto GetAdjustedInputRightPadH(const ProblemDescription& problem)
    {
        int hi              = GetInputHeightHi(problem);
        int ho              = GetOutputHeightHo(problem);
        int y               = GetFilterHeightY(problem);
        int conv_stride_h   = GetAdjustedConvolutionStrideH(problem);
        int conv_dilation_h = GetAdjustedConvolutionDilationH(problem);
        int in_left_pad_h   = GetInputLeftPadH(problem);

        int hi_padded = 1 + (y - 1) * conv_dilation_h + (ho - 1) * conv_stride_h;

        int in_right_pad_h =
            hi_padded > (in_left_pad_h + hi) ? hi_padded - (in_left_pad_h + hi) : 0;

        return in_right_pad_h;
    }

    // adjust right padding size so that filter will not move out-of-bound
    static auto GetAdjustedInputRightPadW(const ProblemDescription& problem)
    {
        int wi              = GetInputWidthWi(problem);
        int wo              = GetOutputWidthWo(problem);
        int x               = GetFilterWidthX(problem);
        int conv_stride_w   = GetAdjustedConvolutionStrideW(problem);
        int conv_dilation_w = GetAdjustedConvolutionDilationW(problem);
        int in_left_pad_w   = GetInputLeftPadW(problem);

        int wi_padded = 1 + (x - 1) * conv_dilation_w + (wo - 1) * conv_stride_w;

        int in_right_pad_w =
            wi_padded > (in_left_pad_w + wi) ? wi_padded - (in_left_pad_w + wi) : 0;

        return in_right_pad_w;
    }
};

} // namespace solver
} // namespace miopen

#endif
