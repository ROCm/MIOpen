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

#ifndef GUARD_CONVOLUTION_CONTEXT_INTERPRETER_HPP_
#define GUARD_CONVOLUTION_CONTEXT_INTERPRETER_HPP_

#include <miopen/env.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/rocm_features.hpp>
#include <algorithm>

namespace miopen {
namespace solver {

// 1. get the original dimension of conv problem
//    (undo the dimeniosn swapping and tensor swapping happened inside ConvolutionContext)
// 2. adjust right padding size so that filter will not move out-of-bound
// 3. adjust stride to 1 if output image size is 1
// 4. adjust dilation to 1 if filter size is 1
struct ConvolutionContextInterpreter
{
    static auto GetGroupCountG(const ConvolutionContext& c) { return c.group_counts; }

    static auto GetBatchN(const ConvolutionContext& c) { return c.batch_sz; }

    static auto GetOutputLayout(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.out_layout;
        else
            return c.in_layout;
    }

    static auto GetOutputChannelK(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.n_outputs;
        else
            return c.n_inputs;
    }

    static auto GetInputLayout(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.in_layout;
        else
            return c.out_layout;
    }

    static auto GetInputChannelC(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.n_inputs;
        else
            return c.n_outputs;
    }

    static auto GetInputDepthDi(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.in_depth;
        else
            return c.out_depth;
    }

    static auto GetInputHeightHi(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.in_height;
        else
            return c.out_height;
    }

    static auto GetInputWidthWi(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.in_width;
        else
            return c.out_width;
    }

    static auto GetOutputDepthDo(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.out_depth;
        else
            return c.in_depth;
    }

    static auto GetOutputHeightHo(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.out_height;
        else
            return c.in_height;
    }

    static auto GetOutputWidthWo(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.out_width;
        else
            return c.in_width;
    }

    static auto GetOutputDataType(const ConvolutionContext& c)
    {
        return c.direction.IsForward() ? c.out_data_type : c.in_data_type;
    }

    static auto GetInputDataType(const ConvolutionContext& c)
    {
        return c.direction.IsForward() ? c.in_data_type : c.out_data_type;
    }

    static auto GetFilterDepthZ(const ConvolutionContext& c) { return c.kernel_size_d; }

    static auto GetFilterLayout(const ConvolutionContext& c) { return c.weights_layout; }

    static auto GetFilterHeightY(const ConvolutionContext& c) { return c.kernel_size_h; }

    static auto GetFilterWidthX(const ConvolutionContext& c) { return c.kernel_size_w; }

    // adjust conv_stride_d to 1 if Do is 1
    static auto GetAdjustedConvolutionStrideD(const ConvolutionContext& c)
    {
        return GetOutputDepthDo(c) > 1 ? c.kernel_stride_d : 1;
    }

    // adjust conv_stride_h to 1 if Ho is 1
    static auto GetAdjustedConvolutionStrideH(const ConvolutionContext& c)
    {
        return GetOutputHeightHo(c) > 1 ? c.kernel_stride_h : 1;
    }

    // adjust conv_stride_w to 1 if Wo is 1
    static auto GetAdjustedConvolutionStrideW(const ConvolutionContext& c)
    {
        return GetOutputWidthWo(c) > 1 ? c.kernel_stride_w : 1;
    }

    // adjust conv_dilation_d to 1 if Z is 1
    static auto GetAdjustedConvolutionDilationD(const ConvolutionContext& c)
    {
        return GetFilterDepthZ(c) > 1 ? c.kernel_dilation_d : 1;
    }

    // adjust conv_dilation_h to 1 if Y is 1
    static auto GetAdjustedConvolutionDilationH(const ConvolutionContext& c)
    {
        return GetFilterHeightY(c) > 1 ? c.kernel_dilation_h : 1;
    }

    // adjust conv_dilation_w to 1 if X is 1
    static auto GetAdjustedConvolutionDilationW(const ConvolutionContext& c)
    {
        return GetFilterWidthX(c) > 1 ? c.kernel_dilation_w : 1;
    }

    static auto GetInputLeftPadD(const ConvolutionContext& c) { return c.pad_d; }

    static auto GetInputLeftPadH(const ConvolutionContext& c) { return c.pad_h; }

    static auto GetInputLeftPadW(const ConvolutionContext& c) { return c.pad_w; }

    // adjust right padding size so that filter will not move out-of-bound
    static auto GetAdjustedInputRightPadD(const ConvolutionContext& c)
    {
        int di              = GetInputDepthDi(c);
        int dout            = GetOutputDepthDo(c);
        int z               = GetFilterDepthZ(c);
        int conv_stride_d   = GetAdjustedConvolutionStrideD(c);
        int conv_dilation_d = GetAdjustedConvolutionDilationD(c);
        int in_left_pad_d   = GetInputLeftPadD(c);

        int di_padded = 1 + (z - 1) * conv_dilation_d + (dout - 1) * conv_stride_d;

        int in_right_pad_d =
            di_padded > (in_left_pad_d + di) ? di_padded - (in_left_pad_d + di) : 0;

        return in_right_pad_d;
    }

    // adjust right padding size so that filter will not move out-of-bound
    static auto GetAdjustedInputRightPadH(const ConvolutionContext& c)
    {
        int hi              = GetInputHeightHi(c);
        int ho              = GetOutputHeightHo(c);
        int y               = GetFilterHeightY(c);
        int conv_stride_h   = GetAdjustedConvolutionStrideH(c);
        int conv_dilation_h = GetAdjustedConvolutionDilationH(c);
        int in_left_pad_h   = GetInputLeftPadH(c);

        int hi_padded = 1 + (y - 1) * conv_dilation_h + (ho - 1) * conv_stride_h;

        int in_right_pad_h =
            hi_padded > (in_left_pad_h + hi) ? hi_padded - (in_left_pad_h + hi) : 0;

        return in_right_pad_h;
    }

    // adjust right padding size so that filter will not move out-of-bound
    static auto GetAdjustedInputRightPadW(const ConvolutionContext& c)
    {
        int wi              = GetInputWidthWi(c);
        int wo              = GetOutputWidthWo(c);
        int x               = GetFilterWidthX(c);
        int conv_stride_w   = GetAdjustedConvolutionStrideW(c);
        int conv_dilation_w = GetAdjustedConvolutionDilationW(c);
        int in_left_pad_w   = GetInputLeftPadW(c);

        int wi_padded = 1 + (x - 1) * conv_dilation_w + (wo - 1) * conv_stride_w;

        int in_right_pad_w =
            wi_padded > (in_left_pad_w + wi) ? wi_padded - (in_left_pad_w + wi) : 0;

        return in_right_pad_w;
    }
};

} // namespace solver
} // namespace miopen

#endif
