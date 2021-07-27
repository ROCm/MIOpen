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

#ifndef GUARD_CK_UTIL_HPP_
#define GUARD_CK_UTIL_HPP_

#include <miopen/env.hpp>
#include <miopen/hip_build_utils.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/rocm_features.hpp>
#include <algorithm>

#include "../composable_kernel/composable_kernel/include/utility/data_type_enum.hpp"
#include "../composable_kernel/host/driver_online/include/convolution_problem_descriptor.hpp"
#include "../composable_kernel/host/driver_online/include/online_driver_common.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CK_USE_AMD_BUFFER_ADDRESSING)

namespace miopen {
namespace solver {

static inline bool is_composable_kernel_supported_hardware(const ConvolutionContext& c)
{
    return (StartsWith(c.GetStream().GetDeviceName(), "gfx803") &&
            c.GetStream().GetMaxComputeUnits() == 64) ||
           StartsWith(c.GetStream().GetDeviceName(), "gfx900") ||
           StartsWith(c.GetStream().GetDeviceName(), "gfx906") ||
           StartsWith(c.GetStream().GetDeviceName(), "gfx908") ||
           StartsWith(c.GetStream().GetDeviceName(), "gfx90a") ||
           StartsWith(c.GetStream().GetDeviceName(), "gfx1030");
}

static inline bool support_amd_buffer_atomic_fadd(const std::string& device_name)
{
    return StartsWith(device_name, "gfx908");
}

static inline auto get_ck_common_compiler_flag(const ConvolutionContext& ctx)
{
    auto compiler_flag = std::string(" --std=c++17");

    // atomic-fadd
    compiler_flag += std::string(" -DCK_USE_AMD_BUFFER_ATOMIC_FADD=") +
                     (support_amd_buffer_atomic_fadd(ctx.GetStream().GetDeviceName()) ? '1' : '0');

    compiler_flag +=
        std::string(" -DCK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM=") +
        (miopen::IsDisabled(MIOPEN_DEBUG_CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM{}) ? '0' : '1');

    // enable or disable buffer load/store
    compiler_flag += std::string(" -DCK_USE_AMD_BUFFER_ADDRESSING=") +
                     (miopen::IsDisabled(MIOPEN_DEBUG_CK_USE_AMD_BUFFER_ADDRESSING{}) ? '0' : '1');

    return compiler_flag;
}

// 1. get the original dimension of conv problem
//    (undo the dimeniosn swapping happened inside ConvolutionContext)
// 2. adjust right padding size to align with the way implicit GEMM deal with padding
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

    // adjust right padding size to align with the way implicit GEMM deal with padding
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

    // adjust right padding size to align with the way implicit GEMM deal with padding
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

    // adjust right padding size to align with the way implicit GEMM deal with padding
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

static inline auto get_ck_convolution_problem_descriptor(const ConvolutionContext& ctx)
{
    ck::DataTypeEnum_t ck_datatype;

    if(ctx.IsFp32())
        ck_datatype = ck::DataTypeEnum_t::Float;
    else if(ctx.IsFp16())
        ck_datatype = ck::DataTypeEnum_t::Half;
    else if(ctx.IsBfp16())
        ck_datatype = ck::DataTypeEnum_t::BFloat16;
    else
        ck_datatype = ck::DataTypeEnum_t::Unknown;

    return ck_driver::ConvolutionProblemDescriptor{
        ConvolutionContextInterpreter::GetBatchN(ctx),
        ConvolutionContextInterpreter::GetOutputChannelK(ctx),
        ConvolutionContextInterpreter::GetInputChannelC(ctx),
        ConvolutionContextInterpreter::GetFilterHeightY(ctx),
        ConvolutionContextInterpreter::GetFilterWidthX(ctx),
        ConvolutionContextInterpreter::GetInputHeightHi(ctx),
        ConvolutionContextInterpreter::GetInputWidthWi(ctx),
        ConvolutionContextInterpreter::GetOutputHeightHo(ctx),
        ConvolutionContextInterpreter::GetOutputWidthWo(ctx),
        ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx),
        ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx),
        ConvolutionContextInterpreter::GetAdjustedConvolutionDilationH(ctx),
        ConvolutionContextInterpreter::GetAdjustedConvolutionDilationW(ctx),
        ConvolutionContextInterpreter::GetInputLeftPadH(ctx),
        ConvolutionContextInterpreter::GetInputLeftPadW(ctx),
        ConvolutionContextInterpreter::GetAdjustedInputRightPadH(ctx),
        ConvolutionContextInterpreter::GetAdjustedInputRightPadW(ctx),
        ck_datatype,
        ck_datatype,
        ck_datatype};
}

} // namespace solver
} // namespace miopen

#endif
