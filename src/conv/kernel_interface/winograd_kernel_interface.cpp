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

#include <miopen/conv/kernel_interface/winograd_kernel_interface.hpp>

#include <miopen/buffer_info.hpp>
#include <miopen/conv/problem_description.hpp>

namespace miopen {

namespace {

template <class Tdst, class Tsrc>
bool AssignAndCheck(Tdst& dst_v, Tsrc src_v) noexcept
{
    static_assert(std::is_integral_v<Tsrc>);
    static_assert(std::is_integral_v<Tdst>);

    dst_v = src_v;

    if(dst_v != src_v)
        return false;

    if constexpr(std::numeric_limits<Tsrc>::is_signed)
    {
        if constexpr(std::numeric_limits<Tdst>::is_signed)
            return (dst_v >= 0 && src_v >= 0) || (dst_v < 0 && src_v < 0);
        else
            return src_v >= 0;
    }
    else if constexpr(std::numeric_limits<Tdst>::is_signed)
    {
        return dst_v >= 0;
    }

    return true;
}

} // namespace

bool WinoShaderArgsV2::SetConvParams(const conv::ProblemDescription& problem)
{
    if(!problem.Is2d())
        return false;
    if(problem.GetBias() != 0)
        return false;
    if(!(problem.GetInStrideW() == 1 && problem.GetWeightsStrideW() == 1 &&
         problem.GetOutStrideW() == 1))
    {
        return false;
    }

    if(!AssignAndCheck(G, problem.GetGroupCount()))
        return false;

    const auto in_c_per_group  = problem.GetInChannels() / G;
    const auto out_c_per_group = problem.GetOutChannels() / G;

    if(!problem.IsDirectionBackwardWrW())
    {
        if(!AssignAndCheck(N, problem.GetBatchSize()))
            return false;
        if(!AssignAndCheck(Cg, in_c_per_group))
            return false;
        if(!AssignAndCheck(H, problem.GetInHeight()))
            return false;
        if(!AssignAndCheck(W, problem.GetInWidth()))
            return false;
        if(!AssignAndCheck(Kg, out_c_per_group))
            return false;
        if(!AssignAndCheck(R, problem.GetWeightsHeight()))
            return false;
        if(!AssignAndCheck(S, problem.GetWeightsWidth()))
            return false;
        if(!AssignAndCheck(out_h, problem.GetOutHeight()))
            return false;
        if(!AssignAndCheck(out_w, problem.GetOutWidth()))
            return false;
    }
    else
    {
        if(!AssignAndCheck(N, out_c_per_group))
            return false;
        if(!AssignAndCheck(Cg, problem.GetBatchSize()))
            return false;
        if(!AssignAndCheck(H, problem.GetOutHeight()))
            return false;
        if(!AssignAndCheck(W, problem.GetOutWidth()))
            return false;
        if(!AssignAndCheck(Kg, in_c_per_group))
            return false;
        if(!AssignAndCheck(R, problem.GetInHeight()))
            return false;
        if(!AssignAndCheck(S, problem.GetInWidth()))
            return false;
        if(!AssignAndCheck(out_h, problem.GetWeightsHeight()))
            return false;
        if(!AssignAndCheck(out_w, problem.GetWeightsWidth()))
            return false;
    }

    if(!problem.IsDirectionBackwardData())
    {
        if(!AssignAndCheck(pad_h, problem.GetPadH()))
            return false;
        if(!AssignAndCheck(pad_w, problem.GetPadW()))
            return false;
    }
    else
    {
        if(!AssignAndCheck(pad_h, problem.GetBackwardPadH()))
            return false;
        if(!AssignAndCheck(pad_w, problem.GetBackwardPadW()))
            return false;
    }

    if(problem.GetInBatchStride() > std::numeric_limits<uint32_t>::max() ||
       problem.GetInChannelStride() > std::numeric_limits<uint32_t>::max() ||
       problem.GetInStrideH() > std::numeric_limits<uint32_t>::max())
        return false;
    if(problem.GetWeightsStrideK() > std::numeric_limits<uint32_t>::max() ||
       problem.GetWeightsStrideC() > std::numeric_limits<uint32_t>::max() ||
       problem.GetWeightsStrideH() > std::numeric_limits<uint32_t>::max())
        return false;
    if(problem.GetOutBatchStride() > std::numeric_limits<uint32_t>::max() ||
       problem.GetOutChannelStride() > std::numeric_limits<uint32_t>::max() ||
       problem.GetOutStrideH() > std::numeric_limits<uint32_t>::max())
        return false;

    return true;
}

void WinoShaderArgsV2::SetStrides(const conv::ProblemDescription& problem)
{
    MemLayout_t d_layout, o_layout, f_layout;

    if(!problem.IsDirectionBackwardWrW())
    {
        d_layout = GetGroupConvLayout(GetMemLayout_t(problem.GetInLayout()), true);
        o_layout = GetGroupConvLayout(GetMemLayout_t(problem.GetOutLayout()), true);
        // clang-format off
        f_layout = GetGroupConvLayout(problem.IsDirectionForward() ? MemLayout_t::NCHW
                                                                   : GetSwappedNCLayout(MemLayout_t::NCHW), false);
        // clang-format on
    }
    else
    {
        d_layout =
            GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(problem.GetInLayout())), true);
        o_layout =
            GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(problem.GetOutLayout())), false);
        f_layout = GetGroupConvLayout(GetSwappedNCLayout(MemLayout_t::NCHW), true);
    }

    // TODO Make a constructor that takes unsigned int
    BuffInfo d_buf(d_layout, N, Cg, H, W, G, GetTypeSize(problem.GetInDataType()));
    BuffInfo o_buf(o_layout, N, Kg, out_h, out_w, G, GetTypeSize(problem.GetOutDataType()));
    BuffInfo f_buf(f_layout, Kg, Cg, R, S, G, GetTypeSize(problem.GetWeightsDataType()));

    const auto& d_strides = d_buf.stride;
    const auto& f_strides = f_buf.stride;
    const auto& o_strides = o_buf.stride;

    d_N_stride = d_strides.nk;
    d_C_stride = d_strides.c;
    d_H_stride = d_strides.h;
    d_G_stride = d_strides.g;

    f_K_stride = f_strides.nk;
    f_C_stride = f_strides.c;
    f_R_stride = f_strides.h;
    f_G_stride = f_strides.g;

    o_N_stride = o_strides.nk;
    o_K_stride = o_strides.c;
    o_H_stride = o_strides.h;
    o_G_stride = o_strides.g;
}

void WinoShaderArgsV2::SetActivParams(miopenActivationMode_t mode)
{
    // Fused activation parameters
    // clang-format off
    switch(mode)
    {
    case miopenActivationPASTHRU:
        activation_mode = WinoShaderActivationModeV2_t::IDENTITY;
        break;
    case miopenActivationLOGISTIC:
        activation_mode = WinoShaderActivationModeV2_t::SIGMOID;
        break;
    case miopenActivationTANH:
        activation_mode = WinoShaderActivationModeV2_t::SCALED_TANH;
        break;
    case miopenActivationLEAKYRELU:
        activation_mode = WinoShaderActivationModeV2_t::LEAKY_RELU;
        break;
    case miopenActivationRELU:
        activation_mode = WinoShaderActivationModeV2_t::RELU;
        break;
    default:
        MIOPEN_THROW(miopenStatusInternalError);
    }
    // clang-format on
}

void WinoShaderArgsV2::SetShaderParams(uint32_t n_groups_,
                                       WinoShaderFlagsV2 flags_,
                                       uint8_t sync_limit_,
                                       uint8_t sync_period_) noexcept
{
    n_groups    = n_groups_;
    flags64     = flags_;
    sync_limit  = sync_limit_;
    sync_period = sync_period_;
}

} // namespace miopen
