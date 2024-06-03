/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include "mha_common.hpp"

#include <miopen/mha/solvers.hpp>
#include <miopen/mha/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>

#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue.hpp"

#include <algorithm>
#include <vector>
#include <tuple>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_ATTN_NAIVE_FWD)

namespace miopen {

namespace solver {

namespace mha {

namespace { 


bool MhaCKForward::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                              const miopen::mha::ProblemDescription& problem) const
{
    // It's important to have this check before problem.GetDescsForward() call
    if(!problem.IsForward())
    {
        return false;
    }

    const miopen::mha::MhaInputDescsForward& descsFwd = problem.GetDescsForward();

    auto [N, H, S, D] = miopen::tien<4>(descsFwd.kDesc.GetLengths());

    return MIOPEN_USE_GEMM                                                 //
           && !miopen::IsDisabled(MIOPEN_ENV(MIOPEN_DEBUG_ATTN_NAIVE_FWD)) //
           && S <= std::numeric_limits<uint32_t>::max()                    //
           && descsFwd.kDesc.IsPacked()                                    //
           && descsFwd.qDesc.IsPacked()                                    //
           && descsFwd.vDesc.IsPacked()                                    //
           && descsFwd.oDesc.IsPacked()                                    //
           && descsFwd.mDesc.IsPacked()                                    //
           && descsFwd.zInvDesc.IsPacked()                                 //
           && descsFwd.mDesc.GetType() == miopenFloat                      //
           && descsFwd.zInvDesc.GetType() == miopenFloat                   //
           && descsFwd.kDesc.GetType() == descsFwd.qDesc.GetType()         //
           && descsFwd.kDesc.GetType() == descsFwd.vDesc.GetType()         //
           && descsFwd.kDesc.GetType() == descsFwd.oDesc.GetType()         //
           && ((descsFwd.kDesc.GetType() == miopenFloat)                   //
               || (USE_ROCBLAS_EX3                                         //
                   && (MIOPEN_FP8_IEEE_EXPONENT_BIAS == 0)                 //
                   && (descsFwd.kDesc.GetType() == miopenFloat8)));        //
}

std::size_t MhaCKForward::GetWorkspaceSize([[maybe_unused]] const ExecutionContext& context,
                                         const miopen::mha::ProblemDescription& problem) const
{
    const auto& kDesc = problem.GetDescsForward().kDesc;
    return SplitBufferToWorkspace(kDesc.GetLengths(), kDesc.GetType()).GetSize();
}

ConvSolution MhaCKForward::GetSolution(const ExecutionContext& context,
                                     const miopen::mha::ProblemDescription& problem) const
{
    // arg 1
    auto fmha_traits = fmha_fwd_traits{hdim_q,
                                       hdim_v,
                                       data_type,
                                       mode == mode_enum::group,
                                       is_v_rowmajor,
                                       mask.type,
                                       bias.type,
                                       lse,
                                       squant};
    // arg 2
    auto fmha_args = [&]() {
        assert(nhead % nhead_k == 0);
        /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
        ///       seqlen_k] in this example, hence both the 'batch_stride_bias' &
        ///       'nhead_stride_bias' are 0.
        // setup stride_* arguments
        const ck_tile::index_t stride_q = (i_perm ? hdim_q : nhead * hdim_q);
        const ck_tile::index_t stride_k = (i_perm ? hdim_q : nhead_k * hdim_q);
        const ck_tile::index_t stride_v = [&]() {
            if(is_v_rowmajor)
                return i_perm ? hdim_v : nhead_k * hdim_v;
            else
                return i_perm ? shape_seqlen_k : nhead_k * shape_seqlen_k;
        }();
        const ck_tile::index_t stride_bias = (i_perm ? shape_seqlen_k : 1 * shape_seqlen_k);
        const ck_tile::index_t stride_o    = (o_perm ? hdim_v : nhead * hdim_v);
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_k = (i_perm ? shape_seqlen_k * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_v = [&]() {
            if(is_v_rowmajor)
                return i_perm ? shape_seqlen_k * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * shape_seqlen_k : shape_seqlen_k;
        }();
        const ck_tile::index_t nhead_stride_bias =
            (i_perm ? 0 * shape_seqlen_q * shape_seqlen_k : 0 * shape_seqlen_k);
        const ck_tile::index_t nhead_stride_lse = (shape_seqlen_q * 1);
        const ck_tile::index_t nhead_stride_o   = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q    = (nhead * shape_seqlen_q * hdim_q);
        const ck_tile::index_t batch_stride_k    = (nhead_k * shape_seqlen_k * hdim_q);
        const ck_tile::index_t batch_stride_v    = (nhead_k * hdim_v * shape_seqlen_k);
        const ck_tile::index_t batch_stride_bias = (0 * nhead * shape_seqlen_q * shape_seqlen_k);
        const ck_tile::index_t batch_stride_lse  = (nhead * shape_seqlen_q * 1);
        const ck_tile::index_t batch_stride_o    = (nhead * shape_seqlen_q * hdim_v);

        return fmha_fwd_args{q_buf.GetDeviceBuffer(),
                             k_buf.GetDeviceBuffer(),
                             v_buf.GetDeviceBuffer(),
                             bias.type == bias_enum::alibi ? alibi_slope_buf.GetDeviceBuffer()
                                                           : bias_buf.GetDeviceBuffer(),
                             lse_buf.GetDeviceBuffer(),
                             o_buf.GetDeviceBuffer(),
                             seqstart_q.GetDeviceBuffer(),
                             seqstart_k.GetDeviceBuffer(),
                             nullptr,
                             shape_seqlen_q,
                             shape_seqlen_k,
                             batch,
                             max_seqlen_q,
                             hdim_q,
                             hdim_v,
                             nhead,
                             nhead_k,
                             scale_s,
                             scale_p,
                             scale_o,
                             stride_q,
                             stride_k,
                             stride_v,
                             bias.type == bias_enum::alibi ? (bias.rank_info == 0 ? 0 : nhead)
                                                           : stride_bias,
                             stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_lse,
                             nhead_stride_o,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             batch_stride_bias,
                             batch_stride_lse,
                             batch_stride_o,
                             mask.left,
                             mask.right,
                             static_cast<ck_tile::index_t>(mask.type)};
    }();

    // arg 3 
    float ave_time = fmha_fwd(fmha_traits, fmha_args, stream_config);
}

bool MhaCKForward::MayNeedWorkspace() const { return true; }

} // namespace mha

} // namespace solver

} // namespace miopen
