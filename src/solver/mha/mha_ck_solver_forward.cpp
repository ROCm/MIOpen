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
#include "ck_tile/ops/fmha_fwd.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck/stream_config.hpp"

#include <algorithm>
#include <vector>
#include <tuple>
#include <iostream>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_ATTN_NAIVE_CK_FWD)

namespace miopen {

namespace solver {

namespace mha {

bool MhaCKForward::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                                const miopen::mha::ProblemDescription& problem) const
{
    return true;
}

std::size_t MhaCKForward::GetWorkspaceSize([[maybe_unused]] const ExecutionContext& context,
                                           const miopen::mha::ProblemDescription& problem) const
{
    return 0;
}

ConvSolution MhaCKForward::GetSolution(const ExecutionContext& context,
                                       const miopen::mha::ProblemDescription& problem) const
{
    auto result         = ConvSolution{miopenStatusSuccess};
    result.workspace_sz = 0;

    const miopen::mha::MhaInputDescsForward& descsFwd = problem.GetDescsForward();

    uint64_t N, H, S, D;
    std::tie(N, H, S, D) = miopen::tien<4>(descsFwd.kDesc.GetLengths());

    descsFwd.kDesc.GetType();

    ck_tile::index_t seqlen_q = S;
    ck_tile::index_t seqlen_k = S;
    ck_tile::index_t hdim_q   = D;
    ck_tile::index_t hdim_v   = D;
    ck_tile::index_t nhead    = H;
    ck_tile::index_t nhead_k  = H;

    // currently we assume both sqlen_q and seqlen_k as S
    // no mask for now
    // fp8 currenly only supports fp8
    bool store_loss    = false;
    bool is_v_rowmajor = false;

    // input permute
    bool i_perm = true; // if true, will be batch * nhead * seqlen * hdim
    // output permute
    bool o_perm = true; // if false, will be batch * seqlen * nhead * hdim

    // mode_enum::batch or mode_enum::group
    // auto mode         = mode_enum::batch;
    ck_tile::index_t batch = N;

    bool squant   = true; // fp8 quantization
    float range_q = 1;
    float range_k = 1;
    float range_v = 1;
    float range_p = 1;
    float range_o = 1;

    float scale_s = 1.0;
    float scale_p = 1.0;
    float scale_o = 1.0;

    float dtype_max = ck_tile::type_convert<float>(ck_tile::numeric<ck_tile::fp8_t>::max());

    if(squant)
    {
        scale_s = scale_s * (range_q / dtype_max) * (range_k / dtype_max);
        scale_p = dtype_max / range_p;
        scale_o = range_p * range_v / range_o / dtype_max;
    }

    // if mode is mode_enum::batch
    const ck_tile::index_t shape_seqlen_q = seqlen_q;
    const ck_tile::index_t shape_seqlen_k = seqlen_k;
    const ck_tile::index_t max_seqlen_q   = seqlen_k;
    const ck_tile::index_t max_seqlen_k   = seqlen_k;
    result.invoker_factory                = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::mha::InvokeParams>();
            const auto& dataFwd   = params.GetDataForward();

            // arg 1
            auto fmha_traits = fmha_fwd_traits{hdim_q,
                                               hdim_v,
                                               "fp8", // data_type in string
                                               false, // mode == mode_enum::group, // is_group_mode
                                               is_v_rowmajor,
                                               mask_enum::no_mask, // no mask for now
                                               bias_enum::no_bias, // no bias
                                               false,
                                               store_loss,
                                               squant};

            // arg 2
            auto fmha_args = [&]() {
                const ck_tile::index_t stride_q = (i_perm ? hdim_q : nhead * hdim_q);
                const ck_tile::index_t stride_k = (i_perm ? hdim_q : nhead_k * hdim_q);
                
                const ck_tile::index_t stride_v = [&]() {
                    if(is_v_rowmajor)
                        return i_perm ? hdim_v : nhead_k * hdim_v;
                    else
                        return i_perm ? shape_seqlen_k : nhead_k * shape_seqlen_k;
                }();
                const ck_tile::index_t stride_o = (o_perm ? hdim_v : nhead * hdim_v);
                const ck_tile::index_t nhead_stride_q = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
                const ck_tile::index_t nhead_stride_k = (i_perm ? shape_seqlen_k * hdim_q : hdim_q);
                const ck_tile::index_t nhead_stride_v = [&]() {
                    if(!is_v_rowmajor)
                        return i_perm ? shape_seqlen_k * hdim_v : hdim_v;
                    else
                        return i_perm ? hdim_v * shape_seqlen_k : shape_seqlen_k;
                }();
                const ck_tile::index_t nhead_stride_bias =
                    (i_perm ? 0 * shape_seqlen_q * shape_seqlen_k : 0 * shape_seqlen_k);
                const ck_tile::index_t nhead_stride_lse = (shape_seqlen_q * 1);
                // setup batch_stride_* arguments
                const ck_tile::index_t batch_stride_q = (nhead * shape_seqlen_q * hdim_q);
                const ck_tile::index_t batch_stride_k = (nhead_k * shape_seqlen_k * hdim_q);
                const ck_tile::index_t batch_stride_v = (nhead_k * hdim_v * shape_seqlen_k);
                const ck_tile::index_t batch_stride_bias =
                    (0 * nhead * shape_seqlen_q * shape_seqlen_k);
                float p_drop = 0.0f;
                bool s_randval = false;
                uint64_t drop_seed = 1; // seed for random number generator
                uint64_t drop_offset = 0; // offset for random number generator
                // This is tuning parameter
                int num_splits = 1;

                const ck_tile::index_t stride_randval = (max_seqlen_k);
                const ck_tile::index_t stride_o_acc   = hdim_v;
                const ck_tile::index_t nhead_stride_randval = (shape_seqlen_q * max_seqlen_k);
                const ck_tile::index_t nhead_stride_lse_acc = max_seqlen_q;
                const ck_tile::index_t nhead_stride_o_acc   = (max_seqlen_q * hdim_v);
                const ck_tile::index_t nhead_stride_o       = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);

                const ck_tile::index_t batch_stride_randval = (nhead * shape_seqlen_q * max_seqlen_k);
                const ck_tile::index_t batch_stride_lse     = (nhead * max_seqlen_q);
                const ck_tile::index_t batch_stride_lse_acc = (nhead * max_seqlen_q);
                const ck_tile::index_t batch_stride_o_acc   = (nhead * max_seqlen_q * hdim_v);
                const ck_tile::index_t batch_stride_o       = (nhead * shape_seqlen_q * hdim_v);

                const ck_tile::index_t split_stride_lse_acc = (batch * nhead * max_seqlen_q);
                const ck_tile::index_t split_stride_o_acc   = (batch * nhead * max_seqlen_q * hdim_v);

                // if num_splits = 1
                float o_acc = 0.0;
                // if num_splits > 1
                // we will need tensor<float>{num_splits, n, h, s, d});

                return fmha_fwd_args{dataFwd.qData, // q_ptr
                                     dataFwd.kData, // k_ptr
                                     dataFwd.vData, // v_ptr
                                     nullptr, //       bias_ptr  (no bias for now)
                                     nullptr, //       rand_val_pr loss store (no loss for now)
                                     nullptr,//        lse_acc_ptr (no loss for now)
                                     &o_acc, //       o_acc_ptr        
                                     nullptr, //       lse_ptr (no loss for now)
                                     dataFwd.oData,//        o_ptr
                                     nullptr,//        seqstart_q_ptr
                                     nullptr,//        seqstart_k_ptr
                                     nullptr,//        seqlen_k_ptr (null is ok)
                                     shape_seqlen_q,
                                     shape_seqlen_k,
                                     batch,
                                     max_seqlen_q, // need to replace with max_seqlen_q
                                     hdim_q,
                                     hdim_v,
                                     nhead,
                                     nhead_k,
                                     num_splits, // 
                                     scale_s, //0.0000173579,
                                     scale_p,
                                     scale_o,
                                     stride_q,
                                     stride_k,
                                     stride_v,
                                     0, // for now bias_enum::no_bias
                                     stride_randval,
                                     stride_o_acc,
                                     stride_o,
                                     nhead_stride_q,
                                     nhead_stride_k,
                                     nhead_stride_v,
                                     nhead_stride_bias,
                                     nhead_stride_randval,
                                     nhead_stride_lse,
                                     nhead_stride_lse_acc,
                                     nhead_stride_o_acc,
                                     nhead_stride_o,
                                     batch_stride_q,
                                     batch_stride_k,
                                     batch_stride_v,
                                     batch_stride_bias,
                                     batch_stride_randval,
                                     batch_stride_lse,
                                     batch_stride_lse_acc,
                                     batch_stride_o_acc,
                                     batch_stride_o,
                                     split_stride_lse_acc,
                                     split_stride_o_acc,
                                     0, // mask.left (no mask for now)
                                     0, // mask.right (no mask for now)
                                     static_cast<ck_tile::index_t>(mask_enum::no_mask),
                                     p_drop, // float value
                                     s_randval, // bool flag
                                     {drop_seed, drop_offset}};
            }();

            int stream_warmup = 1; // number of iterations before benchmark the kernel
            int stream_repeat = 0;   // number of iterations to benchmark the kernel
            bool kname        = false; // print kernel name
 
            ck_tile::stream_config stream_config_tmp{
                nullptr/*stream_id*/, true/*time_kernel*/, /* log_level = */ (kname ? 1 : 0), stream_warmup, stream_repeat};

            fmha_fwd(fmha_traits, fmha_args, stream_config_tmp);
        };
    };
    return result;
}

bool MhaCKForward::MayNeedWorkspace() const { return false; }

} // namespace mha

} // namespace solver

} // namespace miopen
