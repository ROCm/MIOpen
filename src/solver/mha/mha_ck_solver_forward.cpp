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
#include "ck/stream_config.hpp"

#include <algorithm>
#include <vector>
#include <tuple>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_ATTN_NAIVE_CK_FWD)

namespace miopen {

namespace solver {

namespace mha {

enum class mask_enum
{
    no_mask = 0,
    mask_top_left,
    mask_bottom_right,
    window_generic,
};

// keep sync with BlockAttentionBiasEnum
enum class bias_enum
{
    no_bias          = 0,
    elementwise_bias = 1,
    alibi            = 2,
};

struct fmha_fwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    void* lse_ptr;
    void* o_ptr;
    const void* seqstart_q_ptr;
    const void* seqstart_k_ptr;
    const void* seqlen_k_ptr;
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;
    float scale_s;
    float scale_p;
    float scale_o;
    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;
};

struct stream_config
{
    hipStream_t stream_id_ = nullptr;
    bool time_kernel_      = false;
    int log_level_         = 0;
    int cold_niters_       = 3;
    int nrepeat_           = 10;
};

struct fmha_fwd_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    mask_enum mask_type;
    bias_enum bias_type; // 0:no bias, 1:elementwise bias, 2:alibi. sync with BlockAttentionBiasEnum
    bool has_lse;
    bool do_fp8_static_quant;
    // TODO: padding check is inside this api
};
float fmha_fwd(fmha_fwd_traits, fmha_fwd_args, const stream_config&);

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
    // mask_info  ck_mask = mask_info::decode(0, seqlen_q, seqlen_k);
    // bias_info ck_bias = bias_info::decode(0);

    bool store_loss    = false;
    bool is_v_rowmajor = true;

    // input permute
    bool i_perm = true; // if true, will be batch * nhead * seqlen * hdim
    // output permute
    bool o_perm = false; // if false, will be batch * seqlen * nhead * hdim

    // mode_enum::batch or mode_enum::group
    // auto mode         = mode_enum::batch;
    ck_tile::index_t batch = N;

    bool squant   = true; // fp8 quantization
    float range_q = 16;
    float range_k = 16;
    float range_v = 16;
    float range_p = 1;
    float range_o = 16;

    float scale_s;
    float scale_p;
    float scale_o;

    float dtype_max = ck_tile::type_convert<float>(ck_tile::numeric<ck_tile::fp8_t>::max());

    if(squant)
    {
        scale_s = scale_s * (range_q / dtype_max) * (range_k / dtype_max);
        scale_p = dtype_max / range_p;
        // scale_p = [max(fp8_t)/range_o] * [range_p/max(fp8_t)] * [range_v/max(fp8_t)]
        scale_o = range_p * range_v / range_o / dtype_max;
    }

    // if mode is mode_enum::batch
    const ck_tile::index_t shape_seqlen_q = seqlen_q;
    const ck_tile::index_t shape_seqlen_k = seqlen_k;
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
                // const ck_tile::index_t stride_bias = (i_perm ? shape_seqlen_k : 1 *
                // shape_seqlen_k);
                const ck_tile::index_t stride_o = (o_perm ? hdim_v : nhead * hdim_v);
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
                const ck_tile::index_t nhead_stride_o = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
                // setup batch_stride_* arguments
                const ck_tile::index_t batch_stride_q = (nhead * shape_seqlen_q * hdim_q);
                const ck_tile::index_t batch_stride_k = (nhead_k * shape_seqlen_k * hdim_q);
                const ck_tile::index_t batch_stride_v = (nhead_k * hdim_v * shape_seqlen_k);
                const ck_tile::index_t batch_stride_bias =
                    (0 * nhead * shape_seqlen_q * shape_seqlen_k);
                const ck_tile::index_t batch_stride_lse = (nhead * shape_seqlen_q * 1);
                const ck_tile::index_t batch_stride_o   = (nhead * shape_seqlen_q * hdim_v);

                return fmha_fwd_args{dataFwd.qData,
                                     dataFwd.kData,
                                     dataFwd.vData,
                                     nullptr, // bias will revisit lattter
                                     nullptr, // loss store buffer will revisit latter
                                     dataFwd.oData,
                                     nullptr, // seqstart_q nullptr wrl
                                     nullptr, // seqstart_k nullptr wrl
                                     nullptr,
                                     shape_seqlen_q,
                                     shape_seqlen_k,
                                     batch,
                                     shape_seqlen_q, // need to replace with max_seqlen_q
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
                                     0, // for now bias_enum::no_bias
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
                                     0, // mask.left
                                     0, // mask.right
                                     static_cast<ck_tile::index_t>(mask_enum::no_mask)};
            }();

            // int stream_warmup = arg_parser.get_int("warmup");
            // int stream_repeat = arg_parser.get_int("repeat");
            // bool kname        = arg_parser.get_bool("kname");

            int stream_warmup = 5; // number of iterations before benchmark the kernel
            int stream_repeat = 20;   // number of iterations to benchmark the kernel
            bool kname        = true; // print kernel name

            stream_config stream_config_tmp{
                nullptr, true, /* log_level = */ (kname ? 1 : 0), stream_warmup, stream_repeat};

            // arg 3
            float ave_time = fmha_fwd(fmha_traits, fmha_args, stream_config_tmp);
        };
    };
}

bool MhaCKForward::MayNeedWorkspace() const { return false; }

} // namespace mha

} // namespace solver

} // namespace miopen
