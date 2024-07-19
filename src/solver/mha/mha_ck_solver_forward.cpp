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

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_ATTN_CK_FWD)

int numSplitsHeuristic(int batch_nhead_mblocks, int num_SMs, int num_n_blocks, int max_splits)
{
    // If we have enough to almost fill the SMs, then just use 1 split
    if(batch_nhead_mblocks >= 0.8f * num_SMs)
    {
        return 1;
    }
    max_splits           = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 ||
               ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for(int num_splits = 1; num_splits <= max_splits; num_splits++)
    {
        if(!is_split_eligible(num_splits))
        {
            efficiency.push_back(0.f);
        }
        else
        {
            float n_waves = float(batch_nhead_mblocks * num_splits) / num_SMs;
            float eff     = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if(eff > max_efficiency)
            {
                max_efficiency = eff;
            }
            efficiency.push_back(eff);
        }
    }
    for(int num_splits = 1; num_splits <= max_splits; num_splits++)
    {
        if(!is_split_eligible(num_splits))
        {
            continue;
        }
        if(efficiency[num_splits - 1] >= 0.85 * max_efficiency)
        {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

int determineNumSplits(
    int batch, int nhead, int max_seqlen_q, int hdim_v, float p_drop, int num_splits)
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return num_splits;
    }

    hipDeviceProp_t props{};
    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return num_splits;
    }

    // tile size should match the generate.py
    const int kM0 = 64;
    const int kN1 = hdim_v;

    const int num_m_blocks = ck_tile::integer_divide_ceil(max_seqlen_q, kM0);
    const int num_n_blocks = ck_tile::integer_divide_ceil(hdim_v, kN1);

    if(num_splits < 1 && p_drop == 0.0f)
    {
        return numSplitsHeuristic(
            batch * nhead * num_m_blocks, props.multiProcessorCount * 2, num_n_blocks, 128);
    }

    return num_splits;
}

namespace miopen {

namespace solver {

namespace mha {

bool MhaCKForward::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                                const miopen::mha::ProblemDescription& problem) const
{
    ::miopen::mha::MhaInputDescsForward mha_des = problem.GetDescsForward();
    const auto& lens                            = mha_des.qDesc.GetLengths();
    auto [N, H, S, D]                           = std::tie(lens[0], lens[1], lens[2], lens[3]);
    if(D <= 256 && S % 128 == 0 && D % 64 == 0)
        return true;
    return false;
}

// based on num_splits
std::size_t MhaCKForward::GetWorkspaceSize([[maybe_unused]] const ExecutionContext& context,
                                           const miopen::mha::ProblemDescription& problem) const
{
    ::miopen::mha::MhaInputDescsForward mha_des = problem.GetDescsForward();
    // VDesc since its Values dim that is required
    const auto& lens  = mha_des.vDesc.GetLengths();
    auto [N, H, S, D] = std::tie(lens[0], lens[1], lens[2], lens[3]);

    int num_splits = determineNumSplits(N, H, S, D, 0.0, 1 /*default num splits*/);

    int byte_size_of_float = sizeof(float);

    if(num_splits == 1)
    {
        return byte_size_of_float;
    }
    else
    {
        return num_splits * N * H * S * D * byte_size_of_float;
    }
}

ConvSolution MhaCKForward::GetSolution(const ExecutionContext& context,
                                       const miopen::mha::ProblemDescription& problem) const
{
    auto result         = ConvSolution{miopenStatusSuccess};
    result.workspace_sz = 0;
    // heuristics find optimum num_splits for given problem

    const miopen::mha::MhaInputDescsForward& descsFwd = problem.GetDescsForward();

    auto [N, H, S, D] = miopen::tien<4>(descsFwd.kDesc.GetLengths());

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

    // currenly ck's fp8 only supports batch mode
    bool is_group_mode = false;

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
            auto workSpace        = params.GetWorkspace();

            // arg 1
            auto fmha_traits = fmha_fwd_traits{
                hdim_q,
                hdim_v,
                "fp8", // data_type in string (todo: change this based on problem description)
                is_group_mode, // mode == mode_enum::group,
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
                const ck_tile::index_t stride_o       = (o_perm ? hdim_v : nhead * hdim_v);
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
                float p_drop         = 0.0f;
                bool s_randval       = false;
                uint64_t drop_seed   = 1; // seed for random number generator
                uint64_t drop_offset = 0; // offset for random number generator
                // This is tuning parameter
                int num_splits = 1;

                const ck_tile::index_t stride_randval       = (max_seqlen_k);
                const ck_tile::index_t stride_o_acc         = hdim_v;
                const ck_tile::index_t nhead_stride_randval = (shape_seqlen_q * max_seqlen_k);
                const ck_tile::index_t nhead_stride_lse_acc = max_seqlen_q;
                const ck_tile::index_t nhead_stride_o_acc   = (max_seqlen_q * hdim_v);
                const ck_tile::index_t nhead_stride_o = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);

                const ck_tile::index_t batch_stride_randval =
                    (nhead * shape_seqlen_q * max_seqlen_k);
                const ck_tile::index_t batch_stride_lse     = (nhead * max_seqlen_q);
                const ck_tile::index_t batch_stride_lse_acc = (nhead * max_seqlen_q);
                const ck_tile::index_t batch_stride_o_acc   = (nhead * max_seqlen_q * hdim_v);
                const ck_tile::index_t batch_stride_o       = (nhead * shape_seqlen_q * hdim_v);

                const ck_tile::index_t split_stride_lse_acc = (batch * nhead * max_seqlen_q);
                const ck_tile::index_t split_stride_o_acc = (batch * nhead * max_seqlen_q * hdim_v);

                return fmha_fwd_args{dataFwd.qData, // q_ptr
                                     dataFwd.kData, // k_ptr
                                     dataFwd.vData, // v_ptr
                                     nullptr, //       bias_ptr  (no bias for now)
                                     nullptr, //       rand_val_pr loss store (no loss for now)
                                     nullptr, //        lse_acc_ptr (no loss for now)
                                     workSpace, //       o_acc_ptr
                                     nullptr,   //       lse_ptr (no loss for now)
                                     dataFwd.oData, //        o_ptr
                                     nullptr,       //        seqstart_q_ptr
                                     nullptr,       //        seqstart_k_ptr
                                     nullptr, //        seqlen_k_ptr (null is ok)
                                     shape_seqlen_q,
                                     shape_seqlen_k,
                                     batch,
                                     max_seqlen_q, // need to replace with max_seqlen_q
                                     hdim_q,
                                     hdim_v,
                                     nhead,
                                     nhead_k,
                                     num_splits,
                                     scale_s,
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
                                     p_drop,    // float value
                                     s_randval, // bool flag
                                     {drop_seed, drop_offset}};
            }();

            int stream_warmup = 1; // number of iterations before benchmark the kernel
            int stream_repeat = 0; // number of iterations to benchmark the kernel
            bool kname        = false; // print kernel name

            ck_tile::stream_config stream_config_tmp{nullptr /*stream_id*/,
                                                     true /*time_kernel*/,
                                                     /* log_level = */ (kname ? 1 : 0),
                                                     stream_warmup,
                                                     stream_repeat};

            fmha_fwd(fmha_traits, fmha_args, stream_config_tmp);
        };
    };
    return result;
}

bool MhaCKForward::MayNeedWorkspace() const { return false; }

} // namespace mha

} // namespace solver

} // namespace miopen
