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

#include <miopen/mha/solvers.hpp>
#include <miopen/mha/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/hipoc_kernel.hpp>

#ifdef MIOPEN_USE_COMPOSABLEKERNEL
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/fmha_fwd.hpp"
#include "ck/stream_config.hpp"
#endif

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_FA_CK_V2_FWD)

namespace miopen {

namespace solver {

namespace mha {

static std::string Convert(miopenDataType_t dataType)
{
    switch(dataType)
    {
    case miopenHalf: {
        return "fp16";
    }
    case miopenBFloat16: {
        return "bfp16";
    }
    default: {
        MIOPEN_THROW("Unsupported datatype provided");
    }
    }
}

bool MhaCKFlashAttentionV2Forward::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::mha::ProblemDescription& problem) const
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    if(!problem.IsForward())
    {
        return false;
    }

    if(!StartsWith(context.GetStream().GetDeviceName(), "gfx94"))
    {
        return false;
    }

    auto& descsFwd            = problem.GetDescsForward();
    auto [N_k, H_k, S_k, D_k] = miopen::tien<4>(descsFwd.kDesc.GetLengths());
    auto [N_q, H_q, S_q, D_q] = miopen::tien<4>(descsFwd.qDesc.GetLengths());

    return !env::disabled(MIOPEN_DEBUG_FA_CK_V2_FWD) //
           && H_q == H_k   // Replace with H_q % H_k == 0 once we add support for MQA & GQA.
           && H_q <= 256   //
           && H_q % 8 == 0 // No padding support yet which means it needs to be multiple of 8.
           && descsFwd.kDesc.IsPacked()                 //
           && descsFwd.qDesc.IsPacked()                 //
           && descsFwd.vDesc.IsPacked()                 //
           && descsFwd.oDesc.IsPacked()                 //
           && descsFwd.biasDesc.IsPacked()              //
           && descsFwd.biasDesc.GetType() == miopenHalf //
           && descsFwd.kDesc.GetType() == miopenHalf    //
           && descsFwd.qDesc.GetType() == miopenHalf    //
           && descsFwd.vDesc.GetType() == miopenHalf    //
           && descsFwd.oDesc.GetType() == miopenHalf;   //
#else
    return false;
#endif
}

std::size_t MhaCKFlashAttentionV2Forward::GetWorkspaceSize(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::mha::ProblemDescription& problem) const
{
    return 0;
}

ConvSolution
MhaCKFlashAttentionV2Forward::GetSolution([[maybe_unused]] const ExecutionContext& context,
                                          const miopen::mha::ProblemDescription& problem) const
{
    auto result         = ConvSolution{miopenStatusSuccess};
    result.workspace_sz = 0;

    const miopen::mha::MhaInputDescsForward& descsFwd = problem.GetDescsForward();
    auto [N_k, H_k, S_k, D_k] = miopen::tien<4>(descsFwd.kDesc.GetLengths());
    auto [N_q, H_q, S_q, D_q] = miopen::tien<4>(descsFwd.qDesc.GetLengths());
    auto [N_v, H_v, S_v, D_v] = miopen::tien<4>(descsFwd.vDesc.GetLengths());

    ck_tile::index_t batch    = N_q;
    ck_tile::index_t seqlen_q = S_q;
    ck_tile::index_t seqlen_k = S_k;
    ck_tile::index_t hdim_q   = D_q;
    ck_tile::index_t hdim_v   = D_v;
    ck_tile::index_t nhead    = H_q;
    ck_tile::index_t nhead_k  = H_k;
    ck_tile::index_t nhead_q  = H_q;

    bool is_group_mode = false;
    bool o_perm = true, i_perm = true; // if true, will be batch * nhead * seqlen * hdim

    float scale_s = descsFwd.scale;
    float scale_p = 1.0;
    float scale_o = 1.0;

    const ck_tile::index_t shape_seqlen_q = seqlen_q;
    const ck_tile::index_t shape_seqlen_k = seqlen_k;
    const ck_tile::index_t max_seqlen_q   = seqlen_q;
    const ck_tile::index_t max_seqlen_k   = seqlen_k;

    fmha_fwd_traits fmha_traits;
    fmha_traits.hdim_q              = hdim_q;
    fmha_traits.hdim_v              = hdim_v;
    fmha_traits.data_type           = Convert(descsFwd.qDesc.GetType());
    fmha_traits.is_group_mode       = is_group_mode;
    fmha_traits.is_v_rowmajor       = false;
    fmha_traits.mask_type           = mask_enum::no_mask;
    fmha_traits.has_lse             = false;
    fmha_traits.is_v_rowmajor       = false;
    fmha_traits.do_fp8_static_quant = false;

    fmha_fwd_args fmha_args;
    fmha_args.batch          = batch;
    fmha_args.hdim_q         = hdim_q;
    fmha_args.hdim_v         = hdim_v;
    fmha_args.nhead_q        = nhead_q;
    fmha_args.nhead_k        = nhead_k;
    fmha_args.stride_q       = (i_perm ? hdim_q : nhead * hdim_q);
    fmha_args.stride_k       = (i_perm ? hdim_q : nhead_k * hdim_q);
    fmha_args.stride_v       = (i_perm ? shape_seqlen_k : nhead_k * shape_seqlen_k);
    fmha_args.batch_stride_q = (nhead * shape_seqlen_q * hdim_q);
    fmha_args.batch_stride_k = (nhead_k * shape_seqlen_k * hdim_q);
    fmha_args.batch_stride_v = (nhead_k * hdim_v * shape_seqlen_k);
    fmha_args.seqlen_k       = shape_seqlen_k;
    fmha_args.max_seqlen_q   = max_seqlen_q;

    // These are used for group mode, and we are in batch right now.
    fmha_args.seqstart_q_ptr = nullptr;
    fmha_args.seqstart_k_ptr = nullptr;

    // Batch does not support padding, and we aren't using kvcache yet.
    fmha_args.seqlen_k_ptr = nullptr;

    fmha_args.scale_s           = scale_s;
    fmha_args.scale_p           = scale_p;
    fmha_args.scale_o           = scale_o;
    fmha_args.stride_bias       = (i_perm ? shape_seqlen_k : 1 * shape_seqlen_k);
    fmha_args.stride_o          = (o_perm ? hdim_v : nhead * hdim_v);
    fmha_args.nhead_stride_bias = i_perm ? 0 * shape_seqlen_q * shape_seqlen_k : 0 * shape_seqlen_k;
    fmha_args.nhead_stride_lse  = shape_seqlen_q;
    fmha_args.nhead_stride_o    = o_perm ? shape_seqlen_q * hdim_v : hdim_v;
    fmha_args.window_size_left  = 0;
    fmha_args.window_size_right = 0;
    fmha_args.mask_type         = static_cast<ck_tile::index_t>(fmha_traits.mask_type);

    fmha_args.s_randval = false;
    // Since we aren't storing the random values these will be unused for now.
    fmha_args.stride_randval       = max_seqlen_k;
    fmha_args.nhead_stride_randval = (shape_seqlen_q * max_seqlen_k);
    fmha_args.batch_stride_randval = (nhead * shape_seqlen_q * max_seqlen_k);

    result.invoker_factory = [=](const std::vector<Kernel>&) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::mha::InvokeParams>();
            const auto& dataFwd   = params.GetDataForward();

            fmha_fwd_traits fmha_runtime_traits = fmha_traits;
            fmha_fwd_args fmha_runtime_args     = fmha_args;

            fmha_runtime_traits.bias_type =
                dataFwd.biasData != nullptr ? bias_enum::elementwise_bias : bias_enum::no_bias;
            fmha_runtime_traits.has_dropout = dataFwd.dropoutProbabilityData != nullptr;

            float probability = 0;
            uint64_t seed     = 0;
            uint64_t offset   = 0;
            if(fmha_runtime_traits.has_dropout)
            {
                hipMemcpy(&probability,
                          dataFwd.dropoutProbabilityData,
                          sizeof(float),
                          hipMemcpyKind::hipMemcpyDeviceToHost);
                hipMemcpy(&seed,
                          dataFwd.dropoutSeedData,
                          sizeof(uint64_t),
                          hipMemcpyKind::hipMemcpyDeviceToHost);
                hipMemcpy(&offset,
                          dataFwd.dropoutOffsetData,
                          sizeof(uint64_t),
                          hipMemcpyKind::hipMemcpyDeviceToHost);
            }
            fmha_runtime_args.p_drop           = probability;
            fmha_runtime_args.drop_seed_offset = {seed, offset};

            fmha_runtime_args.bias_ptr     = dataFwd.biasData;
            fmha_runtime_args.q_ptr        = dataFwd.qData;
            fmha_runtime_args.k_ptr        = dataFwd.kData;
            fmha_runtime_args.v_ptr        = dataFwd.vData;
            fmha_runtime_args.rand_val_ptr = nullptr;
            fmha_runtime_args.o_ptr        = dataFwd.oData;

            // Create stream_config, and set it to not time kernel.
            ck_tile::stream_config stream_config;
            stream_config.stream_id_ = handle_.GetStream();

            {
                HipEventProfiler profiler(handle_);
                fmha_fwd(fmha_runtime_traits, fmha_runtime_args, stream_config);
            }
        };
    };

    return result;
}

bool MhaCKFlashAttentionV2Forward::MayNeedWorkspace() const { return false; }

} // namespace mha

} // namespace solver

} // namespace miopen
