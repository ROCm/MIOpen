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

#if MIOPEN_USE_COMPOSABLEKERNEL
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/fmha_fwd.hpp"
#include "ck/stream_config.hpp"
#endif

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_FA_CK_V2_FWD)

namespace miopen {

namespace solver {

namespace mha {

#if MIOPEN_USE_COMPOSABLEKERNEL
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
#endif

bool MhaCKFlashAttentionV2Forward::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::mha::ProblemDescription& problem) const
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    if(!problem.IsForward())
    {
        return false;
    }

    auto deviceName = context.GetStream().GetDeviceName();
    if(!StartsWith(deviceName, "gfx94") && deviceName != "gfx90a")
    {
        return false;
    }

    const auto& descsFwd      = problem.GetDescsForward();
    auto [N_k, H_k, S_k, D_k] = miopen::tien<4>(descsFwd.kDesc.GetLengths());
    auto [N_stride_k, H_stride_k, S_stride_k, D_stride_k] =
        miopen::tien<4>(descsFwd.kDesc.GetStrides());

    auto [N_q, H_q, S_q, D_q] = miopen::tien<4>(descsFwd.qDesc.GetLengths());
    auto [N_stride_q, H_stride_q, S_stride_q, D_stride_q] =
        miopen::tien<4>(descsFwd.qDesc.GetStrides());

    auto [N_stride_v, H_stride_v, S_stride_v, D_stride_v] =
        miopen::tien<4>(descsFwd.vDesc.GetStrides());

    auto [N_stride_o, H_stride_o, S_stride_o, D_stride_o] =
        miopen::tien<4>(descsFwd.oDesc.GetStrides());

    return !env::disabled(MIOPEN_DEBUG_FA_CK_V2_FWD) //
           && H_q == H_k   // Replace with H_q % H_k == 0 once we add support for MQA & GQA.
           && D_q <= 256   //
           && D_q % 8 == 0 //
           && descsFwd.kDesc.IsPacked()              //
           && descsFwd.qDesc.IsPacked()              //
           && descsFwd.vDesc.IsPacked()              //
           && descsFwd.oDesc.IsPacked()              //
           && descsFwd.kDesc.GetType() == miopenHalf //
           && descsFwd.qDesc.GetType() == miopenHalf //
           && descsFwd.vDesc.GetType() == miopenHalf //
           && descsFwd.oDesc.GetType() == miopenHalf //
           && D_stride_k == 1                        // CK requires D stride as 1.
           && D_stride_q == 1 && D_stride_v == 1 && D_stride_o == 1;
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
#if MIOPEN_USE_COMPOSABLEKERNEL
    auto result         = ConvSolution{miopenStatusSuccess};
    result.workspace_sz = 0;

    const miopen::mha::MhaInputDescsForward& descsFwd = problem.GetDescsForward();
    auto [N_k, H_k, S_k, D_k] = miopen::tien<4>(descsFwd.kDesc.GetLengths());
    auto [N_stride_k, H_stride_k, S_stride_k, D_stride_k] =
        miopen::tien<4>(descsFwd.kDesc.GetStrides());

    auto [N_q, H_q, S_q, D_q] = miopen::tien<4>(descsFwd.qDesc.GetLengths());
    auto [N_stride_q, H_stride_q, S_stride_q, D_stride_q] =
        miopen::tien<4>(descsFwd.qDesc.GetStrides());

    auto [N_v, H_v, S_v, D_v] = miopen::tien<4>(descsFwd.vDesc.GetLengths());
    auto [N_stride_v, H_stride_v, S_stride_v, D_stride_v] =
        miopen::tien<4>(descsFwd.vDesc.GetStrides());

    auto [N_stride_o, H_stride_o, S_stride_o, D_stride_o] =
        miopen::tien<4>(descsFwd.oDesc.GetStrides());

    float scale_s = descsFwd.scale;
    float scale_p = 1.0;
    float scale_o = 1.0;

    fmha_fwd_traits fmha_traits;
    fmha_traits.hdim_q        = D_q;
    fmha_traits.hdim_v        = D_v;
    fmha_traits.data_type     = Convert(descsFwd.qDesc.GetType());
    fmha_traits.is_group_mode = false;
    // is_v_rowmajor relates to the layout of the V tensor. Row major means NHSD, and Col major
    // means NHDS.
    fmha_traits.is_v_rowmajor       = true;
    fmha_traits.mask_type           = mask_enum::no_mask;
    fmha_traits.has_lse             = false;
    fmha_traits.do_fp8_static_quant = false;
    fmha_traits.has_dropout         = false;
    fmha_traits.bias_type           = bias_enum::no_bias;

    fmha_fwd_args fmha_args;
    fmha_args.hdim_q               = D_q;
    fmha_args.hdim_v               = D_v;
    fmha_args.seqlen_k             = S_k;
    fmha_args.seqlen_q             = S_q;
    fmha_args.max_seqlen_q         = S_q;
    fmha_args.nhead_q              = H_q;
    fmha_args.nhead_k              = H_k;
    fmha_args.batch                = N_q;
    fmha_args.stride_q             = S_stride_q;
    fmha_args.stride_k             = S_stride_k;
    fmha_args.stride_v             = S_stride_v;
    fmha_args.stride_o             = S_stride_o;
    fmha_args.stride_bias          = 0;
    fmha_args.stride_randval       = S_q;
    fmha_args.nhead_stride_q       = H_stride_q;
    fmha_args.nhead_stride_k       = H_stride_k;
    fmha_args.nhead_stride_v       = H_stride_v;
    fmha_args.nhead_stride_o       = H_stride_o;
    fmha_args.nhead_stride_lse     = S_q;
    fmha_args.nhead_stride_bias    = 0;
    fmha_args.nhead_stride_randval = S_q * S_k;
    fmha_args.batch_stride_q       = N_stride_q;
    fmha_args.batch_stride_k       = N_stride_k;
    fmha_args.batch_stride_v       = N_stride_v;
    fmha_args.batch_stride_o       = N_stride_o;
    fmha_args.batch_stride_lse     = H_q * S_q;
    fmha_args.batch_stride_bias    = 0;
    fmha_args.batch_stride_randval = H_q * S_q * S_k;

    // These are used for group mode, and we are in batch right now.
    fmha_args.seqstart_q_ptr = nullptr;
    fmha_args.seqstart_k_ptr = nullptr;

    // Batch does not support padding, and we aren't using kvcache yet.
    fmha_args.seqlen_k_ptr = nullptr;

    fmha_args.s_randval         = false;
    fmha_args.scale_s           = scale_s;
    fmha_args.scale_p           = scale_p;
    fmha_args.scale_o           = scale_o;
    fmha_args.window_size_left  = 0;
    fmha_args.window_size_right = 0;
    fmha_args.mask_type         = static_cast<ck_tile::index_t>(fmha_traits.mask_type);

    result.invoker_factory = [=](const std::vector<Kernel>&) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::mha::InvokeParams>();
            const auto& dataFwd   = params.GetDataForward();

            fmha_fwd_traits fmha_runtime_traits = fmha_traits;
            fmha_fwd_args fmha_runtime_args     = fmha_args;

            fmha_runtime_args.q_ptr        = dataFwd.qData;
            fmha_runtime_args.k_ptr        = dataFwd.kData;
            fmha_runtime_args.v_ptr        = dataFwd.vData;
            fmha_runtime_args.o_ptr        = dataFwd.oData;
            fmha_runtime_args.rand_val_ptr = nullptr;
            fmha_runtime_args.bias_ptr     = nullptr;
            fmha_runtime_args.lse_ptr      = nullptr;

            // Top-left causal mask
            if(dataFwd.mask == miopenMhaMask_t::miopenMhaMaskCausal)
            {
                fmha_runtime_traits.mask_type = mask_enum::mask_top_left;
                fmha_runtime_args.mask_type =
                    static_cast<ck_tile::index_t>(mask_enum::mask_top_left);
                fmha_runtime_args.window_size_left  = -1;
                fmha_runtime_args.window_size_right = 0;
            }

            fmha_runtime_traits.has_dropout = false;
            float probability               = 0;

            // TODO : Change API to take in probability value as host side value instead of device
            // pointer to match CK API. Calling a blocking hipMemcpy will cause issues with stream,
            // and isn't async.

            fmha_runtime_args.p_drop = probability;
            // fmha_runtime_args.drop_seed_offset =
            //     std::make_pair(dataFwd.dropoutSeedData),
            //                    dataFwd.dropoutOffsetData);

            // using dataFwd.dropoutSeedData gpu pointer was causing compiler error
            // since dropout is disabled for now, placing 0.
            fmha_runtime_args.drop_seed_offset = std::make_pair(0, 0);

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
#else
    return ConvSolution{miopenStatusNotImplemented};
#endif
}

bool MhaCKFlashAttentionV2Forward::MayNeedWorkspace() const { return false; }

} // namespace mha

} // namespace solver

} // namespace miopen
