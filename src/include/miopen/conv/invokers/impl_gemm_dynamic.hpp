/*******************************************************************************
*
* MIT License
*
* Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#pragma once

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/handle.hpp>
#include <miopen/invoker.hpp>
#include <miopen/kernel.hpp>
#include <miopen/conv/context.hpp>
#include <miopen/conv/asm_implicit_gemm.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/numeric.hpp>

#include <vector>

namespace miopen {
namespace conv {

struct magic_div_u32
{
    uint32_t magic;
    uint8_t shift;
};

using magic_div_u32_t = magic_div_u32;

/*
*
* numer / denom = quotient, reminder
*
* use magic number to do integer division of uint32 (acctually INT32_MAX, the 31 bit divisoin)
* most algorithm to compute uint32 need branching if cover all 32 bit of uint32.
* since we compute the magic number on host side, implement the division in gpu side, it is better
* not use branching
* hence add more restriction to numer and denom, to be 1 bit less. hence need less-or-equal than
* INT32_MAX
*
* magic_div_u32_gen() compute from input arg d, to get a magic and a shift.
* to use the value, below is a example host-side code to do this
*
* // host side version
* static inline uint32_t magic_div_mulhi_u32(uint32_t x, uint32_t y) {
*     uint64_t xl = x, yl = y;
*     uint64_t rl = xl * yl;
*     return (uint32_t)(rl >> 32);
* }
* uint32_t magic_div_u32_do(uint32_t numer, const struct magic_div_u32_t *denom) {
*     uint32_t tmp = magic_div_mulhi_u32(denom->magic, numer);
*     return (tmp + numer) >> denom->shift;
* }
*
*/
static inline magic_div_u32_t magic_div_u32_gen(uint32_t d)
{
    assert(d >= 1 && d <= INT32_MAX);
    uint8_t shift;
    for(shift = 0; shift < 32; shift++)
        if((1U << shift) >= d)
            break;

    constexpr uint64_t one = 1;
    uint64_t magic         = ((one << 32) * ((one << shift) - d)) / d + 1;
    assert(magic <= 0xffffffffUL);

    return {static_cast<uint32_t>(magic), shift};
}

static inline uint32_t magic_div_u32_pack_shift(uint8_t s0, uint8_t s1, uint8_t s2, uint8_t s3)
{
    uint32_t shift_0 = s0;
    uint32_t shift_1 = s1;
    uint32_t shift_2 = s2;
    uint32_t shift_3 = s3;
    return (shift_3 << 24) | (shift_2 << 16) | (shift_1 << 8) | shift_0;
}

template <typename T>
inline std::vector<OpKernelArg>
ComputeDynamicIGemmForwardKernelArgs(const ProblemDescription& conv_problem, const T& cfg);

template <>
inline std::vector<OpKernelArg>
ComputeDynamicIGemmForwardKernelArgs<int>(const ProblemDescription& conv_problem, const int& cfg)
{
    std::vector<OpKernelArg> opArgs;
    // clang-format off
    int hi          = conv_problem.GetInHeight();
    int wi          = conv_problem.GetInWidth();
    int n           = conv_problem.GetInBatchSize();
    int k           = conv_problem.GetOutChannels();
    int c           = conv_problem.GetInChannels();
    int ho          = conv_problem.GetOutHeight();
    int wo          = conv_problem.GetOutWidth();
    int stride_h    = conv_problem.GetKernelStrideH();
    int stride_w    = conv_problem.GetKernelStrideW();
    int dilation_h  = conv_problem.GetDilationH();
    int dilation_w  = conv_problem.GetDilationW();
    int pad_h       = conv_problem.GetPadH();
    int pad_w       = conv_problem.GetPadW();
    int y           = conv_problem.GetWeightsHeight();
    int x           = conv_problem.GetWeightsWidth();
    int pack0       = cfg;
    // clang-format on

    opArgs.emplace_back(hi);
    opArgs.emplace_back(wi);
    opArgs.emplace_back(n);
    opArgs.emplace_back(k);
    opArgs.emplace_back(c);
    opArgs.emplace_back(ho);
    opArgs.emplace_back(wo);
    opArgs.emplace_back(stride_h);
    opArgs.emplace_back(stride_w);
    opArgs.emplace_back(dilation_h);
    opArgs.emplace_back(dilation_w);
    opArgs.emplace_back(pad_h);
    opArgs.emplace_back(pad_w);
    opArgs.emplace_back(y);
    opArgs.emplace_back(x);
    opArgs.emplace_back(pack0);

    return opArgs;
}

template <>
inline std::vector<OpKernelArg>
ComputeDynamicIGemmForwardKernelArgs<solver::TunableImplicitGemmGTCDynamic_t>(
    const ProblemDescription& conv_problem, const solver::TunableImplicitGemmGTCDynamic_t& cfg)
{
    std::vector<OpKernelArg> opArgs;
    // clang-format off
    int hi          = conv_problem.GetInHeight();
    int wi          = conv_problem.GetInWidth();
    int n           = conv_problem.GetInBatchSize();
    int k           = conv_problem.GetOutChannels();
    int c           = conv_problem.GetInChannels();
    int ho          = conv_problem.GetOutHeight();
    int wo          = conv_problem.GetOutWidth();
    int stride_h    = conv_problem.GetKernelStrideH();
    int stride_w    = conv_problem.GetKernelStrideW();
    int dilation_h  = conv_problem.GetDilationH();
    int dilation_w  = conv_problem.GetDilationW();
    int pad_h       = conv_problem.GetPadH();
    int pad_w       = conv_problem.GetPadW();
    int y           = conv_problem.GetWeightsHeight();
    int x           = conv_problem.GetWeightsWidth();
    int group       = conv_problem.GetGroupCount();
    int pack0       = 0;
    // clang-format on

    int gemm_m =
        ((k / group + cfg.gemm_m_per_block - 1) / cfg.gemm_m_per_block) * cfg.gemm_m_per_block;
    int nxe = cfg.nxe;
    int nxb = cfg.nxb;
    int b =
        nxe == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb; // pad to nxb modulo when nxe != 0

    // init magic division parameters
    uint32_t nb_n0          = cfg.tensor_b_cluster_lengths[2] * cfg.tensor_b_thread_lengths[2];
    uint32_t nb_n1b         = cfg.tensor_b_cluster_lengths[3] * cfg.tensor_b_thread_lengths[3];
    uint32_t unmerge_sub_n  = cfg.gemm_n_per_block / nxb;
    uint32_t unmerge_sub_n1 = unmerge_sub_n / nb_n0;

    magic_div_u32_t mdiv_0 = magic_div_u32_gen(gemm_m / cfg.gemm_m_per_block);
    magic_div_u32_t mdiv_1 = magic_div_u32_gen(b * unmerge_sub_n1 / nb_n1b);
    magic_div_u32_t mdiv_2 = magic_div_u32_gen(y * x);
    magic_div_u32_t mdiv_3 = magic_div_u32_gen(x);
    magic_div_u32_t mdiv_4 = magic_div_u32_gen(b);
    magic_div_u32_t mdiv_5 = magic_div_u32_gen(wo);
    magic_div_u32_t mdiv_6 =
        magic_div_u32_gen((n * b * (gemm_m)) / (cfg.gemm_m_per_block * cfg.gemm_n_per_block));

    uint32_t magic_0 = mdiv_0.magic;
    uint32_t magic_1 = mdiv_1.magic;
    uint32_t magic_2 = mdiv_2.magic;
    uint32_t magic_3 = mdiv_3.magic;
    uint32_t magic_4 = mdiv_4.magic;
    uint32_t magic_5 = mdiv_5.magic;
    uint32_t magic_6 = mdiv_6.magic;
    uint32_t shift_pack_0 =
        magic_div_u32_pack_shift(mdiv_0.shift, mdiv_1.shift, mdiv_2.shift, mdiv_3.shift);
    uint32_t shift_pack_1 = magic_div_u32_pack_shift(mdiv_4.shift, mdiv_5.shift, mdiv_6.shift, 0);

    opArgs.emplace_back(hi);
    opArgs.emplace_back(wi);
    opArgs.emplace_back(n);
    opArgs.emplace_back(k);
    opArgs.emplace_back(c);
    opArgs.emplace_back(ho);
    opArgs.emplace_back(wo);
    opArgs.emplace_back(stride_h);
    opArgs.emplace_back(stride_w);
    opArgs.emplace_back(dilation_h);
    opArgs.emplace_back(dilation_w);
    opArgs.emplace_back(pad_h);
    opArgs.emplace_back(pad_w);
    opArgs.emplace_back(y);
    opArgs.emplace_back(x);
    if(conv_problem.IsFp16())
    {
        opArgs.emplace_back(group);
        opArgs.emplace_back(magic_0);
        opArgs.emplace_back(magic_1);
        opArgs.emplace_back(magic_2);
        opArgs.emplace_back(magic_3);
        opArgs.emplace_back(magic_4);
        opArgs.emplace_back(magic_5);
        opArgs.emplace_back(magic_6);
        opArgs.emplace_back(shift_pack_0);
        opArgs.emplace_back(shift_pack_1);
    }
    opArgs.emplace_back(pack0);

    return opArgs;
}

template <typename T>
static inline InvokerFactory MakeImplGemmDynamicForwardInvokerFactory(const ConvolutionContext& ctx,
                                                                      const T& cfg)
{
    const auto& conv_problem = ctx.conv_problem;
    auto opShapeArgs         = ComputeDynamicIGemmForwardKernelArgs<T>(conv_problem, cfg);
    return [opShapeArgs](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            const auto k            = handle.Run(kernels[0]);

            std::vector<OpKernelArg> opArgs;
            opArgs.reserve(3 + opShapeArgs.size()); // Avoids vector resize.
            opArgs.emplace_back(tensors.in);
            opArgs.emplace_back(tensors.w);
            opArgs.emplace_back(tensors.out);

            std::transform(opShapeArgs.begin(),
                           opShapeArgs.end(),
                           std::back_inserter(opArgs),
                           [](const OpKernelArg& arg) { return arg; });

            k(opArgs);
        };
    };
}

InvokerFactory MakeImplGemmDynamicForward1x1InvokerFactory(const ConvolutionContext& ctx);

template <typename T = int>
InvokerFactory MakeImplGemmDynamicBackwardDataInvokerFactory(const ConvolutionContext& ctx,
                                                             const T& cfg);

template <>
InvokerFactory MakeImplGemmDynamicBackwardDataInvokerFactory<int>(const ConvolutionContext& ctx,
                                                                  const int& cfg);

template <>
InvokerFactory
MakeImplGemmDynamicBackwardDataInvokerFactory<solver::TunableImplicitGemmGTCDynamic_t>(
    const ConvolutionContext& ctx, const solver::TunableImplicitGemmGTCDynamic_t& cfg);

} // namespace conv
} // namespace miopen
