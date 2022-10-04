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
#include <miopen/solver.hpp>
#include <miopen/magic_div.hpp>
#include <vector>

namespace miopen {
namespace conv {

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

    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(0); // placeholder
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

    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(0); // placeholder
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

    opArgs.emplace_back(pack0);

    return opArgs;
}

template <typename T>
static inline InvokerFactory
MakeImplGemmDynamicForwardInvokerFactory(const miopen::ProblemDescription& problem, const T& cfg)
{
    const auto& conv_problem = problem.conv_problem;
    auto opArgs              = ComputeDynamicIGemmForwardKernelArgs<T>(conv_problem, cfg);
    return [opArgs](const std::vector<Kernel>& kernels) mutable {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
            decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            const auto k            = handle.Run(kernels[0]);

            opArgs[0] = OpKernelArg(tensors.in);
            opArgs[1] = OpKernelArg(tensors.w);
            opArgs[2] = OpKernelArg(tensors.out);

            k(opArgs);
        };
    };
}

InvokerFactory
MakeImplGemmDynamicForward1x1InvokerFactory(const miopen::ProblemDescription& problem);

template <typename T = int>
InvokerFactory
MakeImplGemmDynamicBackwardDataInvokerFactory(const miopen::ProblemDescription& problem,
                                              const T& cfg);

template <>
InvokerFactory
MakeImplGemmDynamicBackwardDataInvokerFactory<int>(const miopen::ProblemDescription& problem,
                                                   const int& cfg);

template <>
InvokerFactory
MakeImplGemmDynamicBackwardDataInvokerFactory<solver::TunableImplicitGemmGTCDynamic_t>(
    const miopen::ProblemDescription& problem, const solver::TunableImplicitGemmGTCDynamic_t& cfg);

InvokerFactory MakeImplGemmDynamicForwardXdlopsNHWCInvokerFactory(
    const ConvolutionContext& ctx,
    const miopen::ProblemDescription& problem,
    const solver::PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC& config);
InvokerFactory MakeImplGemmDynamicBackwardDataXdlopsNHWCInvokerFactory(
    const ConvolutionContext& ctx,
    const miopen::ProblemDescription& problem,
    const solver::PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC& config);
InvokerFactory MakeImplGemmDynamicForwardDlopsNCHWCInvokerFactory(
    const miopen::ProblemDescription& problem,
    const solver::PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC& config);
} // namespace conv
} // namespace miopen
