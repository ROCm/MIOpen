/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <miopen/convolution.hpp>

#include <miopen/config.h>
#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/find_controls.hpp>
#include <miopen/handle.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/solver.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_layout.hpp>
#include <miopen/algorithm.hpp>

#include <cassert>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <ostream>

#include <boost/range/combine.hpp>
#include <boost/range/adaptors.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_WINOGRAD)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_GEMM)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_FFT)

namespace miopen {

ConvolutionDescriptor::ConvolutionDescriptor(std::size_t spatial_dim,
                                             miopenConvolutionMode_t c_mode,
                                             miopenPaddingMode_t p_mode,
                                             const std::vector<int>& p_pads,
                                             const std::vector<int>& p_strides,
                                             const std::vector<int>& p_dilations,
                                             const std::vector<int>& p_trans_output_pads,
                                             int p_group_count,
                                             float p_lowp_quant)
    : spatialDim(spatial_dim),
      mode(c_mode),
      paddingMode(p_mode),
      pads(p_pads),
      strides(p_strides),
      dilations(p_dilations),
      trans_output_pads(p_trans_output_pads),
      group_count(p_group_count),
      lowp_quant(p_lowp_quant)
{
    if(pads.size() != spatial_dim || strides.size() != spatial_dim ||
       dilations.size() != spatial_dim || trans_output_pads.size() != spatial_dim ||
       miopen::any_of(pads, [](auto v) { return v < 0; }) ||
       miopen::any_of(strides, [](auto v) { return v < 1; }) ||
       miopen::any_of(dilations, [](auto v) { return v < 1; }))
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "Invalid parameters, check usage. MIOPEN expects padding "
                     ">= 0, stride >= 1, dilation >= 1 and the same dilation "
                     "factor for horizontal and vertical direction");
    }
    if(!(mode == miopenConvolution || mode == miopenTranspose))
    {
        if(mode == miopenGroupConv || mode == miopenDepthwise)
        {
            mode = miopenConvolution;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm, "Convolution mode not supported");
        }
    }
    if(!(paddingMode == miopenPaddingSame || paddingMode == miopenPaddingValid ||
         paddingMode == miopenPaddingDefault))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Padding mode not supported");
    }
}

ConvolutionDescriptor::ConvolutionDescriptor(const std::vector<int>& p_pads,
                                             const std::vector<int>& p_strides,
                                             const std::vector<int>& p_dilations,
                                             const std::vector<int>& p_trans_output_pads,
                                             int p_group_count,
                                             float p_lowp_quant)
    : ConvolutionDescriptor{p_pads.size(),
                            miopenConvolution,
                            miopenPaddingDefault,
                            p_pads,
                            p_strides,
                            p_dilations,
                            p_trans_output_pads,
                            p_group_count,
                            p_lowp_quant}
{
}

std::size_t ConvolutionDescriptor::GetSpatialDimension() const { return spatialDim; }

const std::vector<int>& ConvolutionDescriptor::GetConvPads() const { return pads; }

const std::vector<int>& ConvolutionDescriptor::GetConvStrides() const { return strides; }

const std::vector<int>& ConvolutionDescriptor::GetConvDilations() const { return dilations; }

const std::vector<int>& ConvolutionDescriptor::GetTransposeConvPads() const
{
    return trans_output_pads;
}

int ConvolutionDescriptor::GetGroupCount() const { return group_count; }

TensorDescriptor
ConvolutionDescriptor::GetForwardOutputTensorWithLayout(const TensorDescriptor& xDesc,
                                                        const TensorDescriptor& wDesc,
                                                        const std::string& yLayout,
                                                        miopenDataType_t yType) const
{
    const std::size_t spatial_dim = GetSpatialDimension();

    assert(xDesc.GetLengths().size() == spatial_dim + 2);
    assert(wDesc.GetLengths().size() == spatial_dim + 2);

    if(xDesc.GetType() != wDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Types do not match for the filter");
    }

    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = miopen::tie_pick<0, 1>{}(xDesc.GetLengths());

    auto in_spatial = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);

    std::size_t wei_k, wei_c;
    std::tie(wei_k, wei_c) = miopen::tie_pick<0, 1>{}(wDesc.GetLengths());

    auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);

    if(wDesc.GetLayout_str() == "CHWNc")
    {
        std::tie(wei_k, wei_c) = miopen::tie_pick<3, 0>{}(wDesc.GetLengths());
        wei_spatial            = boost::adaptors::slice(wDesc.GetLengths(), 1, 1 + spatial_dim);
    }

    if(mode == miopenConvolution)
    {
        // for depthwise conv wei_c must be 1 while group_count must be wei_c
        if((group_count == 1 && in_c != wei_c) ||
           (group_count > 1 && (in_c % wei_c != 0 || wei_k % (in_c / wei_c) != 0)))
        {
            MIOPEN_THROW(miopenStatusBadParm, "Channels do not match for the filter");
        }
    }
    else if(mode == miopenTranspose)
    {
        if(in_c != wei_k || (group_count > 1 && (wei_k % group_count != 0)))
        {
            MIOPEN_THROW(miopenStatusBadParm, "Channels do not match for the filter");
        }

        if(miopen::any_of(boost::combine(GetTransposeConvPads(), GetConvStrides()), [](auto v) {
               auto trans_conv_pad = boost::get<0>(v);
               auto stride         = boost::get<1>(v);
               return trans_conv_pad >= stride;
           }))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Output shape doesn't match due to invalid output padding");
        }
    }

    std::size_t out_c;
    std::vector<std::size_t> out_lens(spatial_dim + 2);

    auto out_spatial = boost::adaptors::slice(out_lens, 2, 2 + spatial_dim);

    if(paddingMode == miopenPaddingSame && mode == miopenConvolution &&
       miopen::all_of(GetConvDilations(), [](auto v) { return v == 1; }))
    {
        out_c = wei_k;

        for(int i = 0; i < spatial_dim; ++i)
        {
            out_spatial[i] = miopen::integer_division_ceil(in_spatial[i], GetConvStrides()[i]);
        }
    }
    else if(paddingMode == miopenPaddingValid && mode == miopenConvolution &&
            miopen::all_of(GetConvDilations(), [](auto v) { return v == 1; }))
    {
        out_c = wei_k;

        for(int i = 0; i < spatial_dim; ++i)
        {
            out_spatial[i] = miopen::integer_division_ceil(
                std::ptrdiff_t(in_spatial[i]) - wei_spatial[i] + 1, GetConvStrides()[i]);
        }
    }
    else if(paddingMode == miopenPaddingDefault || paddingMode == miopenPaddingSame ||
            paddingMode == miopenPaddingValid)
    {
        if(mode == miopenTranspose)
        {
            out_c = wei_c * group_count;

            for(int i = 0; i < spatial_dim; ++i)
            {
                out_spatial[i] = std::max<std::ptrdiff_t>(
                    1,
                    GetConvStrides()[i] * (std::ptrdiff_t(in_spatial[i]) - 1) + 1 +
                        GetConvDilations()[i] * (std::ptrdiff_t(wei_spatial[i]) - 1) -
                        2 * GetConvPads()[i] + GetTransposeConvPads()[i]);
            }
        }
        else
        {
            out_c = wei_k / wDesc.GetVectorLength();

            for(int i = 0; i < spatial_dim; ++i)
            {
                out_spatial[i] = std::max<std::ptrdiff_t>(
                    1,
                    (ptrdiff_t(in_spatial[i]) -
                     (1 + GetConvDilations()[i] * (std::ptrdiff_t(wei_spatial[i]) - 1)) +
                     2 * GetConvPads()[i]) /
                            GetConvStrides()[i] +
                        1);
            }
        }
    }
    else
        MIOPEN_THROW(miopenStatusInvalidValue, "Invalid Padding Mode!");

    out_lens[0] = in_n;
    out_lens[1] = out_c;

    const std::string default_layout = tensor_layout_get_default(xDesc.GetSize());
    std::vector<std::size_t> out_strides;
    tensor_layout_to_strides(
        out_lens, default_layout, yLayout, xDesc.GetVectorLength(), out_strides);
    return {(xDesc.GetType() == miopenInt8 || xDesc.GetType() == miopenInt8x4
                 ? (yType == miopenInt32 ? yType : miopenFloat)
                 : xDesc.GetType()),
            xDesc.GetLayout_t(),
            out_lens,
            out_strides};
}

TensorDescriptor ConvolutionDescriptor::GetForwardOutputTensor(const TensorDescriptor& xDesc,
                                                               const TensorDescriptor& wDesc,
                                                               miopenDataType_t yType) const
{
    // output layout same as input
    const std::string default_layout = tensor_layout_get_default(xDesc.GetSize());
    const std::string in_layout      = xDesc.GetLayout(default_layout);
    return GetForwardOutputTensorWithLayout(xDesc, wDesc, in_layout, yType);
}

/// There is assumption that if Winograd is applicable and granularity loss is low, then there is no
/// advantage in trying other algorithms as those either slower or use more workspace. This allows
/// for some related host-side optimizations.
///
/// These optimizations are kind of cutting corners, but advantages are quite high.
bool ConvolutionDescriptor::IsWinograd3x3SupportedAndFast(miopen::ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_WINOGRAD{}))
        return false;

    // Disable this performance optimization when we want to run some specific Solver.
    // Other Solvers will be skipped anyway.
    if(GetEnvFindOnlySolver())
        return false;

    // Filter out configs where 3x3 Winograd does not have high WTI.
    if(!(ctx.n_outputs >= 16 && ctx.n_outputs % 2 == 0))
        return false;

    return solver::ConvBinWinograd3x3U{}.IsApplicable(ctx);
}

std::size_t
ConvolutionDescriptor::WrwGetValidWorkSpaceSizeGemm(const TensorDescriptor& dyDesc,
                                                    const TensorDescriptor& xDesc,
                                                    const TensorDescriptor& dwDesc) const
{
#if MIOPEN_USE_GEMM
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
        return 0;

    const auto ctx =
        ConvolutionContext{xDesc, dwDesc, dyDesc, *this, conv::Direction::BackwardWeights};
    decltype(auto) gemm_ws_sz_pairs = AllGemmWorkspaceSize(ctx);

    if(!gemm_ws_sz_pairs.empty())
    {
        decltype(auto) gemm_ws_szs =
            gemm_ws_sz_pairs | boost::adaptors::transformed([](const auto& p) { return p.second; });
        return *std::max_element(gemm_ws_szs.begin(), gemm_ws_szs.end());
    }
#else
    std::ignore = dyDesc;
    std::ignore = xDesc;
    std::ignore = dwDesc;
#endif

    return 0;
}

std::size_t ConvolutionDescriptor::ForwardGetWorkSpaceSize(Handle& handle,
                                                           const TensorDescriptor& wDesc,
                                                           const TensorDescriptor& xDesc,
                                                           const TensorDescriptor& yDesc) const
{
    MIOPEN_LOG_I("");

    auto ctx = ConvolutionContext{xDesc, wDesc, yDesc, *this, conv::Direction::Forward};
    ctx.SetStream(&handle);
    ctx.DetectRocm();
    ctx.SetupFloats();
    ctx.do_search             = false;
    ctx.disable_perfdb_access = true;

    while(findMode.IsFast(ctx) || findMode.IsHybrid(ctx))
    {
        /// \section ffind_gwss_why_not_0
        /// Basically we can return 0 here because
        /// * (A) Find() emulated by Immediate mode does not execute kernels.
        /// * (B) We expect that applications read output of Find() and
        ///   allocate WS for Run phase as indicated there
        ///   (in miopenConvAlgoPerf_t::memory).
        ///
        /// However there are some known apps that allocate WS once
        /// (using size returned by *this* call) and then re-use
        /// the same workspace for Run phase. That is why we shall return
        /// actually required workspace here.
        size_t count;
        miopenConvSolution_t sol;
        bool fallback;
        GetForwardSolutions(handle, wDesc, xDesc, yDesc, 1, &count, &sol, &fallback);
        if(count < 1 || (findMode.IsHybrid(ctx) && fallback))
        {
            ctx.use_dynamic_solutions_only = findMode.IsDynamicHybrid(ctx);
            break; // Fall down to Normal Find.
        }
        MIOPEN_LOG_I2(sol.workspace_size);
        return sol.workspace_size;
    }

    if(IsWinograd3x3SupportedAndFast(ctx))
    {
        AutoUseFastDynamicSolutions tmp{ctx};
        const auto ws = ForwardBackwardDataGetWorkSpaceSizeWinograd(ctx);
        MIOPEN_LOG_I2(ws);
        return ws;
    }
    const size_t workspace_size_winograd = ForwardBackwardDataGetWorkSpaceSizeWinograd(ctx);
    const size_t direct_workspace        = ForwardBackwardDataGetWorkSpaceSizeDirect(ctx);
    const size_t implicit_gemm_workspace = ForwardBackwardGetWorkSpaceSizeImplicitGemm(ctx);

    size_t workspace_size_gemm = 0;
#if MIOPEN_USE_GEMM
    if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
    {
        decltype(auto) gemm_ws_sz_pairs = AllGemmWorkspaceSize(ctx);

        if(!gemm_ws_sz_pairs.empty())
        {
            decltype(auto) gemm_ws_szs =
                gemm_ws_sz_pairs |
                boost::adaptors::transformed([](const auto& p) { return p.second; });
            workspace_size_gemm = *std::max_element(gemm_ws_szs.begin(), gemm_ws_szs.end());
        }

        if(miopen::any_of(GetConvDilations(), [](auto v) { return v > 1; }))
        {
            return std::max({workspace_size_gemm,
                             direct_workspace,
                             implicit_gemm_workspace,
                             workspace_size_winograd});
        }
    }
#endif

    const size_t workspace_size_fft = ForwardBackwardDataGetWorkSpaceSizeFFT(ctx);

    const size_t workspace_size = std::max({workspace_size_fft,
                                            workspace_size_gemm,
                                            direct_workspace,
                                            implicit_gemm_workspace,
                                            workspace_size_winograd});

    MIOPEN_LOG_I2(workspace_size);
    return workspace_size;
}

std::size_t
ConvolutionDescriptor::BackwardDataGetWorkSpaceSize(Handle& handle,
                                                    const TensorDescriptor& wDesc,
                                                    const TensorDescriptor& dyDesc,
                                                    const TensorDescriptor& dxDesc) const
{
    MIOPEN_LOG_I("");

    auto ctx = ConvolutionContext{dxDesc, wDesc, dyDesc, *this, conv::Direction::BackwardData};
    ctx.SetStream(&handle);
    ctx.DetectRocm();
    ctx.SetupFloats();
    ctx.do_search             = false;
    ctx.disable_perfdb_access = true;

    while(findMode.IsFast(ctx) || findMode.IsHybrid(ctx))
    {
        /// \ref ffind_gwss_why_not_0
        size_t count;
        miopenConvSolution_t sol;
        bool fallback;
        GetBackwardSolutions(handle, dyDesc, wDesc, dxDesc, 1, &count, &sol, &fallback);
        if(count < 1 || (findMode.IsHybrid(ctx) && fallback))
        {
            ctx.use_dynamic_solutions_only = findMode.IsDynamicHybrid(ctx);
            break; // Fall down to Normal Find.
        }
        MIOPEN_LOG_I2(sol.workspace_size);
        return sol.workspace_size;
    }

    if(IsWinograd3x3SupportedAndFast(ctx))
    {
        AutoUseFastDynamicSolutions tmp{ctx};
        const auto ws = ForwardBackwardDataGetWorkSpaceSizeWinograd(ctx);
        MIOPEN_LOG_I2(ws);
        return ws;
    }

    const size_t workspace_size_winograd = ForwardBackwardDataGetWorkSpaceSizeWinograd(ctx);
    const size_t direct_workspace        = ForwardBackwardDataGetWorkSpaceSizeDirect(ctx);
    const size_t implicit_gemm_workspace = ForwardBackwardGetWorkSpaceSizeImplicitGemm(ctx);

    size_t workspace_size_gemm = 0;

#if MIOPEN_USE_GEMM
    size_t tmp_max_workspace =
        std::max({direct_workspace, implicit_gemm_workspace, workspace_size_winograd});
    if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
    {
        decltype(auto) gemm_ws_sz_pairs = AllGemmWorkspaceSize(ctx);

        if(!gemm_ws_sz_pairs.empty())
        {
            decltype(auto) gemm_ws_szs =
                gemm_ws_sz_pairs |
                boost::adaptors::transformed([](const auto& p) { return p.second; });
            workspace_size_gemm = *std::max_element(gemm_ws_szs.begin(), gemm_ws_szs.end());
        }

        if(miopen::any_of(GetConvDilations(), [](auto v) { return v > 1; }))
        {
            return std::max({workspace_size_gemm, tmp_max_workspace});
        }
    }
#endif

    const size_t workspace_size_fft = ForwardBackwardDataGetWorkSpaceSizeFFT(ctx);

    const size_t workspace_size = std::max({workspace_size_fft,
                                            workspace_size_gemm,
                                            direct_workspace,
                                            implicit_gemm_workspace,
                                            workspace_size_winograd});
    MIOPEN_LOG_I2(workspace_size);
    return workspace_size;
}

std::size_t ConvolutionDescriptor::BackwardWeightsGetWorkSpaceSizeGEMM(
    const miopen::ConvolutionContext& ctx) const
{
#if MIOPEN_USE_GEMM
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
        return 0;

    decltype(auto) gemm_ws_sz_pairs = AllGemmWorkspaceSize(ctx);

    if(!gemm_ws_sz_pairs.empty())
    {
        decltype(auto) gemm_ws_szs =
            gemm_ws_sz_pairs | boost::adaptors::transformed([](const auto& p) { return p.second; });
        return *std::max_element(gemm_ws_szs.begin(), gemm_ws_szs.end());
    }
#else
    std::ignore = ctx;
#endif

    return 0;
}

std::size_t ConvolutionDescriptor::ForwardBackwardGetWorkSpaceSizeImplicitGemm(
    const miopen::ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM{}))
    {
        return 0;
    }

    try
    {
        const auto sz_v = FindAllImplicitGemmWorkspaceSizes(ctx);
        std::size_t sz  = 0;
        for(const auto& pr : sz_v)
        {
            if(sz < pr.second)
            {
                MIOPEN_LOG_I2(sz << " < " << pr.second);
                sz = pr.second;
            }
        }
        return sz;
    }
    catch(const miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return 0;
    }
}

std::size_t ConvolutionDescriptor::ForwardBackwardDataGetWorkSpaceSizeDirect(
    const miopen::ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{}))
    {
        return 0;
    }

    try
    {
        const auto sz_v = AllDirectForwardBackwardDataWorkspaceSize(ctx);
        std::size_t sz  = 0;
        for(const auto& pr : sz_v)
        {
            if(sz < pr.second)
            {
                MIOPEN_LOG_I2(sz << " < " << pr.second); // solution.workspace_sz);
                sz = pr.second;                          // solution.workspace_sz;
            }
        }
        return sz;
    }
    catch(const miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return 0;
    }
}

std::size_t ConvolutionDescriptor::ForwardBackwardDataGetWorkSpaceSizeFFT(
    const miopen::ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_FFT{}))
        return 0;

    try
    {
        const auto all_ws_sz = AllFFTForwardBackwardDataWorkspaceSize(ctx);
        std::size_t sz       = 0;
        for(const auto& pair : all_ws_sz)
        {
            if(sz < pair.second)
            {
                MIOPEN_LOG_I2(sz << " < " << pair.second);
                sz = pair.second;
            }
        }
        return sz;
    }
    catch(const miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return 0;
    }
}

std::size_t ConvolutionDescriptor::ForwardBackwardDataGetWorkSpaceSizeWinograd(
    const miopen::ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_WINOGRAD{}))
        return 0;

    try
    {
        const auto sz_v = FindAllWinogradWorkspaceSizes(ctx);
        std::size_t sz  = 0;
        for(const auto& pr : sz_v)
        {
            if(sz < pr.second)
            {
                MIOPEN_LOG_I2(sz << " < " << pr.second);
                sz = pr.second;
            }
        }
        return sz;
    }
    catch(const miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return 0;
    }
}

std::size_t ConvolutionDescriptor::BackwardWeightsGetWorkSpaceSizeDirect(
    const miopen::ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{}))
        return 0;

    try
    {
        const auto sz_v = AllDirectBwdWrW2DWorkspaceSize(ctx);
        std::size_t sz  = 0;
        for(const auto& pr : sz_v)
        {
            if(sz < pr.second)
            {
                MIOPEN_LOG_I2(sz << " < " << pr.second);
                sz = pr.second;
            }
        }
        return sz;
    }
    catch(const miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return 0;
    }
}

std::size_t ConvolutionDescriptor::BackwardWeightsGetWorkSpaceSizeWinograd(
    const miopen::ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_WINOGRAD{}))
        return 0;

    try
    {
        if(ctx.do_search)
            MIOPEN_THROW("Auto-tune is not supported in the get workspace size");
        const auto sz_v = FindWinogradWrWWorkspaceSizes(ctx);
        std::size_t sz  = 0;
        for(const auto& pr : sz_v)
        {
            if(sz < pr.second)
            {
                MIOPEN_LOG_I2(sz << " < " << pr.second);
                sz = pr.second;
            }
        }
        return sz;
    }
    catch(const miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return 0;
    }
}

std::size_t ConvolutionDescriptor::BackwardWeightsGetWorkSpaceSizeImplicitGemm(
    const miopen::ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM{}))
        return 0;

    try
    {
        if(ctx.do_search)
            MIOPEN_THROW("Auto-tune is not supported in the get workspace size");
        const auto sz_v = FindImplicitGemmWrWWorkspaceSizes(ctx);
        std::size_t sz  = 0;
        for(const auto& pr : sz_v)
        {
            if(sz < pr.second)
            {
                MIOPEN_LOG_I2(sz << " < " << pr.second);
                sz = pr.second;
            }
        }
        return sz;
    }
    catch(const miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return 0;
    }
}

std::size_t
ConvolutionDescriptor::BackwardWeightsGetWorkSpaceSize(Handle& handle,
                                                       const TensorDescriptor& dyDesc,
                                                       const TensorDescriptor& xDesc,
                                                       const TensorDescriptor& dwDesc) const
{
    MIOPEN_LOG_I("");
    auto ctx = ConvolutionContext(xDesc, dwDesc, dyDesc, *this, conv::Direction::BackwardWeights);
    while(findMode.IsFast(ctx) || findMode.IsHybrid(ctx))
    {
        /// \ref ffind_gwss_why_not_0
        size_t count;
        miopenConvSolution_t sol;
        bool fallback;
        GetWrwSolutions(handle, dyDesc, xDesc, dwDesc, 1, &count, &sol, &fallback);
        if(count < 1 || (findMode.IsHybrid(ctx) && fallback))
        {
            ctx.use_dynamic_solutions_only = findMode.IsDynamicHybrid(ctx);
            break; // Fall down to Normal Find.
        }
        MIOPEN_LOG_I2(sol.workspace_size);
        return sol.workspace_size;
    }

    ctx.SetStream(&handle);
    ctx.DetectRocm();
    ctx.SetupFloats();
    ctx.do_search             = false;
    ctx.disable_perfdb_access = true;

    const size_t workspace_size = std::max({BackwardWeightsGetWorkSpaceSizeImplicitGemm(ctx),
                                            BackwardWeightsGetWorkSpaceSizeWinograd(ctx),
                                            BackwardWeightsGetWorkSpaceSizeDirect(ctx),
                                            BackwardWeightsGetWorkSpaceSizeGEMM(ctx)});
    MIOPEN_LOG_I2(workspace_size);
    return workspace_size;
}

std::ostream& operator<<(std::ostream& stream, const ConvolutionDescriptor& c)
{
    stream << "conv" << c.spatialDim << "d, ";
    MIOPEN_LOG_ENUM(stream, c.mode, miopenConvolution, miopenTranspose) << ", ";
    MIOPEN_LOG_ENUM(
        stream, c.paddingMode, miopenPaddingDefault, miopenPaddingSame, miopenPaddingValid)
        << ", ";

    LogRange(stream << "{", c.GetConvPads(), ", ") << "}, ";
    LogRange(stream << "{", c.GetConvStrides(), ", ") << "}, ";
    LogRange(stream << "{", c.GetConvDilations(), ", ") << "}, ";

    if(c.group_count > 1)
    {
        stream << c.group_count << ", ";
    }

    if(c.mode == miopenTranspose)
    {
        LogRange(stream << "{", c.GetTransposeConvPads(), ", ") << "}, ";
    }

    return stream;
}

void ConvolutionAttribute::Set(miopenConvolutionAttrib_t attr, int value)
{
    if(attr == MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL)
    {
        if(value < -1 || value > 1)
            MIOPEN_THROW(miopenStatusBadParm,
                         "[Set conv attribute] Error: Attempt to set invalid value of "
                         "MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL: " +
                             std::to_string(value));
        gfx90aFp16alt.value = value;
    }
    else if(attr == MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC)
    {
        if(value < 0 || value > 1)
            MIOPEN_THROW(miopenStatusBadParm,
                         "[Set conv attribute] Error: Attemp to set invalid value for "
                         "MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC: " +
                             std::to_string(value));
        deterministic.value = value;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "[Set conv attribute] Error: Attribute [" +
                         std::to_string(static_cast<int>(attr)) + "] does not exist.");
    }
}

int ConvolutionAttribute::Get(miopenConvolutionAttrib_t attr) const
{
    if(attr == MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL)
        return gfx90aFp16alt.value;
    else if(attr == MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC)
        return deterministic.value;
    MIOPEN_THROW(miopenStatusBadParm,
                 "[Get conv attribute] Error: Attribute [" +
                     std::to_string(static_cast<int>(attr)) + "] does not exist.");
}

} // namespace miopen
