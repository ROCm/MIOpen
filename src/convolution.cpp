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

#include <miopen/any_solver.hpp>
#include <miopen/config.h>
#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/find_controls.hpp>
#include <miopen/handle.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_layout.hpp>
#include <miopen/algorithm.hpp>

#include <nlohmann/json.hpp>

#include <cassert>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <ostream>

#include <boost/range/combine.hpp>
#include <boost/range/adaptors.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_WINOGRAD)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_GEMM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_FFT)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK)

namespace miopen {

namespace {

std::size_t GetMaxWorkSpaceSize(const std::vector<std::pair<std::string, std::size_t>>& values)
{
    std::size_t sz = 0;
    for(const auto& pr : values)
    {
        if(sz < pr.second)
        {
            MIOPEN_LOG_I2(sz << " < " << pr.second);
            sz = pr.second;
        }
    }
    return sz;
}

std::size_t GetWorkSpaceSizeGEMM(const miopen::ExecutionContext& ctx,
                                 const conv::ProblemDescription& problem)
{
#if MIOPEN_USE_GEMM
    if(env::disabled(MIOPEN_DEBUG_CONV_GEMM) ||
       miopen::any_of(problem.GetConv().GetConvDilations(), [](auto v) { return v > 1; }))
        return 0;

    return GetMaxWorkSpaceSize(AllGemmWorkspaceSize(ctx, problem));
#else
    std::ignore = ctx;
    std::ignore = problem;
    return 0;
#endif
}

std::size_t GetWorkSpaceSizeImplicitGemm(const miopen::ExecutionContext& ctx,
                                         const conv::ProblemDescription& problem)
{
    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM))
        return 0;
    return GetMaxWorkSpaceSize(FindAllImplicitGemmWorkspaceSizes(ctx, problem));
}

std::size_t GetWorkSpaceSizeDirect(const miopen::ExecutionContext& ctx,
                                   const conv::ProblemDescription& problem)
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT))
        return 0;
    return GetMaxWorkSpaceSize(AllDirectForwardBackwardDataWorkspaceSize(ctx, problem));
}

std::size_t GetWorkSpaceSizeFFT(const miopen::ExecutionContext& ctx,
                                const conv::ProblemDescription& problem)
{
    if(env::disabled(MIOPEN_DEBUG_CONV_FFT))
        return 0;
    return GetMaxWorkSpaceSize(AllFFTForwardBackwardDataWorkspaceSize(ctx, problem));
}

std::size_t GetWorkSpaceSizeWinograd(const miopen::ExecutionContext& ctx,
                                     const conv::ProblemDescription& problem)
{
    if(env::disabled(MIOPEN_DEBUG_CONV_WINOGRAD))
        return 0;
    return GetMaxWorkSpaceSize(FindAllWinogradWorkspaceSizes(ctx, problem));
}

std::size_t GetWorkSpaceSizeDirectWrW(const miopen::ExecutionContext& ctx,
                                      const conv::ProblemDescription& problem)
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT))
        return 0;
    return GetMaxWorkSpaceSize(AllDirectBwdWrW2DWorkspaceSize(ctx, problem));
}

std::size_t GetWorkSpaceSizeWinogradWrW(const miopen::ExecutionContext& ctx,
                                        const conv::ProblemDescription& problem)
{
    if(env::disabled(MIOPEN_DEBUG_CONV_WINOGRAD))
        return 0;
    return GetMaxWorkSpaceSize(FindWinogradWrWWorkspaceSizes(ctx, problem));
}

std::size_t GetWorkSpaceSizeImplicitGemmWrW(const miopen::ExecutionContext& ctx,
                                            const conv::ProblemDescription& problem)
{
    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM))
        return 0;
    return GetMaxWorkSpaceSize(FindImplicitGemmWrWWorkspaceSizes(ctx, problem));
}

} // namespace

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

    std::size_t out_c = 0;
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
                        2 * static_cast<std::ptrdiff_t>(GetConvPads()[i]) +
                        GetTransposeConvPads()[i]);
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
                     2 * static_cast<std::ptrdiff_t>(GetConvPads()[i])) /
                            GetConvStrides()[i] +
                        1);
            }
        }
    }
    else
        MIOPEN_THROW(miopenStatusInvalidValue, "Invalid Padding Mode!");

    out_lens[0] = in_n;
    out_lens[1] = out_c;

    const std::string default_layout = tensor_layout_get_default(xDesc.GetNumDims());
    std::vector<std::size_t> out_strides;
    tensor_layout_to_strides(
        out_lens, default_layout, yLayout, xDesc.GetVectorLength(), out_strides);
    return {(xDesc.GetType() == miopenInt8
                 ? (yType)
                 : xDesc.GetType()), // TODO: This function overrides the output type with
                                     // essentially the input which is incorrect.
            xDesc.GetLayout_t(),
            out_lens,
            out_strides};
}

TensorDescriptor ConvolutionDescriptor::GetForwardOutputTensor(const TensorDescriptor& xDesc,
                                                               const TensorDescriptor& wDesc,
                                                               miopenDataType_t yType) const
{
    // output layout same as input
    const std::string in_layout = xDesc.GetLayout_str();
    return GetForwardOutputTensorWithLayout(xDesc, wDesc, in_layout, yType);
}

/// There is assumption that if Winograd is applicable and granularity loss is low, then there is no
/// advantage in trying other algorithms as those either slower or use more workspace. This allows
/// for some related host-side optimizations.
///
/// These optimizations are kind of cutting corners, but advantages are quite high.
bool ConvolutionDescriptor::IsWinograd3x3SupportedAndFast(
    const miopen::ExecutionContext& ctx, const conv::ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_WINOGRAD))
        return false;

    // Disable this performance optimization when we want to run some specific Solver.
    // Other Solvers will be skipped anyway.
    if(GetEnvFindOnlySolver())
        return false;

    // Filter out configs where 3x3 Winograd does not have high WTI.
    if(!(problem.GetOutChannels() >= 16 && problem.GetOutChannels() % 2 == 0))
        return false;

    return solver::conv::ConvBinWinograd3x3U{}.IsApplicable(ctx, problem);
}

std::size_t ConvolutionDescriptor::GetWorkSpaceSize(ExecutionContext ctx,
                                                    const conv::ProblemDescription& problem) const
{
    MIOPEN_LOG_I2("");

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
        auto fallback        = bool{};
        const auto solutions = GetSolutions(ctx, problem, 1, &fallback);
        if(solutions.empty() || ((findMode.IsHybrid(ctx) && fallback) &&
                                 !env::enabled(MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK)))
        {
            ctx.use_dynamic_solutions_only = findMode.IsDynamicHybrid(ctx);
            break; // Fall down to Normal Find.
        }
        const auto id             = solver::Id{solutions.front().solution_id};
        const auto& s             = id.GetSolver();
        const auto workspace_size = s.GetWorkspaceSize(ctx, problem);

        MIOPEN_LOG_I(workspace_size);
        return workspace_size;
    }

    size_t workspace_size;

    if(problem.GetDirection() != conv::Direction::BackwardWeights)
    {
        if(IsWinograd3x3SupportedAndFast(ctx, problem))
        {
            ctx.use_dynamic_solutions_only = true;
            workspace_size                 = GetWorkSpaceSizeWinograd(ctx, problem);
        }
        else
        {
            workspace_size = std::max({GetWorkSpaceSizeFFT(ctx, problem),
                                       GetWorkSpaceSizeGEMM(ctx, problem),
                                       GetWorkSpaceSizeDirect(ctx, problem),
                                       GetWorkSpaceSizeImplicitGemm(ctx, problem),
                                       GetWorkSpaceSizeWinograd(ctx, problem)});
        }
    }
    else
    {
        workspace_size = std::max({GetWorkSpaceSizeGEMM(ctx, problem),
                                   GetWorkSpaceSizeDirectWrW(ctx, problem),
                                   GetWorkSpaceSizeImplicitGemmWrW(ctx, problem),
                                   GetWorkSpaceSizeWinogradWrW(ctx, problem)});
    }

    MIOPEN_LOG_I(workspace_size);
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

void to_json(nlohmann::json& json, const ConvolutionAttribute::Gfx90aFp16alt& attribute)
{
    json = {{"value", attribute.value}};
}

void from_json(const nlohmann::json& json, ConvolutionAttribute::Gfx90aFp16alt& attribute)
{
    json.at("value").get_to(attribute.value);
}

void ConvolutionAttribute::Set(miopenConvolutionAttrib_t attr, int value)
{
    if(attr == MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL)
    {
        if(value < -1 || value > 1)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "[Set conv attribute] Error: Attempt to set invalid value of "
                         "MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL: " +
                             std::to_string(value));
        }
        gfx90aFp16alt.value = value;
    }
    else if(attr == MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC)
    {
        if(value < 0 || value > 1)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "[Set conv attribute] Error: Attemp to set invalid value for "
                         "MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC: " +
                             std::to_string(value));
        }
        deterministic.value = value;
    }
    else if(attr == MIOPEN_CONVOLUTION_ATTRIB_FP8_ROUNDING_MODE)
    {
        const auto rounding_mode = static_cast<miopenF8RoundingMode_t>(value);
        if(rounding_mode != miopenF8RoundingModeStochastic &&
           rounding_mode != miopenF8RoundingModeStandard)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "[Set conv attribute] Error: Attempt to set invalid value for "
                         "MIOPEN_CONVOLUTION_ATTRIB_FP8_ROUNDING_MODE" +
                             std::to_string(value));
        }
        fp8rounding_mode.rounding_mode = rounding_mode;
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
    else if(attr == MIOPEN_CONVOLUTION_ATTRIB_FP8_ROUNDING_MODE)
        return static_cast<int>(fp8rounding_mode.rounding_mode);
    else if(attr == MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC)
        return deterministic.value;
    MIOPEN_THROW(miopenStatusBadParm,
                 "[Get conv attribute] Error: Attribute [" +
                     std::to_string(static_cast<int>(attr)) + "] does not exist.");
}

void to_json(nlohmann::json& json, const ConvolutionAttribute& conv)
{
    json = {{"gfx90aFp16alt", conv.gfx90aFp16alt}};
}

void from_json(const nlohmann::json& json, ConvolutionAttribute& conv)
{
    json.at("gfx90aFp16alt").get_to(conv.gfx90aFp16alt);
}

void to_json(nlohmann::json& json, const ConvolutionDescriptor& conv)
{
    json = nlohmann::json{
        {"spatialDim", conv.spatialDim},
        {"mode", conv.mode},
        {"paddingMode", conv.paddingMode},
        {"pads", conv.pads},
        {"strides", conv.strides},
        {"dilations", conv.dilations},
        {"transOutputPads", conv.trans_output_pads},
        {"groupCount", conv.group_count},
        {"lowpQuant", conv.lowp_quant},
        {"attribute", conv.attribute},
    };
}

void from_json(const nlohmann::json& json, ConvolutionDescriptor& conv)
{
    json.at("spatialDim").get_to(conv.spatialDim);
    json.at("mode").get_to(conv.mode);
    json.at("paddingMode").get_to(conv.paddingMode);
    json.at("pads").get_to(conv.pads);
    json.at("strides").get_to(conv.strides);
    json.at("dilations").get_to(conv.dilations);
    json.at("transOutputPads").get_to(conv.trans_output_pads);
    json.at("groupCount").get_to(conv.group_count);
    json.at("lowpQuant").get_to(conv.lowp_quant);
    json.at("attribute").get_to(conv.attribute);
}

} // namespace miopen
