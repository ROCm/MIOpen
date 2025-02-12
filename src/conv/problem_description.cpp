/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <miopen/conv/problem_description.hpp>

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/tensor_layout.hpp>

#include <sstream>

namespace miopen {

std::string
EncodeDataTypesForKey(miopenDataType_t in, miopenDataType_t weights, miopenDataType_t out)
{
    if(in == weights && in == out)
        return GetDataTypeName(in);
    return GetDataTypeName(in) + GetDataTypeName(weights) + GetDataTypeName(out);
}

namespace conv {
namespace {

std::function<void(std::ostream&)>
PrintDHW(char sep, unsigned spatial_dims, int64_t depth, int64_t height, int64_t width)
{
    return [=](std::ostream& stream) {
        if(spatial_dims > 2)
            stream << depth << sep;
        stream << height << sep << width;
    };
}

std::ostream& operator<<(std::ostream& stream, std::function<void(std::ostream&)>&& manipulator)
{
    manipulator(stream);
    return stream;
}

} // namespace

miopenAlphaBetaCase_t ClassifyAlphaBeta(const Scalar& alpha, const Scalar& beta)
{
    // double since we are comparing
    double alpha_val = alpha.GetAsDouble();
    double beta_val  = beta.GetAsDouble();

    bool alpha_one  = (alpha_val == 1.0);
    bool alpha_zero = (alpha_val == 0.0);
    bool beta_zero  = (beta_val == 0.0);

    if(alpha_one && beta_zero)
    {
        return DEFAULT;
    }

    if(!alpha_one && beta_zero)
    {
        return SCALE;
    }

    if(!alpha_zero && !beta_zero)
    {
        return BILINEAR;
    }

    return ERROR_STATE;
}

std::string ProblemDescription::GetDirectionStr() const
{
    std::string s;

    switch(GetDirection())
    {
    case Direction::Forward: s = "F"; break;
    case Direction::BackwardData: s = "B"; break;
    case Direction::BackwardWeights: s = "W"; break;
    default: assert(false);
    }

    return s;
}

std::string ProblemDescription::GetAlphaBetaCaseStr() const
{
    switch(GetAlphaBetaCase())
    {
    case BILINEAR: return "Bilinear";
    case SCALE: return "Scale";
    case DEFAULT: return "Default";
    default: MIOPEN_THROW(miopenStatusInvalidValue, "Alpha Beta Case in ERROR_STATE");
    }
}

void ProblemDescription::HeuristicUpdateLayouts()
{
    static const std::vector<std::string> supported_layouts = {"NCHW", "NHWC", "CHWN", "NCDHW"};

    for(const std::string& layout : supported_layouts)
    {
        if(in.IsPossibleLayout4D5D(layout) && out.IsPossibleLayout4D5D(layout) &&
           weights.IsPossibleLayout4D5D(layout))
        {
            in_layout      = layout;
            weights_layout = layout;
            out_layout     = layout;
            return;
        }
    }
    // If we did not find consistent layout, leave them as-is
}

void ProblemDescription::MakeNetworkConfig(std::string& conf_key) const
{
    std::ostringstream ss;

    ss << GetInChannels();
    ss << 'x' << PrintDHW('x', GetSpatialDims(), GetInDepth(), GetInHeight(), GetInWidth());
    ss << 'x'
       << PrintDHW('x', GetSpatialDims(), GetWeightsDepth(), GetWeightsHeight(), GetWeightsWidth());
    ss << 'x' << GetOutChannels();
    ss << 'x' << PrintDHW('x', GetSpatialDims(), GetOutDepth(), GetOutHeight(), GetOutWidth());
    ss << 'x' << GetInBatchSize();
    if((GetInLayout() == "NCHW" && GetWeightsLayout() == "NCHW" && GetOutLayout() == "NCHW") ||
       (GetInLayout() == "NCDHW" && GetWeightsLayout() == "NCDHW" && GetOutLayout() == "NCDHW"))
    {
        ss << 'x' << GetInLayout();
    }
    else
    {
        ss << 'x' << GetInLayout();
        ss << 'x' << GetWeightsLayout();
        ss << 'x' << GetOutLayout();
    }
    ss << 'x' << EncodeDataTypesForKey(GetInDataType(), GetWeightsDataType(), GetOutDataType());

    std::ostringstream optional;
    if(const auto ct = GetInCastType())
        optional << "ci" << GetDataTypeName(*ct);
    if(const auto ct = GetWeightsCastType())
        optional << "cw" << GetDataTypeName(*ct);
    if(const auto ct = GetOutCastType())
        optional << "co" << GetDataTypeName(*ct);
    if(!optional.str().empty())
    {
        ss << 'x' << optional.str();
    }

    ss << 'x' << PrintDHW('x', GetSpatialDims(), GetPadD(), GetPadH(), GetPadW());
    ss << 'x'
       << PrintDHW(
              'x', GetSpatialDims(), GetKernelStrideD(), GetKernelStrideH(), GetKernelStrideW());
    ss << 'x' << PrintDHW('x', GetSpatialDims(), GetDilationD(), GetDilationH(), GetDilationW());
    ss << 'x' << GetGroupCount();
    ss << 'x' << GetDirectionStr();
    ss << 'x' << GetAlphaBetaCaseStr();

    conf_key = ss.str();
}

void ProblemDescription::Serialize(std::ostream& stream) const
{
    const auto sep = '-';
    // Problem description with default layout
    // 576-4-4-1x1-192-4-4-8-1x1-2x2-3x3-0-NCHW-FP32-F
    // Problem description with non-default layout
    // 576-4-4-1x1-192-4-4-8-1x1-2x2-3x3-0-NHWC-NCHW-NCHW-FP32-F
    // clang-format off
    stream << GetInChannels();
    stream << sep << PrintDHW(sep, GetSpatialDims(), GetInDepth(), GetInHeight(), GetInWidth());
    stream << sep << PrintDHW('x', GetSpatialDims(), GetWeightsDepth(), GetWeightsHeight(), GetWeightsWidth());
    stream << sep << GetOutChannels();
    stream << sep << PrintDHW(sep, GetSpatialDims(), GetOutDepth(), GetOutHeight(), GetOutWidth());
    stream << sep << GetInBatchSize();
    stream << sep << PrintDHW('x', GetSpatialDims(), GetPadD(), GetPadH(), GetPadW());
    stream << sep << PrintDHW('x', GetSpatialDims(), GetKernelStrideD(), GetKernelStrideH(), GetKernelStrideW());
    stream << sep << PrintDHW('x', GetSpatialDims(), GetDilationD(), GetDilationH(), GetDilationW());
    stream << sep << GetBias();
    if ((GetInLayout() == "NCHW" && GetWeightsLayout() == "NCHW" && GetOutLayout() == "NCHW")
        || (GetInLayout() == "NCDHW" && GetWeightsLayout() == "NCDHW" && GetOutLayout() == "NCDHW"))
    {
        stream << sep << GetInLayout();
    } else {
        stream << sep << GetInLayout();
        stream << sep << GetWeightsLayout();
        stream << sep << GetOutLayout();
    }
    stream << sep << EncodeDataTypesForKey(GetInDataType(), GetWeightsDataType(), GetOutDataType());
    stream << sep << GetDirectionStr();

    // clang-format on
    // New performance config entries shall come into variable/optional part of db key.
    // This is to support backward compatibility with previous versions of databases.
    std::ostringstream optional;
    {
        // Group count > 1 identifies Group/Depthwise modes.
        if(GetGroupCount() != 1)
            optional << "_g" << GetGroupCount();

        if(const auto ct = GetInCastType())
            optional << "_ci" << GetDataTypeName(*ct);
        if(const auto ct = GetWeightsCastType())
            optional << "_cw" << GetDataTypeName(*ct);
        if(const auto ct = GetOutCastType())
            optional << "_co" << GetDataTypeName(*ct);
    }
    if(!optional.str().empty())
    {
        stream << optional.str();
    }
}

bool ProblemDescription::IsLayoutDefault() const
{
    if(GetSpatialDims() == 2)
    {
        return (in_layout == "NCHW") && (out_layout == "NCHW") && (weights_layout == "NCHW");
    }
    else
    {
        return (in_layout == "NCDHW") && (out_layout == "NCDHW") && (weights_layout == "NCDHW");
    }
}

bool ProblemDescription::IsLayoutNHWC() const
{
    if(GetSpatialDims() == 2)
    {
        return (in_layout == "NHWC") && (out_layout == "NHWC") && (weights_layout == "NHWC");
    }
    else
    {
        return (in_layout == "NDHWC") && (out_layout == "NDHWC") && (weights_layout == "NDHWC");
    }
}

bool ProblemDescription::IsLayoutNCHWc() const
{
    return GetSpatialDims() == 2 && (IsNCHWc_NCHWc() || IsNCHWc_CHWNc());
}

bool ProblemDescription::IsNCHWc_NCHWc() const
{
    return GetInLayout() == "NCHWc" && GetWeightsLayout() == "NCHWc" && GetOutLayout() == "NCHWc";
}

bool ProblemDescription::IsNCHWc_CHWNc() const
{
    return GetInLayout() == "NCHWc" && GetWeightsLayout() == "CHWNc" && GetOutLayout() == "NCHWc";
}

void ProblemDescription::SetupFloats(ExecutionContext& ctx) const
{
    if(IsFp32() || IsFp16() || IsBfp16() || IsInt8() || IsFp8() || IsBfp8())
    {
        ctx.general_compile_options += GetDataTypeKernelParams(GetInDataType());
        return;
    }

    MIOPEN_LOG_W("Unsupported data types configuration: "
                 << GetDataTypeName(GetInDataType()) << "x" << GetDataTypeName(GetWeightsDataType())
                 << "x" << GetDataTypeName(GetOutDataType()));
}

std::string ProblemDescription::ComputeLayout(const TensorDescriptor& td) const
{
    return td.GetLayout_str();
}

std::string ProblemDescription::ComputeInLayout() const { return ComputeLayout(in); }

std::string ProblemDescription::ComputeOutLayout() const { return ComputeLayout(out); }

std::string ProblemDescription::ComputeWeightsLayout() const { return ComputeLayout(weights); }

} // namespace conv
} // namespace miopen
