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

std::function<void(std::ostream&)>
PrintDHW(char sep, int spatial_dims, int depth, int height, int width)
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

void ProblemDescription::BuildConfKey(std::string& conf_key) const
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
    ss << 'x' << PrintDHW('x', GetSpatialDims(), GetPadD(), GetPadH(), GetPadW());
    ss << 'x'
       << PrintDHW(
              'x', GetSpatialDims(), GetKernelStrideD(), GetKernelStrideH(), GetKernelStrideW());
    ss << 'x' << PrintDHW('x', GetSpatialDims(), GetDilationD(), GetDilationH(), GetDilationW());
    ss << 'x' << GetGroupCount();

    switch(GetDirection())
    {
    case Direction::Forward: ss << 'x' << "F"; break;
    case Direction::BackwardData: ss << 'x' << "B"; break;
    case Direction::BackwardWeights: ss << 'x' << "W"; break;
    }

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
    }else {
        stream << sep << GetInLayout();
        stream << sep << GetWeightsLayout();
        stream << sep << GetOutLayout();
    }
    stream << sep << EncodeDataTypesForKey(GetInDataType(), GetWeightsDataType(), GetOutDataType());

    switch(GetDirection())
    {
    case Direction::Forward: stream << sep << "F"; break;
    case Direction::BackwardData: stream << sep << "B"; break;
    case Direction::BackwardWeights: stream << sep << "W"; break;
    }

    // clang-format on
    // New performance config entries shall come into variable/optional part of db key.
    // This is to support backward compatibility with previous versions of databases.
    std::ostringstream optional;
    {
        // Group count > 1 identifies Group/Depthwise modes.
        if(GetGroupCount() != 1)
            optional << 'g' << GetGroupCount();
    }
    if(!optional.str().empty())
    {
        stream << '_' << optional.str();
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

} // namespace conv
} // namespace miopen
