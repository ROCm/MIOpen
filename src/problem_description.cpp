#include <miopen/problem_description.hpp>

#include <miopen/convolution.hpp>

#include <functional>
#include <sstream>
#include <tuple>

namespace miopen {

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

int ProblemDescription::mloBuildConf_Key(std::string& conf_key) const
{
    conv_problem.BuildConfKey(conf_key);
    return (0);
}

bool ProblemDescription::IsLayoutDefault() const { return conv_problem.IsLayoutDefault(); }

bool ProblemDescription::IsLayoutNHWC() const
{
    if(GetSpatialDims() == 2)
    {
        return (GetInLayout() == "NHWC") && (GetOutLayout() == "NHWC") &&
               (GetWeightsLayout() == "NHWC");
    }
    else
    {
        return (GetInLayout() == "NDHWC") && (GetOutLayout() == "NDHWC") &&
               (GetWeightsLayout() == "NDHWC");
    }
}

bool ProblemDescription::IsLayoutNCHWC() const
{
    return ((GetSpatialDims() == 2) && (GetInLayout() == "NCHWc") && (GetOutLayout() == "NCHWc") &&
            (GetWeightsLayout() == "NCHWc")) ||
           ((GetSpatialDims() == 2) && (GetInLayout() == "NCHWc") && (GetOutLayout() == "NCHWc") &&
            (GetWeightsLayout() == "CHWNc"));
}

void ProblemDescription::Serialize(std::ostream& stream) const
{
    const auto sep = '-';
    // Problem description with default NCHW-NCHW-NCHW layout
    // 576-4-4-1x1-192-4-4-8-1x1-2x2-3x3-0-NCHW-FP32-F
    // Problem description with non-default layout
    // 576-4-4-1x1-192-4-4-8-1x1-2x2-3x3-0-NHWC-NCHW-NCHW-FP32-F
    // clang-format off
    stream << GetInChannels();
    stream << sep << PrintDHW(sep, GetSpatialDims(), GetInDepth(), GetInHeight(), GetInWidth());
    stream << sep << PrintDHW('x', GetSpatialDims(), GetWeightsDepth(), GetWeightsHeight(), GetWeightsWidth());
    stream << sep << GetOutChannels();
    stream << sep << PrintDHW(sep, GetSpatialDims(), GetOutDepth(), GetOutHeight(), GetOutWidth());
    stream << sep << GetBatchSize();
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
    stream << sep << (direction.IsForward() ? "F" : direction.IsBackwardData() ? "B" : "W");
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

ProblemDescription::ProblemDescription(const TensorDescriptor& in,
                                       const TensorDescriptor& weights,
                                       const TensorDescriptor& out,
                                       const ConvolutionDescriptor& conv,
                                       conv::Direction dir,
                                       int bias_)
    : ProblemDescription(dir == conv::Direction::Forward
                             ? conv::ProblemDescription{in, weights, out, conv, dir, bias_}
                             : conv::ProblemDescription{out, weights, in, conv, dir, bias_})
{
}

ProblemDescription::ProblemDescription(conv::ProblemDescription desc)
    : conv_problem(std::move(desc)),
      direction(conv_problem.GetDirection())
{
}

} // namespace miopen
