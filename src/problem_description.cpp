#include <miopen/problem_description.hpp>

#include <miopen/convolution.hpp>

#include <functional>
#include <sstream>
#include <tuple>

namespace miopen {

std::string
EncodeDataTypesForKey(miopenDataType_t in, miopenDataType_t weights, miopenDataType_t out)
{
    if(in == weights && in == out)
        return GetDataTypeName(in);
    return GetDataTypeName(in) + GetDataTypeName(weights) + GetDataTypeName(out);
}

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
    std::ostringstream ss;

    ss << n_inputs;
    ss << 'x' << PrintDHW('x', spatial_dims, in_depth, in_height, in_width);
    ss << 'x' << PrintDHW('x', spatial_dims, kernel_size_d, kernel_size_h, kernel_size_w);
    ss << 'x' << n_outputs;
    ss << 'x' << PrintDHW('x', spatial_dims, out_depth, out_height, out_width);
    ss << 'x' << batch_sz;
    ss << 'x' << in_layout;
    ss << 'x' << EncodeDataTypesForKey(in_data_type, weights_data_type, out_data_type);
    ss << 'x' << PrintDHW('x', spatial_dims, pad_d, pad_h, pad_w);
    ss << 'x' << PrintDHW('x', spatial_dims, kernel_stride_d, kernel_stride_h, kernel_stride_w);
    ss << 'x'
       << PrintDHW('x', spatial_dims, kernel_dilation_d, kernel_dilation_h, kernel_dilation_w);
    ss << 'x' << group_counts;
    ss << 'x' << (direction.IsForward() ? "1" : "0");

    conf_key = ss.str();

    return (0);
}

void ProblemDescription::Serialize(std::ostream& stream) const
{
    if(!direction.IsKnown())
        MIOPEN_THROW("!direction.IsKnown()");
    const auto sep = '-';
    // 576-4-4-1x1-192-4-4-8-1x1-2x2-3x3-0-NCHW-FP32-F
    // clang-format off
    stream << n_inputs;
    stream << sep << PrintDHW(sep, spatial_dims, in_depth, in_height, in_width);
    stream << sep << PrintDHW('x', spatial_dims, kernel_size_d, kernel_size_h, kernel_size_w);
    stream << sep << n_outputs;
    stream << sep << PrintDHW(sep, spatial_dims, out_depth, out_height, out_width);
    stream << sep << batch_sz;
    stream << sep << PrintDHW('x', spatial_dims, pad_d, pad_h, pad_w);
    stream << sep << PrintDHW('x', spatial_dims, kernel_stride_d, kernel_stride_h, kernel_stride_w);
    stream << sep << PrintDHW('x', spatial_dims, kernel_dilation_d, kernel_dilation_h, kernel_dilation_w);
    stream << sep << bias;
    stream << sep << in_layout;
    stream << sep << EncodeDataTypesForKey(in_data_type, weights_data_type, out_data_type);
    stream << sep << (direction.IsForward() ? "F" : direction.IsBackwardData() ? "B" : "W");
    // clang-format on
    // New performance config entries shall come into variable/optional part of db key.
    // This is to support backward compatibility with previous versions of databases.
    std::ostringstream optional;
    {
        // Group count > 1 identifies Group/Depthwise modes.
        if(group_counts != 1)
            optional << 'g' << group_counts;
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
                                       int dir,
                                       int bias_)
    : bias(bias_)
{
    direction.Set(dir);

    setConvDescr(conv);
    SetDescFromMLDesc(spatial_dims, *this, in, &ProblemDescription::setInputDescr);
    SetDescFromMLDesc(spatial_dims, *this, weights, &ProblemDescription::setWeightsDescr);
    SetDescFromMLDesc(spatial_dims, *this, out, &ProblemDescription::setOutputDescr);
}

std::tuple<int, int, int> GetDHW(int spatial_dims, const std::vector<int>& data)
{
    if(spatial_dims == 2)
        return std::make_tuple(0, data[0], data[1]);
    return std::make_tuple(data[0], data[1], data[2]);
}

void ProblemDescription::setConvDescr(const ConvolutionDescriptor& conv)
{
    spatial_dims = conv.spatialDim;
    std::tie(pad_d, pad_h, pad_w) = GetDHW(spatial_dims, conv.GetConvPads());
    std::tie(kernel_stride_d, kernel_stride_h, kernel_stride_w) =
        GetDHW(spatial_dims, conv.GetConvStrides());
    std::tie(kernel_dilation_d, kernel_dilation_h, kernel_dilation_w) =
        GetDHW(spatial_dims, conv.GetConvDilations());
    group_counts = conv.group_count;
}

} // namespace miopen
