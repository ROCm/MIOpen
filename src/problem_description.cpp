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
    if(spatial_dims == 2)
    {
        return (in_layout == "NHWC") && (out_layout == "NHWC") && (weights_layout == "NHWC");
    }
    else
    {
        return (in_layout == "NDHWC") && (out_layout == "NDHWC") && (weights_layout == "NDHWC");
    }
}

void ProblemDescription::Serialize(std::ostream& stream) const
{
    if(!direction.IsKnown())
        MIOPEN_THROW("!direction.IsKnown()");
    const auto sep = '-';
    // Problem description with default NCHW-NCHW-NCHW layout
    // 576-4-4-1x1-192-4-4-8-1x1-2x2-3x3-0-NCHW-FP32-F
    // Problem description with non-default layout
    // 576-4-4-1x1-192-4-4-8-1x1-2x2-3x3-0-NHWC-NCHW-NCHW-FP32-F
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
    if ((in_layout == "NCHW" && weights_layout == "NCHW" && out_layout == "NCHW")
        || (in_layout == "NCDHW" && weights_layout == "NCDHW" && out_layout == "NCDHW"))
    {
        stream << sep << in_layout;
    } else {
        stream << sep << in_layout;
        stream << sep << weights_layout;
        stream << sep << out_layout;
    }
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
                                       conv::Direction dir,
                                       int bias_)
    : ProblemDescription(dir == conv::Direction::Forward
                             ? conv::ProblemDescription{in, weights, out, conv, dir, bias_}
                             : conv::ProblemDescription{out, weights, in, conv, dir, bias_})
{
}

ProblemDescription::ProblemDescription(conv::ProblemDescription desc)
    : conv_problem(std::move(desc)),
      spatial_dims(conv_problem.GetSpatialDims()),

      n_inputs(conv_problem.GetInChannels()),
      in_height(conv_problem.GetInHeight()),
      in_width(conv_problem.GetInWidth()),
      in_depth(conv_problem.GetInDepth()),

      kernel_size_h(conv_problem.GetWeightsHeight()),
      kernel_size_w(conv_problem.GetWeightsWidth()),
      kernel_size_d(conv_problem.GetWeightsDepth()),

      n_outputs(conv_problem.GetOutChannels()),
      out_height(conv_problem.GetOutHeight()),
      out_width(conv_problem.GetOutWidth()),
      out_depth(conv_problem.GetOutDepth()),

      batch_sz(conv_problem.GetInBatchSize()),
      pad_h(conv_problem.GetPadH()),
      pad_w(conv_problem.GetPadW()),
      pad_d(conv_problem.GetPadD()),
      kernel_stride_h(conv_problem.GetKernelStrideH()),
      kernel_stride_w(conv_problem.GetKernelStrideW()),
      kernel_stride_d(conv_problem.GetKernelStrideD()),
      kernel_dilation_h(conv_problem.GetDilationH()),
      kernel_dilation_w(conv_problem.GetDilationW()),
      kernel_dilation_d(conv_problem.GetDilationD()),
      bias(conv_problem.GetBias()),
      in_layout(conv_problem.GetInLayout()),
      weights_layout(conv_problem.GetWeightsLayout()),
      out_layout(conv_problem.GetOutLayout()),
      in_data_type(conv_problem.GetInDataType()),
      weights_data_type(conv_problem.GetWeightsDataType()),
      out_data_type(conv_problem.GetOutDataType()),
      bot_sz(conv_problem.GetInSize()),
      top_sz(conv_problem.GetOutSize()),
      weights_sz(conv_problem.GetWeightsSize()),
      bias_sz(conv_problem.GetBias()),
      in_stride(conv_problem.GetInStrideH()),
      out_stride(conv_problem.GetOutStrideH()),
      in_channel_stride(conv_problem.GetInChannelStride()),
      in_batch_stride(conv_problem.GetInBatchStride()),
      out_channel_stride(conv_problem.GetOutChannelStride()),
      out_batch_stride(conv_problem.GetOutBatchStride()),
      group_counts(conv_problem.GetGroupCount()),
      direction(conv_problem.GetDirection())
{
}

} // namespace miopen
