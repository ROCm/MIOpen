#include <miopen/problem_description.hpp>

#include <miopen/convolution.hpp>

#include <sstream>

static std::string
EncodeDataTypesForKey(miopenDataType_t in, miopenDataType_t weights, miopenDataType_t out)
{
    if(in == weights && in == out)
        return miopen::GetDataTypeName(in);
    return miopen::GetDataTypeName(in) + miopen::GetDataTypeName(weights) +
           miopen::GetDataTypeName(out);
}

int miopen::ProblemDescription::mloBuildConf_Key(std::string& conf_key) const
{
    std::ostringstream ss;

    ss << n_inputs << 'x';
    ss << in_height << 'x';
    ss << in_width << 'x';
    ss << kernel_size_h << 'x';
    ss << kernel_size_w << 'x';
    ss << n_outputs << 'x';
    ss << out_height << 'x';
    ss << out_width << 'x';
    ss << batch_sz << 'x';
    ss << in_layout << 'x';
    ss << EncodeDataTypesForKey(in_data_type, weights_data_type, out_data_type) << 'x';
    ss << pad_h << 'x';
    ss << pad_w << 'x';
    ss << kernel_stride_h << 'x';
    ss << kernel_stride_w << 'x';
    ss << kernel_dilation_h << 'x';
    ss << kernel_dilation_w << 'x';
    ss << group_counts << 'x';
    ss << (direction.IsForward() ? "1" : "0");

    conf_key = ss.str();

    return (0);
}

void miopen::ProblemDescription::Serialize(std::ostream& stream) const
{
    if(!direction.IsKnown())
        MIOPEN_THROW("!direction.IsKnown()");
    const auto sep = '-';
    // clang-format off
        // 576-4-4-1x1-192-4-4-8-1x1-2x2-3x3-0-NCHW-FP32-F
        stream
            << n_inputs << sep << in_height << sep << in_width
            << sep << kernel_size_h << 'x' << kernel_size_w
            << sep << n_outputs << sep << out_height << sep << out_width
            << sep << batch_sz
            << sep << pad_h << 'x' << pad_w
            << sep << kernel_stride_h << 'x' << kernel_stride_w
            << sep << kernel_dilation_h << 'x' << kernel_dilation_w
            << sep << bias
            << sep << in_layout
            << sep << EncodeDataTypesForKey(in_data_type, weights_data_type, out_data_type)
            << sep << (direction.IsForward() ? "F"
                     : direction.IsBackwardData() ? "B" : "W"); // clang-format on
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

miopen::ProblemDescription::ProblemDescription(const TensorDescriptor& in,
                                               const TensorDescriptor& weights,
                                               const TensorDescriptor& out,
                                               const ConvolutionDescriptor& conv,
                                               int dir,
                                               int bias_)
    : bias(bias_)
{
    direction.Set(dir);

    SetDescFromMLDesc(*this, in, &ProblemDescription::setInputDescr);
    SetDescFromMLDesc(*this, weights, &ProblemDescription::setWeightsDescr);
    SetDescFromMLDesc(*this, out, &ProblemDescription::setOutputDescr);
    setConvDescr(conv);
}

void miopen::ProblemDescription::setConvDescr(const ConvolutionDescriptor& conv)
{
    pad_h             = conv.GetConvPads()[0];
    pad_w             = conv.GetConvPads()[1];
    kernel_stride_h   = conv.GetConvStrides()[0];
    kernel_stride_w   = conv.GetConvStrides()[1];
    kernel_dilation_h = conv.GetConvDilations()[0];
    kernel_dilation_w = conv.GetConvDilations()[1];
    group_counts      = conv.group_count;
}
