#include <miopen/problem_description.hpp>

#include <miopen/convolution.hpp>

/***********************************************************************************************************

 * Internal implementation of the direct conv configuration search

 ************************************************************************************************************/

/*
   the search db is a text file with the name defined by the device characteristics.
   each line is a key/value pair, separated by a space:
   32x16x16x3x3x64x16x16x100xNCHWxFP32x1 16.16.16.16.1.4.8.4.1
   or
   64x8x8x5x5x32x8x8x100xNCHWxFP32x0 16.16.8.8.2.4.1.1.4

   key format (all values are separted by x):
   n input maps
   input height
   input width
   filter height
   filter width
   n output maps
   output height
   output width
   batch size
   tensors' layout
   tensprs' data type
   direction (1 - forward, 0 - backward)

Note:
for backward direction - input and output are reversed.

value format (all values are separated by .):
vertical group size
horizontal group size
input block vertical size
input block horizontal size
output tile vertical size
output tile horizaontal size
n of output tiles
n of input blocks
n batchs (stacks) processed by the group
*/

int miopen::ProblemDescription::mloBuildConf_Key(std::string& conf_key) const
{

    conf_key =
        std::to_string(static_cast<long long>(n_inputs)) + std::string("x") +
        std::to_string(static_cast<long long>(in_height)) + std::string("x") +
        std::to_string(static_cast<long long>(in_width)) + std::string("x") +
        std::to_string(static_cast<long long>(kernel_size1)) + std::string("x") +
        std::to_string(static_cast<long long>(kernel_size0)) + std::string("x") +
        std::to_string(static_cast<long long>(n_outputs)) + std::string("x") +
        std::to_string(static_cast<long long>(out_height)) + std::string("x") +
        std::to_string(static_cast<long long>(out_width)) + std::string("x") +
        std::to_string(static_cast<long long>(batch_sz)) + std::string("x") + in_layout +
        std::string("x") + in_data_type + std::string("x") +
        (direction.IsForward() ? "1" : "0"); /// \todo Shall we separate keys for WrW convolutions?
    return (0);
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
    setConvDescr(conv.pad_h, conv.pad_w, conv.u, conv.v, conv.dilation_h, conv.dilation_w);
}
