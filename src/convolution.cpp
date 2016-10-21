#include <mlopen/convolution.hpp>
#include <mlopen/errors.hpp>

namespace mlopen {

ConvolutionDescriptor::ConvolutionDescriptor(int p_pad_h, int p_pad_w, int p_u, int p_v, int p_upscalex, int p_upscaley) 
: mode(mlopenConvolution), pad_h(p_pad_h), pad_w(p_pad_w), u(p_u), v(p_v), upscalex(p_upscalex), upscaley(p_upscaley) 
{
	if(pad_h < 0 || pad_w < 0 || u < 0 || v < 0) {
		MLOPEN_THROW(mlopenStatusBadParm, "Parameters to filter cannot be negative");
	}
}

ConvolutionDescriptor::ConvolutionDescriptor(mlopenConvolutionMode_t p_mode, int p_pad_h, int p_pad_w, int p_u, int p_v, int p_upscalex, int p_upscaley)
: mode(p_mode), pad_h(p_pad_h), pad_w(p_pad_w), u(p_u), v(p_v), upscalex(p_upscalex), upscaley(p_upscaley)
{
	if(pad_h < 0 || pad_w < 0 || u < 0 || v < 0) {
		MLOPEN_THROW(mlopenStatusBadParm, "Parameters to filter cannot be negative");
	}
}

std::tuple<int, int, int, int> ConvolutionDescriptor::GetForwardOutputDim(
	const TensorDescriptor& inputTensorDesc, 
	const TensorDescriptor& filterDesc) 
const
{
	assert(inputTensorDesc.GetLengths().size() == 4);
	assert(filterDesc.GetLengths().size() == 4);

	if (inputTensorDesc.GetType() != filterDesc.GetType()) {
		MLOPEN_THROW(mlopenStatusBadParm, "Types do not match for the filter");
	}

	int input_n;
	int input_c;
	int input_h;
	int input_w;

	std::tie(input_n, input_c, input_h, input_w) = mlopen::tie4(inputTensorDesc.GetLengths());

	int filter_k;
	int filter_c;
	int filter_h;
	int filter_w;
	
	std::tie(filter_k, filter_c, filter_h, filter_w) = mlopen::tie4(filterDesc.GetLengths());

	if(input_c != filter_c) {
		MLOPEN_THROW(mlopenStatusBadParm, "Channels do not match for the filter");
	}

	return std::make_tuple(
		input_n, 
		filter_k, 
		(input_h - filter_h + 2*pad_h) / u + 1, 
		(input_w - filter_w + 2*pad_w) / v + 1
	);
}

std::tuple<int, int, int, int> ConvolutionDescriptor::GetBackwardOutputDim(
	const TensorDescriptor& outputTensorDesc, 
	const TensorDescriptor& filterDesc) 
const
{
	assert(outputTensorDesc.GetLengths().size() == 4);
	assert(filterDesc.GetLengths().size() == 4);

	if (outputTensorDesc.GetType() != filterDesc.GetType()) {
		MLOPEN_THROW(mlopenStatusBadParm, "Types do not match for the filter");
	}

	int output_n;
	int output_c;
	int output_h;
	int output_w;

	std::tie(output_n, output_c, output_h, output_w) = mlopen::tie4(outputTensorDesc.GetLengths());

	int filter_k;
	int filter_c;
	int filter_h;
	int filter_w;
	
	std::tie(filter_k, filter_c, filter_h, filter_w) = mlopen::tie4(filterDesc.GetLengths());

	if(output_c != filter_k) {
		MLOPEN_THROW(mlopenStatusBadParm, "Channels do not match for the filter");
	}

	return std::make_tuple(
		output_n, 
		filter_c, 
		u * (output_h - 1) - 2*pad_h + filter_h, 
		v * (output_w - 1) - 2*pad_w + filter_w
	);
}

TensorDescriptor ConvolutionDescriptor::GetForwardOutputTensor(
	const TensorDescriptor& inputTensorDesc, 
	const TensorDescriptor& filterDesc) const
{
	auto dims = this->GetForwardOutputDim(inputTensorDesc, filterDesc);
	return TensorDescriptor(inputTensorDesc.GetType(), {
		std::get<0>(dims),
		std::get<1>(dims),
		std::get<2>(dims),
		std::get<3>(dims)});
}

TensorDescriptor ConvolutionDescriptor::GetBackwardOutputTensor(
	const TensorDescriptor& outputTensorDesc, 
	const TensorDescriptor& filterDesc) const
{
	auto dims = this->GetBackwardOutputDim(outputTensorDesc, filterDesc);
	return TensorDescriptor(outputTensorDesc.GetType(), {
		std::get<0>(dims),
		std::get<1>(dims),
		std::get<2>(dims),
		std::get<3>(dims)});
}
} // namespace mlopen
