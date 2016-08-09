#include <mlopen/convolution.hpp>
#include <mlopen/mlo_internal.hpp>

namespace mlopen {

ConvolutionDescriptor::ConvolutionDescriptor(int p_pad_h, int p_pad_w, int p_u, int p_v, int p_upscalex, int p_upscaley) 
: mode(mlopenConvolution), pad_h(p_pad_h), pad_w(p_pad_w), u(p_u), v(p_v), upscalex(p_upscalex), upscaley(p_upscaley) 
{}

ConvolutionDescriptor::ConvolutionDescriptor(mlopenConvolutionMode_t p_mode, int p_pad_h, int p_pad_w, int p_u, int p_v, int p_upscalex, int p_upscaley)
: mode(p_mode), pad_h(p_pad_h), pad_w(p_pad_w), u(p_u), v(p_v), upscalex(p_upscalex), upscaley(p_upscaley)
{}

mlopenStatus_t ConvolutionDescriptor::GetForwardOutputDim(const mlopen::TensorDescriptor& inputTensorDesc,
			const mlopen::TensorDescriptor& filterDesc,
			int *n,
			int *c,
			int *h, 
			int *w) {

	assert(inputTensorDesc.GetLengths().size() == 4);
	assert(filterDesc.GetLengths().size() == 4);

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
		return mlopenStatusBadParm;
	}

	*n = input_n;
	*c = filter_k;
	*h = (input_h - filter_h + 2*pad_h) / u + 1;
	*w = (input_w - filter_w + 2*pad_w) / v + 1;

	return mlopenStatusSuccess;
}
}
