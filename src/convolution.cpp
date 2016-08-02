#include <mlopen/convolution.hpp>
#include <mlopen/mlo_internal.hpp>

mlopenConvolutionDescriptor::mlopenConvolutionDescriptor() : _pad_h(0), _pad_w(0), _u(1), _v(1), _upscalex(0), _upscaley(0) {
	printf("In convolution Ctor\n");
	_mode = mlopenConvolution;
}

mlopenStatus_t mlopenConvolutionDescriptor::GetForwardOutputDim(const mlopenTensorDescriptor_t inputTensorDesc,
			const mlopenTensorDescriptor_t filterDesc,
			int *n,
			int *c,
			int *h, 
			int *w) {
	
	if(inputTensorDesc == NULL || filterDesc == NULL) {
		return mlopenStatusBadParm;
	}

	int input_n;
	int input_c;
	int input_h;
	int input_w;

	std::tie(input_n, input_c, input_h, input_w) = tie4(inputTensorDesc->GetLengths());

	int filter_k;
	int filter_c;
	int filter_h;
	int filter_w;
	
	std::tie(filter_k, filter_c, filter_h, filter_w) = tie4(filterDesc->GetLengths());

	if(input_c != filter_c) {
		return mlopenStatusBadParm;
	}

	*n = input_n;
	*c = filter_k;
	*h = (input_h - filter_h + 2*_pad_h) / _u + 1;
	*w = (input_w - filter_w + 2*_pad_w) / _v + 1;

	return mlopenStatusSuccess;
}
