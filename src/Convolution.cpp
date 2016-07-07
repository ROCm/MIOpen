#include "Convolution.hpp"
#include "mlo_internal.hpp"

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
	
	printf("To be implemented (GetForwardOutputDim)\n");

	return mlopenStatusSuccess;
}

