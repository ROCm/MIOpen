#include <mlopen/convolution.hpp>
#include <mlopen/errors.hpp>

namespace mlopen {

	size_t ConvolutionDescriptor::ForwardGetWorkSpaceSizeFFT(
		const TensorDescriptor& wDesc,
		const TensorDescriptor& xDesc,
		const TensorDescriptor& yDesc) const
{
	return 0;
}

} // namespace mlopen
