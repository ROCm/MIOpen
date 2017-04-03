#include <mlopen/convolution.hpp>
#include <mlopen/convolution_fft.hpp>
#include <mlopen/errors.hpp>

namespace mlopen {

	size_t ConvolutionDescriptor::ForwardGetWorkSpaceSizeFFT(
		const TensorDescriptor& wDesc,
		const TensorDescriptor& xDesc,
		const TensorDescriptor& yDesc) const
{

	int in_n, in_c, in_h, in_w;
	std::tie(in_n, in_c, in_h, in_w) = mlopen::tie4(xDesc.GetLengths());

	int out_n, out_c, out_h, out_w;
	std::tie(out_n, out_c, out_h, out_w) = mlopen::tie4(yDesc.GetLengths());

	int wei_k, wei_c, wei_h, wei_w;
	std::tie(wei_k, wei_c, wei_h, wei_w) = mlopen::tie4(wDesc.GetLengths());

	bool supported = true;

	// FFT convolutions only works for specific config(s)
	// coverage to expand gradually

	supported = ((in_n < 1) || (in_n > 512)) ? false : supported;
	supported = ((wei_k < 1) || (wei_k > 512)) ? false : supported;
	supported = (std::tie(in_c, in_h, in_w) != std::make_tuple(64, 27, 27)) ? false : supported;
	supported = (std::tie(wei_c, wei_h, wei_w) != std::make_tuple(64, 5, 5)) ? false : supported;
	supported = (std::tie(out_h, out_w) != std::make_tuple(27, 27)) ? false : supported;
	supported = (std::tie(pad_h, pad_w, u, v) != std::make_tuple(2, 2, 1, 1)) ? false : supported;
	supported = (yDesc.GetType() != mlopenFloat) ? false : supported;

	const int N = FFTConvParams::N;
	const int Padding = FFTConvParams::TransposePadding;

	if(supported)
	{
		int temp_size1 = (in_c*in_n + Padding) + (wei_k*wei_c + Padding);
		int temp_size2 = (out_n*out_c + Padding);
		int temp_size = temp_size1 > temp_size2 ? temp_size1 : temp_size2;

		return 2*2*N*temp_size*sizeof(float);
	}
	else
		return 0;
}

} // namespace mlopen
