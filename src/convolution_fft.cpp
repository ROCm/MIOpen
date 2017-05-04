#include <miopen/convolution.hpp>
#include <miopen/convolution_fft.hpp>
#include <miopen/errors.hpp>
#include <miopen/env.hpp>

namespace miopen {

	size_t ConvolutionDescriptor::ForwardGetWorkSpaceSizeFFT(
		const TensorDescriptor& wDesc,
		const TensorDescriptor& xDesc,
		const TensorDescriptor& yDesc) const
{
	int in_n, in_c, in_h, in_w;
	std::tie(in_n, in_c, in_h, in_w) = miopen::tie4(xDesc.GetLengths());

	int out_n, out_c, out_h, out_w;
	std::tie(out_n, out_c, out_h, out_w) = miopen::tie4(yDesc.GetLengths());

	int wei_k, wei_c, wei_h, wei_w;
	std::tie(wei_k, wei_c, wei_h, wei_w) = miopen::tie4(wDesc.GetLengths());

	bool supported = true;

	// FFT convolutions only works for specific config(s)
	// coverage to expand gradually

	supported = ((in_n < 1) || (in_n > 512)) ? false : supported;
	supported = ((wei_k < 1) || (wei_k > 512)) ? false : supported;
	supported = ((in_c*in_n)%16 != 0) ? false : supported;
	supported = ((wei_c*wei_k)%16 != 0) ? false : supported;
	supported = ((out_c*out_n)%16 != 0) ? false : supported;

	supported = (
					(std::tie(in_h, in_w) != std::make_tuple(28, 28)) &&
					(std::tie(in_h, in_w) != std::make_tuple(27, 27)) &&
					(std::tie(in_h, in_w) != std::make_tuple(14, 14))
				) ? false : supported;

	supported = (std::tie(wei_h, wei_w) != std::make_tuple(5, 5)) ? false : supported;
	supported = (std::tie(pad_h, pad_w, u, v) != std::make_tuple(2, 2, 1, 1)) ? false : supported;
	supported = (yDesc.GetType() != miopenFloat) ? false : supported;

	const int N = FFTConvParams::TileSize(in_h, in_w);
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

} // namespace miopen
