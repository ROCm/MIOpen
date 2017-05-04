#ifndef GUARD_MIOPEN_CONVOLUTION_FFT_HPP_
#define GUARD_MIOPEN_CONVOLUTION_FFT_HPP_

namespace miopen {

struct FFTConvParams
{
	static int TileSize(int in_h, int in_w)
	{
		int NX, NY; // fft tile width, height

		if( (in_h == 14) && (in_w == 14) )
		{
			NY = 18;
			NX = 18;
		}
		else
		{
			NY = 32;
			NX = 32;
		}

		return NY*(1 + NX/2);
	}

	static const int TransposePadding = 64;
	static const int NumKernels = 7;
};

} // namespace miopen

#endif // GUARD_MIOPEN_CONVOLUTION_FFT_HPP_
