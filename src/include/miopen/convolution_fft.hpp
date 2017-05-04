#ifndef GUARD_MIOPEN_CONVOLUTION_FFT_HPP_
#define GUARD_MIOPEN_CONVOLUTION_FFT_HPP_

namespace miopen {

struct FFTConvParams
{
	static int TileSize(int in_h, int in_w)
	{
		int NX, NY, NXc, N;

		if( (in_h == 14) && (in_w == 14) )
		{
			NY = 18; // fft tile height
			NX = 18; // fft tile width
			NXc = (1 + NX/2);
			N = NY*NXc;
		}
		else
		{
			NY = 32; // fft tile height
			NX = 32; // fft tile width
			NXc = (1 + NX/2);
			N = NY*NXc;
		}

		return N;
	}

	static const int TransposePadding = 64;
	static const int NumKernels = 7;
};

} // namespace miopen

#endif // GUARD_MIOPEN_CONVOLUTION_FFT_HPP_
