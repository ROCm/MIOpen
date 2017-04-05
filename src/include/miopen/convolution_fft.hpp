#ifndef GUARD_MIOPEN_CONVOLUTION_FFT_HPP_
#define GUARD_MIOPEN_CONVOLUTION_FFT_HPP_

namespace miopen {

struct FFTConvParams
{
	static const int NY = 32; // fft tile height
	static const int NX = 32; // fft tile width
	static const int NXc = (1 + NX/2);
	static const int N = NY*NXc;

	static const int TransposePadding = 64;
	static const int NumKernels = 7;
};

} // namespace miopen

#endif // GUARD_MIOPEN_CONVOLUTION_FFT_HPP_
