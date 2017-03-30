#ifndef GUARD_MLOPEN_CONVOLUTION_FFT_HPP_
#define GUARD_MLOPEN_CONVOLUTION_FFT_HPP_

namespace mlopen {

struct FFTConvParams
{
	static const int NY = 32; // fft tile height
	static const int NX = 32; // fft tile width
	static const int NXc = (1 + NX/2);
	static const int N = NY*NXc;

	static const int TransposePadding = 64;
	static const int NumKernels = 7;
};

} // namespace mlopen

#endif // GUARD_MLOPEN_CONVOLUTION_FFT_HPP_
