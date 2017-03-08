#ifndef GUARD_MLOPEN_CONVOLUTION_FFT_HPP_
#define GUARD_MLOPEN_CONVOLUTION_FFT_HPP_

namespace mlopen {

struct FFTConvParams
{
	static const int NY = 32; // nearest pow2, 27+5-1
	static const int NX = 32; // nearest pow2, 27+5-1
	static const int NXc = (1 + NX/2);
	static const int N = NY*NXc;

	static const int TransposePadding = 64;
};

} // namespace mlopen

#endif // GUARD_MLOPEN_CONVOLUTION_FFT_HPP_
