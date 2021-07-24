#ifndef CONVOLUTION_PROBLEM_DESCRIPTOR
#define CONVOLUTION_PROBLEM_DESCRIPTOR

namespace ck_driver {

struct ConvolutionProblemDescriptor
{
    int N;
    int K;
    int C;
    int Y;
    int X;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int ConvStrideH;
    int ConvStrideW;
    int ConvDilationH;
    int ConvDilationW;
    int InLeftPadH;
    int InLeftPadW;
    int InRightPadH;
    int InRightPadW;

    std::size_t CalculateFlop() const { return 2L * N * K * C * Y * X * Ho * Wo; }
};

} // namespace ck_driver
#endif
