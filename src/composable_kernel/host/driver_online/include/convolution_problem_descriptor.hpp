#ifndef CONVOLUTION_PROBLEM_DESCRIPTOR
#define CONVOLUTION_PROBLEM_DESCRIPTOR

namespace ck {

struct ConvolutionProblemDescriptor
{
    int N, K, C, Y, X, Hi, Wi, Ho, Wo;
    int ConvStrideH, ConvStrideW;
    int ConvDilationH, ConvDilationW;
    int InLeftPadH, InLeftPadW;
    int InRightPadH, InRightPadW;
};

} // namespace ck
#endif
