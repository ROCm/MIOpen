#ifndef CONVOLUTION_PROBLEM_DESCRIPTOR
#define CONVOLUTION_PROBLEM_DESCRIPTOR

namespace ck {
namespace driver {

struct ConvolutionProblemDescriptor
{
    ConvolutionProblemDescriptor() = default;

    ConvolutionProblemDescriptor(int N_,
                                 int K_,
                                 int C_,
                                 int Y_,
                                 int X_,
                                 int Hi_,
                                 int Wi_,
                                 int Ho_,
                                 int Wo_,
                                 int ConvStrideH_,
                                 int ConvStrideW_,
                                 int ConvDilationH_,
                                 int ConvDilationW_,
                                 int InLeftPadH_,
                                 int InLeftPadW_,
                                 int InRightPadH_,
                                 int InRightPadW_,
                                 ck::DataTypeEnum_t InDataTypeEnum_,
                                 ck::DataTypeEnum_t WeiDataTypeEnum_,
                                 ck::DataTypeEnum_t OutDataTypeEnum_)
        : N{N_},
          K{K_},
          C{C_},
          Y{Y_},
          X{X_},
          Hi{Hi_},
          Wi{Wi_},
          Ho{Ho_},
          Wo{Wo_},
          ConvStrideH{ConvStrideH_},
          ConvStrideW{ConvStrideW_},
          ConvDilationH{ConvDilationH_},
          ConvDilationW{ConvDilationW_},
          InLeftPadH{InLeftPadH_},
          InLeftPadW{InLeftPadW_},
          InRightPadH{InRightPadH_},
          InRightPadW{InRightPadW_},
          InDataTypeEnum{InDataTypeEnum_},
          WeiDataTypeEnum{WeiDataTypeEnum_},
          OutDataTypeEnum{OutDataTypeEnum_}
    {
    }

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

    ck::DataTypeEnum_t InDataTypeEnum;
    ck::DataTypeEnum_t WeiDataTypeEnum;
    ck::DataTypeEnum_t OutDataTypeEnum;

    std::size_t CalculateFlop() const { return 2L * N * K * C * Y * X * Ho * Wo; }
};

} // namespace driver
} // namespace ck
#endif
