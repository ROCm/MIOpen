#ifndef COMPILE_PARAMETER_CONV_IGEMM_FWD_V6R1_DLOPS_NCHW_KCYX_NKHW_HPP
#define COMPILE_PARAMETER_CONV_IGEMM_FWD_V6R1_DLOPS_NCHW_KCYX_NKHW_HPP

#include <numeric>

namespace ck_driver {

struct CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw
{
    int ABDatatype;
    int AccDatatype;
    int CDatatype;

    int BlockSize;

    int GN0;
    int GK1;

    int GM1PerBlockGM11;
    int GN1PerBlockGN11;
    int GK0PerBlock;

    int BM1PerThreadBM11;
    int BN1PerThreadBN11;
    int BK0PerThread;

    std::array<int, 2> BM10BN10ThreadClusterBM10Xs;
    std::array<int, 2> BM10BN10ThreadClusterBN10Xs;

    std::array<int, 5> ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1;
    std::array<int, 5> ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1;
    std::array<int, 5> ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1;
    std::array<int, 5> ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1;

    std::array<int, 5> BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1;
    std::array<int, 5> BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1;
    std::array<int, 5> BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1;
    std::array<int, 5> BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1;

    int CThreadTransferDstScalarPerVector;

    bool HasMainKBlockLoop;
    bool HasDoubleTailKBlockLoop;

    auto GetCompileParameterString() const
    {
        // clang-format off
        return
            " -DCK_PARAM_A_B_DATATYPE=" + 
                std::to_string(ABDatatype) + 
            " -DCK_PARAM_ACC_DATATYPE=" + 
                std::to_string(AccDatatype) +
            " -DCK_PARAM_C_DATATYPE=" + 
                std::to_string(CDatatype) + 
            " -DCK_PARAM_BlockSize=" +
                std::to_string(BlockSize) +
            " -DCK_PARAM_GN0=" +
                std::to_string(GN0) +
            " -DCK_PARAM_GK1=" +
                std::to_string(GK1) +
            " -DCK_PARAM_GM1PerBlockGM11=" +
                std::to_string(GM1PerBlockGM11) +
            " -DCK_PARAM_GN1PerBlockGN11=" +
                std::to_string(GN1PerBlockGN11) +
            " -DCK_PARAM_GK0PerBlock=" + 
                std::to_string(GK0PerBlock) +
            " -DCK_PARAM_BM1PerThreadBM11=" +
                std::to_string(BM1PerThreadBM11) +
            " -DCK_PARAM_BN1PerThreadBN11=" +
                std::to_string(BN1PerThreadBN11) +
            " -DCK_PARAM_BK0PerThread=" +
                std::to_string(BK0PerThread) +
            " -DCK_PARAM_BM10BN10ThreadClusterBM10Xs=" +
                std::to_string(BM10BN10ThreadClusterBM10Xs[0]) + "," +
                std::to_string(BM10BN10ThreadClusterBM10Xs[1]) +
            " -DCK_PARAM_BM10BN10ThreadClusterBN10Xs=" +
                std::to_string(BM10BN10ThreadClusterBN10Xs[0]) + "," +
                std::to_string(BM10BN10ThreadClusterBN10Xs[1]) +
            " -DCK_PARAM_ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1=" +
                std::to_string(ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[0]) + "," +
                std::to_string(ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[1]) + "," +
                std::to_string(ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[2]) + "," +
                std::to_string(ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[3]) + "," +
                std::to_string(ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[4]) +
            " -DCK_PARAM_ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1=" +
                std::to_string(ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[0]) + "," +
                std::to_string(ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[1]) + "," +
                std::to_string(ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[2]) + "," +
                std::to_string(ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[3]) + "," +
                std::to_string(ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[4]) +
            " -DCK_PARAM_ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1=" +
                std::to_string(ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[0]) +  "," +
                std::to_string(ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[1]) + "," +
                std::to_string(ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[2]) + "," +
                std::to_string(ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[3]) + "," +
                std::to_string(ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[4]) +
            " -DCK_PARAM_ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1=" +
                std::to_string(ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[0]) + "," +
                std::to_string(ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[1]) + "," +
                std::to_string(ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[2]) + "," +
                std::to_string(ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[3]) + "," +
                std::to_string(ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[4]) +
            " -DCK_PARAM_BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1=" +
                std::to_string(BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[0]) + "," +
                std::to_string(BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[1]) + "," +
                std::to_string(BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[2]) + "," +
                std::to_string(BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[3]) + "," +
                std::to_string(BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[4]) +
            " -DCK_PARAM_BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1=" +
                std::to_string(BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[0]) + "," +
                std::to_string(BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[1]) + "," +
                std::to_string(BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[2]) + "," +
                std::to_string(BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[3]) + "," +
                std::to_string(BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[4]) +
            " -DCK_PARAM_BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1=" +
                std::to_string(BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[0]) + "," +
                std::to_string(BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[1]) + "," +
                std::to_string(BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[2]) + "," +
                std::to_string(BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[3]) + "," +
                std::to_string(BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[4]) +
            " -DCK_PARAM_BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1=" +
                std::to_string(BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[0]) + "," +
                std::to_string(BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[1]) + "," +
                std::to_string(BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[2]) + "," +
                std::to_string(BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[3]) + "," +
                std::to_string(BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[4]) +
            " -DCK_PARAM_CThreadTransferDstScalarPerVector=" +
                std::to_string(CThreadTransferDstScalarPerVector) +
            " -DCK_PARAM_HAS_MAIN_KBLOCK_LOOP=" +
                std::to_string(HasMainKBlockLoop) + 
            " -DCK_PARAM_HAS_DOUBLE_TAIL_KBLOCK_LOOP=" +
                std::to_string(HasDoubleTailKBlockLoop);
        // clang-format on
    }
};

// TODO
const static std::vector<CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw>
    compile_param_list_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw{
        // clang-format off
        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, true},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, true},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, true},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, {8, 2}, {8, 2}, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, true},
        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, true},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, true},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, true},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, {8, 2}, {8, 2}, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, true},

        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, false},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, false},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, false},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, {8, 2}, {8, 2}, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, false},
        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, false},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, false},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, false},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, {8, 2}, {8, 2}, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, false},

        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, true},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, true},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, true},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, {8, 2}, {8, 2}, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, true},
        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, true},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, true},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, true},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, {8, 2}, {8, 2}, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, true},

        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, false},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, false},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, false},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, {8, 2}, {8, 2}, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, false},
        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, false},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, false},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, {8, 2}, {8, 2}, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, false},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, {8, 2}, {8, 2}, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, false}
        // clang-format on
    };

// TODO make this common interface and write specs for it
struct ConvIgemmFwdV6r1DlopsNchwKcyxNkhw
{
    static bool IsApplicable(const ConvolutionProblemDescriptor& conv_problem_desc)
    {
        for(auto compile_param : compile_param_list_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw)
        {
            if(IsValidCompileParameter(conv_problem_desc, compile_param))
            {
                return true;
            }
        }

        return false;
    }

    static bool
    IsValidCompileParameter(const ConvolutionProblemDescriptor& conv_problem_desc,
                            const CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw& compile_param)
    {
        const int N  = conv_problem_desc.N;
        const int K  = conv_problem_desc.K;
        const int C  = conv_problem_desc.C;
        const int Y  = conv_problem_desc.Y;
        const int X  = conv_problem_desc.X;
        const int Hi = conv_problem_desc.Hi;
        const int Wi = conv_problem_desc.Wi;
        const int Ho = conv_problem_desc.Ho;
        const int Wo = conv_problem_desc.Wo;

        const int GK1  = compile_param.GK1;
        const int GN0  = compile_param.GN0;
        const int GM11 = compile_param.GM1PerBlockGM11;
        const int GN11 = compile_param.GN1PerBlockGN11;

        const int BM11 = compile_param.BM1PerThreadBM11;
        const int BN11 = compile_param.BN1PerThreadBN11;

        const int C0 = GK1;
        const int N0 = GN0;

        if(!(C % C0 == 0))
            return false;

        const int C1 = C / C0;

        if(!(N % N0 == 0))
            return false;

        const int N1 = N / N0;

        const int GM0 = 1;
        const int GM1 = K;
        const int GN1 = N1 * Ho * Wo;
        const int GK0 = C1 * Y * X;

        // check gridwise contraction
        {
            if(!(GM1 % GM11 == 0 && GN1 % GN11 == 0 && GK0 % compile_param.GK0PerBlock == 0))
                return false;

            const bool has_main_k_block_loop =
                ((GK0 + compile_param.GK0PerBlock) / (2 * compile_param.GK0PerBlock) > 1);

            const bool has_double_tail_k_block_loop = ((GK0 / compile_param.GK0PerBlock) % 2 == 0);

            if(!(has_main_k_block_loop == compile_param.HasMainKBlockLoop &&
                 has_double_tail_k_block_loop == compile_param.HasDoubleTailKBlockLoop))
                return false;
        }

        // check A blockwise copy
        {
            const auto block_slice_lengths =
                std::array<int, 5>{compile_param.GK0PerBlock, GM0, 1, GM11, GK1};
            const auto& cluster_lengths =
                compile_param.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1;
            const auto& thread_slice_lengths =
                compile_param.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1;
            const auto& src_vector_lengths =
                compile_param.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1;
            const auto& dst_vector_lengths =
                compile_param.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1;

            // check number of working thread
            const int num_work_thread = std::accumulate(
                cluster_lengths.begin(), cluster_lengths.end(), 1, std::multiplies<int>{});

            if(!(compile_param.BlockSize >= num_work_thread))
                return false;

            // check block slice lengths vs thread slice lengths vs cluster lengths
            for(int i = 0; i < 5; ++i)
            {
                if(!(cluster_lengths[i] * thread_slice_lengths[i] == block_slice_lengths[i]))
                    return false;
            }

            // check thread slice lengths vs vector lengths
            for(int i = 0; i < 5; ++i)
            {
                if(!(thread_slice_lengths[i] % src_vector_lengths[i] == 0))
                    return false;

                if(!(thread_slice_lengths[i] % dst_vector_lengths[i] == 0))
                    return false;
            }

            // check Src vectorization, GK0 is global mem vector dim
            if(!(src_vector_lengths[1] == 1 && src_vector_lengths[2] == 1 &&
                 src_vector_lengths[3] == 1 && src_vector_lengths[4] == 1))
                return false;

            // check Dst vectorization, {GM11, GK1} are LDS vector dims
            if(dst_vector_lengths[4] == GK1)
            { // vectorize on {GM11, GK1}
                if(!(GM11 % dst_vector_lengths[3] == 0))
                    return false;
            }
            else
            { // vectorize on {GK1} only
                if(!(GK1 % dst_vector_lengths[4] == 0))
                    return false;

                if(!(dst_vector_lengths[3] == 1))
                    return false;
            }
        }

        // check B blockwise copy
        {
            const auto block_slice_lengths =
                std::array<int, 5>{compile_param.GK0PerBlock, GN0, 1, GN11, GK1};
            const auto& cluster_lengths =
                compile_param.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1;
            const auto& thread_slice_lengths =
                compile_param.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1;
            const auto& src_vector_lengths =
                compile_param.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1;
            const auto& dst_vector_lengths =
                compile_param.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1;

            // check number of working thread
            const int num_work_thread = std::accumulate(
                cluster_lengths.begin(), cluster_lengths.end(), 1, std::multiplies<int>{});

            if(!(compile_param.BlockSize >= num_work_thread))
                return false;

            // check block slice lengths vs thread slice lengths vs cluster lengths
            for(int i = 0; i < 5; ++i)
            {
                if(!(cluster_lengths[i] * thread_slice_lengths[i] == block_slice_lengths[i]))
                    return false;
            }

            // check thread slice lengths vs vector lengths
            for(int i = 0; i < 5; ++i)
            {
                if(!(thread_slice_lengths[i] % src_vector_lengths[i] == 0 &&
                     thread_slice_lengths[i] % dst_vector_lengths[i] == 0))
                    return false;
            }

            // check Src vectorization: {GN11} is global mem vector dim
            if(!(src_vector_lengths[0] == 1 && src_vector_lengths[1] == 1 &&
                 src_vector_lengths[2] == 1 && src_vector_lengths[4] == 1))
                return false;

            // check Src tensor layout related vectorization
            if(conv_problem_desc.ConvDilationW == 1 && conv_problem_desc.InLeftPadW == 0 &&
               conv_problem_desc.InRightPadW == 0)
            {
                if(!(Wo % src_vector_lengths[3] == 0))
                    return false;
            }
            else
            {
                if(!(src_vector_lengths[3] == 1))
                    return false;
            }

            // check Dst vectorization: {GN11, GK1} are LDS vector dims
            if(dst_vector_lengths[4] == GK1)
            { // vectorize on {GN11, GK1}
                if(!(GN11 % dst_vector_lengths[3] == 0))
                    return false;
            }
            else
            { // vectorize on {GK1} only
                if(!(dst_vector_lengths[3] == 1))
                    return false;

                if(!(GK1 % dst_vector_lengths[4] == 0))
                    return false;
            }
        }

        // check blockwise GEMM
        {
            const int BM10 = std::accumulate(compile_param.BM10BN10ThreadClusterBM10Xs.begin(),
                                             compile_param.BM10BN10ThreadClusterBM10Xs.end(),
                                             1,
                                             std::multiplies<int>{});

            const int BN10 = std::accumulate(compile_param.BM10BN10ThreadClusterBN10Xs.begin(),
                                             compile_param.BM10BN10ThreadClusterBN10Xs.end(),
                                             1,
                                             std::multiplies<int>{});

            if(!(compile_param.BlockSize == BM10 * BN10))
                return false;

            const int BM = GM0 * GM11;
            const int BN = GN0 * GN11;

            const int BM1 = BM10 * BM11;
            const int BN1 = BN10 * BN11;

            if(!(BM % BM1 == 0 && BN % BN1 == 0))
                return false;

            const int BM0 = BM / BM1;
            const int BN0 = BN / BN1;

            // blockwise GEMM currently only support BM0 == 2 && BN0 == 2
            if(!(BM0 == 2 && BN0 == 2))
                return false;

            if(!(compile_param.GK0PerBlock % compile_param.BK0PerThread == 0))
                return false;
        }

        // check C threadwise copy
        {
            // {BN11} or {BN} or {BN1} or {GN11} is Dst vector dim
            const int dst_vector_len_gn11 = compile_param.CThreadTransferDstScalarPerVector;

            // check slice length vs Dst vector length:
            if(!(BN11 % dst_vector_len_gn11 == 0 && GN11 % dst_vector_len_gn11 == 0))
                return false;

            // check Dst memory layout related vectorization:
            if(!((Ho * Wo) % compile_param.CThreadTransferDstScalarPerVector == 0))
                return false;
        }

        return true;
    };

    static auto GetDefaultCompileParameter(const ConvolutionProblemDescriptor& conv_problem_desc)
    {
        for(auto compile_param : compile_param_list_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw)
        {
            if(IsValidCompileParameter(conv_problem_desc, compile_param))
            {
                return std::make_tuple(true, compile_param);
            }
        }

        return std::make_tuple(false, CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw{});
    }

    static int GetBlockSize(const ConvolutionProblemDescriptor&,
                            const CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw& compile_param)
    {
        return compile_param.BlockSize;
    }

    static int GetGridSize(const ConvolutionProblemDescriptor& conv_problem_desc,
                           const CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw& compile_param)
    {
        const int N  = conv_problem_desc.N;
        const int C  = conv_problem_desc.C;
        const int K  = conv_problem_desc.K;
        const int Ho = conv_problem_desc.Ho;
        const int Wo = conv_problem_desc.Wo;

        const int N0 = compile_param.GN0;
        const int N1 = N / N0;

        const int C0 = compile_param.GK1;
        const int C1 = C / C0;

        const int GM0 = 1;
        const int GM1 = K;

        const int GN1 = N1 * Ho * Wo;

        const int GM11 = compile_param.GM1PerBlockGM11;
        const int GN11 = compile_param.GN1PerBlockGN11;

        const int GM10 = GM1 / GM11;
        const int GN10 = GN1 / GN11;

        return GM10 * GN10;
    }

    static std::size_t GetWorkSpaceSize(const ConvolutionProblemDescriptor&,
                                        const CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw&)
    {
        return 4096L;
    }
};

} // namespace ck_driver
#endif
