#ifndef COMPILE_PARAMETER_CONV_IGEMM_FWD_V6R1_DLOPS_NCHW_KCYX_NKHW_HPP
#define COMPILE_PARAMETER_CONV_IGEMM_FWD_V6R1_DLOPS_NCHW_KCYX_NKHW_HPP

namespace ck {

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

    int BM10BN10ThreadClusterBM100;
    int BM10BN10ThreadClusterBN100;
    int BM10BN10ThreadClusterBM101;
    int BM10BN10ThreadClusterBN101;

    int ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[5];
    int ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[5];
    int ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[5];
    int ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[5];

    int BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[5];
    int BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[5];
    int BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[5];
    int BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[5];

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
            " -DCK_PARAM_BM10BN10ThreadClusterBM100=" +
                std::to_string(BM10BN10ThreadClusterBM100) +
            " -DCK_PARAM_BM10BN10ThreadClusterBN100=" +
                std::to_string(BM10BN10ThreadClusterBN100) +
            " -DCK_PARAM_BM10BN10ThreadClusterBM101=" +
                std::to_string(BM10BN10ThreadClusterBM101) +
            " -DCK_PARAM_BM10BN10ThreadClusterBN101=" +
                std::to_string(BM10BN10ThreadClusterBN101) +
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
        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, true},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, true},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, true},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, 2, 2, 8, 8, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, true},
        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, true},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, true},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, true},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, 2, 2, 8, 8, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, true},

        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, false},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, false},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, false},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, 2, 2, 8, 8, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, true, false},
        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, false},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, false},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, false},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, 2, 2, 8, 8, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, true, false},

        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, true},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, true},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, true},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, 2, 2, 8, 8, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, true},
        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, true},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, true},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, true},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, 2, 2, 8, 8, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, true},

        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, false},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, false},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, false},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, 2, 2, 8, 8, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, false, false},
        {70, 70, 70, 256, 1, 1, 128, 128,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, { 2, 1, 1, 128, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, false},
        {70, 70, 70, 256, 2, 1, 128,  64,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, { 4, 1, 1,  64, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, false},
        {70, 70, 70, 256, 4, 1, 128,  32,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1,  32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, false},
        {70, 70, 70, 256, 8, 1, 128,  16, 16, 4, 4, 1, 2, 2, 8, 8, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1,  16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 4, false, false}
        // clang-format on
    };

// TODO make this common interface and write specs for it
struct ConvIgemmFwdV6r1DlopsNchwKcyxNkhw
{
    // TODO
    static bool IsApplicable(const ConvolutionProblemDescriptor& conv_problem_desc) { return true; }

    // TODO
    static bool
    IsValidCompileParameter(const ConvolutionProblemDescriptor& conv_problem_desc,
                            const CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw& compile_param)
    {
        {
            const int C = conv_problem_desc.C;
            const int Y = conv_problem_desc.Y;
            const int X = conv_problem_desc.X;

            const int C0 = compile_param.GK1;
            const int C1 = C / C0;

            const int GK0 = C1 * Y * X;

            const bool has_main_k_block_loop =
                ((GK0 + compile_param.GK0PerBlock) / (2 * compile_param.GK0PerBlock) > 1);

            const bool has_double_tail_k_block_loop = ((GK0 / compile_param.GK0PerBlock) % 2 == 0);

            if(!(has_main_k_block_loop == compile_param.HasMainKBlockLoop &&
                 has_double_tail_k_block_loop == compile_param.HasDoubleTailKBlockLoop))
            {
                return false;
            }
        }

        return true;
    };

    static auto GetDefaultCompileParameter(const ConvolutionProblemDescriptor& conv_problem_desc)
    {
        for(int i = 0; i < compile_param_list_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw.size(); ++i)
        {
            const auto compile_param =
                compile_param_list_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw[i];

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

        // GN0 is tunable
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

} // namespace ck
#endif
