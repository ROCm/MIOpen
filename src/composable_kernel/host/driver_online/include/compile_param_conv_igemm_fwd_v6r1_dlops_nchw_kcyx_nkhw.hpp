#ifndef COMPILE_PARAMETER_CONV_IGEMM_FWD_V6R1_DLOPS_NCHW_KCYX_NKHW_HPP
#define COMPILE_PARAMETER_CONV_IGEMM_FWD_V6R1_DLOPS_NCHW_KCYX_NKHW_HPP

namespace ck {
namespace kernel_compile_parameter {

struct CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw
{
    int BlockSize = 256;

    int GN0 = 4;
    int GK1 = 1;

    int GM1PerBlockGM11 = 128;
    int GN1PerBlockGN11 = 32;
    int GK0PerBlock     = 8;

    int BM1PerThreadBM11 = 4;
    int BN1PerThreadBN11 = 4;
    int BK0PerThread     = 1;

    int BM10BN10ThreadClusterBM100 = 2;
    int BM10BN10ThreadClusterBN100 = 2;
    int BM10BN10ThreadClusterBM101 = 8;
    int BM10BN10ThreadClusterBN101 = 8;

    int ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[5]     = {4, 1, 1, 1, 1};
    int ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[5]   = {2, 1, 1, 128, 1};
    int ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[5] = {4, 1, 1, 1, 1};
    int ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[5] = {1, 1, 1, 1, 1};

    int BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[5]     = {1, 4, 1, 1, 1};
    int BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[5]   = {8, 1, 1, 32, 1};
    int BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[5] = {1, 1, 1, 1, 1};
    int BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[5] = {1, 1, 1, 1, 1};

    int CThreadTransferDstScalarPerVector = 1;

    auto GetCompileParameterString() const
    {
        // clang-format off
        return
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
                std::to_string(CThreadTransferDstScalarPerVector);
        // clang-format on
    }
};

const static std::vector<CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw>
    compile_param_list_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw{
        // clang-format off
    {256, 4, 1, 128, 32,  8, 4, 4, 1, 2, 2, 8, 8, {4, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 4, 1, 1, 1}, { 8, 1, 1, 32, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1},
    {256, 8, 1, 128, 16, 16, 4, 4, 1, 2, 2, 8, 8, {8, 1, 1, 1, 1}, {2, 1, 1, 128, 1}, {4, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 8, 1, 1, 1}, {16, 1, 1, 16, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1}
        // clang-format on
    };

} // namespace kernel_compile_parameter
} // namespace ck
#endif
