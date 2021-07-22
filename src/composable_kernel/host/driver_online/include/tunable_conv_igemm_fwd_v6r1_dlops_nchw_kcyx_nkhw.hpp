#ifndef TUNABLE_CONV_IGEMM_FWD_V6R1_DLOPS_NCHW_KCYX_NKHW_HPP
#define TUNABLE_CONV_IGEMM_FWD_V6R1_DLOPS_NCHW_KCYX_NKHW_HPP

struct tunable_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw
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

    std::vector<int> ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1     = {4, 1, 1, 1, 1};
    std::vector<int> ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1   = {2, 1, 1, 128, 1};
    std::vector<int> ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1 = {4, 1, 1, 1, 1};
    std::vector<int> ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1 = {1, 1, 1, 1, 1};

    std::vector<int> BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1     = {1, 4, 1, 1, 1};
    std::vector<int> BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1   = {8, 1, 1, 32, 1};
    std::vector<int> BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1 = {1, 1, 1, 1, 1};
    std::vector<int> BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1 = {1, 1, 1, 1, 1};

    int CThreadTransferDstScalarPerVector = 1;
};
#endif
