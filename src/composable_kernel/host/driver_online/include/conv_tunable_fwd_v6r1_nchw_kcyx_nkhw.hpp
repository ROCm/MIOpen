#ifndef CONV_TUNABLE_FWD_V6R1_NCHW_KCYX_NKHW_HPP
#define CONV_TUNABLE_FWD_V6R1_NCHW_KCYX_NKHW_HPP

struct tunable_dyn_conv_fwd_v6r1_nchw_kcyx_nkhw
{
    int32_t BlockSize = 256;

    int32_t GN0 = 4;
    int32_t GK1 = 1;

    int32_t GM1PerBlockGM11 = 128;
    int32_t GN1PerBlockGN11 = 32;
    int32_t GK0PerBlock     = 8;

    int32_t BM1PerThreadBM11 = 4;
    int32_t BN1PerThreadBN11 = 4;
    int32_t BK0PerThread     = 1;

    int32_t BM10BN10ThreadClusterBM100 = 2;
    int32_t BM10BN10ThreadClusterBN100 = 2;
    int32_t BM10BN10ThreadClusterBM101 = 8;
    int32_t BM10BN10ThreadClusterBN101 = 8;

    std::array<int32_t, 5> ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1 = {4, 1, 1, 1, 1};
    std::array<int32_t, 5> ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1 = {
        2, 1, 1, 128, 1};
    std::array<int32_t, 5> ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1 = {
        4, 1, 1, 1, 1};
    std::array<int32_t, 5> ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1 = {
        1, 1, 1, 1, 1};

    std::array<int32_t, 5> BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1 = {1, 4, 1, 1, 1};
    std::array<int32_t, 5> BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1 = {
        8, 1, 1, 32, 1};
    std::array<int32_t, 5> BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1 = {
        1, 1, 1, 1, 1};
    std::array<int32_t, 5> BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1 = {
        1, 1, 1, 1, 1};

    int32_t CThreadTransferDstScalarPerVector = 1;
};
#endif
