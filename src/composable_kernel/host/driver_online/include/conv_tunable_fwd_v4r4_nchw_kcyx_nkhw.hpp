#ifndef CONV_TUNABLE_FWD_V4R4_NCHW_KCYX_NKHW_HPP
#define CONV_TUNABLE_FWD_V4R4_NCHW_KCYX_NKHW_HPP

struct tunable_dyn_conv_fwd_v4r4_nchw_kcyx_nkhw
{
    int32_t BlockSize;

    int32_t MPerBlock;
    int32_t NPerBlock;
    int32_t KPerBlock;

    int32_t M1PerThread;
    int32_t N1PerThread;
    int32_t KPerThread;

    int32_t M1N1ThreadClusterM10;
    int32_t M1N1ThreadClusterN10;
    int32_t M1N1ThreadClusterM11;
    int32_t M1N1ThreadClusterN11;

    std::array<int32_t, 3> ABlockTransferThreadSliceLengths_K_M0_M1;
    std::array<int32_t, 3> ABlockTransferThreadClusterLengths_K_M0_M1;
    std::array<int32_t, 3> ABlockTransferThreadClusterArrangeOrder;
    std::array<int32_t, 3> ABlockTransferSrcAccessOrder;
    int32_t ABlockTransferSrcVectorDim;
    int32_t ABlockTransferSrcScalarPerVector;
    int32_t ABlockTransferDstScalarPerVector_M1;
    bool AThreadTransferSrcResetCoordinateAfterRun;

    std::array<int32_t, 3> BBlockTransferThreadSliceLengths_K_N0_N1;
    std::array<int32_t, 3> BBlockTransferThreadClusterLengths_K_N0_N1;
    std::array<int32_t, 3> BBlockTransferThreadClusterArrangeOrder;
    std::array<int32_t, 3> BBlockTransferSrcAccessOrder;
    int32_t BBlockTransferSrcVectorDim;
    int32_t BBlockTransferSrcScalarPerVector;
    int32_t BBlockTransferDstScalarPerVector_N1;
    bool BThreadTransferSrcResetCoordinateAfterRun;

    std::array<int32_t, 6> CThreadTransferSrcDstAccessOrder;
    int32_t CThreadTransferSrcDstVectorDim;
    int32_t CThreadTransferDstScalarPerVector;
};

static tunable_dyn_conv_fwd_v4r4_nchw_kcyx_nkhw default_tunable_dyn_conv_fwd_v4r4_nchw_kcyx_nkhw = {
    256,       128,       128, 8, 4,         4,           1,
    8,         8,         2,   2, {4, 1, 1}, {2, 1, 128}, {2, 1, 0},
    {2, 1, 0}, 0,         4,   1, false,     {4, 1, 1},   {2, 1, 128},
    {0, 1, 2}, {0, 1, 2}, 2,   1, 1,         false,       {3, 4, 5, 0, 1, 2},
    5,         1};
#endif
