#ifndef CONV_TUNABLE_FWD_V4R4_XDLOPS_NCHW_KCYX_NKHW_HPP
#define CONV_TUNABLE_FWD_V4R4_XDLOPS_NCHW_KCYX_NKHW_HPP

struct tunable_dyn_conv_fwd_v4r4_xdlops_nchw_kcyx_nkhw
{
    int BlockSize;

    int MPerBlock;
    int NPerBlock;
    int KPerBlock;

    int MPerWave;
    int NPerWave;
    int K1;

    int MRepeat;
    int NRepeat;

    std::array<int, 3> ABlockTransferThreadSliceLengths_K0_M_K1;
    std::array<int, 3> ABlockTransferThreadClusterLengths_K0_M_K1;
    std::array<int, 3> ABlockTransferThreadClusterArrangeOrder;
    std::array<int, 3> ABlockTransferSrcAccessOrder;
    int ABlockTransferSrcVectorDim;
    int ABlockTransferSrcScalarPerVector;
    int ABlockTransferDstScalarPerVector_K1;
    bool AThreadTransferSrcResetCoordinateAfterRun;

    std::array<int, 3> BBlockTransferThreadSliceLengths_K0_N_K1;
    std::array<int, 3> BBlockTransferThreadClusterLengths_K0_N_K1;
    std::array<int, 3> BBlockTransferThreadClusterArrangeOrder;
    std::array<int, 3> BBlockTransferSrcAccessOrder;
    int BBlockTransferSrcVectorDim;
    int BBlockTransferSrcScalarPerVector;
    int BBlockTransferDstScalarPerVector_K1;
    bool BThreadTransferSrcResetCoordinateAfterRun;

    std::array<int, 8> CThreadTransferSrcDstAccessOrder;
    int CThreadTransferSrcDstVectorDim;
    int CThreadTransferDstScalarPerVector;
};

static tunable_dyn_conv_fwd_v4r4_xdlops_nchw_kcyx_nkhw
    default_tunable_dyn_conv_fwd_v4r4_xdlops_nchw_kcyx_nkhw = {
        256,                      // BlockSize
        128,                      // MPerBlock,
        128,                      // NPerBlock,
        4,                        // KPerBlock,
        32,                       // MPerWave,
        32,                       // NPerWave,
        4,                        // K1,
        2,                        // MRepeat,
        2,                        // NRepeat,
        {1, 2, 4},                // ABlockTransferThreadSliceLengths_K0_M_K1,
        {4, 64, 1},               // ABlockTransferThreadClusterLengths_K0_M_K1,
        {1, 0, 2},                // ABlockTransferThreadClusterArrangeOrder,
        {1, 0, 2},                // ABlockTransferSrcAccessOrder,
        2,                        // ABlockTransferSrcVectorDim
        1,                        // ABlockTransferSrcScalarPerVector,
        4,                        // ABlockTransferDstScalarPerVector_K1,
        false,                    // AThreadTransferSrcResetCoordinateAfterRun,
        {1, 2, 4},                // BBlockTransferThreadSliceLengths_K0_N_K1,
        {4, 64, 1},               // BBlockTransferThreadClusterLengths_K0_N_K1,
        {0, 2, 1},                // BBlockTransferThreadClusterArrangeOrder,
        {1, 0, 2},                // BBlockTransferSrcAccessOrder,
        1,                        // BBlockTransferSrcVectorDim
        1,                        // BBlockTransferSrcScalarPerVector
        4,                        // BBlockTransferDstScalarPerVector_K1
        false,                    // BThreadTransferSrcResetCoordinateAfterRun
        {3, 0, 1, 2, 7, 5, 4, 6}, // CThreadTransferSrcDstAccessOrder
        7,                        // CThreadTransferSrcDstVectorDim,
        1                         // CThreadTransferDstScalarPerVector
};
#endif
