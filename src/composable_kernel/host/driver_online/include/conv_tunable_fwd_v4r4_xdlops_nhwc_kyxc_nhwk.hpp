#ifndef CONV_TUNABLE_FWD_V4R4_XDLOPS_NHWC_KYXC_NHWK_HPP
#define CONV_TUNABLE_FWD_V4R4_XDLOPS_NHWC_KYXC_NHWK_HPP

struct tunable_dyn_conv_fwd_v4r4_xdlops_nhwc_kyxc_nhwk
{
    int32_t BlockSize;

    int32_t MPerBlock;
    int32_t NPerBlock;
    int32_t KPerBlock;

    int32_t MPerWave;
    int32_t NPerWave;
    int32_t K1;

    int32_t MRepeat;
    int32_t NRepeat;

    std::array<int32_t, 3> ABlockTransferThreadSliceLengths_K0_M_K1;
    std::array<int32_t, 3> ABlockTransferThreadClusterLengths_K0_M_K1;
    std::array<int32_t, 3> ABlockTransferThreadClusterArrangeOrder;
    std::array<int32_t, 3> ABlockTransferSrcAccessOrder;
    int32_t ABlockTransferSrcVectorDim;
    int32_t ABlockTransferSrcScalarPerVector;
    int32_t ABlockTransferDstScalarPerVector_K1;
    bool AThreadTransferSrcResetCoordinateAfterRun;

    std::array<int32_t, 3> BBlockTransferThreadSliceLengths_K0_N_K1;
    std::array<int32_t, 3> BBlockTransferThreadClusterLengths_K0_N_K1;
    std::array<int32_t, 3> BBlockTransferThreadClusterArrangeOrder;
    std::array<int32_t, 3> BBlockTransferSrcAccessOrder;
    int32_t BBlockTransferSrcVectorDim;
    int32_t BBlockTransferSrcScalarPerVector;
    int32_t BBlockTransferDstScalarPerVector_K1;
    bool BThreadTransferSrcResetCoordinateAfterRun;

    std::array<int32_t, 8> CThreadTransferSrcDstAccessOrder;
    int32_t CThreadTransferSrcDstVectorDim;
    int32_t CThreadTransferDstScalarPerVector;
};

static tunable_dyn_conv_fwd_v4r4_xdlops_nhwc_kyxc_nhwk
    default_tunable_dyn_conv_fwd_v4r4_xdlops_nhwc_kyxc_nhwk = {
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
        4,                        // ABlockTransferSrcScalarPerVector,
        4,                        // ABlockTransferDstScalarPerVector_K1,
        false,                    // AThreadTransferSrcResetCoordinateAfterRun,
        {1, 2, 4},                // BBlockTransferThreadSliceLengths_K0_N_K1,
        {4, 64, 1},               // BBlockTransferThreadClusterLengths_K0_N_K1,
        {1, 0, 2},                // BBlockTransferThreadClusterArrangeOrder,
        {1, 0, 2},                // BBlockTransferSrcAccessOrder,
        2,                        // BBlockTransferSrcVectorDim
        4,                        // BBlockTransferSrcScalarPerVector
        4,                        // BBlockTransferDstScalarPerVector_K1
        false,                    // BThreadTransferSrcResetCoordinateAfterRun
        {2, 3, 0, 1, 7, 5, 4, 6}, // CThreadTransferSrcDstAccessOrder
        7,                        // CThreadTransferSrcDstVectorDim,
        1                         // CThreadTransferDstScalarPerVector
};
#endif
