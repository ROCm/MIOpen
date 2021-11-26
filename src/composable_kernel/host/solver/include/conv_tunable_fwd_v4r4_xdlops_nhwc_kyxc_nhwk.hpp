/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef CONV_TUNABLE_FWD_V4R4_XDLOPS_NHWC_KYXC_NHWK_HPP
#define CONV_TUNABLE_FWD_V4R4_XDLOPS_NHWC_KYXC_NHWK_HPP

struct tunable_dyn_conv_fwd_v4r4_xdlops_nhwc_kyxc_nhwk
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
