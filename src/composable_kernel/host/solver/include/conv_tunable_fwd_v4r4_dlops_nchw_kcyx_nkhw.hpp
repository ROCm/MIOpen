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
#ifndef CONV_TUNABLE_FWD_V4R4_DLOPS_NCHW_KCYX_NKHW_HPP
#define CONV_TUNABLE_FWD_V4R4_DLOPS_NCHW_KCYX_NKHW_HPP

struct tunable_dyn_conv_fwd_v4r4_dlops_nchw_kcyx_nkhw
{
    int BlockSize;

    int MPerBlock;
    int NPerBlock;
    int KPerBlock;

    int M1PerThread;
    int N1PerThread;
    int KPerThread;

    int M1N1ThreadClusterM10;
    int M1N1ThreadClusterN10;
    int M1N1ThreadClusterM11;
    int M1N1ThreadClusterN11;

    std::array<int, 3> ABlockTransferThreadSliceLengths_K_M0_M1;
    std::array<int, 3> ABlockTransferThreadClusterLengths_K_M0_M1;
    std::array<int, 3> ABlockTransferThreadClusterArrangeOrder;
    std::array<int, 3> ABlockTransferSrcAccessOrder;
    int ABlockTransferSrcVectorDim;
    int ABlockTransferSrcScalarPerVector;
    int ABlockTransferDstScalarPerVector_M1;
    bool AThreadTransferSrcResetCoordinateAfterRun;

    std::array<int, 3> BBlockTransferThreadSliceLengths_K_N0_N1;
    std::array<int, 3> BBlockTransferThreadClusterLengths_K_N0_N1;
    std::array<int, 3> BBlockTransferThreadClusterArrangeOrder;
    std::array<int, 3> BBlockTransferSrcAccessOrder;
    int BBlockTransferSrcVectorDim;
    int BBlockTransferSrcScalarPerVector;
    int BBlockTransferDstScalarPerVector_N1;
    bool BThreadTransferSrcResetCoordinateAfterRun;

    std::array<int, 6> CThreadTransferSrcDstAccessOrder;
    int CThreadTransferSrcDstVectorDim;
    int CThreadTransferDstScalarPerVector;
};

static tunable_dyn_conv_fwd_v4r4_dlops_nchw_kcyx_nkhw
    default_tunable_dyn_conv_fwd_v4r4_dlops_nchw_kcyx_nkhw = {
        256,       128,       128, 8, 4,         4,           1,
        8,         8,         2,   2, {4, 1, 1}, {2, 1, 128}, {2, 1, 0},
        {2, 1, 0}, 0,         4,   1, false,     {4, 1, 1},   {2, 1, 128},
        {0, 1, 2}, {0, 1, 2}, 2,   1, 1,         false,       {3, 4, 5, 0, 1, 2},
        5,         1};
#endif
