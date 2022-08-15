/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include "static_kernel_common_header.hpp"
#include "static_kernel_gridwise_convolution_implicit_gemm_v4r1_gnchw_gkcyx_gnkhw_lds_double_buffer.hpp"
#include "static_kernel_gridwise_convolution_implicit_gemm_v4r1_fp16_bfp16_gnchw_gkcyx_gnkhw_lds_double_buffer.hpp"
#include "float_types.h"

extern "C" __global__
    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_implicit_gemm_v4r1_gnchw_gkcyx_gnkhw_lds_double_buffer(
        const FLOAT* const __restrict__ p_in_global,
        const FLOAT* const __restrict__ p_wei_global,
        FLOAT* const __restrict__ p_out_global)
{
    using namespace ck;

    // read params: problem decription
    constexpr index_t N  = CK_PARAM_PROBLEM_N;
    constexpr index_t K  = CK_PARAM_PROBLEM_K;
    constexpr index_t C  = CK_PARAM_PROBLEM_C;
    constexpr index_t Hi = CK_PARAM_PROBLEM_HI;
    constexpr index_t Wi = CK_PARAM_PROBLEM_WI;
    constexpr index_t Ho = CK_PARAM_PROBLEM_HO;
    constexpr index_t Wo = CK_PARAM_PROBLEM_WO;
    constexpr index_t Y  = CK_PARAM_PROBLEM_Y;
    constexpr index_t X  = CK_PARAM_PROBLEM_X;

    constexpr index_t ConvStrideH = CK_PARAM_PROBLEM_CONV_STRIDE_H;
    constexpr index_t ConvStrideW = CK_PARAM_PROBLEM_CONV_STRIDE_W;

    constexpr index_t ConvDilationH = CK_PARAM_PROBLEM_CONV_DILATION_H;
    constexpr index_t ConvDilationW = CK_PARAM_PROBLEM_CONV_DILATION_W;

    constexpr index_t LeftPadH = CK_PARAM_PROBLEM_LEFT_PAD_H;
    constexpr index_t LeftPadW = CK_PARAM_PROBLEM_LEFT_PAD_W;

    constexpr index_t RightPadH = CK_PARAM_PROBLEM_RIGHT_PAD_H;
    constexpr index_t RightPadW = CK_PARAM_PROBLEM_RIGHT_PAD_W;

    // read params: tunable params
    constexpr index_t BlockSize = CK_PARAM_TUNABLE_BLOCK_SIZE;

    constexpr index_t BPerBlock = CK_PARAM_TUNABLE_B_PER_BLOCK;
    constexpr index_t KPerBlock = CK_PARAM_TUNABLE_K_PER_BLOCK;
    constexpr index_t EPerBlock = CK_PARAM_TUNABLE_E_PER_BLOCK;

    // read params: dependent params
    constexpr index_t GridSize = CK_PARAM_DEPENDENT_GRID_SIZE;

    constexpr index_t GroupCounts = CK_PARAM_PROBLEM_CONV_GROUP_COUNTS;

    constexpr auto CPerGroup = C / GroupCounts;
    constexpr auto KPerGroup = K / GroupCounts;

// calculate dependent params amd heuristic params
#if CK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD
    constexpr auto ConvDirection = ConvolutionDirection::Forward;

    constexpr auto in_gnchw_desc =
        make_native_tensor_descriptor(Sequence<GroupCounts, N, CPerGroup, Hi, Wi>{},
                                      Sequence<CPerGroup * Hi * Wi, C * Hi * Wi, Hi * Wi, Wi, 1>{});

    constexpr auto wei_gkcyx_desc =
        make_native_tensor_descriptor_packed(Sequence<GroupCounts, KPerGroup, CPerGroup, Y, X>{});

    constexpr auto out_gnkhw_desc =
        make_native_tensor_descriptor(Sequence<GroupCounts, N, KPerGroup, Ho, Wo>{},
                                      Sequence<KPerGroup * Ho * Wo, K * Ho * Wo, Ho * Wo, Wo, 1>{});

    using ConvStrides   = Sequence<ConvStrideH, ConvStrideW>;
    using ConvDilations = Sequence<ConvDilationH, ConvDilationW>;
#elif CK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT
    constexpr auto ConvDirection = ConvolutionDirection::BackwardWeight;

    // In the WrW direction the filter is the output, while the output image is the input being
    // convolved with the (original) input image. This requires that the tensordescriptors be
    // swapped
    // To reuse the fwd kernel for this operation we need to swap the n and c dimension of the
    // input descriptor, the n and k dimension of the output descriptor
    // This change is necessary so that reduction dimensions are consistent with the requirement
    // of the wrw convolution when used in a fwd context
    constexpr auto tmp_in_gnchw_desc =
        make_native_tensor_descriptor(Sequence<GroupCounts, N, CPerGroup, Hi, Wi>{},
                                      Sequence<CPerGroup * Hi * Wi, C * Hi * Wi, Hi * Wi, Wi, 1>{});

    constexpr auto tmp_wei_gkcyx_desc =
        make_native_tensor_descriptor_packed(Sequence<GroupCounts, KPerGroup, CPerGroup, Y, X>{});

    constexpr auto tmp_out_gnkhw_desc =
        make_native_tensor_descriptor(Sequence<GroupCounts, N, KPerGroup, Ho, Wo>{},
                                      Sequence<KPerGroup * Ho * Wo, K * Ho * Wo, Ho * Wo, Wo, 1>{});

    // swapping tensors and dimensions for bwd-weight pass
    constexpr auto in_gnchw_desc =
        reorder_tensor_descriptor_given_upper2lower(tmp_in_gnchw_desc, Sequence<0, 2, 1, 3, 4>{});
    constexpr auto wei_gkcyx_desc =
        reorder_tensor_descriptor_given_upper2lower(tmp_out_gnkhw_desc, Sequence<0, 2, 1, 3, 4>{});
    constexpr auto out_gnkhw_desc =
        reorder_tensor_descriptor_given_upper2lower(tmp_wei_gkcyx_desc, Sequence<0, 2, 1, 3, 4>{});

    // swap stride and dilation
    using ConvDilations = Sequence<ConvStrideH, ConvStrideW>;
    using ConvStrides   = Sequence<ConvDilationH, ConvDilationW>;
#else
    static_assert(false,
                  "wrong! convolution direction should be either forward of backward-weight");
#endif

    using LeftPads  = Sequence<LeftPadH, LeftPadW>;
    using RightPads = Sequence<RightPadH, RightPadW>;

    constexpr index_t GemmMPerThreadSubC = CK_PARAM_GEMM_M_PER_THREAD_SUB_C;
    constexpr index_t GemmNPerThreadSubC = CK_PARAM_GEMM_N_PER_THREAD_SUB_C;
    constexpr index_t GemmMLevel0Cluster = CK_PARAM_GEMM_M_LEVEL0_CLUSTER;
    constexpr index_t GemmNLevel0Cluster = CK_PARAM_GEMM_N_LEVEL0_CLUSTER;
    constexpr index_t GemmMLevel1Cluster = CK_PARAM_GEMM_M_LEVEL1_CLUSTER;
    constexpr index_t GemmNLevel1Cluster = CK_PARAM_GEMM_N_LEVEL1_CLUSTER;
    constexpr index_t GemmKPerThreadLoop = 1;

    constexpr index_t GemmNRepeat = CK_PARAM_GEMM_N_REPEAT;
    constexpr index_t N1          = GemmNRepeat;
    constexpr index_t N2          = GemmNPerThreadSubC;

    constexpr index_t InBlockCopyClusterLengths_E  = CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_E;
    constexpr index_t InBlockCopyClusterLengths_B  = CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_B;
    constexpr index_t InBlockCopyClusterLengths_N1 = CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_N1;
    constexpr index_t InBlockCopyClusterLengths_N2 = CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_N2;

    constexpr index_t InBlockCopySubLengths_E  = EPerBlock / InBlockCopyClusterLengths_E;
    constexpr index_t InBlockCopySubLengths_B  = BPerBlock / InBlockCopyClusterLengths_B;
    constexpr index_t InBlockCopySubLengths_N1 = N1 / InBlockCopyClusterLengths_N1;
    constexpr index_t InBlockCopySubLengths_N2 = N2 / InBlockCopyClusterLengths_N2;

    constexpr index_t WeiBlockCopyClusterLengths_E = CK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_E;
    constexpr index_t WeiBlockCopyClusterLengths_K = CK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_K;
    constexpr index_t WeiBlockCopySubLengths_E     = EPerBlock / WeiBlockCopyClusterLengths_E;
    constexpr index_t WeiBlockCopySubLengths_K     = KPerBlock / WeiBlockCopyClusterLengths_K;

#if MIOPEN_USE_FP32

    constexpr index_t GemmDataPerReadA = GemmMPerThreadSubC;
    constexpr index_t GemmDataPerReadB = GemmNPerThreadSubC;

    using InBlockCopySubLengths_G_E_N1_B_N2     = Sequence<1,
                                                       InBlockCopySubLengths_E,
                                                       InBlockCopySubLengths_N1,
                                                       InBlockCopySubLengths_B,
                                                       InBlockCopySubLengths_N2>;
    using InBlockCopyClusterLengths_G_E_N1_B_N2 = Sequence<1,
                                                           InBlockCopyClusterLengths_E,
                                                           InBlockCopyClusterLengths_N1,
                                                           InBlockCopyClusterLengths_B,
                                                           InBlockCopyClusterLengths_N2>;

    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 2, 4, 3>; // [G, E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 1, 3, 2, 4>; // [G, E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3, 4>; // [G, E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = CK_PARAM_IN_BLOCK_COPY_SRC_DATA_PER_READ_B;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = CK_PARAM_IN_BLOCK_COPY_DST_DATA_PER_WRITE_N2;

    using WeiBlockCopySubLengths_G_E_K =
        Sequence<1, WeiBlockCopySubLengths_E, WeiBlockCopySubLengths_K>;
    using WeiBlockCopyClusterLengths_G_E_K =
        Sequence<1, WeiBlockCopyClusterLengths_E, WeiBlockCopyClusterLengths_K>;

    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<0, 2, 1>; // [G, K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<0, 2, 1>; // [G, K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1, 2>; // [G, E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E = CK_PARAM_WEI_BLOCK_COPY_SRC_DATA_PER_READ_E;
    constexpr index_t WeiBlockCopyDstDataPerWrite_EPack =
        CK_PARAM_WEI_BLOCK_COPY_DST_DATA_PER_WRITE_K;

#elif MIOPEN_USE_FP16 || MIOPEN_USE_BFP16

    constexpr index_t EPACK            = CK_PARAM_EPACK_LENGTH;
    constexpr index_t GemmDataPerReadA = 1;
    constexpr index_t GemmDataPerReadB = 1;

    using InBlockCopySubLengths_G_E_N1_B_N2_EPack     = Sequence<1,
                                                             InBlockCopySubLengths_E,
                                                             InBlockCopySubLengths_N1,
                                                             InBlockCopySubLengths_B,
                                                             InBlockCopySubLengths_N2,
                                                             EPACK>;
    using InBlockCopyClusterLengths_G_E_N1_B_N2_EPack = Sequence<1,
                                                                 InBlockCopyClusterLengths_E,
                                                                 InBlockCopyClusterLengths_N1,
                                                                 InBlockCopyClusterLengths_B,
                                                                 InBlockCopyClusterLengths_N2,
                                                                 1>;

    constexpr index_t InBlockCopySrcDataPerRead_B = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_EPack =
        CK_PARAM_IN_BLOCK_COPY_DST_DATA_PER_WRITE_EPACK;

    // EPACK  - E dimension is folded into 2 dimensions E and EPACK
    using InBlockCopyThreadClusterArrangeOrder =
        Sequence<0, 1, 2, 4, 3, 5>;                               // [G, E, N1, N2, B, EPACK]
    using InBlockCopySrcAccessOrder = Sequence<0, 1, 2, 4, 3, 5>; // [G, E, N1, N2, B, EPACK]
    using InBlockCopyDstAccessOrder = Sequence<0, 1, 2, 3, 4, 5>; // [G, E, N1, B, N2, EPACK]

    using WeiBlockCopySubLengths_G_E_K_EPack =
        Sequence<1, WeiBlockCopySubLengths_E, WeiBlockCopySubLengths_K, EPACK>;
    using WeiBlockCopyClusterLengths_G_E_K_EPack =
        Sequence<1, WeiBlockCopyClusterLengths_E, WeiBlockCopyClusterLengths_K, 1>;

    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<0, 2, 1, 3>; // [G, K, E, EPACK]
    using WeiBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [G, K, E, EPACK]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [G, E, K, EPACK]

    constexpr index_t WeiBlockCopySrcDataPerRead_E = CK_PARAM_WEI_BLOCK_COPY_SRC_DATA_PER_READ_E;
    constexpr index_t WeiBlockCopyDstDataPerWrite_EPack =
        CK_PARAM_WEI_BLOCK_COPY_DST_DATA_PER_WRITE_EPACK;
#else
    static_assert(false, "wrong! Only kperblock could be 32/64/128 not supported");
#endif

    constexpr auto gridwise_conv =
#if MIOPEN_USE_FP32
        GridwiseConvolutionImplicitGemm_v4r1_gnchw_gkcyx_gnkhw_lds_double_buffer<
            GridSize,
            BlockSize,
            FLOAT,
            FLOAT_ACCUM,
            decltype(in_gnchw_desc),
            decltype(wei_gkcyx_desc),
            decltype(out_gnkhw_desc),
            ConvStrides,
            ConvDilations,
            LeftPads,
            RightPads,
            ConvDirection,
            BPerBlock,
            KPerBlock,
            EPerBlock,
            GemmNRepeat,
            GemmMPerThreadSubC,
            GemmNPerThreadSubC,
            GemmMLevel0Cluster,
            GemmNLevel0Cluster,
            GemmMLevel1Cluster,
            GemmNLevel1Cluster,
            GemmKPerThreadLoop,
            GemmDataPerReadA,
            GemmDataPerReadB,
            InBlockCopySubLengths_G_E_N1_B_N2,
            InBlockCopyClusterLengths_G_E_N1_B_N2,
            InBlockCopyThreadClusterArrangeOrder,
            InBlockCopySrcAccessOrder,
            InBlockCopyDstAccessOrder,
            InBlockCopySrcDataPerRead_B,
            InBlockCopyDstDataPerWrite_N2,
            WeiBlockCopySubLengths_G_E_K,
            WeiBlockCopyClusterLengths_G_E_K,
            WeiBlockCopyThreadClusterArrangeOrder,
            WeiBlockCopySrcAccessOrder,
            WeiBlockCopyDstAccessOrder,
            WeiBlockCopySrcDataPerRead_E,
            WeiBlockCopyDstDataPerWrite_EPack>{};
#elif MIOPEN_USE_FP16 || MIOPEN_USE_BFP16
        GridwiseConvolutionImplicitGemm_v4r1_fp16_bfp16_gnchw_gkcyx_gnkhw_lds_double_buffer<
            GridSize,
            BlockSize,
            FLOAT,
            FLOAT_ACCUM,
            decltype(in_gnchw_desc),
            decltype(wei_gkcyx_desc),
            decltype(out_gnkhw_desc),
            ConvStrides,
            ConvDilations,
            LeftPads,
            RightPads,
            ConvDirection,
            BPerBlock,
            KPerBlock,
            EPerBlock,
            GemmNRepeat,
            EPACK,
            GemmMPerThreadSubC,
            GemmNPerThreadSubC,
            GemmMLevel0Cluster,
            GemmNLevel0Cluster,
            GemmMLevel1Cluster,
            GemmNLevel1Cluster,
            GemmKPerThreadLoop,
            GemmDataPerReadA,
            GemmDataPerReadB,
            InBlockCopySubLengths_G_E_N1_B_N2_EPack,
            InBlockCopyClusterLengths_G_E_N1_B_N2_EPack,
            InBlockCopyThreadClusterArrangeOrder,
            InBlockCopySrcAccessOrder,
            InBlockCopyDstAccessOrder,
            InBlockCopySrcDataPerRead_B,
            InBlockCopyDstDataPerWrite_EPack,
            WeiBlockCopySubLengths_G_E_K_EPack,
            WeiBlockCopyClusterLengths_G_E_K_EPack,
            WeiBlockCopyThreadClusterArrangeOrder,
            WeiBlockCopySrcAccessOrder,
            WeiBlockCopyDstAccessOrder,
            WeiBlockCopySrcDataPerRead_E,
            WeiBlockCopyDstDataPerWrite_EPack>{};
#else
        static_assert(false, "wrong! Only fp32, fp16 and bfp16 are supported.");
#endif
    gridwise_conv.Run(p_in_global, p_wei_global, p_out_global);
}
