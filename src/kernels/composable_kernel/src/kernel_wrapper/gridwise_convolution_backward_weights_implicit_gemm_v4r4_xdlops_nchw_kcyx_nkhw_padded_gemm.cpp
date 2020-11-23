#include "common_header.hpp"
#include "gridwise_convolution_backward_weights_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw_padded_gemm.hpp"
#include "float_types.h"

extern "C" __global__
    __launch_bounds__(CK_PARAM_DEPENDENT_BLOCK_SIZE) void gridwise_convolution_backward_weights_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw_padded_gemm(
        const FLOAT* const __restrict__ p_in_global,
        const FLOAT* const __restrict__ p_out_global,
        FLOAT_ACCUM* const __restrict__ p_wei_global)
{
    using namespace ck;

    // read params: problem description
    constexpr index_t G  = CK_PARAM_PROBLEM_G;
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

    constexpr index_t InLeftPadH = CK_PARAM_PROBLEM_IN_LEFT_PAD_H;
    constexpr index_t InLeftPadW = CK_PARAM_PROBLEM_IN_LEFT_PAD_W;

    constexpr index_t InRightPadH = CK_PARAM_PROBLEM_IN_RIGHT_PAD_H;
    constexpr index_t InRightPadW = CK_PARAM_PROBLEM_IN_RIGHT_PAD_W;

    constexpr auto CPerGroup = C / G;

    constexpr auto in_n_c_hi_wi_desc =
        make_native_tensor_descriptor_packed(Sequence<N, C, Hi, Wi>{});
    constexpr auto wei_k_cpergroup_y_x_desc =
        make_native_tensor_descriptor_packed(Sequence<K, CPerGroup, Y, X>{});
    constexpr auto out_n_k_ho_wo_desc =
        make_native_tensor_descriptor_packed(Sequence<N, K, Ho, Wo>{});

    using ConvStrides   = Sequence<ConvStrideH, ConvStrideW>;
    using ConvDilations = Sequence<ConvDilationH, ConvDilationW>;

    using InLeftPads  = Sequence<InLeftPadH, InLeftPadW>;
    using InRightPads = Sequence<InRightPadH, InRightPadW>;

    // read params: tunning parameters
    constexpr index_t GemmMPerBlock = CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK;
    constexpr index_t GemmNPerBlock = CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK;
    constexpr index_t GemmKPerBlock = CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK;
    constexpr index_t GemmMPerWave  = CK_PARAM_TUNABLE_GEMM_M_PER_WAVE;
    constexpr index_t GemmNPerWave  = CK_PARAM_TUNABLE_GEMM_N_PER_WAVE;
    constexpr index_t GemmKPack     = CK_PARAM_TUNABLE_GEMM_KPACK;

    // read params: dependent parameters
    constexpr index_t BlockSize  = CK_PARAM_DEPENDENT_BLOCK_SIZE;
    constexpr index_t GridSize   = CK_PARAM_DEPENDENT_GRID_SIZE;
    constexpr index_t GemmKBlock = CK_PARAM_GEMM_K_BLOCK;

    // A matrix copy
    constexpr index_t GemmABlockCopyClusterLengths_GemmK =
        CK_PARAM_DEPENDENT_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K;
    constexpr index_t GemmABlockCopyClusterLengths_GemmM =
        CK_PARAM_DEPENDENT_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M;
    constexpr index_t GemmABlockCopyClusterLengths_GemmKPack =
        CK_PARAM_DEPENDENT_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_KPACK;

    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmABlockCopyClusterLengths_GemmK;
    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmM =
        GemmMPerBlock / GemmABlockCopyClusterLengths_GemmM;
    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmKPack =
        GemmKPack / GemmABlockCopyClusterLengths_GemmKPack;

    using GemmABlockCopyClusterLengths_GemmG_GemmK_GemmM_GemmKPack =
        Sequence<1,
                 GemmABlockCopyClusterLengths_GemmK,
                 GemmABlockCopyClusterLengths_GemmM,
                 GemmABlockCopyClusterLengths_GemmKPack>;
    using GemmABlockCopySubLengths_GemmG_GemmK_GemmM_GemmKPack =
        Sequence<1,
                 GemmABlockCopyThreadSliceLengths_GemmK,
                 GemmABlockCopyThreadSliceLengths_GemmM,
                 GemmABlockCopyThreadSliceLengths_GemmKPack>;

    using GemmABlockCopyThreadClusterArrangeOrder =
        Sequence<0, 2, 1, 3>;                                  // [GemmG, GemmM, GemmK, GemmKPack]
    using GemmABlockCopySrcAccessOrder = Sequence<0, 2, 1, 3>; // [GemmG, GemmM, GemmK, GemmKPack]
    using GemmABlockCopyDstAccessOrder = Sequence<0, 1, 2, 3>; // [GemmG, GemmK, GemmM, GemmKPack]

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmKPack =
        CK_PARAM_DEPENDENT_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_KPACK;

    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmKPack =
        CK_PARAM_DEPENDENT_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK;

    // B matrix Copy
    constexpr index_t GemmBBlockCopyClusterLengths_GemmK =
        CK_PARAM_DEPENDENT_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K;
    constexpr index_t GemmBBlockCopyClusterLengths_GemmN =
        CK_PARAM_DEPENDENT_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N;
    constexpr index_t GemmBBlockCopyClusterLengths_GemmKPack =
        CK_PARAM_DEPENDENT_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_KPACK;

    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmBBlockCopyClusterLengths_GemmK;
    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmN =
        GemmNPerBlock / GemmBBlockCopyClusterLengths_GemmN;
    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmKPack =
        GemmKPack / GemmBBlockCopyClusterLengths_GemmKPack;

    using GemmBBlockCopyClusterLengths_GemmG_GemmK_GemmN_GemmKPack =
        Sequence<1,
                 GemmBBlockCopyClusterLengths_GemmK,
                 GemmBBlockCopyClusterLengths_GemmN,
                 GemmBBlockCopyClusterLengths_GemmKPack>;
    using GemmBBlockCopySubLengths_GemmG_GemmK_GemmN_GemmKPack =
        Sequence<1,
                 GemmBBlockCopyThreadSliceLengths_GemmK,
                 GemmBBlockCopyThreadSliceLengths_GemmN,
                 GemmBBlockCopyThreadSliceLengths_GemmKPack>;

    using GemmBBlockCopyThreadClusterArrangeOrder =
        Sequence<0, 2, 1, 3>;                                  // [GemmG, GemmK, GemmKPack, GemmN]
    using GemmBBlockCopySrcAccessOrder = Sequence<0, 2, 1, 3>; // [GemmG, GemmK, GemmKPack, GemmN]
    using GemmBBlockCopyDstAccessOrder = Sequence<0, 1, 2, 3>; // [GemmG, GemmK, GemmN, GemmKPack]

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmKPack =
        CK_PARAM_DEPENDENT_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_KPACK;

    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmKPack =
        CK_PARAM_DEPENDENT_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK;

    // gridwise GEMM
    constexpr auto wkgrp_schd_order = NBlock1MBlock0;
    constexpr auto GemmMPad         = CK_GEMM_M_PAD;
    constexpr auto GemmNPad         = CK_GEMM_N_PAD;
    constexpr auto GemmKTotalPad    = CK_GEMM_K_TOTAL_PAD;

    constexpr auto gridwise_conv =
        GridwiseConvolutionBackwardWeightsImplicitGemm_v4r4_xdlops_nchw_kcyx_nkhw_padded_gemm<
            GridSize,
            BlockSize,
            FLOAT,       // Input data type
            FLOAT_ACCUM, // Acc data type
            FLOAT_ACCUM, // Ouput data type
            decltype(in_n_c_hi_wi_desc),
            decltype(wei_k_cpergroup_y_x_desc),
            decltype(out_n_k_ho_wo_desc),
            G,
            ConvStrides,
            ConvDilations,
            InLeftPads,
            InRightPads,
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmMPerWave,
            GemmNPerWave,
            GemmKPack,
            GemmABlockCopySubLengths_GemmG_GemmK_GemmM_GemmKPack,
            GemmABlockCopyClusterLengths_GemmG_GemmK_GemmM_GemmKPack,
            GemmABlockCopyThreadClusterArrangeOrder,
            GemmABlockCopySrcAccessOrder,
            GemmABlockCopyDstAccessOrder,
            GemmABlockCopySrcDataPerRead_GemmKPack,
            GemmABlockCopyDstDataPerWrite_GemmKPack,
            GemmBBlockCopySubLengths_GemmG_GemmK_GemmN_GemmKPack,
            GemmBBlockCopyClusterLengths_GemmG_GemmK_GemmN_GemmKPack,
            GemmBBlockCopyThreadClusterArrangeOrder,
            GemmBBlockCopySrcAccessOrder,
            GemmBBlockCopyDstAccessOrder,
            GemmBBlockCopySrcDataPerRead_GemmKPack,
            GemmBBlockCopyDstDataPerWrite_GemmKPack,
            GemmMPad,
            GemmNPad,
            GemmKTotalPad,
            wkgrp_schd_order,
            GemmKBlock>{};
    gridwise_conv.Run(p_in_global, p_out_global, p_wei_global);
}
