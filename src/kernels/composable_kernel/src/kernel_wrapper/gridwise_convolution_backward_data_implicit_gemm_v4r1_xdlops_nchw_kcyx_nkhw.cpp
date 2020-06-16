#include "common_header.hpp"
#include "gridwise_convolution_backward_data_implicit_gemm_v4r1_xdlops_nchw_kcyx_nkhw.hpp"
#include "gridwise_convolution_backward_data_implicit_gemm_v4r1_xdlops_fp16_bfp16_nchw_kcyx_nkhw.hpp"
#include "float_types.h"

extern "C" __global__
    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_backward_data_implicit_gemm_v4r1_xdlops_nchw_kcyx_nkhw(
        const FLOAT* const __restrict__ p_out_global,
        const FLOAT* const __restrict__ p_wei_global,
        FLOAT* const __restrict__ p_in_global)
{
    using namespace ck;

    // read problem parameters
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

    constexpr index_t BlockSize = CK_PARAM_TUNABLE_BLOCK_SIZE;
    constexpr index_t GridSize  = CK_PARAM_DEPENDENT_GRID_SIZE;

    constexpr index_t GemmMPerBlock = CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK;
    constexpr index_t GemmNPerBlock = CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK;
    constexpr index_t GemmKPerBlock = CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK;

    constexpr auto in_nchw_desc  = make_native_tensor_descriptor_packed(Sequence<N, C, Hi, Wi>{});
    constexpr auto wei_kcyx_desc = make_native_tensor_descriptor_packed(Sequence<K, C, Y, X>{});
    constexpr auto out_nkhw_desc = make_native_tensor_descriptor_packed(Sequence<N, K, Ho, Wo>{});

    using ConvStrides   = Sequence<ConvStrideH, ConvStrideW>;
    using ConvDilations = Sequence<ConvDilationH, ConvDilationW>;

    using InLeftPads  = Sequence<InLeftPadH, InLeftPadW>;
    using InRightPads = Sequence<InRightPadH, InRightPadW>;

    // A matrix
    constexpr index_t GemmABlockCopyClusterLengths_GemmK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K;

    constexpr index_t GemmABlockCopyClusterLengths_GemmM =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M;

    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmABlockCopyClusterLengths_GemmK;

    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmM =
        GemmMPerBlock / GemmABlockCopyClusterLengths_GemmM;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmM =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_M;

    // B matrix
    constexpr index_t GemmBBlockCopyClusterLengths_GemmK =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K;

    constexpr index_t GemmBBlockCopyClusterLengths_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N;

    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmBBlockCopyClusterLengths_GemmK;

    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmN =
        GemmNPerBlock / GemmBBlockCopyClusterLengths_GemmN;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N;

#if MIOPEN_USE_FP32
    // A matrix
    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM =
        Sequence<GemmABlockCopyThreadSliceLengths_GemmK, GemmABlockCopyThreadSliceLengths_GemmM>;

    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM =
        Sequence<GemmABlockCopyClusterLengths_GemmK, GemmABlockCopyClusterLengths_GemmM>;

    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M;

    // B matrix
    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN =
        Sequence<GemmBBlockCopyThreadSliceLengths_GemmK, GemmBBlockCopyThreadSliceLengths_GemmN>;

    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN =
        Sequence<GemmBBlockCopyClusterLengths_GemmK, GemmBBlockCopyClusterLengths_GemmN>;

    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N;

#elif MIOPEN_USE_FP16 || MIOPEN_USE_BFP16
    constexpr index_t GemmKPACK = CK_PARAM_KPACK_LENGTH;

    // A matrix
    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM_GemmKPACK =
        Sequence<GemmABlockCopyThreadSliceLengths_GemmK,
                 GemmABlockCopyThreadSliceLengths_GemmM,
                 GemmKPACK>;

    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM_GemmKPACK =
        Sequence<GemmABlockCopyClusterLengths_GemmK, GemmABlockCopyClusterLengths_GemmM, 1>;

    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmKPACK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK;

    // B matrix
    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN_GemmKPACK =
        Sequence<GemmBBlockCopyThreadSliceLengths_GemmK,
                 GemmBBlockCopyThreadSliceLengths_GemmN,
                 GemmKPACK>;

    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN_GemmKPACK =
        Sequence<GemmBBlockCopyClusterLengths_GemmK, GemmBBlockCopyClusterLengths_GemmN, 1>;

    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmKPACK =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK;

#else
    static_assert(false, "wrong! Only fp32, fp16 and bfp16 are supported.");
#endif

    // C matrix
    constexpr auto GemmMPerWave = CK_PARAM_GEMM_M_PER_WAVE;
    constexpr auto GemmNPerWave = CK_PARAM_GEMM_N_PER_WAVE;

    constexpr index_t GemmThreadGemmDataPerReadM = 1;
    constexpr index_t GemmThreadGemmDataPerReadN = 1;

#if MIOPEN_USE_FP32
    constexpr auto gridwise_conv_bwd_data =
        GridwiseConvolutionBackwardDataImplicitGemm_v4r1_xdlops_nchw_kcyx_nkhw<
            GridSize,
            BlockSize,
            FLOAT,
            FLOAT_ACCUM,
            decltype(in_nchw_desc),
            decltype(wei_kcyx_desc),
            decltype(out_nkhw_desc),
            ConvStrides,
            ConvDilations,
            InLeftPads,
            InRightPads,
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmMPerWave,
            GemmNPerWave,
            GemmThreadGemmDataPerReadM,
            GemmThreadGemmDataPerReadN,
            GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
            GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
            GemmABlockCopySrcDataPerRead_GemmM,
            GemmABlockCopyDstDataPerWrite_GemmM,
            GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
            GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmN>{};

    // these decide which GEMM will be called
    constexpr index_t GemmId = CK_PARAM_GEMM_ID;

    gridwise_conv_bwd_data.template Run<GemmId>(p_in_global, p_wei_global, p_out_global);
#elif MIOPEN_USE_FP16 || MIOPEN_USE_BFP16
    constexpr auto gridwise_conv_bwd_data =
        GridwiseConvolutionBackwardDataImplicitGemm_v4r1_xdlops_fp16_bfp16_nchw_kcyx_nkhw<
            GridSize,
            BlockSize,
            FLOAT,
            FLOAT_ACCUM,
            decltype(in_nchw_desc),
            decltype(wei_kcyx_desc),
            decltype(out_nkhw_desc),
            ConvStrides,
            ConvDilations,
            InLeftPads,
            InRightPads,
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmKPACK,
            GemmMPerWave,
            GemmNPerWave,
            GemmThreadGemmDataPerReadM,
            GemmThreadGemmDataPerReadN,
            GemmABlockCopyThreadSliceLengths_GemmK_GemmM_GemmKPACK,
            GemmABlockCopyThreadClusterLengths_GemmK_GemmM_GemmKPACK,
            GemmABlockCopySrcDataPerRead_GemmM,
            GemmABlockCopyDstDataPerWrite_GemmKPACK,
            GemmBBlockCopyThreadSliceLengths_GemmK_GemmN_GemmKPACK,
            GemmBBlockCopyThreadClusterLengths_GemmK_GemmN_GemmKPACK,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmKPACK>{};

    // these decide which GEMM will be called
    constexpr index_t GemmId = CK_PARAM_GEMM_ID;

    gridwise_conv_bwd_data.template Run<GemmId>(p_in_global, p_wei_global, p_out_global);

#else
    static_assert(false, "wrong! Only fp32, fp16 and bfp16 are supported.");
#endif
}
