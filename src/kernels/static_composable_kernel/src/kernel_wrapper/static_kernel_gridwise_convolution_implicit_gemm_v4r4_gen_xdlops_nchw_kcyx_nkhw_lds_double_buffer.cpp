#include "static_kernel_common_header.hpp"
#include "static_kernel_ConstantTensorDescriptor_deprecated.hpp"
#include "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_nchw_kcyx_nkhw_lds_double_buffer.hpp"
#include "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_fp16_bfp16_fwd_nchw_kcyx_nkhw_lds_double_buffer.hpp"
#include "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_fp16_bfp16_wrw_nchw_kcyx_nkhw_lds_double_buffer.hpp"
#include "float_types.h"

extern "C" __global__
    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_nchw_kcyx_nkhw_lds_double_buffer(
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

    // read params: tunable params
    constexpr index_t BlockSize = CK_PARAM_TUNABLE_BLOCK_SIZE;

    constexpr index_t GemmMPerBlock = CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK;
    constexpr index_t GemmNPerBlock = CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK;
    constexpr index_t GemmKPerBlock = CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK;
    constexpr index_t GemmKBlocks   = CK_PARAM_TUNABLE_GEMM_K_BLOCKS;

    // read params: dependent params
    constexpr index_t GridSize = CK_PARAM_DEPENDENT_GRID_SIZE;

    constexpr index_t LeftPadH = CK_PARAM_PROBLEM_LEFT_PAD_H;
    constexpr index_t LeftPadW = CK_PARAM_PROBLEM_LEFT_PAD_W;

    constexpr index_t RightPadH = CK_PARAM_PROBLEM_RIGHT_PAD_H;
    constexpr index_t RightPadW = CK_PARAM_PROBLEM_RIGHT_PAD_W;

    using LeftPads  = Sequence<LeftPadH, LeftPadW>;
    using RightPads = Sequence<RightPadH, RightPadW>;

// calculate dependent params amd heuristic params
#if CK_PARAM_PROBLEM_DIRECTION == 2
    // In the WrW direction the filter is the output, while the output image is the input being
    // convolved with the (original) input image. This requires that the tensordescriptors be
    // swapped
    // To reuse the fwd kernel for this operation we need to swap the n and c dimension of the
    // input descriptor, the n and k dimension of the output descriptor
    // This change is necessary so that reduction dimensions are consistent with the requirement
    // of the wrw convolution when used in a fwd context
    constexpr auto tmp_in_nchw_desc =
        make_native_tensor_descriptor_packed(Sequence<N, C, Hi, Wi>{});
    constexpr auto tmp_wei_kcyx_desc = make_native_tensor_descriptor_packed(Sequence<K, C, Y, X>{});
    constexpr auto tmp_out_nkhw_desc =
        make_native_tensor_descriptor_packed(Sequence<N, K, Ho, Wo>{});
    constexpr auto in_nchw_desc =
        reorder_tensor_descriptor_given_upper2lower(tmp_in_nchw_desc, Sequence<1, 0, 2, 3>{});
    // wei and out are swapped in the solver
    constexpr auto wei_kcyx_desc =
        reorder_tensor_descriptor_given_upper2lower(tmp_out_nkhw_desc, Sequence<1, 0, 2, 3>{});
    constexpr auto out_nkhw_desc =
        reorder_tensor_descriptor_given_upper2lower(tmp_wei_kcyx_desc, Sequence<1, 0, 2, 3>{});
    constexpr auto dir = ImplicitGemmDirection::BackwardWeight;

    // swap stride and dilation
    using ConvDilations = Sequence<ConvStrideH, ConvStrideW>;
    using ConvStrides   = Sequence<ConvDilationH, ConvDilationW>;
#else
    static_assert(GemmKBlocks == 1, "do not support GemmKBlocks > 1 for forward!");
    // calculate dependent params amd heuristic params
    constexpr auto in_nchw_desc  = make_native_tensor_descriptor_packed(Sequence<N, C, Hi, Wi>{});
    constexpr auto wei_kcyx_desc = make_native_tensor_descriptor_packed(Sequence<K, C, Y, X>{});
    constexpr auto out_nkhw_desc = make_native_tensor_descriptor_packed(Sequence<N, K, Ho, Wo>{});

    constexpr auto dir          = ImplicitGemmDirection::ForwardData;
    using ConvStrides           = Sequence<ConvStrideH, ConvStrideW>;
    using ConvDilations         = Sequence<ConvDilationH, ConvDilationW>;
#endif // CK_PARAM_PROBLEM_DIRECTION == 2

    constexpr index_t GemmBBlockCopyClusterLengths_GemmK =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K;
    constexpr index_t GemmBBlockCopyClusterLengths_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N;

    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmBBlockCopyClusterLengths_GemmK;
    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmN =
        GemmNPerBlock / GemmBBlockCopyClusterLengths_GemmN;

    constexpr index_t GemmABlockCopyClusterLengths_GemmK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K;
    constexpr index_t GemmABlockCopyClusterLengths_GemmM =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M;

    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmABlockCopyClusterLengths_GemmK;
    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmM =
        GemmMPerBlock / GemmABlockCopyClusterLengths_GemmM;

#if MIOPEN_USE_FP32
    using GemmABlockCopyThreadSliceLengths_GemmG_GemmK_GemmM =
        Sequence<1, GemmABlockCopyThreadSliceLengths_GemmK, GemmABlockCopyThreadSliceLengths_GemmM>;
    using GemmABlockCopyThreadClusterLengths_GemmG_GemmK_GemmM =
        Sequence<1, GemmABlockCopyClusterLengths_GemmK, GemmABlockCopyClusterLengths_GemmM>;

    using GemmABlockCopyThreadClusterArrangeOrder = Sequence<0, 2, 1>; // [E0, K, E1]
    using GemmABlockCopySrcAccessOrder            = Sequence<0, 2, 1>; // [E0, K, E1]
    using GemmABlockCopyDstAccessOrder            = Sequence<0, 1, 2>; // [E0, E1, K]

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmM =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M;

    using GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN =
        Sequence<1, GemmBBlockCopyThreadSliceLengths_GemmK, GemmBBlockCopyThreadSliceLengths_GemmN>;
    using GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN =
        Sequence<1, GemmBBlockCopyClusterLengths_GemmK, GemmBBlockCopyClusterLengths_GemmN>;

    using GemmBBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 2>; // [E0, E1, B]
    using GemmBBlockCopySrcAccessOrder            = Sequence<0, 1, 2>; // [E0, E1, B]
    using GemmBBlockCopyDstAccessOrder            = Sequence<0, 1, 2>; // [E0, E1, B]

    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_K;

#elif MIOPEN_USE_FP16 || MIOPEN_USE_BFP16
    constexpr index_t GemmKPACK = CK_PARAM_GEMM_KPACK_LENGTH;

    using GemmABlockCopySubLengths_GemmG_GemmK_GemmM_GemmKPACK =
        Sequence<1,
                 GemmABlockCopyThreadSliceLengths_GemmK,
                 GemmABlockCopyThreadSliceLengths_GemmM,
                 GemmKPACK>;
    using GemmABlockCopyClusterLengths_GemmG_GemmK_GemmM_GemmKPACK =
        Sequence<1, GemmABlockCopyClusterLengths_GemmK, GemmABlockCopyClusterLengths_GemmM, 1>;

    using GemmABlockCopyThreadClusterArrangeOrder = Sequence<0, 2, 1, 3>; // [G, M, K, GemmKPACK]
    using GemmABlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [G, M, K, GemmKPACK]
    using GemmABlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [G, K, M, GemmKPACK]

    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmKPACK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK;

    using GemmBBlockCopySubLengths_GemmG_GemmK_GemmN_GemmKPACK =
        Sequence<1,
                 GemmBBlockCopyThreadSliceLengths_GemmK,
                 GemmBBlockCopyThreadSliceLengths_GemmN,
                 GemmKPACK>;
    using GemmBBlockCopyClusterLengths_GemmG_GemmK_GemmN_GemmKPACK =
        Sequence<1, GemmBBlockCopyClusterLengths_GemmK, GemmBBlockCopyClusterLengths_GemmN, 1>;

    using GemmBBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [G, K, GemmKPACK, B]
    using GemmBBlockCopySrcAccessOrder            = Sequence<0, 1, 3, 2>; // [G, K, GemmKPACK, B]
    using GemmBBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [G, K, B, GemmKPACK]

    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmKPACK =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK;

#if CK_PARAM_PROBLEM_DIRECTION == 2
    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_K;
#else
    constexpr index_t GemmABlockCopySrcDataPerRead_GemmKPACK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_KPACK;
#endif // CK_PARAM_PROBLEM_DIRECTION

#endif // MIOPEN_USE_FP16 || MIOPEN_USE_BFP16

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N;

    constexpr auto GemmMPerWave                  = CK_PARAM_GEMM_M_PER_WAVE;
    constexpr auto GemmNPerWave                  = CK_PARAM_GEMM_N_PER_WAVE;
    constexpr index_t GemmThreadGemmDataPerReadM = 1;
    constexpr index_t GemmThreadGemmDataPerReadN = 1;

#if MIOPEN_USE_FP32
    constexpr auto gridwise_conv =
        GridwiseConvolutionImplicitGemm_v4r4_gen_xdlops_nchw_kcyx_nkhw_lds_double_buffer<
            GridSize,
            BlockSize,
            FLOAT,
            FLOAT_ACCUM,
            decltype(in_nchw_desc),
            decltype(wei_kcyx_desc),
            decltype(out_nkhw_desc),
            ConvStrides,
            ConvDilations,
            LeftPads,
            RightPads,
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmKBlocks,
            GemmMPerWave,
            GemmNPerWave,
            GemmThreadGemmDataPerReadM,
            GemmThreadGemmDataPerReadN,
            GemmABlockCopyThreadSliceLengths_GemmG_GemmK_GemmM,
            GemmABlockCopyThreadClusterLengths_GemmG_GemmK_GemmM,
            GemmABlockCopyThreadClusterArrangeOrder,
            GemmABlockCopySrcAccessOrder,
            GemmABlockCopyDstAccessOrder,
            GemmABlockCopySrcDataPerRead_GemmK,
            GemmABlockCopySrcDataPerRead_GemmM,
            GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN,
            GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN,
            GemmBBlockCopyThreadClusterArrangeOrder,
            GemmBBlockCopySrcAccessOrder,
            GemmBBlockCopyDstAccessOrder,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmN,
            dir>{};
    gridwise_conv.Run(p_in_global, p_wei_global, p_out_global);

#elif(MIOPEN_USE_FP16 || MIOPEN_USE_BFP16) && CK_PARAM_PROBLEM_DIRECTION == 2

    // Backward weight in fp16/bfp16 uses atomic add to do reduction along K dimension
    // It requires output blob to be of float as no atomic add exists for fp16/ushort
    constexpr auto gridwise_conv =
        GridwiseConvolutionImplicitGemm_v4r4_gen_xdlops_fp16_bfp16_wrw_nchw_kcyx_nkhw_lds_double_buffer<
            GridSize,
            BlockSize,
            FLOAT,       // Input data type = fp16 (fp16) or ushort (bfp16)
            FLOAT_ACCUM, // Acc data type = float (see float_types.h)
            float, // Output data type = float  (not fp16/ushort) as no atomic add ISA exists for
                   // fp16/ushort.
            decltype(in_nchw_desc),
            decltype(wei_kcyx_desc),
            decltype(out_nkhw_desc),
            ConvStrides,
            ConvDilations,
            LeftPads,
            RightPads,
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmKBlocks,
            GemmKPACK,
            GemmMPerWave,
            GemmNPerWave,
            GemmThreadGemmDataPerReadM,
            GemmThreadGemmDataPerReadN,
            GemmABlockCopySubLengths_GemmG_GemmK_GemmM_GemmKPACK,
            GemmABlockCopyClusterLengths_GemmG_GemmK_GemmM_GemmKPACK,
            GemmABlockCopyThreadClusterArrangeOrder,
            GemmABlockCopySrcAccessOrder,
            GemmABlockCopyDstAccessOrder,
            GemmABlockCopySrcDataPerRead_GemmK,
            GemmABlockCopyDstDataPerWrite_GemmKPACK,
            GemmBBlockCopySubLengths_GemmG_GemmK_GemmN_GemmKPACK,
            GemmBBlockCopyClusterLengths_GemmG_GemmK_GemmN_GemmKPACK,
            GemmBBlockCopyThreadClusterArrangeOrder,
            GemmBBlockCopySrcAccessOrder,
            GemmBBlockCopyDstAccessOrder,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmKPACK,
            dir>{};

    // Output blob is cast to float as no atomic add exists for fp16/ushort
    gridwise_conv.Run(p_in_global, p_wei_global, reinterpret_cast<FLOAT_ACCUM*>(p_out_global));
#elif(MIOPEN_USE_FP16 || MIOPEN_USE_BFP16) && CK_PARAM_PROBLEM_DIRECTION != 2
    // Forward data doesn't use any atomic add so output blob remains of the same type
    // as input blob

    constexpr auto wkgrp_schd_order =
#if MIOPEN_USE_FP16
        NBlock1MBlock0;
#else
        MBlock1NBlock0;
#endif // MIOPEN_USE_FP16

    constexpr auto gridwise_conv =
        GridwiseConvolutionImplicitGemm_v4r4_gen_xdlops_fp16_bfp16_fwd_nchw_kcyx_nkhw_lds_double_buffer<
            GridSize,
            BlockSize,
            FLOAT,       // Input data type = fp16 (fp16) or ushort (bfp16)
            FLOAT_ACCUM, // Acc data type = float (see float_types.h)
            FLOAT,       // Input data type = fp16 (fp16) or ushort (bfp16)
            decltype(in_nchw_desc),
            decltype(wei_kcyx_desc),
            decltype(out_nkhw_desc),
            ConvStrides,
            ConvDilations,
            LeftPads,
            RightPads,
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmKBlocks,
            GemmKPACK,
            GemmMPerWave,
            GemmNPerWave,
            GemmThreadGemmDataPerReadM,
            GemmThreadGemmDataPerReadN,
            GemmABlockCopySubLengths_GemmG_GemmK_GemmM_GemmKPACK,
            GemmABlockCopyClusterLengths_GemmG_GemmK_GemmM_GemmKPACK,
            GemmABlockCopyThreadClusterArrangeOrder,
            GemmABlockCopySrcAccessOrder,
            GemmABlockCopyDstAccessOrder,
            GemmABlockCopySrcDataPerRead_GemmKPACK,
            GemmABlockCopyDstDataPerWrite_GemmKPACK,
            GemmBBlockCopySubLengths_GemmG_GemmK_GemmN_GemmKPACK,
            GemmBBlockCopyClusterLengths_GemmG_GemmK_GemmN_GemmKPACK,
            GemmBBlockCopyThreadClusterArrangeOrder,
            GemmBBlockCopySrcAccessOrder,
            GemmBBlockCopyDstAccessOrder,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmKPACK,
            dir,
            wkgrp_schd_order>{};
    gridwise_conv.Run(p_in_global, p_wei_global, p_out_global);
#else
    static_assert(false, "wrong! Only fp32, fp16 and bfp16 are supported.");
#endif // MIOPEN_USE_FP32
}
