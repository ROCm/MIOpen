#include "common_header.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
#include "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_nchw_kcyx_nkhw_lds_double_buffer.hpp"
#include "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_fp16_bfp16_nchw_kcyx_nkhw_lds_double_buffer.hpp"
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
    // calculate dependent params amd heuristic params
    constexpr auto in_nchw_desc  = make_native_tensor_descriptor_packed(Sequence<N, C, Hi, Wi>{});
    constexpr auto wei_kcyx_desc = make_native_tensor_descriptor_packed(Sequence<K, C, Y, X>{});
    constexpr auto out_nkhw_desc = make_native_tensor_descriptor_packed(Sequence<N, K, Ho, Wo>{});

    constexpr auto dir  = ImplicitGemmDirection::ForwardData;
    using ConvStrides   = Sequence<ConvStrideH, ConvStrideW>;
    using ConvDilations = Sequence<ConvDilationH, ConvDilationW>;
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
    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN =
        Sequence<GemmBBlockCopyThreadSliceLengths_GemmK, GemmBBlockCopyThreadSliceLengths_GemmN>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN =
        Sequence<GemmBBlockCopyClusterLengths_GemmK, GemmBBlockCopyClusterLengths_GemmN>;

    using GemmBBlockCopyThreadClusterArrangeOrder = Sequence<0, 1>; // [E, B]
    using GemmBBlockCopySrcAccessOrder            = Sequence<0, 1>; // [E, B]
    using GemmBBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, B]

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM =
        Sequence<GemmABlockCopyThreadSliceLengths_GemmK, GemmABlockCopyThreadSliceLengths_GemmM>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM =
        Sequence<GemmABlockCopyClusterLengths_GemmK, GemmABlockCopyClusterLengths_GemmM>;

    using GemmABlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using GemmABlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using GemmABlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N;
    constexpr index_t GemmABlockCopySrcDataPerRead_GemmM =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M;

#elif MIOPEN_USE_FP16 || MIOPEN_USE_BFP16
    constexpr index_t GemmKPack = CK_PARAM_GEMM_KPACK_LENGTH;

    using GemmBBlockCopySubLengths_GemmK_GemmN_GemmKPack =
        Sequence<GemmBBlockCopyThreadSliceLengths_GemmK,
                 GemmBBlockCopyThreadSliceLengths_GemmN,
                 GemmKPack>;
    using GemmBBlockCopyClusterLengths_GemmK_GemmN_GemmKPack =
        Sequence<GemmBBlockCopyClusterLengths_GemmK, GemmBBlockCopyClusterLengths_GemmN, 1>;

    using GemmBBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 2>; // [E, B, GemmKPack]
    using GemmBBlockCopySrcAccessOrder            = Sequence<0, 1, 2>; // [E, B, GemmKPack]
    using GemmBBlockCopyDstAccessOrder            = Sequence<0, 1, 2>; // [E, B, GemmKPack]

    using GemmABlockCopySubLengths_GemmK_GemmM_GemmKPack =
        Sequence<GemmABlockCopyThreadSliceLengths_GemmK,
                 GemmABlockCopyThreadSliceLengths_GemmM,
                 GemmKPack>;
    using GemmABlockCopyClusterLengths_GemmK_GemmM_GemmKPack =
        Sequence<GemmABlockCopyClusterLengths_GemmK, GemmABlockCopyClusterLengths_GemmM, 1>;

    using GemmABlockCopyThreadClusterArrangeOrder = Sequence<1, 0, 2>; // [K, E, GemmKPack]
    using GemmABlockCopySrcAccessOrder            = Sequence<1, 0, 2>; // [K, E, GemmKPack]
    using GemmABlockCopyDstAccessOrder            = Sequence<0, 1, 2>; // [E, K, GemmKPack]

    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmKPack =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmKPack =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK;
#endif

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N;
    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_K;

    constexpr auto GemmMPerWave                  = CK_PARAM_GEMM_M_PER_WAVE;
    constexpr auto GemmNPerWave                  = CK_PARAM_GEMM_N_PER_WAVE;
    constexpr auto GemmMWaves                    = GemmMPerBlock / GemmMPerWave;
    constexpr auto GemmNWaves                    = GemmNPerBlock / GemmNPerWave;
    constexpr index_t GemmThreadGemmDataPerReadM = 1;
    constexpr index_t GemmThreadGemmDataPerReadN = 1;

    constexpr auto gridwise_conv =
#if MIOPEN_USE_FP32
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
            GemmMPerWave,
            GemmNPerWave,
            GemmMWaves,
            GemmNWaves,
            GemmThreadGemmDataPerReadM,
            GemmThreadGemmDataPerReadN,
            GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
            GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
            GemmABlockCopyThreadClusterArrangeOrder,
            GemmABlockCopySrcAccessOrder,
            GemmABlockCopyDstAccessOrder,
            GemmABlockCopySrcDataPerRead_GemmK,
            GemmABlockCopySrcDataPerRead_GemmM,
            GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
            GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
            GemmBBlockCopyThreadClusterArrangeOrder,
            GemmBBlockCopySrcAccessOrder,
            GemmBBlockCopyDstAccessOrder,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmN,
            dir>{};
#elif MIOPEN_USE_FP16 || MIOPEN_USE_BFP16
        GridwiseConvolutionImplicitGemm_v4r4_gen_xdlops_fp16_bfp16_nchw_kcyx_nkhw_lds_double_buffer<
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
            GemmNPerBlock,
            GemmMPerBlock,
            GemmKPerBlock,
            GemmKPack,
            GemmMPerWave,
            GemmNPerWave,
            GemmMWaves,
            GemmNWaves,
            GemmThreadGemmDataPerReadM,
            GemmThreadGemmDataPerReadN,
            GemmBBlockCopySubLengths_GemmK_GemmN_GemmKPack,
            GemmBBlockCopyClusterLengths_GemmK_GemmN_GemmKPack,
            GemmBBlockCopyThreadClusterArrangeOrder,
            GemmBBlockCopySrcAccessOrder,
            GemmBBlockCopyDstAccessOrder,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmKPack,
            GemmABlockCopySubLengths_GemmK_GemmM_GemmKPack,
            GemmABlockCopyClusterLengths_GemmK_GemmM_GemmKPack,
            GemmABlockCopyThreadClusterArrangeOrder,
            GemmABlockCopySrcAccessOrder,
            GemmABlockCopyDstAccessOrder,
            GemmABlockCopySrcDataPerRead_GemmK,
            GemmABlockCopyDstDataPerWrite_GemmKPack,
            dir>{};
#else
        static_assert(false, "wrong! Only fp32, fp16 and bfp16 are supported.");
#endif
    gridwise_conv.Run(p_in_global, p_wei_global, p_out_global);
}
