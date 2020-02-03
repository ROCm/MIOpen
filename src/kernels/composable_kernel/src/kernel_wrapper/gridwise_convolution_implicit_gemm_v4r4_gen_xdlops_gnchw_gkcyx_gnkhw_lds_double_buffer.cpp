#include "common_header.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
#include "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_gnchw_gkcyx_gnkhw_lds_double_buffer.hpp"
#include "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_fp16_bfp16_gnchw_gkcyx_gnkhw_lds_double_buffer.hpp"
#include "float_types.h"

extern "C" __global__
    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_gnchw_gkcyx_gnkhw_lds_double_buffer(
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

    constexpr index_t BPerBlock = CK_PARAM_TUNABLE_B_PER_BLOCK;
    constexpr index_t KPerBlock = CK_PARAM_TUNABLE_K_PER_BLOCK;
    constexpr index_t EPerBlock = CK_PARAM_TUNABLE_E_PER_BLOCK;

    // read params: dependent params
    constexpr index_t GridSize = CK_PARAM_DEPENDENT_GRID_SIZE;

    constexpr index_t LeftPadH = CK_PARAM_PROBLEM_LEFT_PAD_H;
    constexpr index_t LeftPadW = CK_PARAM_PROBLEM_LEFT_PAD_W;

    constexpr index_t RightPadH = CK_PARAM_PROBLEM_RIGHT_PAD_H;
    constexpr index_t RightPadW = CK_PARAM_PROBLEM_RIGHT_PAD_W;

    using LeftPads  = Sequence<LeftPadH, LeftPadW>;
    using RightPads = Sequence<RightPadH, RightPadW>;

    constexpr index_t GroupCounts = CK_PARAM_PROBLEM_CONV_GROUP_COUNTS;

    constexpr auto CPerGroup = C / GroupCounts;
    constexpr auto KPerGroup = K / GroupCounts;

// calculate dependent params amd heuristic params
#if CK_PARAM_PROBLEM_DIRECTION == 2
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

    constexpr auto dir = ImplicitGemmDirection::BackwardWeight;

    // swap stride and dilation
    using ConvDilations = Sequence<ConvStrideH, ConvStrideW>;
    using ConvStrides   = Sequence<ConvDilationH, ConvDilationW>;
#else
    // calculate dependent params amd heuristic params
    constexpr auto in_gnchw_desc =
        make_native_tensor_descriptor(Sequence<GroupCounts, N, CPerGroup, Hi, Wi>{},
                                      Sequence<CPerGroup * Hi * Wi, C * Hi * Wi, Hi * Wi, Wi, 1>{});

    constexpr auto wei_gkcyx_desc =
        make_native_tensor_descriptor_packed(Sequence<GroupCounts, KPerGroup, CPerGroup, Y, X>{});

    constexpr auto out_gnkhw_desc =
        make_native_tensor_descriptor(Sequence<GroupCounts, N, KPerGroup, Ho, Wo>{},
                                      Sequence<KPerGroup * Ho * Wo, K * Ho * Wo, Ho * Wo, Wo, 1>{});

    constexpr auto dir  = ImplicitGemmDirection::ForwardData;
    using ConvStrides   = Sequence<ConvStrideH, ConvStrideW>;
    using ConvDilations = Sequence<ConvDilationH, ConvDilationW>;
#endif // CK_PARAM_PROBLEM_DIRECTION == 2

    constexpr index_t InBlockCopyClusterLengths_E = CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_E;
    constexpr index_t InBlockCopyClusterLengths_B = CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_B;

    constexpr index_t InBlockCopySubLengths_E = EPerBlock / InBlockCopyClusterLengths_E;
    constexpr index_t InBlockCopySubLengths_B = BPerBlock / InBlockCopyClusterLengths_B;

    constexpr index_t WeiBlockCopyClusterLengths_E = CK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_E;
    constexpr index_t WeiBlockCopyClusterLengths_K = CK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_K;

    constexpr index_t WeiBlockCopySubLengths_E = EPerBlock / WeiBlockCopyClusterLengths_E;
    constexpr index_t WeiBlockCopySubLengths_K = KPerBlock / WeiBlockCopyClusterLengths_K;

#if MIOPEN_USE_FP32
    using InBlockCopySubLengths_G_E_B =
        Sequence<1, InBlockCopySubLengths_E, InBlockCopySubLengths_B>;
    using InBlockCopyClusterLengths_G_E_B =
        Sequence<1, InBlockCopyClusterLengths_E, InBlockCopyClusterLengths_B>;

    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 2>; // [G, E, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 1, 2>; // [G, E, B]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2>; // [G, E, B]

    using WeiBlockCopySubLengths_G_E_K =
        Sequence<1, WeiBlockCopySubLengths_E, WeiBlockCopySubLengths_K>;
    using WeiBlockCopyClusterLengths_G_E_K =
        Sequence<1, WeiBlockCopyClusterLengths_E, WeiBlockCopyClusterLengths_K>;

    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<0, 2, 1>; // [G, K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<0, 2, 1>; // [G, K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1, 2>; // [G, E, K]

    constexpr index_t InBlockCopyDstDataPerWrite_B  = CK_PARAM_IN_BLOCK_COPY_DST_DATA_PER_WRITE_B;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = CK_PARAM_WEI_BLOCK_COPY_DST_DATA_PER_WRITE_K;
#elif MIOPEN_USE_FP16 || MIOPEN_USE_BFP16
    constexpr index_t EPack = CK_PARAM_EPACK_LENGTH;

    using InBlockCopySubLengths_G_E_B_EPACK =
        Sequence<1, InBlockCopySubLengths_E, InBlockCopySubLengths_B, EPack>;
    using InBlockCopyClusterLengths_G_E_B_EPACK =
        Sequence<1, InBlockCopyClusterLengths_E, InBlockCopyClusterLengths_B, 1>;

    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 2, 3>; // [G, E, B, EPack]
    using InBlockCopySrcAccessOrder            = Sequence<0, 1, 2, 3>; // [G, E, B, EPack]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [G, E, B, EPack]

    using WeiBlockCopySubLengths_G_E_K_EPACK =
        Sequence<1, WeiBlockCopySubLengths_E, WeiBlockCopySubLengths_K, EPack>;
    using WeiBlockCopyClusterLengths_G_E_K_EPACK =
        Sequence<1, WeiBlockCopyClusterLengths_E, WeiBlockCopyClusterLengths_K, 1>;

    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<0, 2, 1, 3>; // [G, K, E, EPack]
    using WeiBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [G, K, E, EPack]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [G, E, K, EPack]

    constexpr index_t InBlockCopyDstDataPerWrite_EPACK =
        CK_PARAM_IN_BLOCK_COPY_DST_DATA_PER_WRITE_EPACK;
    constexpr index_t WeiBlockCopyDstDataPerWrite_EPACK =
        CK_PARAM_WEI_BLOCK_COPY_DST_DATA_PER_WRITE_EPACK;
#endif

    constexpr index_t InBlockCopySrcDataPerRead_B  = CK_PARAM_IN_BLOCK_COPY_SRC_DATA_PER_READ_B;
    constexpr index_t WeiBlockCopySrcDataPerRead_E = CK_PARAM_WEI_BLOCK_COPY_SRC_DATA_PER_READ_E;
    constexpr index_t OutThreadCopyDataPerAccess_B = CK_PARAM_OUT_THREAD_COPY_DATA_PER_ACCESS_B;

    constexpr auto GemmMPerWave        = CK_PARAM_GEMM_M_PER_WAVE;
    constexpr auto GemmNPerWave        = CK_PARAM_GEMM_N_PER_WAVE;
    constexpr auto GemmMWaves          = KPerBlock / GemmMPerWave;
    constexpr auto GemmNWaves          = BPerBlock / GemmNPerWave;
    constexpr index_t GemmDataPerReadA = 1;
    constexpr index_t GemmDataPerReadB = 1;

    constexpr auto gridwise_conv =
#if MIOPEN_USE_FP32
        GridwiseConvolutionImplicitGemm_v4r4_gen_xdlops_gnchw_gkcyx_gnkhw_lds_double_buffer<
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
            BPerBlock,
            KPerBlock,
            EPerBlock,
            GemmMPerWave,
            GemmNPerWave,
            GemmMWaves,
            GemmNWaves,
            GemmDataPerReadA,
            GemmDataPerReadB,
            InBlockCopySubLengths_G_E_B,
            InBlockCopyClusterLengths_G_E_B,
            InBlockCopyThreadClusterArrangeOrder,
            InBlockCopySrcAccessOrder,
            InBlockCopyDstAccessOrder,
            InBlockCopySrcDataPerRead_B,
            InBlockCopyDstDataPerWrite_B,
            WeiBlockCopySubLengths_G_E_K,
            WeiBlockCopyClusterLengths_G_E_K,
            WeiBlockCopyThreadClusterArrangeOrder,
            WeiBlockCopySrcAccessOrder,
            WeiBlockCopyDstAccessOrder,
            WeiBlockCopySrcDataPerRead_E,
            WeiBlockCopyDstDataPerWrite_K,
            OutThreadCopyDataPerAccess_B,
            dir>{};
#elif MIOPEN_USE_FP16 || MIOPEN_USE_BFP16
        GridwiseConvolutionImplicitGemm_v4r4_gen_xdlops_fp16_bfp16_gnchw_gkcyx_gnkhw_lds_double_buffer<
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
            BPerBlock,
            KPerBlock,
            EPerBlock,
            EPack,
            GemmMPerWave,
            GemmNPerWave,
            GemmMWaves,
            GemmNWaves,
            GemmDataPerReadA,
            GemmDataPerReadB,
            InBlockCopySubLengths_G_E_B_EPACK,
            InBlockCopyClusterLengths_G_E_B_EPACK,
            InBlockCopyThreadClusterArrangeOrder,
            InBlockCopySrcAccessOrder,
            InBlockCopyDstAccessOrder,
            InBlockCopySrcDataPerRead_B,
            InBlockCopyDstDataPerWrite_EPACK,
            WeiBlockCopySubLengths_G_E_K_EPACK,
            WeiBlockCopyClusterLengths_G_E_K_EPACK,
            WeiBlockCopyThreadClusterArrangeOrder,
            WeiBlockCopySrcAccessOrder,
            WeiBlockCopyDstAccessOrder,
            WeiBlockCopySrcDataPerRead_E,
            WeiBlockCopyDstDataPerWrite_EPACK,
            OutThreadCopyDataPerAccess_B,
            dir>{};
#else
        static_assert(false, "wrong! Only fp32, fp16 and bfp16 are supported.");
#endif
    gridwise_conv.Run(p_in_global, p_wei_global, p_out_global);
}
