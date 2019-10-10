#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer.hpp"
#include "gridwise_convolution_implicit_gemm_v4_fp16_bfp16_nchw_kcyx_nkhw_lds_double_buffer.hpp"
#include "float_types.h"

extern "C" __global__
    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer(
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
        make_ConstantTensorDescriptor_packed(Sequence<N, C, Hi, Wi>{});
    constexpr auto tmp_wei_kcyx_desc = make_ConstantTensorDescriptor_packed(Sequence<K, C, Y, X>{});
    constexpr auto tmp_out_nkhw_desc =
        make_ConstantTensorDescriptor_packed(Sequence<N, K, Ho, Wo>{});
    constexpr auto in_nchw_desc = tmp_in_nchw_desc.ReorderGivenNew2Old(Sequence<1, 0, 2, 3>{});
    // wei and out are swapped in the solver
    constexpr auto wei_kcyx_desc = tmp_out_nkhw_desc.ReorderGivenNew2Old(Sequence<1, 0, 2, 3>{});
    constexpr auto out_nkhw_desc = tmp_wei_kcyx_desc.ReorderGivenNew2Old(Sequence<1, 0, 2, 3>{});
    constexpr auto dir           = ImplicitGemmDirection::BackwardWeight;

    // swap stride and dilation
    using ConvDilations = Sequence<ConvStrideH, ConvStrideW>;
    using ConvStrides   = Sequence<ConvDilationH, ConvDilationW>;
#else
    constexpr auto in_nchw_desc  = make_ConstantTensorDescriptor_packed(Sequence<N, C, Hi, Wi>{});
    constexpr auto wei_kcyx_desc = make_ConstantTensorDescriptor_packed(Sequence<K, C, Y, X>{});
    constexpr auto out_nkhw_desc = make_ConstantTensorDescriptor_packed(Sequence<N, K, Ho, Wo>{});

    constexpr auto dir  = ImplicitGemmDirection::ForwardData;
    using ConvStrides   = Sequence<ConvStrideH, ConvStrideW>;
    using ConvDilations = Sequence<ConvDilationH, ConvDilationW>;
#endif // CK_PARAM_PROBLEM_DIRECTION == 2

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

    using InBlockCopySubLengths_E_N1_B_N2 = Sequence<InBlockCopySubLengths_E,
                                                     InBlockCopySubLengths_N1,
                                                     InBlockCopySubLengths_B,
                                                     InBlockCopySubLengths_N2>;
    using InBlockCopyClusterLengths_E_N1_B_N2 = Sequence<InBlockCopyClusterLengths_E,
                                                         InBlockCopyClusterLengths_N1,
                                                         InBlockCopyClusterLengths_B,
                                                         InBlockCopyClusterLengths_N2>;

    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = CK_PARAM_IN_BLOCK_COPY_SRC_DATA_PER_READ_B;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = CK_PARAM_IN_BLOCK_COPY_DST_DATA_PER_WRITE_N2;

    using WeiBlockCopySubLengths_E_K = Sequence<WeiBlockCopySubLengths_E, WeiBlockCopySubLengths_K>;
    using WeiBlockCopyClusterLengths_E_K =
        Sequence<WeiBlockCopyClusterLengths_E, WeiBlockCopyClusterLengths_K>;

    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = CK_PARAM_WEI_BLOCK_COPY_SRC_DATE_PER_READ_E;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = CK_PARAM_WEI_BLOCK_COPY_DST_DATE_PER_WRITE_K;

#elif MIOPEN_USE_FP16 || MIOPEN_USE_BFP16
    constexpr index_t GemmDataPerReadA          = 1;
    constexpr index_t GemmDataPerReadB          = 1;
    constexpr index_t EPACK                     = CK_PARAM_EPACK_LENGTH;
    using InBlockCopySubLengths_E_N1_B_N2_EPACK = Sequence<InBlockCopySubLengths_E,
                                                           InBlockCopySubLengths_N1,
                                                           InBlockCopySubLengths_B,
                                                           InBlockCopySubLengths_N2,
                                                           EPACK>;
    using InBlockCopyClusterLengths_E_N1_B_N2_EPACK = Sequence<InBlockCopyClusterLengths_E,
                                                               InBlockCopyClusterLengths_N1,
                                                               InBlockCopyClusterLengths_B,
                                                               InBlockCopyClusterLengths_N2,
                                                               1>;

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = CK_PARAM_IN_BLOCK_COPY_DST_DATA_PER_WRITE_N2;

    // EPACK  - E dimension is folded into 2 dimensions E and EPACK
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2, 4>; // [E, N1, N2, B, EPACK]
    using InBlockCopySrcAccessOrder            = Sequence<0, 1, 3, 2, 4>; // [E, N1, N2, B, EPACK]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3, 4>; // [E, N1, B, N2, EPACK]

    using WeiBlockCopySubLengths_E_K_EPACK =
        Sequence<WeiBlockCopySubLengths_E, WeiBlockCopySubLengths_K, EPACK>;
    using WeiBlockCopyClusterLengths_E_K_EPACK =
        Sequence<WeiBlockCopyClusterLengths_E, WeiBlockCopyClusterLengths_K, 1>;

    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0, 2>; // [K, E, EPACK]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0, 2>; // [K, E, EPACK]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1, 2>; // [E, K, EPACK]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = CK_PARAM_WEI_BLOCK_COPY_SRC_DATE_PER_READ_E;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#else
    static_assert(false, "wrong! Only kperblock could be 32/64/128 not supported");
#endif

#if MIOPEN_USE_FP32
    constexpr auto gridwise_conv =
        GridwiseConvolutionImplicitGemm_v4_nchw_kcyx_nkhw_lds_double_buffer<
            GridSize,
            BlockSize,
            FLOAT,
            FLOAT_ACCUM,
            decltype(in_nchw_desc),
            decltype(wei_kcyx_desc),
            decltype(out_nkhw_desc),
            ConvStrides,
            ConvDilations,
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
            InBlockCopySubLengths_E_N1_B_N2,
            InBlockCopyClusterLengths_E_N1_B_N2,
            InBlockCopyThreadClusterArrangeOrder,
            InBlockCopySrcAccessOrder,
            InBlockCopyDstAccessOrder,
            InBlockCopySrcDataPerRead_B,
            InBlockCopyDstDataPerWrite_N2,
            WeiBlockCopySubLengths_E_K,
            WeiBlockCopyClusterLengths_E_K,
            WeiBlockCopyThreadClusterArrangeOrder,
            WeiBlockCopySrcAccessOrder,
            WeiBlockCopyDstAccessOrder,
            WeiBlockCopySrcDataPerRead_E,
            WeiBlockCopyDstDataPerWrite_K,
            dir>{};
#elif MIOPEN_USE_FP16 || MIOPEN_USE_BFP16
    constexpr auto gridwise_conv =
        GridwiseConvolutionImplicitGemm_v4_fp16_bfp16_nchw_kcyx_nkhw_lds_double_buffer<
            GridSize,
            BlockSize,
            FLOAT,
            FLOAT_ACCUM,
            decltype(in_nchw_desc),
            decltype(wei_kcyx_desc),
            decltype(out_nkhw_desc),
            ConvStrides,
            ConvDilations,
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
            InBlockCopySubLengths_E_N1_B_N2_EPACK,
            InBlockCopyClusterLengths_E_N1_B_N2_EPACK,
            InBlockCopyThreadClusterArrangeOrder,
            InBlockCopySrcAccessOrder,
            InBlockCopyDstAccessOrder,
            InBlockCopySrcDataPerRead_B,
            InBlockCopyDstDataPerWrite_N2,
            WeiBlockCopySubLengths_E_K_EPACK,
            WeiBlockCopyClusterLengths_E_K_EPACK,
            WeiBlockCopyThreadClusterArrangeOrder,
            WeiBlockCopySrcAccessOrder,
            WeiBlockCopyDstAccessOrder,
            WeiBlockCopySrcDataPerRead_E,
            WeiBlockCopyDstDataPerWrite_K,
            dir>{};
#else
    static_assert(false, "wrong! Only fp32, fp16 and bfp16 are supported.");
#endif
    gridwise_conv.Run(p_in_global, p_wei_global, p_out_global);
}
