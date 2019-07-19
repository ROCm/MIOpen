#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer.hpp"

extern "C" __global__ void gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer(
    const float* const __restrict__ p_in_global,
    const float* const __restrict__ p_wei_global,
    float* const __restrict__ p_out_global)
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

    // read params: dependent params
    constexpr index_t GridSize = CK_PARAM_DEPENDENT_GRID_SIZE;

    // calculate dependent params amd heuristic params
    constexpr auto in_nchw_desc  = make_ConstantTensorDescriptor_packed(Sequence<N, C, Hi, Wi>{});
    constexpr auto wei_kcyx_desc = make_ConstantTensorDescriptor_packed(Sequence<K, C, Y, X>{});
    constexpr auto out_nkhw_desc = make_ConstantTensorDescriptor_packed(Sequence<N, K, Ho, Wo>{});

    using ConvStrides   = Sequence<ConvStrideH, ConvStrideW>;
    using ConvDilations = Sequence<ConvDilationH, ConvDilationW>;

    constexpr index_t N1 = 2;
    constexpr index_t N2 = 4;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

#if CK_PARAM_TUNABLE_K_PER_BLOCK == 32
    static_assert(BlockSize == 64, "wrong!");

    constexpr index_t EPerBlock = 4;

    constexpr index_t GemmMLevel0Cluster = 1;

    // all_of(X_Per_Block % (X_Sub_Length * X_Cluster_Length) == 0)
    // accumulate(X_Cluster_Lengths, multiply) == BlockSize
    using InBlockCopySubLengths_E_N1_B_N2     = Sequence<1, 2, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2 = Sequence<4, 1, 16, 1>;

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K     = Sequence<2, 1>;
    using WeiBlockCopyClusterLengths_E_K = Sequence<2, 32>;

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 2;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif CK_PARAM_TUNABLE_K_PER_BLOCK == 64
    static_assert(BlockSize == 128, "wrong!");

    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmMLevel0Cluster = 2;

    using InBlockCopySubLengths_E_N1_B_N2     = Sequence<1, 2, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2 = Sequence<8, 1, 16, 1>;

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K     = Sequence<4, 1>;
    using WeiBlockCopyClusterLengths_E_K = Sequence<2, 64>;

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif CK_PARAM_TUNABLE_K_PER_BLOCK == 128
    static_assert(BlockSize == 256, "wrong!");

    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmMLevel0Cluster = 4;

    using InBlockCopySubLengths_E_N1_B_N2     = Sequence<1, 1, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2 = Sequence<8, 2, 16, 1>;

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K     = Sequence<4, 1>;
    using WeiBlockCopyClusterLengths_E_K = Sequence<2, 128>;

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#else
    static_assert(false, "wrong! not supported");
#endif

    constexpr auto gridwise_conv =
        GridwiseConvolutionImplicitGemm_v4_nchw_kcyx_nkhw_lds_double_buffer<
            GridSize,
            BlockSize,
            float,
            decltype(in_nchw_desc),
            decltype(wei_kcyx_desc),
            decltype(out_nkhw_desc),
            ConvStrides,
            ConvDilations,
            BPerBlock,
            KPerBlock,
            EPerBlock,
            N1,
            N2,
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
            WeiBlockCopyDstDataPerWrite_K>{};

    gridwise_conv.Run(p_in_global, p_wei_global, p_out_global);
}
