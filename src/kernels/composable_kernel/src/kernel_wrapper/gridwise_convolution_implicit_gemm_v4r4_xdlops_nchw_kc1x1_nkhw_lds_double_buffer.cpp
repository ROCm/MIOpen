#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kc1x1_nkhw_lds_double_buffer.hpp"
#include "float_types.h"
#include "implicitgemm_params.hpp"

extern "C" __global__
    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kc1x1_nkhw_lds_double_buffer(
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

    constexpr index_t ConvStrideH = CK_PARAM_PROBLEM_CONV_STRIDE_H;
    constexpr index_t ConvStrideW = CK_PARAM_PROBLEM_CONV_STRIDE_W;

    // read params: tunable params
    constexpr index_t BlockSize = CK_PARAM_TUNABLE_BLOCK_SIZE;

    constexpr index_t BPerBlock = CK_PARAM_TUNABLE_B_PER_BLOCK;
    constexpr index_t KPerBlock = CK_PARAM_TUNABLE_K_PER_BLOCK;
    constexpr index_t EPerBlock = CK_PARAM_TUNABLE_E_PER_BLOCK;

    // read params: dependent params
    constexpr index_t GridSize = CK_PARAM_DEPENDENT_GRID_SIZE;

    // calculate dependent params amd heuristic params
    constexpr auto in_nchw_desc = make_ConstantTensorDescriptor_packed(Sequence<N, C, Hi, Wi>{});
    constexpr auto wei_ck_desc  = make_ConstantTensorDescriptor(Sequence<C, K>{}, Sequence<1, C>{});
    constexpr auto out_nkhw_desc = make_ConstantTensorDescriptor_packed(Sequence<N, K, Ho, Wo>{});

    using ConvStrides = Sequence<ConvStrideH, ConvStrideW>;

    constexpr index_t InBlockCopyClusterLengths_E = CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_E;
    constexpr index_t InBlockCopyClusterLengths_B = CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_B;

    constexpr index_t InBlockCopySubLengths_E = EPerBlock / InBlockCopyClusterLengths_E;
    constexpr index_t InBlockCopySubLengths_B = BPerBlock / InBlockCopyClusterLengths_B;

    constexpr index_t WeiBlockCopyClusterLengths_E = CK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_E;
    constexpr index_t WeiBlockCopyClusterLengths_K = CK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_K;

    constexpr index_t WeiBlockCopySubLengths_E = EPerBlock / WeiBlockCopyClusterLengths_E;
    constexpr index_t WeiBlockCopySubLengths_K = KPerBlock / WeiBlockCopyClusterLengths_K;

    using InBlockCopySubLengths_E_B = Sequence<InBlockCopySubLengths_E, InBlockCopySubLengths_B>;

    using InBlockCopyClusterLengths_E_B =
        Sequence<InBlockCopyClusterLengths_E, InBlockCopyClusterLengths_B>;

    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1>; // [E, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 1>; // [E, B]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, B]

    constexpr index_t InBlockCopyDataPerAccess_B = CK_PARAM_IN_BLOCK_COPY_DATA_PER_ACCESS_B;

    using WeiBlockCopySubLengths_E_K = Sequence<WeiBlockCopySubLengths_E, WeiBlockCopySubLengths_K>;
    using WeiBlockCopyClusterLengths_E_K =
        Sequence<WeiBlockCopyClusterLengths_E, WeiBlockCopyClusterLengths_K>;

    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = CK_PARAM_WEI_BLOCK_COPY_SRC_DATA_PER_READ_E;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = CK_PARAM_WEI_BLOCK_COPY_DST_DATA_PER_WRITE_K;

    constexpr index_t OutThreadCopyDataPerAccess_B = CK_PARAM_OUT_THREAD_COPY_DATA_PER_ACCESS_B;

    constexpr auto GemmMPerWave     = CK_PARAM_GEMM_M_PER_WAVE;
    constexpr auto GemmNPerWave     = CK_PARAM_GEMM_N_PER_WAVE;
    constexpr auto GemmMWaves       = KPerBlock / GemmMPerWave;
    constexpr auto GemmNWaves       = BPerBlock / GemmNPerWave;
    constexpr auto GemmDataPerReadA = 1;
    constexpr auto GemmDataPerReadB = 1;
    constexpr auto EnableXdlops     = CK_ENABLE_XDLOPS == 1;

    constexpr auto gridwise_conv =
        GridwiseConvolutionImplicitGemm_v4r4_xdlops_nchw_kc1x1_nkhw_lds_double_buffer<
            GridSize,
            BlockSize,
            FLOAT,
            decltype(in_nchw_desc),
            decltype(wei_ck_desc),
            decltype(out_nkhw_desc),
            ConvStrides,
            static_cast<ImplicitGemmDirection>(CK_PARAM_PROBLEM_DIRECTION),
            BPerBlock,
            KPerBlock,
            EPerBlock,
            GemmMPerWave,
            GemmNPerWave,
            GemmMWaves,
            GemmNWaves,
            GemmDataPerReadA,
            GemmDataPerReadB,
            EnableXdlops,
            InBlockCopySubLengths_E_B,
            InBlockCopyClusterLengths_E_B,
            InBlockCopyThreadClusterArrangeOrder,
            InBlockCopySrcAccessOrder,
            InBlockCopyDstAccessOrder,
            InBlockCopyDataPerAccess_B,
            WeiBlockCopySubLengths_E_K,
            WeiBlockCopyClusterLengths_E_K,
            WeiBlockCopyThreadClusterArrangeOrder,
            WeiBlockCopySrcAccessOrder,
            WeiBlockCopyDstAccessOrder,
            WeiBlockCopySrcDataPerRead_E,
            WeiBlockCopyDstDataPerWrite_K,
            OutThreadCopyDataPerAccess_B>{};

    gridwise_conv.Run(p_in_global, p_wei_global, p_out_global);
}
