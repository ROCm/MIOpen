#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "gridwise_convolution_implicit_gemm_v4_nchw_kc1x1_nkhw_lds_double_buffer.hpp"
#include "float_types.h"
#include "implicitgemm_params.hpp"

extern "C" __global__
    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_implicit_gemm_v4_nchw_kc1x1_nkhw_lds_double_buffer(
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
    constexpr index_t CPerBlock = CK_PARAM_TUNABLE_E_PER_BLOCK;

    // read params: dependent params
    constexpr index_t GridSize = CK_PARAM_DEPENDENT_GRID_SIZE;

    constexpr auto in_nchw_desc  = make_ConstantTensorDescriptor_packed(Sequence<N, C, Hi, Wi>{});
    constexpr auto out_nkhw_desc = make_ConstantTensorDescriptor_packed(Sequence<N, K, Ho, Wo>{});

    constexpr auto wei_ck_desc = make_ConstantTensorDescriptor(Sequence<C, K>{}, Sequence<1, C>{});

    using ConvStrides = Sequence<ConvStrideH, ConvStrideW>;

    constexpr index_t GemmMPerThreadSubC = CK_PARAM_GEMM_M_PER_THREAD_SUB_C;
    constexpr index_t GemmNPerThreadSubC = CK_PARAM_GEMM_N_PER_THREAD_SUB_C;
    constexpr index_t GemmMLevel0Cluster = CK_PARAM_GEMM_M_LEVEL0_CLUSTER;
    constexpr index_t GemmNLevel0Cluster = CK_PARAM_GEMM_N_LEVEL0_CLUSTER;
    constexpr index_t GemmMLevel1Cluster = CK_PARAM_GEMM_M_LEVEL1_CLUSTER;
    constexpr index_t GemmNLevel1Cluster = CK_PARAM_GEMM_N_LEVEL1_CLUSTER;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = GemmMPerThreadSubC;
    constexpr index_t GemmDataPerReadB   = GemmNPerThreadSubC;

    constexpr index_t GemmNRepeat = 2;
    constexpr index_t N1          = GemmNRepeat;
    constexpr index_t N2          = GemmNPerThreadSubC;

    constexpr index_t InBlockCopyClusterLengths_E  = CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_E;
    constexpr index_t InBlockCopyClusterLengths_B  = CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_B;
    constexpr index_t InBlockCopyClusterLengths_N1 = CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_N1;
    constexpr index_t InBlockCopyClusterLengths_N2 = CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_N2;

    constexpr index_t InBlockCopySubLengths_E  = CPerBlock / InBlockCopyClusterLengths_E;
    constexpr index_t InBlockCopySubLengths_B  = BPerBlock / InBlockCopyClusterLengths_B;
    constexpr index_t InBlockCopySubLengths_N1 = N1 / InBlockCopyClusterLengths_N1;
    constexpr index_t InBlockCopySubLengths_N2 = N2 / InBlockCopyClusterLengths_N2;

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

    constexpr index_t WeiBlockCopyClusterLengths_E = CK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_E;
    constexpr index_t WeiBlockCopyClusterLengths_K = CK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_K;
    constexpr index_t WeiBlockCopySubLengths_E     = CPerBlock / WeiBlockCopyClusterLengths_E;
    constexpr index_t WeiBlockCopySubLengths_K     = KPerBlock / WeiBlockCopyClusterLengths_K;

    using WeiBlockCopySubLengths_E_K = Sequence<WeiBlockCopySubLengths_E, WeiBlockCopySubLengths_K>;
    using WeiBlockCopyClusterLengths_E_K =
        Sequence<WeiBlockCopyClusterLengths_E, WeiBlockCopyClusterLengths_K>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = CK_PARAM_WEI_BLOCK_COPY_SRC_DATE_PER_READ_E;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = CK_PARAM_WEI_BLOCK_COPY_DST_DATE_PER_WRITE_K;

    constexpr auto gridwise_conv =
        GridwiseConvolutionImplicitGemm_v4_nchw_kc1x1_nkhw_lds_double_buffer<
            GridSize,
            BlockSize,
            FLOAT,
            FLOAT_ACCUM,
            decltype(in_nchw_desc),
            decltype(wei_ck_desc),
            decltype(out_nkhw_desc),
            ConvStrides,
            static_cast<ImplicitGemmDirection>(CK_PARAM_PROBLEM_DIRECTION),
            BPerBlock,
            KPerBlock,
            CPerBlock,
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
