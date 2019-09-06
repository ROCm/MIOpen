#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "gridwise_convolution_implicit_gemm_v4_nchw_kc1x1_nkhw_lds_double_buffer.hpp"
#include "float_types.h"

extern "C" __global__ void gridwise_convolution_implicit_gemm_v4_nchw_kc1x1_nkhw_lds_double_buffer(
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

    constexpr index_t Direction = CK_PARAM_PROBLEM_DIRECTION;

    constexpr index_t ConvStrideH = CK_PARAM_PROBLEM_CONV_STRIDE_H;
    constexpr index_t ConvStrideW = CK_PARAM_PROBLEM_CONV_STRIDE_W;

    // read params: tunable params
    constexpr index_t BlockSize = CK_PARAM_TUNABLE_BLOCK_SIZE;

    constexpr index_t BPerBlock = CK_PARAM_TUNABLE_B_PER_BLOCK;
    constexpr index_t KPerBlock = CK_PARAM_TUNABLE_K_PER_BLOCK;
    constexpr index_t CPerBlock = CK_PARAM_TUNABLE_C_PER_BLOCK;

    // read params: dependent params
    constexpr index_t GridSize = CK_PARAM_DEPENDENT_GRID_SIZE;

    // calculate dependent params amd heuristic params
    constexpr auto in_nchw_desc  = make_ConstantTensorDescriptor_packed(Sequence<N, C, Hi, Wi>{});
    constexpr auto out_nkhw_desc = make_ConstantTensorDescriptor_packed(Sequence<N, K, Ho, Wo>{});

    constexpr auto wei_ck_desc_back =
        make_ConstantTensorDescriptor(Sequence<C, K>{}, Sequence<K, 1>{});
    constexpr auto wei_ck_desc_forw =
        make_ConstantTensorDescriptor(Sequence<C, K>{}, Sequence<1, C>{});
    constexpr auto wei_ck_desc = typename std::conditional<Direction,
                                                           decltype(wei_ck_desc_forw),
                                                           decltype(wei_ck_desc_back)>::type{};

    using ConvStrides = Sequence<ConvStrideH, ConvStrideW>;

    constexpr index_t N1 = 2;
    constexpr index_t N2 = 4;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 1, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<8, 2, 16, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<1, 4>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<8, 32>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 1;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 4;

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
            Direction,
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
