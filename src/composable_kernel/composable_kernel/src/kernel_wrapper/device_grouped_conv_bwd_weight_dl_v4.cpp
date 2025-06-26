#include "device_grouped_conv_bwd_weight_dl_v4.hpp"


using namespace ck;
using F16  = ck::half_t;
using F32  = float;
using F64  = double;
using BF16 = ushort;

using InDataType  = F16;
using WeiDataType = F16;
using OutDataType = F16;
using AccDataType = F32;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// using ALayout = ck::tensor_layout::convolution::GNHWC;
// using BLayout = ck::tensor_layout::convolution::GKYXC;
// using ELayout = ck::tensor_layout::convolution::GNHWK

constexpr index_t BlockSize = CK_PARAM_BLOCKSIZE;
// constexpr index_t InDataType = CK_PARAM_INDATATYPE;
// constexpr index_t WeiDataType = CK_PARAM_WEIDATATYPE;
// constexpr index_t OutDataType = CK_PARAM_OUTDATATYPE;
// constexpr index_t AccDataType = CK_PARAM_ACCDATATYPE;
constexpr index_t TileW = CK_PARAM_TILE_W;
constexpr index_t TileH = CK_PARAM_TILE_H;
constexpr index_t FilterSize = CK_PARAM_FILTERSIZE;
constexpr index_t ConvDilationW = CK_PARAM_PROBLEM_CONV_DILATION_W;
constexpr index_t ConvDilationH = CK_PARAM_PROBLEM_CONV_DILATION_H;
constexpr index_t ConvStrideW = CK_PARAM_PROBLEM_CONV_STRIDE_W;
constexpr index_t ConvStrideH = CK_PARAM_PROBLEM_CONV_STRIDE_H;
constexpr index_t ConvPadW = CK_PARAM_PROBLEM_CONV_PAD_W;
constexpr index_t ConvPadH = CK_PARAM_PROBLEM_CONV_PAD_H;
constexpr index_t NBatch = CK_PARAM_NBATCH;
constexpr index_t NumWavePerTile = CK_PARAM_NUMWAVEPERTILE;
constexpr index_t InScalarPerVector = CK_PARAM_INSCALARPERVECTOR;
constexpr index_t OutScalarPerVector = CK_PARAM_OUTSCALARPERVECTOR;
constexpr index_t DstScalarPerVector = CK_PARAM_DSTSCALARPERVECTOR;
constexpr index_t RequirePadding = CK_PARAM_REQUIREPADDING;
constexpr index_t WSplit = CK_PARAM_WSPLIT;

extern "C" __global__
    __launch_bounds__(CK_PARAM_BLOCKSIZE, 1) void kernel_grouped_conv_bwd_weight_dl_v4_run(
                 const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 AccDataType* p_acc_grid,
                 const Array<index_t, 2 + 3> in_g_n_c_wis_lengths, // input
                 const Array<index_t, 2 + 3> in_g_n_c_wis_strides,
                 const Array<index_t, 2 + 3> wei_g_k_c_xs_lengths, // weight
                 const Array<index_t, 2 + 3> wei_g_k_c_xs_strides,
                 const Array<index_t, 2 + 3> out_g_n_k_wos_lengths, // output
                 const Array<index_t, 2 + 3> out_g_n_k_wos_strides,
                 const index_t split_k)
{
  constexpr index_t NDimSpatial = 2;

  using conv2 =
    ck::tensor_operation::device::GridwiseGroupedConv2DBwdWeightDlV4<BlockSize,
                                                                  InDataType,
                                                                  WeiDataType,
                                                                  OutDataType,
                                                                  AccDataType,
                                                                  S<TileH, TileW>,
                                                                  FilterSize,
                                                                //   ck::Tuple<S<1,1>, S<1,1>, S<2,2>>,
                                                                  S<ConvDilationH, ConvDilationW>,  // FilterDilation
                                                                  S<ConvStrideH, ConvStrideW>,  // FilterStride
                                                                  S<ConvPadH, ConvPadW>,  // FilterPadding
                                                                //   InElementOp,
                                                                //   WeiElementOp,
                                                                //   OutElementOp,
                                                                  NBatch,  // N batch
                                                                  NumWavePerTile,  // NumWavePerTile
                                                                  InScalarPerVector,  // InScalarPerVector
                                                                  OutScalarPerVector,  // OutScalarPerVector
                                                                  DstScalarPerVector,  // DstScalarPerVector
                                                                  RequirePadding,
                                                                  WSplit>;

  typename conv2::Argument arg (p_in_grid,
                                p_wei_grid,
                                p_out_grid,
                                p_acc_grid,
                                in_g_n_c_wis_lengths, // input
                                in_g_n_c_wis_strides,
                                wei_g_k_c_xs_lengths, // weight
                                wei_g_k_c_xs_strides,
                                out_g_n_k_wos_lengths, // output
                                out_g_n_k_wos_strides,
                                split_k);

  conv2::Run(arg);
}


extern "C" __global__ void kernel_grouped_conv_bwd_weight_elementwise_run(
    WeiDataType* p_wei_grid,
    const AccDataType* p_acc_grid,
    Array<index_t, 2 + 3> wei_g_k_c_xs_strides,
    Array<index_t, 2 + 3> acc_g_k_c_xs_strides)
{
    const index_t g_idx       = __builtin_amdgcn_readfirstlane(blockIdx.x);
    if (threadIdx.x < FilterSize * FilterSize)
    {
        const index_t x = threadIdx.x % FilterSize;
        const index_t y = threadIdx.x / FilterSize;
        auto* p_wei = p_wei_grid + g_idx * wei_g_k_c_xs_strides[0] + y * wei_g_k_c_xs_strides[3] + x * wei_g_k_c_xs_strides[4];
        auto* p_acc = p_acc_grid + g_idx * acc_g_k_c_xs_strides[0] + y * acc_g_k_c_xs_strides[3] + x * acc_g_k_c_xs_strides[4];
        *p_wei = static_cast<WeiDataType>(*p_acc);
    }
}
