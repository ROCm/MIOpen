#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_convolution_forward_implicit_gemm_v5r1_dlops_nchw_kcyx_nkhw.hpp"
#include "driver_convolution_forward_implicit_gemm_v5r1_dlops_nchw_kcyx_nkhw_outpad.hpp"

template <typename TInWei,
          ck::index_t InWeiVectorSize,
          typename TAcc,
          typename TOut,
          typename InLengths,
          typename WeiLengths,
          typename OutLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_convolution_forward_implicit_gemm_v5r1_dlops_nchw_kcyx_nkhw(
    const InLengths& in_n_c_hi_wi_lengths,
    const WeiLengths& wei_k_c_y_x_lengths,
    const OutLengths& out_n_k_ho_wo_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TInWei>& in_n_c_hi_wi,
    const Tensor<TInWei>& wei_k_c_y_x,
    Tensor<TOut>& out_n_k_ho_wo,
    ck::index_t /* nrepeat */)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto N = out_n_k_ho_wo_lengths[I0];
    const auto K = out_n_k_ho_wo_lengths[I1];
    const auto C = wei_k_c_y_x_lengths[I1];

    const auto Hi = in_n_c_hi_wi_lengths[I2];
    const auto Wi = in_n_c_hi_wi_lengths[I3];

    const auto Ho = out_n_k_ho_wo_lengths[I2];
    const auto Wo = out_n_k_ho_wo_lengths[I3];

    const auto Y = wei_k_c_y_x_lengths[I2];
    const auto X = wei_k_c_y_x_lengths[I3];

    const auto C0 = C / Number<InWeiVectorSize>{};
    const auto C1 = Number<InWeiVectorSize>{};

    const auto K0 = K / Number<InWeiVectorSize>{};
    const auto K1 = Number<InWeiVectorSize>{};

    Tensor<TInWei> in_n_c0_hi_wi_c1(
        HostTensorDescriptor(std::initializer_list<index_t>{N, C0, Hi, Wi, C1}));
    Tensor<TInWei> wei_k_c0_y_x_c1(
        HostTensorDescriptor(std::initializer_list<index_t>{K, C0, Y, X, C1}));
    Tensor<TOut> out_n_k0_ho_wo_k1(
        HostTensorDescriptor(std::initializer_list<index_t>{N, K0, Ho, Wo, K1}));

    auto f_nchw2nc0hwc1 = [&](auto n, auto hi, auto wi, auto c) {
        in_n_c0_hi_wi_c1(n, c / InWeiVectorSize, hi, wi, c % InWeiVectorSize) =
            in_n_c_hi_wi(n, c, hi, wi);
    };

    auto f_kcyx2kc0yxc1 = [&](auto k, auto y, auto x, auto c) {
        wei_k_c0_y_x_c1(k, c / InWeiVectorSize, y, x, c % InWeiVectorSize) =
            wei_k_c_y_x(k, c, y, x);
    };

    make_ParallelTensorFunctor(f_nchw2nc0hwc1, N, Hi, Wi, C)();
    make_ParallelTensorFunctor(f_kcyx2kc0yxc1, K, Y, X, C)();

    DeviceMem in_n_c0_hi_wi_c1_device_buf(sizeof(TInWei) *
                                          in_n_c0_hi_wi_c1.mDesc.GetElementSpace());
    DeviceMem wei_k_c0_y_x_c1_device_buf(sizeof(TInWei) * wei_k_c0_y_x_c1.mDesc.GetElementSpace());
    DeviceMem out_n_k0_ho_wo_k1_device_buf(sizeof(TOut) *
                                           out_n_k0_ho_wo_k1.mDesc.GetElementSpace());

    in_n_c0_hi_wi_c1_device_buf.ToDevice(in_n_c0_hi_wi_c1.mData.data());
    wei_k_c0_y_x_c1_device_buf.ToDevice(wei_k_c0_y_x_c1.mData.data());

    const auto in_n_c0_hi_wi_desc = make_naive_tensor_descriptor_packed(make_tuple(N, C0, Hi, Wi));
    const auto wei_k_c0_y_x_desc  = make_naive_tensor_descriptor_packed(make_tuple(K, C0, Y, X));
    const auto out_n_k0_ho_wo_k1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, K0, Ho, Wo, K1));

#if 1
    // cdata = 64, BlockSize = 64, 16x8x32x4
    constexpr index_t BlockSize = 64;

    constexpr index_t KPerBlock  = 16;
    constexpr index_t HoPerBlock = 8;
    constexpr index_t WoPerBlock = 32;
    constexpr index_t EPerBlock  = 1;

    constexpr index_t KPerThread  = KPerBlock;
    constexpr index_t HoPerThread = 2;
    constexpr index_t WoPerThread = 2;
    constexpr index_t EPerThread  = EPerBlock;

    using ABlockTransferThreadSliceLengths_E_K   = Sequence<3, 1>;
    using ABlockTransferThreadClusterLengths_E_K = Sequence<3 * EPerBlock, KPerBlock>;

    constexpr index_t ABlockTransferSrcScalarPerVector_E = 1;
    constexpr index_t ABlockTransferDstScalarPerVector_K = 1;

    constexpr index_t BThreadTransferSrcScalarPerVector_W = 1;

    constexpr index_t CThreadTransferDstScalarPerVector_W = 16;

    static_assert(KPerThread % CThreadTransferDstScalarPerVector_W == 0, "");
#else
    constexpr index_t BlockSize = 64;

    constexpr index_t KPerBlock  = 16;
    constexpr index_t HoPerBlock = 8;
    constexpr index_t WoPerBlock = 32;
    constexpr index_t EPerBlock  = 1;

    constexpr index_t KPerThread  = 16;
    constexpr index_t HoPerThread = 2;
    constexpr index_t WoPerThread = 2;
    constexpr index_t EPerThread  = EPerBlock;

    using ABlockTransferThreadSliceLengths_E_K   = Sequence<9, 1>;
    using ABlockTransferThreadClusterLengths_E_K = Sequence<EPerBlock, 16>;

    constexpr index_t ABlockTransferSrcScalarPerVector_E = 1;
    constexpr index_t ABlockTransferDstScalarPerVector_K = 1;

    constexpr index_t BThreadTransferSrcScalarPerVector_W = 1;

    constexpr index_t CThreadTransferDstScalarPerVector_W = K1;

    static_assert(KPerThread % CThreadTransferDstScalarPerVector_W == 0, "");
#endif

    constexpr auto conv_driver =
#if 0
        DriverDynamicConvolutionForwardImplicitGemmDlops_v5r1_nchw_kcyx_nkhw_pad
#else
        DriverDynamicConvolutionForwardImplicitGemmDlops_v5r1_nchw_kcyx_nkhw_outpad
#endif
        <BlockSize,
         typename vector_type<TInWei, InWeiVectorSize>::type,
         TAcc,
         TOut,
         KPerBlock,
         HoPerBlock,
         WoPerBlock,
         EPerBlock,
         KPerThread,
         HoPerThread,
         WoPerThread,
         EPerThread,
         ABlockTransferThreadSliceLengths_E_K,
         ABlockTransferThreadClusterLengths_E_K,
         ABlockTransferSrcScalarPerVector_E,
         ABlockTransferDstScalarPerVector_K,
         BThreadTransferSrcScalarPerVector_W,
         CThreadTransferDstScalarPerVector_W>{};

    conv_driver.Run(wei_k_c0_y_x_desc,
                    in_n_c0_hi_wi_desc,
                    out_n_k0_ho_wo_k1_desc,
                    conv_strides,
                    conv_dilations,
                    in_left_pads,
                    in_right_pads,
                    static_cast<typename vector_type<TInWei, InWeiVectorSize>::type*>(
                        wei_k_c0_y_x_c1_device_buf.GetDeviceBuffer()),
                    static_cast<typename vector_type<TInWei, InWeiVectorSize>::type*>(
                        in_n_c0_hi_wi_c1_device_buf.GetDeviceBuffer()),
                    static_cast<TOut*>(out_n_k0_ho_wo_k1_device_buf.GetDeviceBuffer()));

    out_n_k0_ho_wo_k1_device_buf.FromDevice(out_n_k0_ho_wo_k1.mData.data());

    auto f_nk0hwk1_to_nkhw = [&](auto n, auto k, auto ho, auto wo) {
        out_n_k_ho_wo(n, k, ho, wo) =
            out_n_k0_ho_wo_k1(n, k / InWeiVectorSize, ho, wo, k % InWeiVectorSize);
    };

    make_ParallelTensorFunctor(f_nk0hwk1_to_nkhw, N, K, Ho, Wo)();
}
