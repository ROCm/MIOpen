#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "debug.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "conv_common.hpp"
#include "host_conv_bwd_weight.hpp"
#include "device_tensor.hpp"
#include "device_convolution_backward_weight_implicit_gemm_v4r4r2_xdlops_nchw_kcyx_nkhw.hpp"
#include "device_convolution_backward_weight_implicit_gemm_v4r4r4_xdlops_nhwc_kyxc_nhwk.hpp"
#include "device_convolution_backward_weight_implicit_gemm_v4r4r2_xdlops_atomic_nchw_kcyx_nkhw.hpp"
#include "device_convolution_backward_weight_implicit_gemm_v4r4r4_xdlops_atomic_nhwc_kyxc_nhwk.hpp"
#include "device_convolution_backward_weight_implicit_gemm_v4r4r5_xdlops_atomic_nhwc_kyxc_nhwk.hpp"

#define USE_DYNAMIC_MODE 1
#define USE_CONV_WRW_V4R4R2_XDL_NCHW 0
#define USE_CONV_WRW_V4R4R4_XDL_NHWC 0
#define USE_CONV_WRW_V4R4R2_XDL_ATOMIC_NCHW 0
#define USE_CONV_WRW_V4R4R4_XDL_ATOMIC_NHWC 0
#define USE_CONV_WRW_V4R4R5_XDL_ATOMIC_NHWC 1

enum ConvBackwardWeightAlgo
{
    V4R4R2XDLNCHW,       // 0
    V4R4R4XDLNHWC,       // 1
    V4R4R2XDLATOMICNCHW, // 2
    V4R4R4XDLATOMICNHWC, // 3
    V4R4R5XDLATOMICNHWC, // 4
};

int main(int argc, char* argv[])
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};
    constexpr auto I6 = Number<6>{};

#if USE_DYNAMIC_MODE
    // dynamic mode
    if(argc != 23)
    {
        printf("arg1 to 6: layout, algo, do_verification, init_method, do_log, nrepeat\n");
        printf("rest: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, RightPx\n");
        printf("additional: desired_grid_size\n");
        exit(1);
    }

    const ConvTensorLayout layout     = static_cast<ConvTensorLayout>(std::stoi(argv[1]));
    const ConvBackwardWeightAlgo algo = static_cast<ConvBackwardWeightAlgo>(std::stoi(argv[2]));
    const bool do_verification        = std::stoi(argv[3]);
    const int init_method             = std::stoi(argv[4]);
    const bool do_log                 = std::stoi(argv[5]);
    const int nrepeat                 = std::stoi(argv[6]);

    const index_t N  = std::stoi(argv[7]);
    const index_t K  = std::stoi(argv[8]);
    const index_t C  = std::stoi(argv[9]);
    const index_t Y  = std::stoi(argv[10]);
    const index_t X  = std::stoi(argv[11]);
    const index_t Hi = std::stoi(argv[12]);
    const index_t Wi = std::stoi(argv[13]);

    const index_t conv_stride_h   = std::stoi(argv[14]);
    const index_t conv_stride_w   = std::stoi(argv[15]);
    const index_t conv_dilation_h = std::stoi(argv[16]);
    const index_t conv_dilation_w = std::stoi(argv[17]);
    const index_t in_left_pad_h   = std::stoi(argv[18]);
    const index_t in_left_pad_w   = std::stoi(argv[19]);
    const index_t in_right_pad_h  = std::stoi(argv[20]);
    const index_t in_right_pad_w  = std::stoi(argv[21]);

    const index_t desired_grid_size = std::stoi(argv[22]);

    const index_t YEff = (Y - 1) * conv_dilation_h + 1;
    const index_t XEff = (X - 1) * conv_dilation_w + 1;

    const index_t Ho = (Hi + in_left_pad_h + in_right_pad_h - YEff) / conv_stride_h + 1;
    const index_t Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;
#else
    // static mode
    if(argc < 7)
    {
        printf("arg1 to 6: layout, algo, do_verification, init_method, do_log, nrepeat\n");
        exit(1);
    }

    const ConvTensorLayout layout     = static_cast<ConvTensorLayout>(std::stoi(argv[1]));
    const ConvBackwardWeightAlgo algo = static_cast<ConvBackwardWeightAlgo>(std::stoi(argv[2]));
    const bool do_verification        = std::stoi(argv[3]);
    const int init_method             = std::stoi(argv[4]);
    const bool do_log                 = std::stoi(argv[5]);
    const int nrepeat                 = std::stoi(argv[6]);

    constexpr auto N  = Number<128>{};
    constexpr auto C  = Number<128>{};
    constexpr auto Hi = Number<14>{};
    constexpr auto Wi = Number<14>{};
    constexpr auto K  = Number<256>{};
    constexpr auto Y  = Number<3>{};
    constexpr auto X  = Number<3>{};

    constexpr auto conv_stride_h   = I1;
    constexpr auto conv_stride_w   = I1;
    constexpr auto conv_dilation_h = I1;
    constexpr auto conv_dilation_w = I1;
    constexpr auto in_left_pad_h   = I1;
    constexpr auto in_left_pad_w   = I1;
    constexpr auto in_right_pad_h  = I1;
    constexpr auto in_right_pad_w  = I1;

    constexpr auto YEff = (Y - I1) * conv_dilation_h + I1;
    constexpr auto XEff = (X - I1) * conv_dilation_w + I1;

    constexpr auto Ho = (Hi + in_left_pad_h + in_right_pad_h - YEff) / conv_stride_h + I1;
    constexpr auto Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + I1;
#endif

#if 0
    using in_data_t  = float;
    using wei_data_t = float;
    using acc_data_t = float;
    using out_data_t = float;
#elif 1
    using in_data_t   = half_t;
    using out_data_t  = half_t;
    using acc_data_t  = float;
    using wei_data_t  = float;
#elif 1
    using in_data_t  = int8_t;
    using out_data_t = int8_t;
    using acc_data_t = int32_t;
    using wei_data_t = int8_t;
#endif

    std::vector<std::size_t> in_lengths_host(4), wei_lengths_host(4), out_lengths_host(4);

    if(layout == ConvTensorLayout::NCHW)
    {
        in_lengths_host[0]  = static_cast<std::size_t>(N);
        in_lengths_host[1]  = static_cast<std::size_t>(C);
        in_lengths_host[2]  = static_cast<std::size_t>(Hi);
        in_lengths_host[3]  = static_cast<std::size_t>(Wi);
        wei_lengths_host[0] = static_cast<std::size_t>(K);
        wei_lengths_host[1] = static_cast<std::size_t>(C);
        wei_lengths_host[2] = static_cast<std::size_t>(Y);
        wei_lengths_host[3] = static_cast<std::size_t>(X);
        out_lengths_host[0] = static_cast<std::size_t>(N);
        out_lengths_host[1] = static_cast<std::size_t>(K);
        out_lengths_host[2] = static_cast<std::size_t>(Ho);
        out_lengths_host[3] = static_cast<std::size_t>(Wo);
    }
    else if(layout == ConvTensorLayout::NHWC)
    {
        in_lengths_host[0]  = static_cast<std::size_t>(N);
        in_lengths_host[1]  = static_cast<std::size_t>(Hi);
        in_lengths_host[2]  = static_cast<std::size_t>(Wi);
        in_lengths_host[3]  = static_cast<std::size_t>(C);
        wei_lengths_host[0] = static_cast<std::size_t>(K);
        wei_lengths_host[1] = static_cast<std::size_t>(Y);
        wei_lengths_host[2] = static_cast<std::size_t>(X);
        wei_lengths_host[3] = static_cast<std::size_t>(C);
        out_lengths_host[0] = static_cast<std::size_t>(N);
        out_lengths_host[1] = static_cast<std::size_t>(Ho);
        out_lengths_host[2] = static_cast<std::size_t>(Wo);
        out_lengths_host[3] = static_cast<std::size_t>(K);
    }
    else
    {
        std::runtime_error("wrong! not implemented");
    }

    Tensor<in_data_t> in(in_lengths_host);
    Tensor<wei_data_t> wei_device(wei_lengths_host);
    Tensor<wei_data_t> wei_host(wei_lengths_host);
    Tensor<out_data_t> out(out_lengths_host);

    std::cout << "layout: " << layout << std::endl;
    ostream_HostTensorDescriptor(in.mDesc, std::cout << "in: ");
    ostream_HostTensorDescriptor(wei_host.mDesc, std::cout << "wei: ");
    ostream_HostTensorDescriptor(out.mDesc, std::cout << "out: ");
    print_array("InLeftPads", make_tuple(in_left_pad_h, in_left_pad_w));
    print_array("InRightPads", make_tuple(in_right_pad_h, in_right_pad_w));
    print_array("ConvStrides", make_tuple(conv_stride_h, conv_stride_w));
    print_array("ConvDilations", make_tuple(conv_dilation_h, conv_dilation_w));

    std::size_t num_thread = std::thread::hardware_concurrency();

    switch(init_method)
    {
    case 0:
        // no initialization
        break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        out.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        break;
    case 2:
        in.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        out.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        break;
    case 3:
        in.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        out.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        break;
    case 4:
        in.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        out.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        break;
    case 5:
        in.GenerateTensorValue(GeneratorTensor_3<float>{-0.1, 0.1}, num_thread);
        out.GenerateTensorValue(GeneratorTensor_3<float>{-0.1, 0.1}, num_thread);
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_2{1, 5}, num_thread);

        auto gen_out = [](auto... is) {
            return GeneratorTensor_2{1, 5}(is...) * GeneratorTensor_Checkboard{}(is...);
        };
        out.GenerateTensorValue(gen_out, num_thread);
    }

    auto f_make_for_device_nchw = [&]() {
        const auto in_lengths_dev     = make_tuple(N, C, Hi, Wi);
        const auto wei_lengths_dev    = make_tuple(K, C, Y, X);
        const auto out_lengths_dev    = make_tuple(N, K, Ho, Wo);
        const auto conv_strides_dev   = make_tuple(conv_stride_h, conv_stride_w);
        const auto conv_dilations_dev = make_tuple(conv_dilation_h, conv_dilation_w);
        const auto in_left_pads_dev   = make_tuple(in_left_pad_h, in_left_pad_w);
        const auto in_right_pads_dev  = make_tuple(in_right_pad_h, in_right_pad_w);

        return make_tuple(in_lengths_dev,
                          wei_lengths_dev,
                          out_lengths_dev,
                          conv_strides_dev,
                          conv_dilations_dev,
                          in_left_pads_dev,
                          in_right_pads_dev);
    };

    auto f_make_for_device_nhwc = [&]() {
        const auto in_lengths_dev     = make_tuple(N, Hi, Wi, C);
        const auto wei_lengths_dev    = make_tuple(K, Y, X, C);
        const auto out_lengths_dev    = make_tuple(N, Ho, Wo, K);
        const auto conv_strides_dev   = make_tuple(conv_stride_h, conv_stride_w);
        const auto conv_dilations_dev = make_tuple(conv_dilation_h, conv_dilation_w);
        const auto in_left_pads_dev   = make_tuple(in_left_pad_h, in_left_pad_w);
        const auto in_right_pads_dev  = make_tuple(in_right_pad_h, in_right_pad_w);

        return make_tuple(in_lengths_dev,
                          wei_lengths_dev,
                          out_lengths_dev,
                          conv_strides_dev,
                          conv_dilations_dev,
                          in_left_pads_dev,
                          in_right_pads_dev);
    };

    // set zero to wei_device
    wei_device.GenerateTensorValue(GeneratorTensor_0{}, num_thread);
#if USE_CONV_WRW_V4R4R2_XDL_NCHW
    if(algo == ConvBackwardWeightAlgo::V4R4R2XDLNCHW)
    {
        if(layout != ConvTensorLayout::NCHW)
        {
            throw std::runtime_error("wrong! layout");
        }

        const auto tmp = f_make_for_device_nchw();

        device_convolution_backward_weight_implicit_gemm_v4r4r2_xdlops_nchw_kcyx_nkhw<in_data_t,
                                                                                      wei_data_t,
                                                                                      acc_data_t,
                                                                                      out_data_t>(
            tmp[I0],
            tmp[I1],
            tmp[I2],
            tmp[I3],
            tmp[I4],
            tmp[I5],
            tmp[I6],
            in,
            wei_device,
            out,
            nrepeat);
    }
#endif

#if USE_CONV_WRW_V4R4R4_XDL_NHWC
    if(algo == ConvBackwardWeightAlgo::V4R4R4XDLNHWC)
    {
        if(layout != ConvTensorLayout::NHWC)
        {
            throw std::runtime_error("wrong! layout");
        }

        const auto tmp = f_make_for_device_nhwc();

        device_convolution_backward_weight_implicit_gemm_v4r4r4_xdlops_nhwc_kyxc_nhwk<in_data_t,
                                                                                      wei_data_t,
                                                                                      acc_data_t,
                                                                                      out_data_t>(
            tmp[I0],
            tmp[I1],
            tmp[I2],
            tmp[I3],
            tmp[I4],
            tmp[I5],
            tmp[I6],
            in,
            wei_device,
            out,
            nrepeat);
    }
#endif

#if USE_CONV_WRW_V4R4R2_XDL_ATOMIC_NCHW
    if(algo == ConvBackwardWeightAlgo::V4R4R2XDLATOMICNCHW)
    {
        if(layout != ConvTensorLayout::NCHW)
        {
            throw std::runtime_error("wrong! layout");
        }

        const auto tmp = f_make_for_device_nchw();

        device_convolution_backward_weight_implicit_gemm_v4r4r2_xdlops_atomic_nchw_kcyx_nkhw<
            in_data_t,
            wei_data_t,
            acc_data_t,
            out_data_t>(tmp[I0],
                        tmp[I1],
                        tmp[I2],
                        tmp[I3],
                        tmp[I4],
                        tmp[I5],
                        tmp[I6],
                        in,
                        wei_device,
                        out,
                        desired_grid_size,
                        nrepeat);
    }
#endif

#if USE_CONV_WRW_V4R4R4_XDL_ATOMIC_NHWC
    if(algo == ConvBackwardWeightAlgo::V4R4R4XDLATOMICNHWC)
    {
        if(layout != ConvTensorLayout::NHWC)
        {
            throw std::runtime_error("wrong! layout");
        }

        const auto tmp = f_make_for_device_nhwc();

        device_convolution_backward_weight_implicit_gemm_v4r4r4_xdlops_atomic_nhwc_kyxc_nhwk<
            in_data_t,
            wei_data_t,
            acc_data_t,
            out_data_t>(tmp[I0],
                        tmp[I1],
                        tmp[I2],
                        tmp[I3],
                        tmp[I4],
                        tmp[I5],
                        tmp[I6],
                        in,
                        wei_device,
                        out,
                        desired_grid_size,
                        nrepeat);
    }
#endif

#if USE_CONV_WRW_V4R4R5_XDL_ATOMIC_NHWC
    if(algo == ConvBackwardWeightAlgo::V4R4R5XDLATOMICNHWC)
    {
        if(layout != ConvTensorLayout::NHWC)
        {
            throw std::runtime_error("wrong! layout");
        }

        const auto tmp = f_make_for_device_nhwc();

        device_convolution_backward_weight_implicit_gemm_v4r4r5_xdlops_atomic_nhwc_kyxc_nhwk<
            in_data_t,
            wei_data_t,
            acc_data_t,
            out_data_t>(tmp[I0],
                        tmp[I1],
                        tmp[I2],
                        tmp[I3],
                        tmp[I4],
                        tmp[I5],
                        tmp[I6],
                        in,
                        wei_device,
                        out,
                        desired_grid_size,
                        nrepeat);
    }
#endif

    if(do_verification)
    {
        host_direct_convolution_backward_weights(out,
                                                 in,
                                                 wei_host,
                                                 make_tuple(conv_stride_h, conv_stride_w),
                                                 make_tuple(conv_dilation_h, conv_dilation_w),
                                                 make_tuple(in_left_pad_h, in_left_pad_w),
                                                 make_tuple(in_right_pad_h, in_right_pad_w),
                                                 layout);

        check_error(wei_host, wei_device);

        if(do_log)
        {
            LogRangeAsType<float>(std::cout << "out: ", out.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "in : ", in.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "wei_device: ", wei_device.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "wei_host  : ", wei_host.mData, ",") << std::endl;
        }
    }
}
