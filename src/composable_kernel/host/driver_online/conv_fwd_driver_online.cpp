#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "conv_common.hpp"
#include "host_conv.hpp"
#include "device_tensor.hpp"
#include "handle.hpp"
#include "hipCheck.hpp"
#include "online_device_dynamic_convolution_forward_implicit_gemm_v4r4_dlops_nchw_kcyx_nkhw.hpp"
#include "online_device_dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw.hpp"
#include "online_device_dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw.hpp"
#include "online_device_dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nhwc_kyxc_nhwk.hpp"

#define USE_CONV_FWD_V4R4_NCHW 1
#define USE_CONV_FWD_V6R1_NCHW 1
#define USE_CONV_FWD_V4R4_XDLOPS_NCHW 1
#define USE_CONV_FWD_V4R4_XDLOPS_NHWC 1

enum ConvForwardAlgo
{
    V4R4NCHW,    // 0
    V6R1NCHW,    // 1
    V4R4XDLNCHW, // 2
    V4R4XDLNHWC  // 3
};

int main(int argc, char* argv[])
{
    using namespace ck;
    using namespace ck_driver;
    using size_t = std::size_t;

    hipStream_t stream;
    olCompile::Handle* handle;

    MY_HIP_CHECK(hipStreamCreate(&stream));

    handle = new olCompile::Handle(stream);

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};
    constexpr auto I6 = Number<6>{};

    if(argc != 22)
    {
        printf("arg1 to 5: layout, algo, do_verification, init_method, do_log, nrepeat\n");
        printf("rest: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, RightPx\n");
        exit(1);
    }

    const ConvTensorLayout layout = static_cast<ConvTensorLayout>(atoi(argv[1]));
    const ConvForwardAlgo algo    = static_cast<ConvForwardAlgo>(atoi(argv[2]));
    const bool do_verification    = atoi(argv[3]);
    const int init_method         = atoi(argv[4]);
    const bool do_log             = atoi(argv[5]);
    const int nrepeat             = atoi(argv[6]);

    const index_t N  = atoi(argv[7]);
    const index_t K  = atoi(argv[8]);
    const index_t C  = atoi(argv[9]);
    const index_t Y  = atoi(argv[10]);
    const index_t X  = atoi(argv[11]);
    const index_t Hi = atoi(argv[12]);
    const index_t Wi = atoi(argv[13]);

    const index_t conv_stride_h   = atoi(argv[14]);
    const index_t conv_stride_w   = atoi(argv[15]);
    const index_t conv_dilation_h = atoi(argv[16]);
    const index_t conv_dilation_w = atoi(argv[17]);
    const index_t in_left_pad_h   = atoi(argv[18]);
    const index_t in_left_pad_w   = atoi(argv[19]);
    const index_t in_right_pad_h  = atoi(argv[20]);
    const index_t in_right_pad_w  = atoi(argv[21]);

    const index_t YEff = (Y - 1) * conv_dilation_h + 1;
    const index_t XEff = (X - 1) * conv_dilation_w + 1;

    const index_t Ho = (Hi + in_left_pad_h + in_right_pad_h - YEff) / conv_stride_h + 1;
    const index_t Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;

#if 1
    using in_data_t  = float;
    using acc_data_t = float;
    using out_data_t = float;
#elif 0
    using in_data_t  = half_t;
    using acc_data_t = float;
    using out_data_t = half_t;
#elif 1
    using in_data_t  = int8_t;
    using acc_data_t = int32_t;
    using out_data_t = int8_t;
#endif

    std::vector<std::size_t> in_lengths_host(4), wei_lengths_host(4), out_lengths_host(4);

    switch(layout)
    {
    case ConvTensorLayout::NCHW:
        // NCHW
        in_lengths_host[0] = static_cast<std::size_t>(N);
        in_lengths_host[1] = static_cast<std::size_t>(C);
        in_lengths_host[2] = static_cast<std::size_t>(Hi);
        in_lengths_host[3] = static_cast<std::size_t>(Wi);

        wei_lengths_host[0] = static_cast<std::size_t>(K);
        wei_lengths_host[1] = static_cast<std::size_t>(C);
        wei_lengths_host[2] = static_cast<std::size_t>(Y);
        wei_lengths_host[3] = static_cast<std::size_t>(X);

        out_lengths_host[0] = static_cast<std::size_t>(N);
        out_lengths_host[1] = static_cast<std::size_t>(K);
        out_lengths_host[2] = static_cast<std::size_t>(Ho);
        out_lengths_host[3] = static_cast<std::size_t>(Wo);
        break;
    case ConvTensorLayout::NHWC:
        // NHWC
        in_lengths_host[0] = static_cast<std::size_t>(N);
        in_lengths_host[1] = static_cast<std::size_t>(Hi);
        in_lengths_host[2] = static_cast<std::size_t>(Wi);
        in_lengths_host[3] = static_cast<std::size_t>(C);

        wei_lengths_host[0] = static_cast<std::size_t>(K);
        wei_lengths_host[1] = static_cast<std::size_t>(Y);
        wei_lengths_host[2] = static_cast<std::size_t>(X);
        wei_lengths_host[3] = static_cast<std::size_t>(C);

        out_lengths_host[0] = static_cast<std::size_t>(N);
        out_lengths_host[1] = static_cast<std::size_t>(Ho);
        out_lengths_host[2] = static_cast<std::size_t>(Wo);
        out_lengths_host[3] = static_cast<std::size_t>(K);
        break;
    default: throw std::runtime_error("wrong! not implemented");
    }

    Tensor<in_data_t> in(in_lengths_host);
    Tensor<in_data_t> wei(wei_lengths_host);
    Tensor<out_data_t> out_host(out_lengths_host);
    Tensor<out_data_t> out_device(out_lengths_host);

    std::cout << "layout: " << layout << std::endl;
    ostream_HostTensorDescriptor(in.mDesc, std::cout << "in: ");
    ostream_HostTensorDescriptor(wei.mDesc, std::cout << "wei: ");
    ostream_HostTensorDescriptor(out_host.mDesc, std::cout << "out: ");
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
        wei.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        break;
    case 2:
        in.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        wei.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        break;
    case 3:
        in.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        wei.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        break;
    case 4:
        in.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        wei.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        break;
    case 5:
        in.GenerateTensorValue(GeneratorTensor_3<float>{0.0, 1.0}, num_thread);
        wei.GenerateTensorValue(GeneratorTensor_3<float>{-0.5, 0.5}, num_thread);
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_2{1, 5}, num_thread);

        auto gen_wei = [](auto... is) {
            return GeneratorTensor_2{1, 5}(is...) * GeneratorTensor_Checkboard{}(is...);
        };
        wei.GenerateTensorValue(gen_wei, num_thread);
    }

    auto f_make_for_device_nchw = [&]() {
        const auto in_lengths_dev  = make_tuple(N, C, Hi, Wi);
        const auto wei_lengths_dev = make_tuple(K, C, Y, X);
        const auto out_lengths_dev = make_tuple(N, K, Ho, Wo);

        return make_tuple(in_lengths_dev, wei_lengths_dev, out_lengths_dev);
    };

    auto f_make_for_device_nhwc = [&]() {
        const auto in_lengths_dev  = make_tuple(N, Hi, Wi, C);
        const auto wei_lengths_dev = make_tuple(K, Y, X, C);
        const auto out_lengths_dev = make_tuple(N, Ho, Wo, K);

        return make_tuple(in_lengths_dev, wei_lengths_dev, out_lengths_dev);
    };

    const auto conv_strides   = make_tuple(conv_stride_h, conv_stride_w);
    const auto conv_dilations = make_tuple(conv_dilation_h, conv_dilation_w);
    const auto in_left_pads   = make_tuple(in_left_pad_h, in_left_pad_w);
    const auto in_right_pads  = make_tuple(in_right_pad_h, in_right_pad_w);

#if USE_CONV_FWD_V4R4_NCHW
    if(algo == ConvForwardAlgo::V4R4NCHW)
    {
        if(layout != ConvTensorLayout::NCHW)
        {
            throw std::runtime_error("wrong! layout");
        }

        const auto tmp = f_make_for_device_nchw();

        tunable_dyn_conv_fwd_v4r4_dlops_nchw_kcyx_nkhw* tunable =
            &default_tunable_dyn_conv_fwd_v4r4_dlops_nchw_kcyx_nkhw;

        online_device_dynamic_convolution_forward_implicit_gemm_v4r4_dlops_nchw_kcyx_nkhw<
            in_data_t,
            acc_data_t,
            out_data_t>(handle,
                        tmp[I0],
                        tmp[I1],
                        tmp[I2],
                        conv_strides,
                        conv_dilations,
                        in_left_pads,
                        in_right_pads,
                        in,
                        wei,
                        out_device,
                        tunable,
                        nrepeat);
    }
#endif

#if USE_CONV_FWD_V6R1_NCHW
    if(algo == ConvForwardAlgo::V6R1NCHW)
    {
        if(layout != ConvTensorLayout::NCHW)
        {
            throw std::runtime_error("wrong! layout");
        }

        const auto tmp = f_make_for_device_nchw();

#if 1
        const CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw compile_param = {
            get_datatype_enum_from_type<in_data_t>::value,
            get_datatype_enum_from_type<acc_data_t>::value,
            get_datatype_enum_from_type<out_data_t>::value,
            256,
            4,
            1,
            128,
            32,
            8,
            4,
            4,
            1,
            {8, 2},
            {8, 2},
            {4, 1, 1, 1, 1},
            {2, 1, 1, 128, 1},
            {4, 1, 1, 1, 1},
            {1, 1, 1, 1, 1},
            {1, 4, 1, 1, 1},
            {8, 1, 1, 32, 1},
            {1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1},
            4,
            true,
            true};
#elif 0
        const CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw compile_param = {
            get_datatype_enum_from_type<in_data_t>::value,
            get_datatype_enum_from_type<acc_data_t>::value,
            get_datatype_enum_from_type<out_data_t>::value,
            256,
            4,
            2,
            128,
            32,
            8,
            4,
            4,
            1,
            {8, 2},
            {8, 2},
            {4, 1, 1, 1, 2},
            {2, 1, 1, 128, 1},
            {4, 1, 1, 1, 1},
            {1, 1, 1, 1, 1},
            {1, 4, 1, 1, 2},
            {8, 1, 1, 32, 1},
            {1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1},
            4,
            true,
            true};
#elif 1
        const CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw compile_param = {
            get_datatype_enum_from_type<in_data_t>::value,
            get_datatype_enum_from_type<acc_data_t>::value,
            get_datatype_enum_from_type<out_data_t>::value,
            256,
            4,
            4,
            128,
            32,
            8,
            4,
            4,
            1,
            {8, 2},
            {8, 2},
            {4, 1, 1, 1, 4},
            {2, 1, 1, 128, 1},
            {4, 1, 1, 1, 1},
            {1, 1, 1, 1, 1},
            {1, 4, 1, 1, 4},
            {8, 1, 1, 32, 1},
            {1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1},
            4,
            true,
            true};
#endif

        online_device_dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw<
            in_data_t,
            acc_data_t,
            out_data_t>(handle,
                        tmp[I0],
                        tmp[I1],
                        tmp[I2],
                        conv_strides,
                        conv_dilations,
                        in_left_pads,
                        in_right_pads,
                        in,
                        wei,
                        out_device,
                        compile_param,
                        nrepeat);
    }
#endif

#if USE_CONV_FWD_V4R4_XDLOPS_NCHW
    if(algo == ConvForwardAlgo::V4R4XDLNCHW)
    {
        if(layout != ConvTensorLayout::NCHW)
        {
            throw std::runtime_error("wrong! layout");
        }

        const auto tmp = f_make_for_device_nchw();

        tunable_dyn_conv_fwd_v4r4_xdlops_nchw_kcyx_nkhw* tunable =
            &default_tunable_dyn_conv_fwd_v4r4_xdlops_nchw_kcyx_nkhw;

        online_device_dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw<
            in_data_t,
            acc_data_t,
            out_data_t>(handle,
                        tmp[I0],
                        tmp[I1],
                        tmp[I2],
                        conv_strides,
                        conv_dilations,
                        in_left_pads,
                        in_right_pads,
                        in,
                        wei,
                        out_device,
                        tunable,
                        nrepeat);
    }
#endif

#if USE_CONV_FWD_V4R4_XDLOPS_NHWC
    if(algo == ConvForwardAlgo::V4R4XDLNHWC)
    {
        if(layout != ConvTensorLayout::NHWC)
        {
            throw std::runtime_error("wrong! layout");
        }

        const auto tmp = f_make_for_device_nhwc();

        tunable_dyn_conv_fwd_v4r4_xdlops_nhwc_kyxc_nhwk* tunable =
            &default_tunable_dyn_conv_fwd_v4r4_xdlops_nhwc_kyxc_nhwk;

        online_device_dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nhwc_kyxc_nhwk<
            in_data_t,
            acc_data_t,
            out_data_t>(handle,
                        tmp[I0],
                        tmp[I1],
                        tmp[I2],
                        conv_strides,
                        conv_dilations,
                        in_left_pads,
                        in_right_pads,
                        in,
                        wei,
                        out_device,
                        tunable,
                        nrepeat);
    }
#endif

    if(do_verification)
    {
        host_direct_convolution(in,
                                wei,
                                out_host,
                                make_tuple(conv_stride_h, conv_stride_w),
                                make_tuple(conv_dilation_h, conv_dilation_w),
                                make_tuple(in_left_pad_h, in_left_pad_w),
                                make_tuple(in_right_pad_h, in_right_pad_w),
                                layout);

        check_error(out_host, out_device);

#if 0
        if(do_log)
        {
            LogRangeAsType<float>(std::cout << "in : ", in.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "wei: ", wei.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "out_host  : ", out_host.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "out_device: ", out_device.mData, ",") << std::endl;
        }
#endif
    }

    delete handle;
    MY_HIP_CHECK(hipStreamDestroy(stream));
}
