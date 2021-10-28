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
#include "gemm_common.hpp"
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_gemm_xdlops_mk_kn_mn.hpp"
#include "device_gemm_xdlops_mk_nk_mn.hpp"
#include "device_gemm_xdlops_km_kn_mn.hpp"
#include "device_gemm_xdlops_km_nk_mn.hpp"
#include "device_gemm_xdlops_mk_kn_nm.hpp"
#include "device_gemm_xdlops_mk_nk_nm.hpp"
#include "device_gemm_xdlops_km_kn_nm.hpp"
#include "device_gemm_xdlops_km_nk_nm.hpp"

#define USE_GEMM_XDL_MK_KN_MN 1
#define USE_GEMM_XDL_MK_NK_MN 1
#define USE_GEMM_XDL_KM_KN_MN 1
#define USE_GEMM_XDL_KM_NK_MN 1
#define USE_GEMM_XDL_MK_KN_NM 0
#define USE_GEMM_XDL_MK_NK_NM 0
#define USE_GEMM_XDL_KM_KN_NM 0
#define USE_GEMM_XDL_KM_NK_NM 0

enum GemmAlgo
{
    Xdl_MK_KN_MN, // 0
    Xdl_MK_NK_MN, // 1
    Xdl_KM_KN_MN, // 2
    Xdl_KM_NK_MN, // 3
    Xdl_MK_KN_NM, // 4
    Xdl_MK_NK_NM, // 5
    Xdl_KM_KN_NM, // 6
    Xdl_KM_NK_NM, // 7
};

int main(int argc, char* argv[])
{
    using namespace ck;

    if(argc != 12)
    {
        printf("arg1 to 6: layout, algo, do_verification, init_method, do_log, nrepeat\n");
        printf("rest: M, N, K\n");
        printf("debug_driver_gemm_xdlops_v2r3::M01, debug_driver_gemm_xdlops_v2r3::N01\n");
        exit(1);
    }

    const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[1]));
    const auto algo            = static_cast<GemmAlgo>(std::stoi(argv[2]));
    const bool do_verification = std::stoi(argv[3]);
    const int init_method      = std::stoi(argv[4]);
    const bool do_log          = std::stoi(argv[5]);
    const int nrepeat          = std::stoi(argv[6]);

    const index_t M = std::stoi(argv[7]);
    const index_t N = std::stoi(argv[8]);
    const index_t K = std::stoi(argv[9]);

    debug::debug_driver_gemm_xdlops_v2r3::M01 = std::stoi(argv[10]);
    debug::debug_driver_gemm_xdlops_v2r3::N01 = std::stoi(argv[11]);

#if 0
    using ab_data_t  = float;
    using acc_data_t = float;
    using c_data_t   = float;
#elif 1
    using ab_data_t  = half_t;
    using acc_data_t = float;
    using c_data_t   = half_t;
#elif 1
    using ab_data_t  = int8_t;
    using acc_data_t = int32_t;
    using c_data_t   = int8_t;
#endif

    std::vector<std::size_t> a_lengths_host(2), b_lengths_host(2), c_lengths_host(2);
    std::vector<std::size_t> a_strides_host(2), b_strides_host(2), c_strides_host(2);

    // A
    if(layout == GemmMatrixLayout::MK_KN_MN || layout == GemmMatrixLayout::MK_NK_MN ||
       layout == GemmMatrixLayout::MK_KN_NM || layout == GemmMatrixLayout::MK_NK_NM)
    {
        a_lengths_host[0] = static_cast<std::size_t>(M);
        a_lengths_host[1] = static_cast<std::size_t>(K);
        a_strides_host[0] = static_cast<std::size_t>(K);
        a_strides_host[1] = static_cast<std::size_t>(1);
    }
    else
    {
        a_lengths_host[0] = static_cast<std::size_t>(K);
        a_lengths_host[1] = static_cast<std::size_t>(M);
        a_strides_host[0] = static_cast<std::size_t>(M);
        a_strides_host[1] = static_cast<std::size_t>(1);
    }

    // B
    if(layout == GemmMatrixLayout::MK_NK_MN || layout == GemmMatrixLayout::KM_NK_MN ||
       layout == GemmMatrixLayout::MK_NK_NM || layout == GemmMatrixLayout::KM_NK_NM)
    {
        b_lengths_host[0] = static_cast<std::size_t>(N);
        b_lengths_host[1] = static_cast<std::size_t>(K);
        b_strides_host[0] = static_cast<std::size_t>(K);
        b_strides_host[1] = static_cast<std::size_t>(1);
    }
    else
    {
        b_lengths_host[0] = static_cast<std::size_t>(K);
        b_lengths_host[1] = static_cast<std::size_t>(N);
        b_strides_host[0] = static_cast<std::size_t>(N);
        b_strides_host[1] = static_cast<std::size_t>(1);
    }

    // C
    if(layout == GemmMatrixLayout::MK_KN_MN || layout == GemmMatrixLayout::KM_KN_MN ||
       layout == GemmMatrixLayout::MK_NK_MN || layout == GemmMatrixLayout::KM_NK_MN)
    {
        c_lengths_host[0] = static_cast<std::size_t>(M);
        c_lengths_host[1] = static_cast<std::size_t>(N);
        c_strides_host[0] = static_cast<std::size_t>(N);
        c_strides_host[1] = static_cast<std::size_t>(1);
    }
    else
    {
        c_lengths_host[0] = static_cast<std::size_t>(N);
        c_lengths_host[1] = static_cast<std::size_t>(M);
        c_strides_host[0] = static_cast<std::size_t>(M);
        c_strides_host[1] = static_cast<std::size_t>(1);
    }

    Tensor<ab_data_t> a(a_lengths_host, a_strides_host);
    Tensor<ab_data_t> b(b_lengths_host, b_strides_host);
    Tensor<c_data_t> c_host(c_lengths_host, c_strides_host);
    Tensor<c_data_t> c_device(c_lengths_host, c_strides_host);

    std::cout << "layout: " << layout << std::endl;
    ostream_HostTensorDescriptor(a.mDesc, std::cout << "a: ");
    ostream_HostTensorDescriptor(b.mDesc, std::cout << "b: ");
    ostream_HostTensorDescriptor(c_host.mDesc, std::cout << "c: ");

    std::size_t num_thread = std::thread::hardware_concurrency();

    switch(init_method)
    {
    case 0:
        // no initialization
        break;
    case 1:
        a.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        b.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        break;
    case 2:
        a.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        b.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        break;
    case 3:
        a.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        b.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        break;
    case 4:
        a.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        b.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        break;
    default:
        a.GenerateTensorValue(GeneratorTensor_3<float>{0.0, 1.0}, num_thread);
        b.GenerateTensorValue(GeneratorTensor_3<float>{-0.5, 0.5}, num_thread);
    }

#if USE_GEMM_XDL_MK_KN_MN
    if(algo == GemmAlgo::Xdl_MK_KN_MN)
    {
        if(layout != GemmMatrixLayout::MK_KN_MN)
        {
            throw std::runtime_error("wrong! layout");
        }

        device_gemm_xdlops_mk_kn_mn<ab_data_t, acc_data_t, c_data_t>(a, b, c_device, nrepeat);
    }
#endif

#if USE_GEMM_XDL_MK_NK_MN
    if(algo == GemmAlgo::Xdl_MK_NK_MN)
    {
        if(layout != GemmMatrixLayout::MK_NK_MN)
        {
            throw std::runtime_error("wrong! layout");
        }

        device_gemm_xdlops_mk_nk_mn<ab_data_t, acc_data_t, c_data_t>(a, b, c_device, nrepeat);
    }
#endif

#if USE_GEMM_XDL_KM_KN_MN
    if(algo == GemmAlgo::Xdl_KM_KN_MN)
    {
        if(layout != GemmMatrixLayout::KM_KN_MN)
        {
            throw std::runtime_error("wrong! layout");
        }

        device_gemm_xdlops_km_kn_mn<ab_data_t, acc_data_t, c_data_t>(a, b, c_device, nrepeat);
    }
#endif

#if USE_GEMM_XDL_KM_NK_MN
    if(algo == GemmAlgo::Xdl_KM_NK_MN)
    {
        if(layout != GemmMatrixLayout::KM_NK_MN)
        {
            throw std::runtime_error("wrong! layout");
        }

        device_gemm_xdlops_km_nk_mn<ab_data_t, acc_data_t, c_data_t>(a, b, c_device, nrepeat);
    }
#endif

#if USE_GEMM_XDL_MK_KN_NM
    if(algo == GemmAlgo::Xdl_MK_KN_NM)
    {
        if(layout != GemmMatrixLayout::MK_KN_NM)
        {
            throw std::runtime_error("wrong! layout");
        }

        device_gemm_xdlops_mk_kn_nm<ab_data_t, acc_data_t, c_data_t>(a, b, c_device, nrepeat);
    }
#endif

#if USE_GEMM_XDL_MK_NK_NM
    if(algo == GemmAlgo::Xdl_MK_NK_NM)
    {
        if(layout != GemmMatrixLayout::MK_NK_NM)
        {
            throw std::runtime_error("wrong! layout");
        }

        device_gemm_xdlops_mk_nk_nm<ab_data_t, acc_data_t, c_data_t>(a, b, c_device, nrepeat);
    }
#endif

#if USE_GEMM_XDL_KM_KN_NM
    if(algo == GemmAlgo::Xdl_KM_KN_NM)
    {
        if(layout != GemmMatrixLayout::KM_KN_NM)
        {
            throw std::runtime_error("wrong! layout");
        }

        device_gemm_xdlops_km_kn_nm<ab_data_t, acc_data_t, c_data_t>(a, b, c_device, nrepeat);
    }
#endif

#if USE_GEMM_XDL_KM_NK_NM
    if(algo == GemmAlgo::Xdl_KM_NK_NM)
    {
        if(layout != GemmMatrixLayout::KM_NK_NM)
        {
            throw std::runtime_error("wrong! layout");
        }

        device_gemm_xdlops_km_nk_nm<ab_data_t, acc_data_t, c_data_t>(a, b, c_device, nrepeat);
    }
#endif

    if(do_verification)
    {
        host_gemm(a, b, c_host, layout);

        check_error(c_host, c_device);

        if(do_log)
        {
            LogRangeAsType<float>(std::cout << "a : ", a.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "b: ", b.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "c_host  : ", c_host.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "c_device: ", c_device.mData, ",") << std::endl;
        }
    }
}
