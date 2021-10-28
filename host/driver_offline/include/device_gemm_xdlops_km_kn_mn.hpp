#pragma once
#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_gemm_xdlops_v2r3.hpp"

template <typename ABType, typename AccType, typename CType>
void device_gemm_xdlops_km_kn_mn(const Tensor<ABType>& a_k_m,
                                 const Tensor<ABType>& b_k_n,
                                 Tensor<CType>& c_m_n,
                                 ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    DeviceMem a_k_m_device_buf(sizeof(ABType) * a_k_m.mDesc.GetElementSpace());
    DeviceMem b_k_n_device_buf(sizeof(ABType) * b_k_n.mDesc.GetElementSpace());
    DeviceMem c_m_n_device_buf(sizeof(CType) * c_m_n.mDesc.GetElementSpace());

    a_k_m_device_buf.ToDevice(a_k_m.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());
    c_m_n_device_buf.ToDevice(c_m_n.mData.data());

#if 0
    // [M, N, K0, K1] = [256, 128, 4, 4] for fp32
    constexpr index_t BlockSize = 256;

    constexpr index_t MPerBlock = 256;
    constexpr index_t NPerBlock = 128;
    constexpr index_t KPerBlock = 4;

    constexpr index_t MPerXDL = 32;
    constexpr index_t NPerXDL = 32;
    constexpr index_t K1      = 4;

    constexpr index_t MRepeat = 4;
    constexpr index_t NRepeat = 2;

    using ABlockTransferThreadSliceLengths_K0_M_K1   = Sequence<1, 4, 4>;
    using ABlockTransferThreadClusterLengths_K0_M_K1 = Sequence<4, 64, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_M  = 4;
    constexpr index_t ABlockTransferDstScalarPerVector_K1 = 4;

    using BBlockTransferThreadSliceLengths_K0_N_K1   = Sequence<1, 2, 4>;
    using BBlockTransferThreadClusterLengths_K0_N_K1 = Sequence<4, 64, 1>;

    constexpr index_t BBlockTransferSrcScalarPerVector_N  = 2;
    constexpr index_t BBlockTransferDstScalarPerVector_K1 = 4;

    constexpr index_t CThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [128, 256, 4, 4], C = 128, for fp32
    constexpr index_t BlockSize = 256;

    constexpr index_t MPerBlock = 128;
    constexpr index_t NPerBlock = 256;
    constexpr index_t KPerBlock = 4;

    constexpr index_t MPerXDL = 32;
    constexpr index_t NPerXDL = 32;
    constexpr index_t K1      = 4;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 4;

    using ABlockTransferThreadSliceLengths_K0_M_K1   = Sequence<1, 2, 4>;
    using ABlockTransferThreadClusterLengths_K0_M_K1 = Sequence<4, 64, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_M  = 2;
    constexpr index_t ABlockTransferDstScalarPerVector_K1 = 4;

    using BBlockTransferThreadSliceLengths_K0_N_K1   = Sequence<1, 4, 4>;
    using BBlockTransferThreadClusterLengths_K0_N_K1 = Sequence<4, 64, 1>;

    constexpr index_t BBlockTransferSrcScalarPerVector_N  = 4;
    constexpr index_t BBlockTransferDstScalarPerVector_K1 = 4;

    constexpr index_t CThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [128, 128, 4, 4], C = 64, for fp32
    constexpr index_t BlockSize = 256;

    constexpr index_t MPerBlock = 128;
    constexpr index_t NPerBlock = 128;
    constexpr index_t KPerBlock = 4;

    constexpr index_t MPerXDL = 32;
    constexpr index_t NPerXDL = 32;
    constexpr index_t K1      = 4;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 2;

    using ABlockTransferThreadSliceLengths_K0_M_K1   = Sequence<1, 2, 4>;
    using ABlockTransferThreadClusterLengths_K0_M_K1 = Sequence<4, 64, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_M  = 2;
    constexpr index_t ABlockTransferDstScalarPerVector_K1 = 4;

    using BBlockTransferThreadSliceLengths_K0_N_K1   = Sequence<1, 2, 4>;
    using BBlockTransferThreadClusterLengths_K0_N_K1 = Sequence<4, 64, 1>;

    constexpr index_t BBlockTransferSrcScalarPerVector_N  = 2;
    constexpr index_t BBlockTransferDstScalarPerVector_K1 = 4;

    constexpr index_t CThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [128, 64, 4, 4], C = 32, for fp32
    constexpr index_t BlockSize = 256;

    constexpr index_t MPerBlock = 128;
    constexpr index_t NPerBlock = 64;
    constexpr index_t KPerBlock = 4;

    constexpr index_t MPerXDL = 32;
    constexpr index_t NPerXDL = 32;
    constexpr index_t K1      = 4;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 1;

    using ABlockTransferThreadSliceLengths_K0_M_K1   = Sequence<1, 2, 4>;
    using ABlockTransferThreadClusterLengths_K0_M_K1 = Sequence<4, 64, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_M  = 2;
    constexpr index_t ABlockTransferDstScalarPerVector_K1 = 4;

    using BBlockTransferThreadSliceLengths_K0_N_K1   = Sequence<1, 1, 4>;
    using BBlockTransferThreadClusterLengths_K0_N_K1 = Sequence<4, 64, 1>;

    constexpr index_t BBlockTransferSrcScalarPerVector_N  = 1;
    constexpr index_t BBlockTransferDstScalarPerVector_K1 = 4;

    constexpr index_t CThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [64, 128, 4, 4], C = 32, for fp32
    constexpr index_t BlockSize = 256;

    constexpr index_t MPerBlock = 64;
    constexpr index_t NPerBlock = 128;
    constexpr index_t KPerBlock = 4;

    constexpr index_t MPerXDL = 32;
    constexpr index_t NPerXDL = 32;
    constexpr index_t K1      = 4;

    constexpr index_t MRepeat = 1;
    constexpr index_t NRepeat = 2;

    using ABlockTransferThreadSliceLengths_K0_M_K1   = Sequence<1, 1, 4>;
    using ABlockTransferThreadClusterLengths_K0_M_K1 = Sequence<4, 64, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_M  = 1;
    constexpr index_t ABlockTransferDstScalarPerVector_K1 = 4;

    using BBlockTransferThreadSliceLengths_K0_N_K1   = Sequence<1, 2, 4>;
    using BBlockTransferThreadClusterLengths_K0_N_K1 = Sequence<4, 64, 1>;

    constexpr index_t BBlockTransferSrcScalarPerVector_N  = 2;
    constexpr index_t BBlockTransferDstScalarPerVector_K1 = 4;

    constexpr index_t CThreadTransferDstScalarPerVector = 1;
#elif 1
    // [M, N, K0, K1] = [256, 128, 4, 8], C = 128, for fp16
    constexpr index_t BlockSize = 256;

    constexpr index_t MPerBlock = 256;
    constexpr index_t NPerBlock = 128;
    constexpr index_t KPerBlock = 4;

    constexpr index_t MPerXDL = 32;
    constexpr index_t NPerXDL = 32;
    constexpr index_t K1      = 8;

    constexpr index_t MRepeat = 4;
    constexpr index_t NRepeat = 2;

    using ABlockTransferThreadSliceLengths_K0_M_K1   = Sequence<1, 4, 8>;
    using ABlockTransferThreadClusterLengths_K0_M_K1 = Sequence<4, 64, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_M  = 4;
    constexpr index_t ABlockTransferDstScalarPerVector_K1 = 8;

    using BBlockTransferThreadSliceLengths_K0_N_K1   = Sequence<1, 2, 8>;
    using BBlockTransferThreadClusterLengths_K0_N_K1 = Sequence<4, 64, 1>;

    constexpr index_t BBlockTransferSrcScalarPerVector_N  = 2;
    constexpr index_t BBlockTransferDstScalarPerVector_K1 = 8;

    constexpr index_t CThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [128, 256, 4, 8] for fp16
    constexpr index_t BlockSize = 256;

    constexpr index_t MPerBlock = 128;
    constexpr index_t NPerBlock = 256;
    constexpr index_t KPerBlock = 4;

    constexpr index_t MPerXDL = 32;
    constexpr index_t NPerXDL = 32;
    constexpr index_t K1      = 8;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 4;

    using ABlockTransferThreadSliceLengths_K0_M_K1   = Sequence<1, 2, 8>;
    using ABlockTransferThreadClusterLengths_K0_M_K1 = Sequence<4, 64, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_M  = 2;
    constexpr index_t ABlockTransferDstScalarPerVector_K1 = 8;

    using BBlockTransferThreadSliceLengths_K0_N_K1   = Sequence<1, 4, 8>;
    using BBlockTransferThreadClusterLengths_K0_N_K1 = Sequence<4, 64, 1>;

    constexpr index_t BBlockTransferSrcScalarPerVector_N  = 4;
    constexpr index_t BBlockTransferDstScalarPerVector_K1 = 8;

    constexpr index_t CThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [128, 128, 4, 8], C = 128, for fp16
    constexpr index_t BlockSize = 128;

    constexpr index_t MPerBlock = 128;
    constexpr index_t NPerBlock = 128;
    constexpr index_t KPerBlock = 4;

    constexpr index_t MPerXDL = 32;
    constexpr index_t NPerXDL = 32;
    constexpr index_t K1      = 8;

    constexpr index_t MRepeat = 4;
    constexpr index_t NRepeat = 2;

    using ABlockTransferThreadSliceLengths_K0_M_K1   = Sequence<1, 4, 8>;
    using ABlockTransferThreadClusterLengths_K0_M_K1 = Sequence<4, 32, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_M  = 4;
    constexpr index_t ABlockTransferDstScalarPerVector_K1 = 8;

    using BBlockTransferThreadSliceLengths_K0_N_K1   = Sequence<1, 4, 8>;
    using BBlockTransferThreadClusterLengths_K0_N_K1 = Sequence<4, 32, 1>;

    constexpr index_t BBlockTransferSrcScalarPerVector_N  = 4;
    constexpr index_t BBlockTransferDstScalarPerVector_K1 = 8;

    constexpr index_t CThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [128, 128, 4, 8], C = 64, for fp16
    constexpr index_t BlockSize = 256;

    constexpr index_t MPerBlock = 128;
    constexpr index_t NPerBlock = 128;
    constexpr index_t KPerBlock = 4;

    constexpr index_t MPerXDL = 32;
    constexpr index_t NPerXDL = 32;
    constexpr index_t K1      = 8;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 2;

    using ABlockTransferThreadSliceLengths_K0_M_K1   = Sequence<1, 2, 8>;
    using ABlockTransferThreadClusterLengths_K0_M_K1 = Sequence<4, 64, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_M  = 2;
    constexpr index_t ABlockTransferDstScalarPerVector_K1 = 8;

    using BBlockTransferThreadSliceLengths_K0_N_K1   = Sequence<1, 2, 8>;
    using BBlockTransferThreadClusterLengths_K0_N_K1 = Sequence<4, 64, 1>;

    constexpr index_t BBlockTransferSrcScalarPerVector_N  = 2;
    constexpr index_t BBlockTransferDstScalarPerVector_K1 = 8;

    constexpr index_t CThreadTransferDstScalarPerVector = 1;
#elif 1
    // [M, N, K0, K1] = [128, 64, 4, 8], C = 32, for fp16
    constexpr index_t BlockSize = 256;

    constexpr index_t MPerBlock = 128;
    constexpr index_t NPerBlock = 64;
    constexpr index_t KPerBlock = 4;

    constexpr index_t MPerXDL = 32;
    constexpr index_t NPerXDL = 32;
    constexpr index_t K1      = 8;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 1;

    using ABlockTransferThreadSliceLengths_K0_M_K1   = Sequence<1, 2, 8>;
    using ABlockTransferThreadClusterLengths_K0_M_K1 = Sequence<4, 64, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_M  = 2;
    constexpr index_t ABlockTransferDstScalarPerVector_K1 = 8;

    using BBlockTransferThreadSliceLengths_K0_N_K1   = Sequence<1, 1, 8>;
    using BBlockTransferThreadClusterLengths_K0_N_K1 = Sequence<4, 64, 1>;

    constexpr index_t BBlockTransferSrcScalarPerVector_N  = 1;
    constexpr index_t BBlockTransferDstScalarPerVector_K1 = 8;

    constexpr index_t CThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [64, 128, 4, 8], C = 32, for fp16
    constexpr index_t BlockSize = 256;

    constexpr index_t MPerBlock = 64;
    constexpr index_t NPerBlock = 128;
    constexpr index_t KPerBlock = 4;

    constexpr index_t MPerXDL = 32;
    constexpr index_t NPerXDL = 32;
    constexpr index_t K1      = 8;

    constexpr index_t MRepeat = 1;
    constexpr index_t NRepeat = 2;

    using ABlockTransferThreadSliceLengths_K0_M_K1   = Sequence<1, 1, 8>;
    using ABlockTransferThreadClusterLengths_K0_M_K1 = Sequence<4, 64, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_M  = 1;
    constexpr index_t ABlockTransferDstScalarPerVector_K1 = 8;

    using BBlockTransferThreadSliceLengths_K0_N_K1   = Sequence<1, 2, 8>;
    using BBlockTransferThreadClusterLengths_K0_N_K1 = Sequence<4, 64, 1>;

    constexpr index_t BBlockTransferSrcScalarPerVector_N  = 2;
    constexpr index_t BBlockTransferDstScalarPerVector_K1 = 8;

    constexpr index_t CThreadTransferDstScalarPerVector = 1;
#endif

    const auto K = a_k_m.mDesc.GetLengths()[0];
    const auto M = a_k_m.mDesc.GetLengths()[1];
    const auto N = b_k_n.mDesc.GetLengths()[1];

    constexpr auto K1Number = Number<K1>{};
    const auto K0           = K / K1Number;

    const auto a_k0_m_k1_grid_desc =
        make_naive_tensor_descriptor(make_tuple(K0, M, K1Number),
                                     make_tuple(K1Number * a_k_m.mDesc.GetStrides()[0],
                                                a_k_m.mDesc.GetStrides()[1],
                                                a_k_m.mDesc.GetStrides()[0]));

    const auto b_k0_n_k1_grid_desc =
        make_naive_tensor_descriptor(make_tuple(K0, N, K1Number),
                                     make_tuple(K1Number * b_k_n.mDesc.GetStrides()[0],
                                                b_k_n.mDesc.GetStrides()[1],
                                                b_k_n.mDesc.GetStrides()[0]));

    const auto c_m_n_grid_desc = make_naive_tensor_descriptor(
        make_tuple(M, N), make_tuple(c_m_n.mDesc.GetStrides()[0], c_m_n.mDesc.GetStrides()[1]));

    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    constexpr auto a_k0_m_k1_grid_step_hacks = make_tuple(make_tuple(Sequence<0>{},   // 0+: K0
                                                                     Sequence<0>{},   // 1+: M
                                                                     Sequence<0>{}),  // 2+: K1
                                                          make_tuple(Sequence<0>{},   // 0-: K0
                                                                     Sequence<0>{},   // 1-: M
                                                                     Sequence<0>{})); // 2-: K1

    constexpr auto b_k0_n_k1_grid_step_hacks = make_tuple(make_tuple(Sequence<0>{},   // 0+: K0
                                                                     Sequence<0>{},   // 1+: N
                                                                     Sequence<0>{}),  // 2+: K1
                                                          make_tuple(Sequence<0>{},   // 0-: K0
                                                                     Sequence<0>{},   // 1-: N
                                                                     Sequence<0>{})); // 2-: K1

    constexpr auto c_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 0+: M0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1+: N0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 2+: M1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 3+: N1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 4+: M2
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 5+: M3
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 6+: M4
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{}),  // 7+: N2
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 0-: M0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1-: N0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 2-: M1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 3-: N1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 4-: M2
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 5-: M3
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 6-: M4
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{})); // 7-: N2

    constexpr auto a_k0_m_k1_grid_move_slice_window_step_hacks = Sequence<0>{};

    constexpr auto b_k0_n_k1_grid_move_slice_window_step_hacks = Sequence<0>{};

    for(index_t i = 0; i < 5; ++i)
    {
        float ave_time =
            driver_gemm_xdlops_v2r3<BlockSize,
                                    ABType,
                                    AccType,
                                    CType,
                                    InMemoryDataOperationEnum_t::Set,
                                    decltype(a_k0_m_k1_grid_desc),
                                    decltype(b_k0_n_k1_grid_desc),
                                    decltype(c_m_n_grid_desc),
                                    MPerBlock,
                                    NPerBlock,
                                    KPerBlock,
                                    MPerXDL,
                                    NPerXDL,
                                    K1,
                                    MRepeat,
                                    NRepeat,
                                    ABlockTransferThreadSliceLengths_K0_M_K1,
                                    ABlockTransferThreadClusterLengths_K0_M_K1,
                                    Sequence<0, 2, 1>,
                                    Sequence<0, 2, 1>,
                                    1,
                                    ABlockTransferSrcScalarPerVector_M,
                                    ABlockTransferDstScalarPerVector_K1,
                                    false, // don't move back src coordinate after threadwise copy
                                    BBlockTransferThreadSliceLengths_K0_N_K1,
                                    BBlockTransferThreadClusterLengths_K0_N_K1,
                                    Sequence<0, 2, 1>,
                                    Sequence<0, 2, 1>,
                                    1,
                                    BBlockTransferSrcScalarPerVector_N,
                                    BBlockTransferDstScalarPerVector_K1,
                                    false, // don't move back src coordinate after threadwise copy
                                    Sequence<0, 2, 4, 5, 6, 1, 3, 7>,
                                    7,
                                    CThreadTransferDstScalarPerVector,
                                    decltype(a_k0_m_k1_grid_step_hacks),
                                    decltype(b_k0_n_k1_grid_step_hacks),
                                    decltype(c_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks),
                                    decltype(a_k0_m_k1_grid_move_slice_window_step_hacks),
                                    decltype(b_k0_n_k1_grid_move_slice_window_step_hacks),
                                    false, // CAccessOrderMRepeatNRepeat
                                    true,  // ABlockLdsExtraM
                                    true   // BBlockLdsExtraN
                                    >(static_cast<ABType*>(a_k_m_device_buf.GetDeviceBuffer()),
                                      static_cast<ABType*>(b_k_n_device_buf.GetDeviceBuffer()),
                                      static_cast<CType*>(c_m_n_device_buf.GetDeviceBuffer()),
                                      a_k0_m_k1_grid_desc,
                                      b_k0_n_k1_grid_desc,
                                      c_m_n_grid_desc,
                                      debug::debug_driver_gemm_xdlops_v2r3::M01,
                                      debug::debug_driver_gemm_xdlops_v2r3::N01,
                                      a_k0_m_k1_grid_step_hacks,
                                      b_k0_n_k1_grid_step_hacks,
                                      c_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks,
                                      a_k0_m_k1_grid_move_slice_window_step_hacks,
                                      b_k0_n_k1_grid_move_slice_window_step_hacks,
                                      nrepeat);

        float perf = static_cast<float>((std::size_t(2) * M * N * K)) /
                     (std::size_t(1000) * 1000 * 1000) / ave_time;

        std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s" << std::endl;
    }

    // copy result back to host
    c_m_n_device_buf.FromDevice(c_m_n.mData.data());
}
