#ifndef CK_BLOCKWISE_GEMM_XDLOPS_HPP
#define CK_BLOCKWISE_GEMM_XDLOPS_HPP

#include "common_header.hpp"
#include "threadwise_tensor_slice_transfer.hpp"
#include "xdlops_gemm.hpp"
#include "tensor_adaptor.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename AK0MK1BlockDesc,
          typename BK0NK1BlockDesc,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t K1>
struct BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr index_t WaveSize = 64;

    static constexpr index_t MPerBlock = AK0MK1BlockDesc{}.GetLength(I1);
    static constexpr index_t NPerBlock = BK0NK1BlockDesc{}.GetLength(I1);

    static constexpr index_t K0 = BK0NK1BlockDesc{}.GetLength(I0);

    static constexpr auto xdlops_gemm = XdlopsGemm<FloatAB, MPerXDL, NPerXDL, K1>{};

    static constexpr index_t MWaves = MPerBlock / (MRepeat * MPerXDL);
    static constexpr index_t NWaves = NPerBlock / (NRepeat * NPerXDL);

    StaticBufferV2<AddressSpaceEnum_t::Vgpr, vector_type<FloatAcc, 16>, MRepeat * NRepeat, true>
        c_thread_buf_;

    __host__ __device__ constexpr auto& GetCThreadBuffer() { return c_thread_buf_; }

    __device__ static auto GetWaveIdx()
    {
        const index_t thread_id = get_thread_local_1d_id();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(MWaves, NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];

        const auto xdlops_a_idx = xdlops_gemm.CalculateAThreadOriginDataIndex();

        return make_tuple(xdlops_a_idx[I0], 0, waveId_m, xdlops_a_idx[I1], 0);
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_n = wave_idx[I1];

        const auto xdlops_b_idx = xdlops_gemm.CalculateBThreadOriginDataIndex();

        return make_tuple(xdlops_b_idx[I0], 0, waveId_n, xdlops_b_idx[I1], 0);
    }

    template <index_t m0, index_t n0, index_t xdlops_i, index_t blk_i>
    __device__ static auto
        CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>, Number<xdlops_i>, Number<blk_i>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = xdlops_gemm.GetBeginOfThreadBlk(xdlops_i, blk_i);

        constexpr auto mrepeat_mwave_mperxdl_to_m_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, MPerXDL))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        constexpr auto nrepeat_nwave_nperxdl_to_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(NRepeat, NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        const index_t c_thread_m = mrepeat_mwave_mperxdl_to_m_adaptor.CalculateBottomIndex(
            make_tuple(m0, waveId_m, blk_idx[I0]))[I0];
        const index_t c_thread_n = nrepeat_nwave_nperxdl_to_n_adaptor.CalculateBottomIndex(
            make_tuple(n0, waveId_n, blk_idx[I1]))[I0];

        return make_tuple(c_thread_m, c_thread_n);
    }

    __host__ __device__ BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1()
    {
        static_assert(AK0MK1BlockDesc::IsKnownAtCompileTime() &&
                          BK0NK1BlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(AK0MK1BlockDesc{}.GetLength(I0) == BK0NK1BlockDesc{}.GetLength(I0),
                      "wrong! K0 dimension not consistent");

        static_assert(AK0MK1BlockDesc{}.GetLength(I2) == BK0NK1BlockDesc{}.GetLength(I2),
                      "wrong! K1 dimension not consistent");

        static_assert(BlockSize == MWaves * NWaves * WaveSize,
                      "BlockSize != MWaves * NWaves * WaveSize\n");

        static_assert(MPerBlock % (MPerXDL * MRepeat) == 0 && NPerBlock % (NPerXDL * NRepeat) == 0,
                      "wrong!");
    }

    __host__ __device__ static constexpr auto GetCM0N0M1N1M2M3M4N2ThreadDescriptor()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(make_tuple(I1, I1, I1, I1, M0, M1, M2, N));
    }

    __host__ __device__ static constexpr auto GetCM0N0M1N1M2M3M4N2BlockDescriptor()
    {
        constexpr auto c_m0_n0_m1_n1_m2_n2_block_desc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCM0N0M1N1M2M3M4N2Descriptor(c_m0_n0_m1_n1_m2_n2_block_desc);
    }

    template <typename CMNGridDesc>
    __host__ __device__ static constexpr auto
    MakeCM0N0M1N1M2M3M4N2GridDescriptor(const CMNGridDesc& c_m_n_grid_desc)
    {
        const auto c_m0_n0_m1_n1_m2_n2_grid_desc = transform_tensor_descriptor(
            c_m_n_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, MPerXDL)),
                       make_unmerge_transform(make_tuple(NRepeat, NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}));

        return xdlops_gemm.MakeCM0N0M1N1M2M3M4N2Descriptor(c_m0_n0_m1_n1_m2_n2_grid_desc);
    }

    __host__ __device__ static constexpr auto MakeAK0M0M1M2K1BlockDescriptor()
    {
        return transform_tensor_descriptor(
            AK0MK1BlockDesc{},
            make_tuple(make_pass_through_transform(Number<K0>{}),
                       make_unmerge_transform(
                           make_tuple(Number<MRepeat>{}, Number<MWaves>{}, Number<MPerXDL>{})),
                       make_pass_through_transform(Number<K1>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}));
    }

    __host__ __device__ static constexpr auto MakeBK0N0N1N2K1BlockDescriptor()
    {
        return transform_tensor_descriptor(
            BK0NK1BlockDesc{},
            make_tuple(make_pass_through_transform(Number<K0>{}),
                       make_unmerge_transform(
                           make_tuple(Number<NRepeat>{}, Number<NWaves>{}, Number<NPerXDL>{})),
                       make_pass_through_transform(Number<K1>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}));
    }

    static constexpr auto a_k0_m0_m1_m2_k1_block_desc = MakeAK0M0M1M2K1BlockDescriptor();
    static constexpr auto b_k0_n0_n1_n2_k1_block_desc = MakeBK0N0N1N2K1BlockDescriptor();

    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum_t::Vgpr, FloatAB>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum_t::Vgpr, FloatAB>(
            b_thread_desc_.GetElementSpaceSize());

        static_for<0, MRepeat, 1>{}([&](auto m0) {
            // read A
            a_thread_copy_.Run(a_k0_m0_m1_m2_k1_block_desc,
                               make_tuple(I0, m0, I0, I0, I0),
                               a_block_buf,
                               a_thread_desc_,
                               make_tuple(I0, I0, I0, I0, I0),
                               a_thread_buf);

            static_for<0, NRepeat, 1>{}([&](auto n0) {
                // read B
                b_thread_copy_.Run(b_k0_n0_n1_n2_k1_block_desc,
                                   make_tuple(I0, n0, I0, I0, I0),
                                   b_block_buf,
                                   b_thread_desc_,
                                   make_tuple(I0, I0, I0, I0, I0),
                                   b_thread_buf);

                static_for<0, K0, xdlops_gemm.K0PerXdlops>{}([&](auto k0) {
                    vector_type<FloatAB, K1> a_thread_vec;
                    vector_type<FloatAB, K1> b_thread_vec;

                    static_for<0, K1, 1>{}([&](auto i) {
                        a_thread_vec.template AsType<FloatAB>()(i) = a_thread_buf
                            [Number<a_thread_desc_.CalculateOffset(make_tuple(k0, 0, 0, 0, i))>{}];
                        b_thread_vec.template AsType<FloatAB>()(i) = b_thread_buf
                            [Number<b_thread_desc_.CalculateOffset(make_tuple(k0, 0, 0, 0, i))>{}];
                    });

                    using mfma_input_type =
                        typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                    constexpr index_t c_offset = c_thread_desc_.CalculateOffset(make_tuple(m0, n0));

                    xdlops_gemm.template Run(a_thread_vec.template AsType<mfma_input_type>(),
                                             b_thread_vec.template AsType<mfma_input_type>(),
                                             c_thread_buf.GetVector(Number<c_offset>{}));
                });
            });
        });
    }

    private:
    // A[K, M]
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(Number<K0>{}, I1, I1, I1, Number<K1>{}));

    // B[K, N]
    static constexpr auto b_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(Number<K0>{}, I1, I1, I1, Number<K1>{}));

    static constexpr auto c_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{}, Number<NRepeat>{}));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(a_k0_m0_m1_m2_k1_block_desc),
                                                         decltype(a_thread_desc_),
                                                         Sequence<K0, 1, 1, 1, K1>,
                                                         Sequence<0, 1, 2, 3, 4>,
                                                         4,
                                                         K1,
                                                         K1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(b_k0_n0_n1_n2_k1_block_desc),
                                                         decltype(b_thread_desc_),
                                                         Sequence<K0, 1, 1, 1, K1>,
                                                         Sequence<0, 1, 2, 3, 4>,
                                                         4,
                                                         K1,
                                                         K1>;

    AThreadCopy a_thread_copy_{CalculateAThreadOriginDataIndex()};
    BThreadCopy b_thread_copy_{CalculateBThreadOriginDataIndex()};
};

} // namespace ck
#endif
