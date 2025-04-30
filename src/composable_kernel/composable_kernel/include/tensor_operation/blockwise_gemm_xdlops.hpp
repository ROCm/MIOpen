#ifndef CK_BLOCKWISE_GEMM_XDLOPS_HPP
#define CK_BLOCKWISE_GEMM_XDLOPS_HPP

#include "common_header.hpp"
#include "threadwise_tensor_slice_transfer.hpp"
#include "xdlops_gemm.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatAB,
          class ABlockDesc,
          class BBlockDesc,
          index_t MPerWave,
          index_t NPerWave,
          index_t K1>
struct BlockwiseGemmXdlops_km_kn_m0m1m2n_v1
{

    using CIndex = MultiIndex<2>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr index_t WaveSize = 64;

    static constexpr index_t M0 = ABlockDesc{}.GetLength(I1);
    static constexpr index_t M1 = ABlockDesc{}.GetLength(I2);

    static constexpr index_t N0 = BBlockDesc{}.GetLength(I1);
    static constexpr index_t N1 = BBlockDesc{}.GetLength(I2);

    static constexpr auto xdlops_gemm = XdlopsGemm<FloatAB, MPerWave, NPerWave, K1>{};

    static constexpr index_t MWaves = M1 / MPerWave;
    static constexpr index_t NWaves = N1 / NPerWave;

    static constexpr index_t MRepeat = M0;
    static constexpr index_t NRepeat = N0;

    __device__ constexpr auto GetCLayout() const { return xdlops_gemm.GetCLayout(); }

    __device__ constexpr auto GetNumBlks() const { return xdlops_gemm.GetCLayout().GetNumBlks(); }

    __device__ constexpr auto GetBlkSize() const { return xdlops_gemm.GetCLayout().GetBlkSize(); }

    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const index_t thread_id = get_thread_local_1d_id();
        const index_t waveId    = thread_id / WaveSize;
        const index_t laneId    = thread_id % WaveSize;
        const index_t waveId_m  = waveId / NWaves;

        if constexpr(xdlops_gemm.IsKReduction)
        {
            const index_t m_offset = waveId_m * MPerWave + xdlops_gemm.GetBlkTd(laneId);
            const index_t k_offset = xdlops_gemm.GetBlkId(laneId);
            return make_tuple(k_offset, 0, m_offset, 0);
        }
        else
        {
            const index_t m_offset = waveId_m * MPerWave + laneId;
            const index_t k_offset = 0;
            return make_tuple(k_offset, 0, m_offset, 0);
        }
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const index_t thread_id = get_thread_local_1d_id();
        const index_t waveId    = thread_id / WaveSize;
        const index_t laneId    = thread_id % WaveSize;
        const index_t waveId_n  = waveId % NWaves;

        if constexpr(xdlops_gemm.IsKReduction)
        {
            const index_t n_offset = waveId_n * NPerWave + xdlops_gemm.GetBlkTd(laneId);
            const index_t k_offset = xdlops_gemm.GetBlkId(laneId);
            return make_tuple(k_offset, 0, n_offset, 0);
        }
        else
        {
            const index_t n_offset = waveId_n * NPerWave + laneId;
            const index_t k_offset = 0;
            return make_tuple(k_offset, 0, n_offset, 0);
        }
    }

    template <index_t m0, index_t n0, index_t xdlops_i, index_t blk_i>
    __device__ static CIndex
        CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>, Number<xdlops_i>, Number<blk_i>)
    {

        const index_t waveId = get_thread_local_1d_id() / WaveSize;

        const auto thread_mtx_on_blk = xdlops_gemm.GetBeginOfThreadBlk(xdlops_i, blk_i);

        const index_t waveId_m = waveId / NWaves;
        const index_t waveId_n = waveId % NWaves;

        const index_t m_offset = m0 * M1 + waveId_m * MPerWave + thread_mtx_on_blk[I0];
        const index_t n_offset = n0 * N1 + waveId_n * NPerWave + thread_mtx_on_blk[I1];

        return CIndex{m_offset, n_offset};
    }

    __device__ BlockwiseGemmXdlops_km_kn_m0m1m2n_v1()
        : a_thread_copy_{CalculateAThreadOriginDataIndex()},
          b_thread_copy_{CalculateBThreadOriginDataIndex()}
    {
        static_assert(ABlockDesc::IsKnownAtCompileTime() && BBlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ABlockDesc{}.GetLength(I0) == BBlockDesc{}.GetLength(I0),
                      "wrong! K dimension not consistent");

        static_assert(ABlockDesc{}.GetLength(I3) == BBlockDesc{}.GetLength(I3),
                      "wrong! K1 dimension not consistent");

        static_assert(BlockSize == MWaves * NWaves * WaveSize,
                      "BlockSize != MWaves * NWaves * WaveSize\n");

        static_assert(K1 == BBlockDesc{}.GetLength(I3), "K1 is wrong!");

        constexpr index_t KPerBlock = ABlockDesc{}.GetLength(I0);

        static_assert(KPerBlock % xdlops_gemm.KPerXdlops == 0, "KPerBlock is wrong!");

        static_assert(K1 % xdlops_gemm.mfma_type.k_base == 0, "K1 is wrong!");
    }

    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum_t::Vgpr, FloatAB>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum_t::Vgpr, FloatAB>(
            b_thread_desc_.GetElementSpaceSize());

        constexpr index_t KPerBlock = ABlockDesc{}.GetLength(I0);

        vector_type<FloatAB, a_thread_desc_.GetElementSpaceSize()> a_thread_vec;

        vector_type<FloatAB, b_thread_desc_.GetElementSpaceSize()> b_thread_vec;

        static_for<0, KPerBlock, xdlops_gemm.KPerXdlops>{}([&](auto k) {
            // read A
            a_thread_copy_.Run(ABlockDesc{},
                               make_tuple(k, I0, I0, I0),
                               a_block_buf,
                               a_thread_desc_,
                               make_tuple(I0, I0, I0, I0),
                               a_thread_buf);

            // read B
            b_thread_copy_.Run(BBlockDesc{},
                               make_tuple(k, I0, I0, I0),
                               b_block_buf,
                               b_thread_desc_,
                               make_tuple(I0, I0, I0, I0),
                               b_thread_buf);

            using mfma_input_type =
                typename vector_type<FloatAB, xdlops_gemm.mfma_type.k_base>::type;

            static_for<0, a_thread_desc_.GetElementSpaceSize(), 1>{}([&](auto i) {
                a_thread_vec.template AsType<FloatAB>()(Number<i>{}) = a_thread_buf[Number<i>{}];
            });

            static_for<0, b_thread_desc_.GetElementSpaceSize(), 1>{}([&](auto i) {
                b_thread_vec.template AsType<FloatAB>()(Number<i>{}) = b_thread_buf[Number<i>{}];
            });

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    xdlops_gemm.template Run<decltype(a_thread_desc_),
                                             decltype(b_thread_desc_),
                                             decltype(c_thread_desc_),
                                             m0,
                                             n0>(a_thread_vec.template AsType<mfma_input_type>(),
                                                 b_thread_vec.template AsType<mfma_input_type>(),
                                                 c_thread_buf);
                });
            });
        });
    }

    private:
    // A[K, M]
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1, Number<MRepeat>{}, I1, Number<K1>{}));

    // B[K, N]
    static constexpr auto b_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1, Number<NRepeat>{}, I1, Number<K1>{}));

    static constexpr auto c_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{}, Number<NRepeat>{}));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         ABlockDesc,
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, MRepeat, 1, K1>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         K1,
                                                         1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         BBlockDesc,
                                                         decltype(b_thread_desc_),
                                                         Sequence<1, NRepeat, 1, K1>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         K1,
                                                         1>;

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;
};

template <index_t BlockSize,
          typename FloatAB,
          class ABlockDesc,
          class BBlockDesc,
          index_t MPerWave,
          index_t NPerWave,
          index_t K1>
struct BlockwiseGemmXdlops_km_kn_m0m1m2n_v1_2x2pipeline
{

    using CIndex = MultiIndex<2>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto xdlops_gemm = XdlopsGemm<float, MPerWave, NPerWave, K1>{};

    static constexpr index_t WaveSize = 64;

    static constexpr index_t M0 = ABlockDesc{}.GetLength(I1);
    static constexpr index_t M1 = ABlockDesc{}.GetLength(I2);

    static constexpr index_t N0 = BBlockDesc{}.GetLength(I1);
    static constexpr index_t N1 = BBlockDesc{}.GetLength(I2);

    static constexpr index_t MWaves = M1 / MPerWave;
    static constexpr index_t NWaves = N1 / NPerWave;

    static constexpr index_t MRepeat = M0;
    static constexpr index_t NRepeat = N0;

    __device__ constexpr auto GetCLayout() const { return xdlops_gemm.GetCLayout(); }

    __device__ constexpr auto GetNumBlks() const { return xdlops_gemm.GetCLayout().GetNumBlks(); }

    __device__ constexpr auto GetBlkSize() const { return xdlops_gemm.GetCLayout().GetBlkSize(); }

    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const index_t thread_id = get_thread_local_1d_id();
        const index_t waveId    = thread_id / WaveSize;
        const index_t laneId    = thread_id % WaveSize;
        const index_t waveId_m  = waveId / NWaves;

        if constexpr(xdlops_gemm.IsKReduction)
        {
            const index_t m_offset = waveId_m * MPerWave + xdlops_gemm.GetBlkTd(laneId);
            const index_t k_offset = xdlops_gemm.GetBlkId(laneId);
            return make_tuple(k_offset, 0, m_offset, 0);
        }
        else
        {
            const index_t m_offset = waveId_m * MPerWave + laneId;
            const index_t k_offset = 0;
            return make_tuple(k_offset, 0, m_offset, 0);
        }
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const index_t thread_id = get_thread_local_1d_id();
        const index_t waveId    = thread_id / WaveSize;
        const index_t laneId    = thread_id % WaveSize;
        const index_t waveId_n  = waveId % NWaves;

        if constexpr(xdlops_gemm.IsKReduction)
        {
            const index_t n_offset = waveId_n * NPerWave + xdlops_gemm.GetBlkTd(laneId);
            const index_t k_offset = xdlops_gemm.GetBlkId(laneId);
            return make_tuple(k_offset, 0, n_offset, 0);
        }
        else
        {
            const index_t n_offset = waveId_n * NPerWave + laneId;
            const index_t k_offset = 0;
            return make_tuple(k_offset, 0, n_offset, 0);
        }
    }

    template <index_t m0, index_t n0, index_t xdlops_i, index_t blk_i>
    __device__ static CIndex
        CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>, Number<xdlops_i>, Number<blk_i>)
    {

        const index_t waveId = get_thread_local_1d_id() / WaveSize;

        const auto thread_mtx_on_blk = xdlops_gemm.GetBeginOfThreadBlk(xdlops_i, blk_i);

        const index_t waveId_m = waveId / NWaves;
        const index_t waveId_n = waveId % NWaves;

        const index_t m_offset = m0 * M1 + waveId_m * MPerWave + thread_mtx_on_blk[I0];
        const index_t n_offset = n0 * N1 + waveId_n * NPerWave + thread_mtx_on_blk[I1];

        return CIndex{m_offset, n_offset};
    }

    __device__ BlockwiseGemmXdlops_km_kn_m0m1m2n_v1_2x2pipeline()
        : a_thread_copy_{CalculateAThreadOriginDataIndex()},
          b_thread_copy_{CalculateBThreadOriginDataIndex()}
    {
        static_assert(ABlockDesc::IsKnownAtCompileTime() && BBlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ABlockDesc{}.GetLength(I0) == BBlockDesc{}.GetLength(I0),
                      "wrong! K dimension not consistent");

        static_assert(ABlockDesc{}.GetLength(I3) == BBlockDesc{}.GetLength(I3),
                      "wrong! K1 dimension not consistent");

        static_assert(BlockSize == MWaves * NWaves * WaveSize,
                      "BlockSize != MWaves * NWaves * WaveSize\n");

        static_assert(K1 == BBlockDesc{}.GetLength(I3), "K1 is wrong!");

        constexpr index_t KPerBlock = ABlockDesc{}.GetLength(I0);

        static_assert(KPerBlock % xdlops_gemm.KPerXdlops == 0, "KPerBlock is wrong!");

        static_assert(K1 % xdlops_gemm.mfma_type.k_base == 0, "K1 is wrong!");
    }

    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum_t::Vgpr, FloatAB>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum_t::Vgpr, FloatAB>(
            b_thread_desc_.GetElementSpaceSize());

        constexpr index_t KPerBlock = ABlockDesc{}.GetLength(I0);

        // read A_sub_0
        a_thread_copy_.Run(ABlockDesc{},
                           make_tuple(I0, I0, I0, I0),
                           a_block_buf,
                           a_thread_desc_,
                           make_tuple(I0, I0, I0, I0),
                           a_thread_buf);

        // read B_sub_0
        b_thread_copy_.Run(BBlockDesc{},
                           make_tuple(I0, I0, I0, I0),
                           b_block_buf,
                           b_thread_desc_,
                           make_tuple(I0, I0, I0, I0),
                           b_thread_buf);

        // read B_sub_1
        b_thread_copy_.Run(BBlockDesc{},
                           make_tuple(I0, I1, I0, I0),
                           b_block_buf,
                           b_thread_desc_,
                           make_tuple(I0, I1, I0, I0),
                           b_thread_buf);

        // read A_sub_1
        a_thread_copy_.Run(ABlockDesc{},
                           make_tuple(I0, I1, I0, I0),
                           a_block_buf,
                           a_thread_desc_,
                           make_tuple(I0, I1, I0, I0),
                           a_thread_buf);

        // C_sub_00 += transpose(A_sub_0) * B_sub_0
        xdlops_gemm.template Run<decltype(a_thread_desc_),
                                 decltype(b_thread_desc_),
                                 decltype(c_thread_desc_),
                                 0,
                                 0>(a_thread_buf, b_thread_buf, c_thread_buf);

        // C_sub_01 += transpose(A_sub_0) * B_sub_1
        xdlops_gemm.template Run<decltype(a_thread_desc_),
                                 decltype(b_thread_desc_),
                                 decltype(c_thread_desc_),
                                 0,
                                 1>(a_thread_buf, b_thread_buf, c_thread_buf);

        static_for<xdlops_gemm.KPerXdlops, KPerBlock, xdlops_gemm.KPerXdlops>{}([&](auto k) {
            // read A_sub_0
            a_thread_copy_.Run(ABlockDesc{},
                               make_tuple(k, I0, I0, I0),
                               a_block_buf,
                               a_thread_desc_,
                               make_tuple(I0, I0, I0, I0),
                               a_thread_buf);

            // C_sub_10 += transpose(A_sub_1) * B_sub_0
            xdlops_gemm.template Run<decltype(a_thread_desc_),
                                     decltype(b_thread_desc_),
                                     decltype(c_thread_desc_),
                                     1,
                                     0>(a_thread_buf, b_thread_buf, c_thread_buf);

            // read B_sub_0
            b_thread_copy_.Run(BBlockDesc{},
                               make_tuple(k, I0, I0, I0),
                               b_block_buf,
                               b_thread_desc_,
                               make_tuple(I0, I0, I0, I0),
                               b_thread_buf);

            // C_sub_11 += transpose(A_sub_1) * B_sub_1
            xdlops_gemm.template Run<decltype(a_thread_desc_),
                                     decltype(b_thread_desc_),
                                     decltype(c_thread_desc_),
                                     1,
                                     1>(a_thread_buf, b_thread_buf, c_thread_buf);

            // read B_sub_1
            b_thread_copy_.Run(BBlockDesc{},
                               make_tuple(k, I1, I0, I0),
                               b_block_buf,
                               b_thread_desc_,
                               make_tuple(I0, I1, I0, I0),
                               b_thread_buf);

            // read A_sub_1
            a_thread_copy_.Run(ABlockDesc{},
                               make_tuple(k, I1, I0, I0),
                               a_block_buf,
                               a_thread_desc_,
                               make_tuple(I0, I1, I0, I0),
                               a_thread_buf);

            // C_sub_00 += transpose(A_sub_0) * B_sub_0
            xdlops_gemm.template Run<decltype(a_thread_desc_),
                                     decltype(b_thread_desc_),
                                     decltype(c_thread_desc_),
                                     0,
                                     0>(a_thread_buf, b_thread_buf, c_thread_buf);

            // C_sub_01 += transpose(A_sub_0) * B_sub_1
            xdlops_gemm.template Run<decltype(a_thread_desc_),
                                     decltype(b_thread_desc_),
                                     decltype(c_thread_desc_),
                                     0,
                                     1>(a_thread_buf, b_thread_buf, c_thread_buf);
        });

        // C_sub_10 += transpose(A_sub_1) * B_sub_0
        xdlops_gemm.template Run<decltype(a_thread_desc_),
                                 decltype(b_thread_desc_),
                                 decltype(c_thread_desc_),
                                 1,
                                 0>(a_thread_buf, b_thread_buf, c_thread_buf);

        // C_sub_11 += transpose(A_sub_1) * B_sub_1
        xdlops_gemm.template Run<decltype(a_thread_desc_),
                                 decltype(b_thread_desc_),
                                 decltype(c_thread_desc_),
                                 1,
                                 1>(a_thread_buf, b_thread_buf, c_thread_buf);
    }

    private:
    // A[K, M]
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1, Number<MRepeat>{}, I1, Number<K1>{}));

    // B[K, N]
    static constexpr auto b_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1, Number<NRepeat>{}, I1, Number<K1>{}));

    static constexpr auto c_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{}, Number<NRepeat>{}));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         ABlockDesc,
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, K1>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         1, // K1,
                                                         1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         BBlockDesc,
                                                         decltype(b_thread_desc_),
                                                         Sequence<1, 1, 1, K1>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         1, // K1,
                                                         1>;

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;
};

} // namespace ck
#endif
