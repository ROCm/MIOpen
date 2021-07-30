#ifndef CK_BLOCKWISE_GEMM_DLOPS_V3_HPP
#define CK_BLOCKWISE_GEMM_DLOPS_V3_HPP

#include "common_header.hpp"
#include "threadwise_gemm_dlops_v3.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename BlockMatrixA,
          typename BlockMatrixB,
          typename ThreadMatrixC,
          index_t KPerThread,
          index_t HPerThread,
          index_t WPerThread,
          index_t EPerThreadLoop,
          index_t ThreadGemmADataPerRead_K,
          index_t ThreadGemmBDataPerRead_W>
struct BlockwiseGemmDlops_km_kn_m0m1n0n1_v3
{
    struct MatrixIndex
    {
        index_t k;
        index_t h;
        index_t w;
    };

    // HACK: fix this @Jing Zhang
    static constexpr index_t KPerThreadSubC = 4;

    static constexpr auto a_thread_mtx_ = make_dynamic_naive_tensor_descriptor_packed_v2(
        make_tuple(Number<EPerThreadLoop>{}, Number<KPerThreadSubC>{}));

    static constexpr auto b_thread_mtx_ = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(
        Number<EPerThreadLoop>{}, Number<1>{}, Number<HPerThread>{}, Number<WPerThread>{}));

    static constexpr auto c_thread_mtx_ = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(
        Number<KPerThreadSubC>{}, Number<1>{}, Number<HPerThread>{}, Number<WPerThread>{}));

    using AThreadCopy =
        ThreadwiseDynamicTensorSliceTransfer_v4<FloatA,
                                                FloatA,
                                                BlockMatrixA,
                                                decltype(a_thread_mtx_),
                                                Sequence<EPerThreadLoop, KPerThreadSubC>,
                                                Sequence<0, 1>,
                                                1,
                                                ThreadGemmADataPerRead_K,
                                                1>;

    __device__ BlockwiseGemmDlops_km_kn_m0m1n0n1_v3()
        : c_thread_begin_mtx_idx_{GetBeginOfThreadMatrixC(get_thread_local_1d_id())},
          a_thread_copy_{make_tuple(0, c_thread_begin_mtx_idx_.k * KPerThread)}
    {
        static_assert(BlockMatrixA::IsKnownAtCompileTime() &&
                          BlockMatrixB::IsKnownAtCompileTime() &&
                          ThreadMatrixC::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        static_assert(BlockMatrixA{}.GetLength(I0) == BlockMatrixB{}.GetLength(I0),
                      "wrong! K dimension not consistent\n");

        constexpr index_t K = BlockMatrixA{}.GetLength(I1); // A is transposed
        constexpr index_t N = BlockMatrixB{}.GetLength(I1);
        constexpr index_t H = BlockMatrixB{}.GetLength(I2);
        constexpr index_t W = BlockMatrixB{}.GetLength(I3);

        static_assert(K % KPerThread == 0 && H % HPerThread == 0 && W % WPerThread == 0,
                      "wrong! Cannot evenly divide work among\n");

        constexpr auto KThreadCluster = K / KPerThread;
        constexpr auto HThreadCluster = H / HPerThread;
        constexpr auto WThreadCluster = W / WPerThread;

        static_assert(BlockSize == KThreadCluster * HThreadCluster * WThreadCluster,
                      "wrong! wrong blocksize\n");
    }

    __device__ static constexpr auto GetThreadMatrixCLengths()
    {
        return Sequence<KPerThread, 1, HPerThread, WPerThread>{};
    }

    __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t thread_id)
    {
        constexpr index_t H = BlockMatrixB{}.GetLength(Number<2>{});
        constexpr index_t W = BlockMatrixB{}.GetLength(Number<3>{});

        constexpr auto num_w_threads  = W / WPerThread;
        constexpr auto num_h_threads  = H / HPerThread;
        constexpr auto num_hw_threads = num_w_threads * num_h_threads;

        index_t k_thread_id  = thread_id / num_hw_threads;
        index_t hw_thread_id = thread_id % num_hw_threads;

        index_t h_thread_id = hw_thread_id / num_w_threads;
        index_t w_thread_id = hw_thread_id % num_w_threads;

        return MatrixIndex{k_thread_id, h_thread_id, w_thread_id};
    }

    template <typename ABlockBuffer, typename BThreadBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BThreadBuffer& b_thread_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        static_assert(is_same<remove_cv_t<remove_reference_t<typename ABlockBuffer::type>>,
                              remove_cv_t<remove_reference_t<FloatA>>>::value &&
                      is_same<remove_cv_t<remove_reference_t<typename BThreadBuffer::type>>,
                              remove_cv_t<remove_reference_t<FloatB>>>::value &&
                      is_same<remove_cv_t<remove_reference_t<typename CThreadBuffer::type>>,
                              remove_cv_t<remove_reference_t<FloatC>>>::value &&
                      "wrong! inconsistent type");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto a_block_mtx = BlockMatrixA{};

        constexpr auto EPerBlock = a_block_mtx.GetLength(I0);

        // HACK: fix this @Jing Zhang
        constexpr auto HoPerThreadSubC = 2;
        constexpr auto WoPerThreadSubC = 2;

        static_assert(KPerThread % KPerThreadSubC == 0, "");
        static_assert(HPerThread % HoPerThreadSubC == 0, "");
        static_assert(WPerThread % WoPerThreadSubC == 0, "");

        // thread A buffer for GEMM
        StaticBuffer<AddressSpaceEnum_t::Vgpr, FloatA, a_thread_mtx_.GetElementSpaceSize()>
            a_thread_buf;

        constexpr auto threadwise_gemm = ThreadwiseGemmDlops_km_kn_mn_v3<FloatA,
                                                                         FloatB,
                                                                         FloatC,
                                                                         decltype(a_thread_mtx_),
                                                                         decltype(b_thread_mtx_),
                                                                         decltype(c_thread_mtx_),
                                                                         HoPerThreadSubC,
                                                                         WoPerThreadSubC>{};

        static_for<0, EPerBlock, EPerThreadLoop>{}([&](auto e_begin) {
            static_for<0, KPerThread, KPerThreadSubC>{}([&](auto k_begin) {
                a_thread_copy_.Run(a_block_mtx,
                                   make_tuple(e_begin, k_begin),
                                   a_block_buf,
                                   a_thread_mtx_,
                                   make_tuple(I0, I0),
                                   a_thread_buf);

                static_for<0, HPerThread, HoPerThreadSubC>{}([&](auto h_begin) {
                    static_for<0, WPerThread, WoPerThreadSubC>{}([&](auto w_begin) {
                        threadwise_gemm.Run(a_thread_buf,
                                            make_tuple(I0, I0),
                                            b_thread_buf,
                                            make_tuple(e_begin, I0, h_begin, w_begin),
                                            c_thread_buf,
                                            make_tuple(k_begin, I0, h_begin, w_begin));
                    });
                });
            });
        });
    }

    template <typename ABlockSliceMoveStepIdx>
    __device__ void MoveASliceWindow(const BlockMatrixA&,
                                     const ABlockSliceMoveStepIdx& a_block_slice_move_step_idx)
    {
        a_thread_copy_.MoveSrcSliceWindow(BlockMatrixA{}, a_block_slice_move_step_idx);
    }

    private:
    MatrixIndex c_thread_begin_mtx_idx_;

    AThreadCopy a_thread_copy_;
};

} // namespace ck
#endif
