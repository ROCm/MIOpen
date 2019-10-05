#ifndef CK_THREADWISE_GEMM_HPP
#define CK_THREADWISE_GEMM_HPP

#include "common_header.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "math.hpp"

namespace ck {

template <typename Float, class Matrix>
__device__ void threadwise_matrix_set_zero(Matrix, Float* __restrict__ p_thread)
{
    for(index_t i = 0; i < Matrix::NRow(); ++i)
    {
        for(index_t j = 0; j < Matrix::NCol(); ++j)
        {
            const index_t id = Matrix::CalculateOffset(i, j);
            p_thread[id]     = Float(0);
        }
    }
}

template <typename SrcMatrix,
          typename DstMatrix,
          index_t NSliceRow,
          index_t NSliceCol,
          index_t DataPerAccess>
struct ThreadwiseMatrixSliceCopy
{
    __device__ constexpr ThreadwiseMatrixSliceCopy()
    {
        static_assert(SrcMatrix::RowStride() % DataPerAccess == 0 &&
                          DstMatrix::RowStride() % DataPerAccess == 0,
                      "wrong! wrong alignment");
        static_assert(NSliceCol % DataPerAccess == 0,
                      "wrong! should be NSliceCol % DataPerAccess == 0");
    }

    template <typename Data>
    __device__ static void Run(const Data* p_src, Data* p_dst)
    {
        using vector_t = typename vector_type<Data, DataPerAccess>::MemoryType;

        for(index_t i = 0; i < NSliceRow; ++i)
        {
            for(index_t j = 0; j < NSliceCol; j += DataPerAccess)
            {
                const index_t src_index = SrcMatrix::CalculateOffset(i, j);
                const index_t dst_index = DstMatrix::CalculateOffset(i, j);

                *reinterpret_cast<vector_t*>(&p_dst[dst_index]) =
                    *reinterpret_cast<const vector_t*>(&p_src[src_index]);
            }
        }
    }
};

// C += transpose(A) * B
//   Element of matrix can be vectorized data
template <typename MatrixA, typename MatrixB, typename MatrixC>
struct ThreadwiseGemmTransANormalBNormalC
{
    __device__ constexpr ThreadwiseGemmTransANormalBNormalC()
    {
        static_assert(MatrixA::NRow() == MatrixB::NRow() && MatrixA::NCol() == MatrixC::NRow() &&
                          MatrixB::NCol() == MatrixC::NCol(),
                      "wrong!");
    }

    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ static void Run_source(const FloatA* p_a, const FloatB* p_b, FloatC* p_c)
    {
        constexpr index_t M = MatrixC::NRow();
        constexpr index_t N = MatrixC::NCol();
        constexpr index_t K = MatrixA::NRow(); // A is transposed

        for(index_t k = 0; k < K; ++k)
        {
            for(index_t m = 0; m < M; ++m)
            {
                for(index_t n = 0; n < N; ++n)
                {
                    const index_t aindex = MatrixA::CalculateOffset(k, m); // A is transposed
                    const index_t bindex = MatrixB::CalculateOffset(k, n);
                    const index_t cindex = MatrixC::CalculateOffset(m, n);

                    p_c[cindex] +=
                        math::inner_product_with_conversion<FloatC>{}(p_a[aindex], p_b[bindex]);
                }
            }
        }
    }

#if CK_THREADWISE_GEMM_USE_AMD_INLINE_ASM
    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ static void Run_amd_asm(const FloatA* p_a, const FloatB* p_b, FloatC* p_c)
    {
        constexpr index_t M = MatrixC::NRow();
        constexpr index_t N = MatrixC::NCol();
        constexpr index_t K = MatrixA::NRow(); // A is transposed

        static_assert(N == 4 || N == 2, "wrong! this config not supported by asm yet");

        for(index_t k = 0; k < K; ++k)
        {
            for(index_t m = 0; m < M; ++m)
            {
                const index_t aindex = MatrixA::CalculateOffset(k, m); // A is transposed

                static_if<N == 2>{}([&](auto) {
                    const index_t bindex_0 = MatrixB::CalculateOffset(k, 0);
                    const index_t bindex_1 = MatrixB::CalculateOffset(k, 1);

                    const index_t cindex_0 = MatrixC::CalculateOffset(m, 0);
                    const index_t cindex_1 = MatrixC::CalculateOffset(m, 1);

                    __outer_product_1x2(
                        p_a[aindex], p_b[bindex_0], p_b[bindex_1], p_c[cindex_0], p_c[cindex_1]);
                });

                static_if<N == 4>{}([&](auto) {
                    const index_t bindex_0 = MatrixB::CalculateOffset(k, 0);
                    const index_t bindex_1 = MatrixB::CalculateOffset(k, 1);
                    const index_t bindex_2 = MatrixB::CalculateOffset(k, 2);
                    const index_t bindex_3 = MatrixB::CalculateOffset(k, 3);

                    const index_t cindex_0 = MatrixC::CalculateOffset(m, 0);
                    const index_t cindex_1 = MatrixC::CalculateOffset(m, 1);
                    const index_t cindex_2 = MatrixC::CalculateOffset(m, 2);
                    const index_t cindex_3 = MatrixC::CalculateOffset(m, 3);

                    __outer_product_1x4(p_a[aindex],
                                        p_b[bindex_0],
                                        p_b[bindex_1],
                                        p_b[bindex_2],
                                        p_b[bindex_3],
                                        p_c[cindex_0],
                                        p_c[cindex_1],
                                        p_c[cindex_2],
                                        p_c[cindex_3]);
                });
            }
        }
    }
#endif

    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ static void Run(const FloatA* p_a, const FloatB* p_b, FloatC* p_c)
    {
#if CK_THREADWISE_GEMM_USE_AMD_INLINE_ASM
        constexpr bool has_amd_asm = is_same<FloatC, float>{} &&
                                     ((is_same<FloatA, float>{} && is_same<FloatB, float>{}) ||
                                      (is_same<FloatA, half2_t>{} && is_same<FloatB, half2_t>{}) ||
                                      (is_same<FloatA, half4_t>{} && is_same<FloatB, half4_t>{}));

        static_if<has_amd_asm>{}([&](auto fwd) {
            Run_amd_asm(p_a, p_b, fwd(p_c));
        }).Else([&](auto) { Run_source(p_a, p_b, p_c); });
#else
        Run_source(p_a, p_b, p_c);
#endif
    }
};

} // namespace ck
#endif
