#ifndef CK_THREADWISE_GEMM_HPP
#define CK_THREADWISE_GEMM_HPP

#include "common_header.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "float_types.h"

namespace ck {

template <class Float, class Matrix>
__device__ void threadwise_matrix_set_zero(Matrix, Float* __restrict__ p_thread)
{
    for(index_t i = 0; i < Matrix::NRow(); ++i)
    {
        for(index_t j = 0; j < Matrix::NCol(); ++j)
        {
            const index_t id = Matrix::GetOffsetFromMultiIndex(i, j);
            p_thread[id]     = Float(0);
        }
    }
}

template <class Float,
          class SrcMatrix,
          class DstMatrix,
          index_t NRow,
          index_t NCol,
          index_t DataPerRead>
__device__ void threadwise_matrix_copy(SrcMatrix,
                                       const Float* __restrict__ p_src,
                                       DstMatrix,
                                       Float* __restrict__ p_dst,
                                       Sequence<NRow, NCol>,
                                       Number<DataPerRead>)
{
    static_assert(NCol % DataPerRead == 0, "wrong! should be NCol % == DataPerRead == 0");

    constexpr auto src_mtx = SrcMatrix{};
    constexpr auto dst_mtx = DstMatrix{};

    // Depending upon datatype i.e float/half/bfloat16, carry out data movement
    // in appropriate vector_typeized form
    // float - 4, half - 4, bfloat16 - 2
    static_if<std::is_same<Float, float>::value>{}([&](auto) {
        using vector_t = typename vector_type<float, DataPerRead>::MemoryType;

        for(index_t i = 0; i < NRow; ++i)
        {
            for(index_t j = 0; j < NCol; j += DataPerRead)
            {
                const index_t src_index = src_mtx.GetOffsetFromMultiIndex(i, j);
                const index_t dst_index = dst_mtx.GetOffsetFromMultiIndex(i, j);

                *reinterpret_cast<vector_t*>(&p_dst[dst_index]) =
                    *reinterpret_cast<const vector_t*>(&p_src[src_index]);
            }
        }

    }).Else([&](auto) { // fp16/bfp16
        for(index_t i = 0; i < NRow; ++i)
        {
            for(index_t j = 0; j < NCol; ++j)
            {
                const index_t src_index = src_mtx.GetOffsetFromMultiIndex(i, j);
                const index_t dst_index = dst_mtx.GetOffsetFromMultiIndex(i, j);

                *reinterpret_cast<Float*>(&p_dst[dst_index]) =
                    *reinterpret_cast<const Float*>(&p_src[src_index]);
            }
        }
    });
}

template <class Accum>
struct inner_product_with_conversion
{
    __device__ Accum operator()(float a, float b) const
    {
        return CVT_FLOAT2ACCUM(a) * CVT_FLOAT2ACCUM(b);
    }

    __device__ Accum operator()(const vector_type<half, 2>::MemoryType& a,
                                const vector_type<half, 2>::MemoryType& b) const
    {
        const half* p_a_half = reinterpret_cast<const half*>(&a);
        const half* p_b_half = reinterpret_cast<const half*>(&b);

        Accum acc = 0.0;
        for(index_t v = 0; v < 2; ++v)
        {
            acc += CVT_FLOAT2ACCUM(p_a_half[v]) * CVT_FLOAT2ACCUM(p_b_half[v]);
        }

        return acc;
    }

    __device__ Accum operator()(const vector_type<half, 4>::MemoryType& a,
                                const vector_type<half, 4>::MemoryType& b) const
    {
        const half* p_a_half = reinterpret_cast<const half*>(&a);
        const half* p_b_half = reinterpret_cast<const half*>(&b);

        Accum acc = 0.0;
        for(index_t v = 0; v < 4; ++v)
        {
            acc += CVT_FLOAT2ACCUM(p_a_half[v]) * CVT_FLOAT2ACCUM(p_b_half[v]);
        }
        return acc;
    }

    __device__ Accum operator()(const vector_type<ushort, 2>::MemoryType& a,
                                const vector_type<ushort, 2>::MemoryType& b) const
    {
        const ushort* p_a_bfloat16 = reinterpret_cast<const ushort*>(&a);
        const ushort* p_b_bfloat16 = reinterpret_cast<const ushort*>(&b);

        Accum acc = 0.0;
        for(index_t v = 0; v < 2; ++v)
        {
            acc += CVT_FLOAT2ACCUM(p_a_bfloat16[v]) * CVT_FLOAT2ACCUM(p_b_bfloat16[v]);
        }

        return acc;
    }

    __device__ Accum operator()(const vector_type<ushort, 4>::MemoryType& a,
                                const vector_type<ushort, 4>::MemoryType& b) const
    {
        const ushort* p_a_bfloat16 = reinterpret_cast<const ushort*>(&a);
        const ushort* p_b_bfloat16 = reinterpret_cast<const ushort*>(&b);

        Accum acc = 0.0;
        for(index_t v = 0; v < 4; ++v)
        {
            acc += CVT_FLOAT2ACCUM(p_a_bfloat16[v]) * CVT_FLOAT2ACCUM(p_b_bfloat16[v]);
        }
        return acc;
    }
};

template <class MatrixA,
          class MatrixB,
          class MatrixC,
          bool TransA,
          bool TransB,
          bool TransC,
          class FloatA,
          class FloatB,
          class FloatC>
__device__ void threadwise_gemm(MatrixA,
                                integral_constant<bool, TransA>,
                                const FloatA* __restrict__ p_a_thread,
                                MatrixB,
                                integral_constant<bool, TransB>,
                                const FloatB* __restrict__ p_b_thread,
                                MatrixC,
                                integral_constant<bool, TransC>,
                                FloatC* __restrict__ p_c_thread)
{
    static_if<TransA && (!TransB) && (!TransC)>{}([&](auto) {
        constexpr auto a_mtx = MatrixA{};
        constexpr auto b_mtx = MatrixB{};
        constexpr auto c_mtx = MatrixC{};

        constexpr index_t M = c_mtx.NRow();
        constexpr index_t N = c_mtx.NCol();
        constexpr index_t K = a_mtx.NRow(); // A is transposed

        for(index_t k = 0; k < K; ++k)
        {
            for(index_t i = 0; i < M; ++i)
            {
                for(index_t j = 0; j < N; ++j)
                {
                    const index_t aindex = a_mtx.GetOffsetFromMultiIndex(k, i); // A is transposed
                    const index_t bindex = b_mtx.GetOffsetFromMultiIndex(k, j);
                    const index_t cindex = c_mtx.GetOffsetFromMultiIndex(i, j);

                    p_c_thread[cindex] += inner_product_with_conversion<FloatC>{}(
                        p_a_thread[aindex], p_b_thread[bindex]);
                }
            }
        }
    }).Else([&](auto fwd) {
        // not implemented
        static_assert(fwd(false), "wrong! support for this config is not implemented");
    });
}

} // namespace ck
#endif
