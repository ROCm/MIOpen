#ifndef CK_THREADWISE_GEMM_DLOPS_V3_HPP
#define CK_THREADWISE_GEMM_DLOPS_V3_HPP

#include "common_header.hpp"
#include "math.hpp"

namespace ck {

// C[M, N] += transpose(A[K, M]) * B[K, N]
//   Element of matrix can be vectorized data
// Assume:
//   1. ADesc, BDesc, CDesc are known at compile-time
//   2. AOriginIdx, BOriginIdx, COriginIdx are known at compile-time
template <typename FloatA,
          typename FloatB,
          typename FloatC,
          typename ADesc,
          typename BDesc,
          typename CDesc,
          index_t H,
          index_t W,
          typename enable_if<ADesc::IsKnownAtCompileTime() && BDesc::IsKnownAtCompileTime() &&
                                 CDesc::IsKnownAtCompileTime(),
                             bool>::type = false>
struct ThreadwiseGemmDlops_km_kn_mn_v3
{
    template <typename ABuffer,
              typename AOriginIdx,
              typename BBuffer,
              typename BOriginIdx,
              typename CBuffer,
              typename COriginIdx>
    __device__ static void Run(const ABuffer& a_buf,
                               AOriginIdx,
                               const BBuffer& b_buf,
                               BOriginIdx,
                               CBuffer& c_buf,
                               COriginIdx)
    {
        static_assert(ADesc::IsKnownAtCompileTime() && BDesc::IsKnownAtCompileTime() &&
                          CDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(
            is_known_at_compile_time<remove_cv_t<remove_reference_t<AOriginIdx>>>::value &&
                is_known_at_compile_time<remove_cv_t<remove_reference_t<BOriginIdx>>>::value &&
                is_known_at_compile_time<remove_cv_t<remove_reference_t<COriginIdx>>>::value,
            "wrong! AOriginIdx, BOriginIdx, COringinIdx should be known at compile-time");

        static_assert(is_same<remove_cv_t<remove_reference_t<typename ABuffer::type>>,
                              remove_cv_t<remove_reference_t<FloatA>>>::value &&
                      is_same<remove_cv_t<remove_reference_t<typename BBuffer::type>>,
                              remove_cv_t<remove_reference_t<FloatB>>>::value &&
                      is_same<remove_cv_t<remove_reference_t<typename CBuffer::type>>,
                              remove_cv_t<remove_reference_t<FloatC>>>::value &&
                      "wrong! inconsistent type");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr auto E = ADesc{}.GetLength(I0);
        constexpr auto K = ADesc{}.GetLength(I1);

        constexpr auto a_origin_idx = to_multi_index(AOriginIdx{});
        constexpr auto b_origin_idx = to_multi_index(BOriginIdx{});
        constexpr auto c_origin_idx = to_multi_index(COriginIdx{});

        static_for<0, E, 1>{}([&](auto e) {
            static_for<0, K, 1>{}([&](auto k) {
                constexpr index_t a_offset =
                    ADesc{}.CalculateOffset(a_origin_idx + make_tuple(e, k));

                if constexpr(H == 2 && W == 2)
                {
                    constexpr index_t b_offset_0 =
                        BDesc{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 0, 0));
                    constexpr index_t b_offset_1 =
                        BDesc{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 0, 1));
                    constexpr index_t b_offset_2 =
                        BDesc{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 1, 0));
                    constexpr index_t b_offset_3 =
                        BDesc{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 1, 1));

                    constexpr index_t c_offset_0 =
                        CDesc{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 0, 0));
                    constexpr index_t c_offset_1 =
                        CDesc{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 0, 1));
                    constexpr index_t c_offset_2 =
                        CDesc{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 1, 0));
                    constexpr index_t c_offset_3 =
                        CDesc{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 1, 1));

                    amd_assembly_outer_product_1x4(a_buf[Number<a_offset>{}],
                                                   b_buf[Number<b_offset_0>{}],
                                                   b_buf[Number<b_offset_1>{}],
                                                   b_buf[Number<b_offset_2>{}],
                                                   b_buf[Number<b_offset_3>{}],
                                                   c_buf(Number<c_offset_0>{}),
                                                   c_buf(Number<c_offset_1>{}),
                                                   c_buf(Number<c_offset_2>{}),
                                                   c_buf(Number<c_offset_3>{}));
                }
                else if constexpr(H == 4 && W == 1)
                {
                    constexpr index_t b_offset_0 =
                        BDesc{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 0, 0));
                    constexpr index_t b_offset_1 =
                        BDesc{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 1, 0));
                    constexpr index_t b_offset_2 =
                        BDesc{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 2, 0));
                    constexpr index_t b_offset_3 =
                        BDesc{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 3, 0));

                    constexpr index_t c_offset_0 =
                        CDesc{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 0, 0));
                    constexpr index_t c_offset_1 =
                        CDesc{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 1, 0));
                    constexpr index_t c_offset_2 =
                        CDesc{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 2, 0));
                    constexpr index_t c_offset_3 =
                        CDesc{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 3, 0));

                    amd_assembly_outer_product_1x4(a_buf[Number<a_offset>{}],
                                                   b_buf[Number<b_offset_0>{}],
                                                   b_buf[Number<b_offset_1>{}],
                                                   b_buf[Number<b_offset_2>{}],
                                                   b_buf[Number<b_offset_3>{}],
                                                   c_buf(Number<c_offset_0>{}),
                                                   c_buf(Number<c_offset_1>{}),
                                                   c_buf(Number<c_offset_2>{}),
                                                   c_buf(Number<c_offset_3>{}));
                }
                else
                {
                    static_for<0, H, 1>{}([&](auto h) {
                        static_for<0, W, 1>{}([&](auto w) {
                            constexpr index_t b_offset =
                                BDesc{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, h, w));

                            constexpr index_t c_offset =
                                CDesc{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, h, w));

#if 0
                            c_buf(Number<c_offset>{}) += inner_product_with_conversion<FloatC>{}(
                                a_buf[Number<a_offset>{}], b_buf[Number<b_offset>{}]);
#else
                            amd_assembly_inner_product(a_buf[Number<a_offset>{}],
                                                       b_buf[Number<b_offset>{}],
                                                       c_buf(Number<c_offset>{}));
#endif
                        });
                    });
                }
            });
        });
    }
};

} // namespace ck
#endif
