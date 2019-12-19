#ifndef CK_MULTI_INDEX_TRANSFORM_HPP
#define CK_MULTI_INDEX_TRANSFORM_HPP

#include "common_header.hpp"

namespace ck {

template <index_t N>
using MultiIndex = Array<index_t, N>;

template <typename... Xs>
__host__ __device__ constexpr auto make_multi_index(Xs... xs)
{
    return MultiIndex<sizeof...(Xs)>(xs...);
}

template <index_t Length>
struct PassThrough
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<1>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<1>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return Sequence<Length>{}; }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        return idx_up;
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        return idx_up_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool
    IsUpperIndexMappedToValidLowerIndex(const UpperIndex& /* idx_up */)
    {
        return true;
    }
};

// LowerLengths: Sequence<...>
template <typename LowerLengths, typename LeftPads, typename RightPads>
struct Pad
{
    static constexpr index_t nDim = LowerLengths::Size();

    using LowerIndex = MultiIndex<nDim>;
    using UpperIndex = MultiIndex<nDim>;

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDim>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<nDim>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        return LowerLengths{} + LeftPads{} + RightPads{};
    }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        return idx_up - LeftPads{};
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        return idx_up_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ constexpr bool
    IsUpperIndexMappedToValidLowerIndex(const UpperIndex& idx_up) const
    {
#if 0
        struct lambda_no_pad
        {
            __host__ __device__ constexpr bool operator()(index_t x) const { return x == 0; }
        };

        if(sequence_all_of(LeftPads{}, lambda_no_pad{}) &&
           sequence_all_of(RightPads{}, lambda_no_pad{}))
        {
            return true;
        }
        else
#endif
        {
            bool flag = true;

            static_for<0, nDim, 1>{}([&](auto idim) {
                // only check if there is left-padding
                static_if<(LeftPads::At(idim) != 0)>{}(
                    [&](auto) { flag = flag && idx_up[idim] >= LeftPads::At(idim); });

                // only check if there is right-padding
                static_if<(RightPads::At(idim) != 0)>{}([&](auto) {
                    flag = flag && (idx_up[idim] < LeftPads::At(idim) + LowerLengths::At(idim));
                });
            });

            return flag;
        }
    }
};

// LowerLengths: Sequence<...>
template <typename LowerLengths>
struct Merge
{
    static constexpr index_t nDimLow = LowerLengths::Size();
    static constexpr index_t nDimUp  = 1;

    using LowerIndex = MultiIndex<nDimLow>;
    using UpperIndex = MultiIndex<nDimUp>;

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<nDimUp>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        return Sequence<reduce_on_sequence(
            LowerLengths{}, math::multiplies<index_t>{}, Number<1>{})>{};
    }

    // emulate constexpr lambda
    template <typename PseudoLowStrides>
    struct lambda_CalculateLowerIndex
    {
        index_t& itmp;
        LowerIndex& idx_low;

        __host__ __device__ explicit constexpr lambda_CalculateLowerIndex(index_t& itmp_,
                                                                          LowerIndex& idx_low_)
            : itmp(itmp_), idx_low(idx_low_)
        {
        }

        template <typename IDim>
        __host__ __device__ constexpr void operator()(IDim idim) const
        {
            constexpr index_t stride = PseudoLowStrides::At(idim);
            idx_low(idim)            = itmp / stride;
            itmp -= idx_low[idim] * stride;
        }
    };

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        LowerIndex idx_low;

        index_t itmp = idx_up[0];

        constexpr auto pseudo_low_strides =
            reverse_inclusive_scan_sequence(
                LowerLengths::PopFront(), math::multiplies<index_t>{}, Number<1>{})
                .PushBack(Number<1>{});

        static_for<0, nDimLow - 1, 1>{}(
            lambda_CalculateLowerIndex<decltype(pseudo_low_strides)>(itmp, idx_low));

        idx_low(nDimLow - 1) = itmp / pseudo_low_strides[nDimLow - 1];

        return idx_low;
    }

    // idx_low_diff depends on idx_low_old, so idx_low need to be up-to-date
    // If idx_up_diff is known at compile-time, many calculations can be optimized
    // away by compiler
    // This function assume idx_low_old is not out-of-bound
    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& idx_low_old)
    {
        // do nothing if idx_up_diff == 0
        if(idx_up_diff[0] == 0)
        {
            return make_zero_array<index_t, nDimLow>();
        }

        // CalculateLowerIndex(idx_up_diff) has multiple integer divisions.
        //   If idx_up_diff is known at compile-time, the calculation can
        //   be done at compile-time. However, if idx_up_diff is only known
        //   at run-time, then the calculation will also be computed at
        //   run-time, and can be very expensive.
        LowerIndex idx_low_new = idx_low_old + CalculateLowerIndex(idx_up_diff);

        if(idx_up_diff[0] > 0)
        {
            bool carry = false;

            // do carry check in reversed order, starting from lowest dimension
            // don't check the highest dimension
            static_for<0, nDimLow - 1, 1>{}([&](auto ireverse) {
                constexpr index_t i = nDimLow - 1 - ireverse;

                if(carry)
                {
                    ++idx_low_new(i);
                }

                carry = false;

                if(idx_low_new[i] >= LowerLengths::At(i))
                {
                    idx_low_new(i) -= LowerLengths::At(i);
                    carry = true;
                }
            });

            // highest dimension, no out-of-bound check
            if(carry)
            {
                ++idx_low_new(0);
            }
        }
        else if(idx_up_diff[0] < 0)
        {
            bool borrow = false;

            // do borrow check in reversed order, starting from lowest dimension
            // don't check the highest dimension
            static_for<0, nDimLow - 1, 1>{}([&](auto ireverse) {
                constexpr index_t i = nDimLow - 1 - ireverse;

                if(borrow)
                {
                    --idx_low_new(i);
                }

                borrow = false;

                if(idx_low_new[i] < 0)
                {
                    idx_low_new(i) += LowerLengths::At(i);
                    borrow = true;
                }
            });

            // highest dimension, no out-of-bound check
            if(borrow)
            {
                --idx_low_new(0);
            }
        }

        return idx_low_new - idx_low_old;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool
    IsUpperIndexMappedToValidLowerIndex(const UpperIndex& /* idx_up */)
    {
        return true;
    }
};

// UpperLengths: Sequence<...>
template <typename UpperLengths>
struct UnMerge
{
    static constexpr index_t nDimLow = 1;
    static constexpr index_t nDimUp  = UpperLengths::Size();

    using LowerIndex = MultiIndex<nDimLow>;
    using UpperIndex = MultiIndex<nDimUp>;

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<nDimUp>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return UpperLengths{}; }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        LowerIndex idx_low{0};

        constexpr auto pseudo_up_strides =
            reverse_inclusive_scan_sequence(
                UpperLengths::PopFront(), math::multiplies<index_t>{}, Number<1>{})
                .PushBack(Number<1>{});

        static_for<0, nDimUp, 1>{}(
            [&](auto idim) { idx_low(0) += idx_up[idim] * pseudo_up_strides[idim]; });

        return idx_low;
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        return CalculateLowerIndex(idx_up_diff);
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool
    IsUpperIndexMappedToValidLowerIndex(const UpperIndex& /* idx_up */)
    {
        return true;
    }
};

// UpperLengths: Sequence<...>
// Coefficients: Sequence<...>
// idx_low = coefficients[0, ...nDimUp-1] * idx_up[0, ...nDimUp-1] + coefficients[nDimUp]
template <typename UpperLengths, typename Coefficients>
struct Embed
{
    static constexpr index_t nDimLow = 1;
    static constexpr index_t nDimUp  = UpperLengths::Size();

    using LowerIndex = MultiIndex<nDimLow>;
    using UpperIndex = MultiIndex<nDimUp>;

    __host__ __device__ explicit constexpr Embed()
    {
        static_assert(UpperLengths::GetSize() == nDimUp && Coefficients::GetSize() == nDimUp + 1,
                      "wrong! # of dimensions not consistent");
    }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<nDimUp>{}; }

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return UpperLengths{}; }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        LowerIndex idx_low(Coefficients{}[nDimUp]);

        static_for<0, nDimUp, 1>{}(
            [&](auto idim) { idx_low(0) += idx_up[idim] * Coefficients{}[idim]; });

        return idx_low;
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        LowerIndex idx_low_diff{0};

        static_for<0, nDimUp, 1>{}(
            [&](auto idim) { idx_low_diff(0) += idx_up_diff[idim] * Coefficients{}[idim]; });

        return idx_low_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool
    IsUpperIndexMappedToValidLowerIndex(const UpperIndex& /* idx_up */)
    {
        return true;
    }
};

template <index_t LowerLength, index_t VectorSize>
struct Vectorize
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    __host__ __device__ constexpr Vectorize()
    {
        static_assert(VectorSize > 0 && LowerLength % VectorSize == 0,
                      "wrong! cannot evenly divide");
    }

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<1>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<1>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        return Sequence<LowerLength / VectorSize>{};
    }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        return VectorSize * idx_up;
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        return VectorSize * idx_up_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool
    IsUpperIndexMappedToValidLowerIndex(const UpperIndex& /* idx_up */)
    {
        return true;
    }
};

} // namespace ck
#endif
