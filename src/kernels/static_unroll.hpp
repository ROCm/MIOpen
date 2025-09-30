/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef MIOPEN_STATIC_UNROLL_HPP
#define MIOPEN_STATIC_UNROLL_HPP

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

namespace miopen {

template <typename IndexType>
struct static_unroll_impl
{
    struct swallow
    {
        template <typename... Ts>
        __forceinline__ __host__ __device__ constexpr swallow(Ts&&...)
        {
        }
    };

    template <IndexType... Is>
    struct sequence
    {
    };

    template <typename Seq, typename... Seqs>
    struct sequence_merge
    {
        using type = typename sequence_merge<Seq, typename sequence_merge<Seqs...>::type>::type;
    };

    template <IndexType... Xs, IndexType... Ys>
    struct sequence_merge<sequence<Xs...>, sequence<Ys...>>
    {
        using type = sequence<Xs..., Ys...>;
    };

    template <typename Seq>
    struct sequence_merge<Seq>
    {
        using type = Seq;
    };

    template <IndexType NSize, typename F>
    struct sequence_gen
    {
        template <IndexType IBegin, IndexType NRemain, typename G>
        struct sequence_gen_impl
        {
            static constexpr IndexType NRemainLeft  = NRemain / 2;
            static constexpr IndexType NRemainRight = NRemain - NRemainLeft;
            static constexpr IndexType IMiddle      = IBegin + NRemainLeft;

            using type = typename sequence_merge<
                typename sequence_gen_impl<IBegin, NRemainLeft, G>::type,
                typename sequence_gen_impl<IMiddle, NRemainRight, G>::type>::type;
        };

        template <IndexType I, typename G>
        struct sequence_gen_impl<I, 1, G>
        {
            using type = sequence<G{}(I)>;
        };

        template <IndexType I, typename G>
        struct sequence_gen_impl<I, 0, G>
        {
            using type = sequence<>;
        };

        using type = typename sequence_gen_impl<0, NSize, F>::type;
    };

    // arithmetic sequence
    template <IndexType IBegin, IndexType IEnd, IndexType Increment>
    struct arithmetic_sequence_gen
    {
        struct F
        {
            constexpr IndexType operator()(IndexType i) const { return i * Increment + IBegin; }
        };

        using type0 = typename sequence_gen<(IEnd - IBegin) / Increment, F>::type;
        using type1 = sequence<>;

        static constexpr bool kHasContent =
            (Increment > 0 && IBegin < IEnd) || (Increment < 0 && IBegin > IEnd);

        using type = typename std::conditional<kHasContent, type0, type1>::type;
    };

    template <class>
    struct static_for_impl;

    template <IndexType... Is>
    struct static_for_impl<sequence<Is...>>
    {
        template <class F>
        __forceinline__ __host__ __device__ constexpr void operator()(F f) const
        {
            swallow{(f(Is), 0)...};
        }
    };

    template <IndexType NBegin, IndexType NEnd, IndexType Increment>
    struct static_for
    {
        static_assert(Increment != 0 && (NEnd - NBegin) % Increment == 0,
                      "Wrong! should satisfy (NEnd - NBegin) % Increment == 0");
        static_assert((Increment > 0 && NBegin <= NEnd) || (Increment < 0 && NBegin >= NEnd),
                      "Wrong! should (Increment > 0 && NBegin <= NEnd) || (Increment < 0 && "
                      "NBegin >= NEnd)");

        template <class F>
        __forceinline__ __host__ __device__ constexpr void operator()(F f) const
        {
            static_for_impl<typename arithmetic_sequence_gen<NBegin, NEnd, Increment>::type>{}(f);
        }
    };
};

template <typename ItemType, ItemType Start, ItemType End, ItemType Stride>
struct static_nounroll
{
    template <typename Func>
    __forceinline__ __host__ __device__ constexpr static_nounroll(Func&& f)
    {
        ItemType i = Start;
        while(i < static_cast<ItemType>(End))
        {
            f(i);
            i += static_cast<ItemType>(Stride);
        }
    }
};

template <typename ItemType, ItemType Start, ItemType End, ItemType Stride>
struct static_unroll_full
{
    static constexpr ItemType actual_end =
        (End - Start) % Stride == 0 ? End : ((End - Start) / Stride + 1) * Stride;
    template <typename F>
    __forceinline__ __host__ __device__ constexpr static_unroll_full(F f)
    {
        typename static_unroll_impl<ItemType>::template static_for<Start, actual_end, Stride>{}(f);
    }
};

template <typename ItemType, ItemType Start, ItemType End, ItemType Stride, ItemType Hint>
struct static_unroll_count
{
    static_assert(Hint > 0, "Hint must be a positive integer.");
    static constexpr ItemType unroll_end = Start + (Stride * Hint);

    template <typename F>
    __forceinline__ __host__ __device__ constexpr static_unroll_count(F f)
    {
        if constexpr(Hint == 1)
        {
            static_nounroll<ItemType, Start, End, Stride>{f};
        }
        else
        {
            static_unroll_full<ItemType, Start, unroll_end, Stride>{f};
            static_nounroll<ItemType, unroll_end, End, Stride>{f};
        }
    }
};

} // namespace miopen

#endif
