#ifndef CK_FUNCTIONAL2_HPP
#define CK_FUNCTIONAL2_HPP

#include "functional.hpp"
#include "Sequence.hpp"

namespace ck {

template <class>
struct static_for_impl;

template <index_t... Is>
struct static_for_impl<Sequence<Is...>>
{
    template <class F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        swallow{(f(Number<Is>{}), 0)...};
    }
};

// F signature: F(Number<Iter>)
template <index_t NBegin, index_t NEnd, index_t Increment>
struct static_for
{
    __host__ __device__ constexpr static_for()
    {
        static_assert(NBegin <= NEnd, "wrongs! should have NBegin <= NEnd");
        static_assert((NEnd - NBegin) % Increment == 0,
                      "Wrong! should satisfy (NEnd - NBegin) % Increment == 0");
    }

    template <class F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        static_for_impl<typename arithmetic_sequence_gen<NBegin, NEnd, Increment>::type>{}(f);
    }
};

template <class Seq, class Reduce>
struct lambda_accumulate_on_sequence
{
    const Reduce& f;
    index_t& result;

    __host__ __device__ constexpr lambda_accumulate_on_sequence(const Reduce& f_, index_t& result_)
        : f(f_), result(result_)
    {
    }

    template <class IDim>
    __host__ __device__ constexpr index_t operator()(IDim) const
    {
        return result = f(result, Seq::Get(IDim{}));
    }
};

template <class Seq, class Reduce, index_t Init>
__host__ __device__ constexpr index_t
accumulate_on_sequence(Seq, Reduce f, Number<Init> /*initial_value*/)
{
    index_t result = Init;

    static_for<0, Seq::mSize, 1>{}(lambda_accumulate_on_sequence<Seq, Reduce>(f, result));

    return result;
}

} // namespace ck
#endif
