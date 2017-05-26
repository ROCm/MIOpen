#ifndef GUARD_MLOPEN_FUNCTIONAL_HPP
#define GUARD_MLOPEN_FUNCTIONAL_HPP

#include <miopen/each_args.hpp>
#include <miopen/returns.hpp>
#include <utility>

namespace miopen { namespace detail {

template<class F, std::size_t ... Ns>
auto each_i_impl(F f, seq<Ns...>) MIOPEN_RETURNS
(f(std::integral_constant<std::size_t, Ns>{}...));

}

template<class F, class P>
struct by_t
{
    F f;
    P p;
    template<class... Ts>
    auto operator()(Ts&&... xs) const MIOPEN_RETURNS
    (f(p(std::forward<Ts>(xs))...))
};

template<class F, class P>
by_t<F, P> by(F f, P p)
{
    return {std::move(f), std::move(p)};
}

template<class F>
struct sequence_t
{
    F f;
    template<class IntegralConstant>
    auto operator()(IntegralConstant) const MIOPEN_RETURNS
    (detail::each_i_impl(f, typename detail::gens<IntegralConstant::value>::type{}));
};

template<class F>
sequence_t<F> sequence(F f)
{
    return {std::move(f)};
}

} // namespace miopen

#endif
