#ifndef GUARD_MLOPEN_EACH_ARGS_HPP
#define GUARD_MLOPEN_EACH_ARGS_HPP

#include <initializer_list>
#include <type_traits>
#include <utility>
 
namespace mlopen { namespace detail {

template<std::size_t ...>
struct seq 
{
    typedef seq type;
};

template <class, class>
struct merge_seq;

template <std::size_t... Xs, std::size_t... Ys>
struct merge_seq<seq<Xs...>, seq<Ys...>>
: seq<Xs..., (sizeof...(Xs)+Ys)...>
{};

template<std::size_t N>
struct gens 
: merge_seq<
    typename gens<N/2>::type,
    typename gens<N - N/2>::type
> 
{};

template<> struct gens<0> : seq<> {}; 
template<> struct gens<1> : seq<0> {};

template<class F, std::size_t ... Ns, class... Ts>
void each_args_i_impl(F f, seq<Ns...>, Ts&&... xs)
{
    (void)std::initializer_list<int>{(f(std::integral_constant<std::size_t, Ns>{}, std::forward<Ts>(xs)), 0)...};
}

}

template<class F, class... Ts>
void each_args_i(F f, Ts&&... xs)
{
    detail::each_args_i_impl(f, typename detail::gens<sizeof...(Ts)>::type{}, std::forward<Ts>(xs)...);
}

template<class F, class... Ts>
void each_args(F f, Ts&&... xs)
{
    (void)std::initializer_list<int>{(f(std::forward<Ts>(xs)), 0)...};
}

}

#endif
