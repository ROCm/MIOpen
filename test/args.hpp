
#include <mlopen/each_args.hpp>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <type_traits>
#include <functional>
#include <cassert>
#include <iso646.h>

namespace args {

using string_map = std::unordered_map<std::string, std::vector<std::string>>;

template<class IsKeyword>
string_map parse(std::vector<std::string> as, IsKeyword is_keyword)
{
    string_map result;

    std::string flag;
    for(auto&& x:as)
    {
        if (is_keyword(x))
        {
            flag = x;
            result[flag]; // Ensure the flag exists
        }
        else
        {
            result[flag].push_back(x);
        }
    }
    return result;
}

template<int N>
struct rank : rank<N-1> {};

template<>
struct rank<0> {};

namespace detail {


template<class T>
auto is_container(args::rank<1>, T&& x) -> decltype(
    x.insert(x.end(), *x.begin()), std::true_type{}
);

template<class T>
std::false_type is_container(args::rank<0>, T&&);

template<class T, class U>
auto is_streamable(args::rank<1>, T&& x, U&& y) -> decltype(
    (x >> y), std::true_type{}
);

template<class T, class U>
std::false_type is_streamable(args::rank<0>, T&&, U&&);

}

template<class T>
struct is_container
: decltype(detail::is_container(args::rank<1>{}, std::declval<T>()))
{};

template<class T>
struct is_streamable
: decltype(detail::is_streamable(args::rank<1>{}, std::declval<std::istream>(), std::declval<typename std::add_lvalue_reference<T>::type>()))
{};

#define ARGS_REQUIRES(...) bool RequiresBool ## __LINE__ = true, typename std::enable_if<(RequiresBool ## __LINE__ && (__VA_ARGS__)), int>::type = 0

template<class T>
struct value_parser
{
    template<ARGS_REQUIRES(is_streamable<T>{} and not std::is_pointer<T>{})>
    static T apply(const std::string& x)
    {
        T result;
        std::stringstream ss;
        ss.str(x);
        ss >> result;
        return result;
    }
};

struct any_value
{
    std::string s;

    template<class T, class=decltype(value_parser<T>::apply(std::string{}))>
    operator T() const
    {
        return value_parser<T>::apply(s);
    }
};

template<class T, std::size_t ... Ns, class Data>
auto any_construct_impl(rank<1>, mlopen::detail::seq<Ns...>, const Data& d) -> decltype(T(any_value{d[Ns]}...))
{
    return T(any_value{d[Ns]}...);
}

template<class T, std::size_t ... Ns, class Data>
T any_construct_impl(rank<0>, mlopen::detail::seq<Ns...>, const Data&)
{
    std::abort(); // TODO: Throw exception
}

template<class T, std::size_t N, class Data>
T any_construct(const Data& d)
{
    return any_construct_impl<T>(rank<1>{}, typename mlopen::detail::gens<N>::type{}, d);
}

struct write_value
{
    template<class T>
    using is_multi_value = std::integral_constant<bool, (is_container<T>{} and not std::is_convertible<T, std::string>{})>;

    template<class Container, ARGS_REQUIRES(is_multi_value<Container>{})>
    void operator()(Container& result, std::vector<std::string> params) const
    {
        using value_type = typename Container::value_type;
        for(auto&& s:params)
        {
            result.insert(result.end(), value_parser<value_type>::apply(s));
        }
    }

    template<class T, ARGS_REQUIRES(not is_multi_value<T>{} and is_streamable<T>{})>
    void operator()(T& result, std::vector<std::string> params) const
    {
        assert(params.size() > 0);
        result = value_parser<T>::apply(params.back());
    }

    template<class T, ARGS_REQUIRES(not is_multi_value<T>{} and not is_streamable<T>{})>
    void operator()(T& result, std::vector<std::string> params) const
    {
        switch(params.size())
        {
            case 0: { result = any_construct<T, 0>(params); break; }
            case 1: { result = any_construct<T, 1>(params); break; }
            case 2: { result = any_construct<T, 2>(params); break; }
            case 3: { result = any_construct<T, 3>(params); break; }
            case 4: { result = any_construct<T, 4>(params); break; }
            case 5: { result = any_construct<T, 5>(params); break; }
            case 6: { result = any_construct<T, 6>(params); break; }
            case 7: { result = any_construct<T, 7>(params); break; }
            default:
                std::abort(); // TODO: Throw exception
        }
    }
};

}

