
#include <unordered_map>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <type_traits>
#include <functional>
#include <iso646.h>

namespace args {

template<int N>
struct rank : rank<N-1> {};

template<>
struct rank<0> {};

namespace detail {


template<class T>
auto is_container(args::rank<1>, T&& x) -> decltype(
    x.insert(x.end(), *x.begin()), std::true_type{}
)
{
    return {};
}

template<class T>
std::false_type is_container(args::rank<0>, T&&)
{
    return {};
}

}

template<class T>
struct is_container
: decltype(detail::is_container(args::rank<1>{}, std::declval<T>()))
{};

template<class T>
struct value_parser
{
    static T apply(const std::string& x)
    {
        T result;
        std::stringstream ss;
        ss.str(x);
        ss >> result;
        return result;
    }
};


template<class T, typename std::enable_if<(not is_container<T>{} or std::is_convertible<T, std::string>{}), int>::type = 0>
void write_value(T& result, const std::string& x)
{
    result = value_parser<T>::apply(x);
}

template<class T, typename std::enable_if<(is_container<T>{} and not std::is_convertible<T, std::string>{}), int>::type = 0>
void write_value(T& result, const std::string& x)
{
    result.insert(result.end(), value_parser<typename T::value_type>::apply(x));
}

struct parse_visitor
{
    using string_map = std::unordered_map<std::string, std::vector<std::string>>;
    template<class T>
    void operator()(const string_map& data, T& x, std::string s) const
    {
        auto it = data.find(s);
        if (it != data.end())
            for(auto&& value:it->second) write_value(x, value);
    }
};

template<class T>
void parse(T& cmd, std::vector<std::string> args)
{
    std::unordered_map<std::string, std::vector<std::string>> data;

    std::string flag;
    for(auto&& x:args)
    {
        if (x[0] == '-')
        {
            flag = x;
        }
        else
        {
            data[flag].push_back(x);
        }
    }
    cmd.visit(std::bind(parse_visitor{}, std::ref(data), std::placeholders::_1, std::placeholders::_2));
    cmd.run();
}

template<class T>
void parse(int argc, char const *argv[])
{
    std::vector<std::string> as(argv+1, argv+argc);
    T cmd{};
    parse(cmd, as);
}

}

