#ifndef GUARD_MIOPEN_REPLACE_HPP
#define GUARD_MIOPEN_REPLACE_HPP

#include <string>
#include <numeric>
#include <algorithm>

namespace miopen {

inline std::string ReplaceString(std::string subject, const std::string& search, const std::string& replace) 
{
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
    return subject;
}

inline bool EndsWith(const std::string& value, const std::string& suffix)
{
    if (suffix.size() > value.size()) return false;
    else return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

template<class Strings>
inline std::string JoinStrings(Strings strings, std::string delim)
{
    auto it = strings.begin();
    if (it == strings.end()) return "";

    auto nit = std::next(it);
    return std::accumulate(nit, strings.end(), *it, [&](std::string x, std::string y)
    {
        return x + delim + y;
    });
}

template<class F>
inline std::string TransformString(std::string s, F f)
{
    std::transform(s.begin(), s.end(), s.begin(), f);
    return s;
}

inline std::string ToUpper(std::string s)
{
    return TransformString(std::move(s), ::toupper);
}

inline bool StartsWith(const std::string& value, const std::string& prefix)
{
    if (prefix.size() > value.size()) return false;
    else return std::equal(prefix.begin(), prefix.end(), value.begin());
}

inline std::string RemovePrefix(std::string s, std::string prefix)
{
    if (StartsWith(s, prefix)) return s.substr(prefix.length());
    else return s;
}

} // namespace miopen

#endif
