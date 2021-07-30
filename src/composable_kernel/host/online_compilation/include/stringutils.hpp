/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_OLC_STRINGUTILS_HPP
#define GUARD_OLC_STRINGUTILS_HPP

#include <algorithm>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>
#include <sstream>

#define OLC_STRINGIZE_1(...) #__VA_ARGS__
#define OLC_STRINGIZE(...) OLC_STRINGIZE_1(__VA_ARGS__)

namespace olCompile {

inline std::string
ReplaceString(std::string subject, const std::string& search, const std::string& replace)
{
    size_t pos = 0;
    while((pos = subject.find(search, pos)) != std::string::npos)
    {
        subject.replace(pos, search.length(), replace);
        pos += replace.length();
    }
    return subject;
}

inline bool EndsWith(const std::string& value, const std::string& suffix)
{
    if(suffix.size() > value.size())
        return false;
    else
        return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

template <class Strings>
inline std::string JoinStrings(Strings strings, std::string delim)
{
    auto it = strings.begin();
    if(it == strings.end())
        return "";

    auto nit = std::next(it);
    return std::accumulate(
        nit, strings.end(), *it, [&](std::string x, std::string y) { return x + delim + y; });
}

template <class F>
static inline std::string TransformString(std::string s, F f)
{
    std::transform(s.begin(), s.end(), s.begin(), f);
    return s;
}

inline std::string ToUpper(std::string s) { return TransformString(std::move(s), ::toupper); }

inline bool StartsWith(const std::string& value, const std::string& prefix)
{
    if(prefix.size() > value.size())
        return false;
    else
        return std::equal(prefix.begin(), prefix.end(), value.begin());
}

inline std::string RemovePrefix(std::string s, std::string prefix)
{
    if(StartsWith(s, prefix))
        return s.substr(prefix.length());
    else
        return s;
}

inline std::vector<std::string> SplitSpaceSeparated(const std::string& in)
{
    std::istringstream ss(in);
    std::istream_iterator<std::string> begin(ss), end;
    return {begin, end};
}

inline std::vector<std::string> SplitSpaceSeparated(const std::string& in,
                                                    const std::vector<std::string>& dontSplitAfter)
{
    std::vector<std::string> rv;
    std::istringstream ss(in);
    std::string s;
    while(ss >> s)
    {
        if(std::any_of(dontSplitAfter.begin(), dontSplitAfter.end(), [&](const auto& dont) {
               return dont == s;
           }))
        {
            std::string s2;
            if(ss >> s2)
            {
                s += std::string(" ").append(s2); // Exactly one space is important.
                rv.push_back(s);
                continue;
            }
            throw std::runtime_error("Error parsing string: '" + in + '\'');
        }
        rv.push_back(s);
    }
    return rv;
}

} // namespace olCompile

#endif // GUARD_OLC_STRINGUTILS_HPP
