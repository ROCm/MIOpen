#ifndef GUARD_MLOPEN_REPLACE_HPP
#define GUARD_MLOPEN_REPLACE_HPP

#include <string>

namespace mlopen {

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

} // namespace mlopen

#endif
