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

}

#endif
