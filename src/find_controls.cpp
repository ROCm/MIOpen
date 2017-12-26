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

#include <ostream>

#include <miopen/find_controls.hpp>
#include <miopen/logger.hpp>
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_FIND_ENFORCE)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_FIND_ENFORCE_SCOPE)

namespace miopen {

namespace {

template<typename T>
bool operator<=(const T& lhs, const int& rhs)
{
    return static_cast<int>(lhs) <= rhs;
}

template<typename T>
bool operator<=(const int& lhs, const T& rhs)
{
    return lhs <= static_cast<int>(rhs);
}

const char* FindEnforce2CString(const FindEnforce mode)
{
    switch(mode)
    {
    case FindEnforce::None: return "NONE";
    case FindEnforce::DbUpdate: return "DB_UPDATE";
    case FindEnforce::Search: return "SEARCH";
    case FindEnforce::SearchDbUpdate: return "SEARCH_DB_UPDATE";
    case FindEnforce::DbClean: return "CLEAN";
    }
    return "<Unknown>";
}

FindEnforce GetFindEnforceImpl()
{
    const char* const p_asciz = miopen::GetStringEnv(MIOPEN_FIND_ENFORCE{});
    if(!p_asciz)
        return FindEnforce::Default_;
    std::string str = p_asciz;
    for(auto& c : str)
        c = toupper(static_cast<unsigned char>(c));
    if(str == "NONE")
        return FindEnforce::None;
    else if(str == "DB_UPDATE")
        return FindEnforce::DbUpdate;
    else if(str == "SEARCH")
        return FindEnforce::Search;
    else if(str == "SEARCH_DB_UPDATE")
        return FindEnforce::SearchDbUpdate;
    else if(str == "DB_CLEAN")
        return FindEnforce::DbClean;
    else
    { // Nop. Fall down & try numerics.
    }
    const int val = miopen::Value(MIOPEN_FIND_ENFORCE{});
    if(FindEnforce::First_ <= val && val <= FindEnforce::Last_)
        return static_cast<FindEnforce>(val);
    MIOPEN_LOG_E("Wrong MIOPEN_FIND_ENFORCE, using default.");
    return FindEnforce::Default_;
}

} // namespace

FindEnforce GetFindEnforce()
{
    static const FindEnforce val = GetFindEnforceImpl();
    return val;
}

std::ostream& operator<<(std::ostream& os, const FindEnforce sm)
{
    return os << FindEnforce2CString(sm) << " (" << static_cast<int>(sm) << ')';
}

namespace {

const char* FindEnforceScope2CString(const FindEnforceScope mode)
{
    switch(mode)
    {
    case FindEnforceScope::All: return "ALL";
    case FindEnforceScope::ConvFwd: return "CONV_FWD";
    case FindEnforceScope::ConvBwd: return "CONV_BWD";
    case FindEnforceScope::ConvWrW: return "CONV_WRW";
    }
    return "<Unknown>";
}

FindEnforceScope GetFindEnforceScopeImpl()
{
    const char* const p_asciz = miopen::GetStringEnv(MIOPEN_FIND_ENFORCE_SCOPE{});
    if(!p_asciz)
        return FindEnforceScope::Default_;
    std::string str = p_asciz;
    for(auto& c : str)
        c = toupper(static_cast<unsigned char>(c));
    if(str == "ALL")
        return FindEnforceScope::All;
    else if(str == "CONV_FWD")
        return FindEnforceScope::ConvFwd;
    else if(str == "CONV_BWD")
        return FindEnforceScope::ConvBwd;
    else if(str == "CONV_WRW")
        return FindEnforceScope::ConvWrW;
    else
    { // Nop. Fall down & try numerics.
    }
    const int val = miopen::Value(MIOPEN_FIND_ENFORCE_SCOPE{});
    if(FindEnforceScope::First_ <= val && val <= FindEnforceScope::Last_)
        return static_cast<FindEnforceScope>(val);
    MIOPEN_LOG_E("Wrong MIOPEN_FIND_ENFORCE_SCOPE, using default.");
    return FindEnforceScope::Default_;
}

} // namespace

FindEnforceScope GetFindEnforceScope()
{
    static const FindEnforceScope val = GetFindEnforceScopeImpl();
    return val;
}

std::ostream& operator<<(std::ostream& os, const FindEnforceScope sm)
{
    return os << FindEnforceScope2CString(sm) << " (" << static_cast<int>(sm) << ')';
}

} // namespace miopen
