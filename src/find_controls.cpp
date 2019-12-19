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

#include <miopen/find_controls.hpp>
#include <miopen/logger.hpp>
#include <miopen/env.hpp>
#include <miopen/solver_id.hpp>

#include <ostream>
#include <cstdlib>
#include <cstring>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_FIND_ENFORCE)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_FIND_ENFORCE_SCOPE)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_FIND_ONLY_SOLVER)

namespace miopen {

namespace debug {

bool FindEnforceDisable = false;

} // namespace debug

namespace {

const char* ToCString(const FindEnforceAction mode)
{
    switch(mode)
    {
    case FindEnforceAction::None: return "NONE";
    case FindEnforceAction::DbUpdate: return "DB_UPDATE";
    case FindEnforceAction::Search: return "SEARCH";
    case FindEnforceAction::SearchDbUpdate: return "SEARCH_DB_UPDATE";
    case FindEnforceAction::DbClean: return "CLEAN";
    }
    return "<Unknown>";
}

FindEnforceAction GetFindEnforceActionImpl()
{
    const char* const p_asciz = miopen::GetStringEnv(MIOPEN_FIND_ENFORCE{});
    if(p_asciz == nullptr)
        return FindEnforceAction::Default_;
    std::string str = p_asciz;
    for(auto& c : str)
        c = toupper(static_cast<unsigned char>(c));
    if(str == "NONE")
        return FindEnforceAction::None;
    else if(str == "DB_UPDATE")
        return FindEnforceAction::DbUpdate;
    else if(str == "SEARCH")
        return FindEnforceAction::Search;
    else if(str == "SEARCH_DB_UPDATE")
        return FindEnforceAction::SearchDbUpdate;
    else if(str == "DB_CLEAN")
        return FindEnforceAction::DbClean;
    else
    { // Nop. Fall down & try numerics.
    }
    const auto val = static_cast<FindEnforceAction>(miopen::Value(MIOPEN_FIND_ENFORCE{}));
    if(FindEnforceAction::First_ <= val && val <= FindEnforceAction::Last_)
        return val;
    MIOPEN_LOG_NQE("Wrong MIOPEN_FIND_ENFORCE, using default.");
    return FindEnforceAction::Default_;
}

FindEnforceAction GetFindEnforceAction()
{
    static const FindEnforceAction val = GetFindEnforceActionImpl();
    return val;
}

const char* ToCString(const FindEnforceScope mode)
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
    if(p_asciz == nullptr)
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
    const auto val = static_cast<FindEnforceScope>(miopen::Value(MIOPEN_FIND_ENFORCE_SCOPE{}));
    if(FindEnforceScope::First_ <= val && val <= FindEnforceScope::Last_)
        return val;
    MIOPEN_LOG_NQE("Wrong MIOPEN_FIND_ENFORCE_SCOPE, using default.");
    return FindEnforceScope::Default_;
}

FindEnforceScope GetFindEnforceScope()
{
    static const FindEnforceScope val = GetFindEnforceScopeImpl();
    return val;
}

solver::Id GetEnvFindOnlySolverImpl()
{
    static_assert(miopen::solver::Id::invalid_value == 0, "miopen::solver::Id::invalid_value == 0");
    const char* const p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_FIND_ONLY_SOLVER{});
    if(p_asciz != nullptr && strlen(p_asciz) > 0)
    {
        auto numeric_id = std::strtoul(p_asciz, nullptr, 10);
        if(errno == ERANGE || numeric_id == 0)
        { // Assume string in the environment. Try to convert it to numeric id.
            errno      = 0;
            numeric_id = solver::Id{p_asciz}.Value();
        }
        else
        { // Non-zero value, assume numeric id. Check if it denotes a solver.
            numeric_id = solver::Id{solver::Id{numeric_id}.ToString()}.Value();
        }
        if(numeric_id != 0)
            MIOPEN_LOG_NQI(numeric_id);
        else
            MIOPEN_THROW(miopenStatusBadParm, "Invalid value of MIOPEN_DEBUG_FIND_ONLY_SOLVER");
        return {numeric_id};
    }
    return {};
}

} // namespace

FindEnforce::FindEnforce()
{
    action = GetFindEnforceAction();
    scope  = GetFindEnforceScope();
}

std::ostream& operator<<(std::ostream& os, const FindEnforce& val)
{
    return os << ToCString(val.action) << "(" << static_cast<int>(val.action) << "), "
              << ToCString(val.scope) << "(" << static_cast<int>(val.scope) << ')';
}

solver::Id GetEnvFindOnlySolver()
{
    static const auto once = GetEnvFindOnlySolverImpl();
    return once;
}

} // namespace miopen
