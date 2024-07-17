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

#include <miopen/miopen.h>
#include <miopen/miopen_internal.h>
#include <miopen/logger.hpp>
#include <miopen/env.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/stringutils.hpp>

#include <boost/optional.hpp>

#include <ostream>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <optional>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_FIND_ENFORCE)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEBUG_FIND_ONLY_SOLVER)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_FIND_MODE)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_FIND_MODE_FUSION)

namespace miopen {

namespace debug {

// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
MIOPEN_EXPORT bool FindEnforceDisable = false;

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
    auto str = env::value(MIOPEN_FIND_ENFORCE);
    if(str.empty())
        return FindEnforceAction::Default_;
    for(auto& c : str)
        c = toupper(static_cast<unsigned char>(c));
    if(str == "NONE")
    {
        return FindEnforceAction::None;
    }
    else if(str == "DB_UPDATE")
    {
        return FindEnforceAction::DbUpdate;
    }
    else if(str == "SEARCH")
    {
        return FindEnforceAction::Search;
    }
    else if(str == "SEARCH_DB_UPDATE")
    {
        return FindEnforceAction::SearchDbUpdate;
    }
    else if(str == "DB_CLEAN")
    {
        return FindEnforceAction::DbClean;
    }
    else
    { // Nop. Fall down & try numerics.
    }
    const auto val = static_cast<FindEnforceAction>(stoul(str));
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

boost::optional<std::vector<solver::Id>> GetEnvFindOnlySolverImpl()
{
    static_assert(miopen::solver::Id::invalid_value == 0, "miopen::solver::Id::invalid_value == 0");
    const auto slv_str = env::value(MIOPEN_DEBUG_FIND_ONLY_SOLVER);
    std::vector<solver::Id> res;
    if(!slv_str.empty())
    {
        const auto solver_list = miopen::SplitDelim(slv_str, ';');
        for(const auto& kinder : solver_list)
        {
            auto numeric_id = std::strtoul(kinder.c_str(), nullptr, 10);
            if(errno == ERANGE || numeric_id == 0)
            { // Assume string in the environment. Try to convert it to numeric id.
                errno      = 0;
                numeric_id = solver::Id{kinder}.Value();
            }
            else
            { // Non-zero value, assume numeric id. Check if it denotes a solver.
                numeric_id = solver::Id{solver::Id{numeric_id}.ToString()}.Value();
            }
            if(numeric_id != 0)
            {
                MIOPEN_LOG_NQI(numeric_id);
            }
            else
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Invalid value of MIOPEN_DEBUG_FIND_ONLY_SOLVER: " + kinder);
            }
            const auto id = solver::Id{numeric_id};
            if(id.IsValid())
            {
                res.push_back(id);
            }
            else
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Invalid MIOPEN_DEBUG_FIND_ONLY_SOVER specified: " + kinder);
            }
        }
    }
    if(res.empty())
        return boost::none;
    else
        return {res};
}

} // namespace

FindEnforce::FindEnforce() { action = GetFindEnforceAction(); }

std::ostream& operator<<(std::ostream& os, const FindEnforce& val)
{
    return os << ToCString(val.action) << '(' << static_cast<int>(val.action) << ')';
}

boost::optional<std::vector<solver::Id>> GetEnvFindOnlySolver()
{
    if(miopen::debug::IsWarmupOngoing)
        return boost::none;
    static const auto once = GetEnvFindOnlySolverImpl();
    return once;
}

namespace {

const char* ToCString(const FindMode::Values mode)
{
    switch(mode)
    {
    case FindMode::Values::Normal: return "NORMAL";
    case FindMode::Values::Fast: return "FAST";
    case FindMode::Values::Hybrid: return "HYBRID";
    case FindMode::Values::DeprecatedFastHybrid: break;
    case FindMode::Values::DynamicHybrid: return "DYNAMIC_HYBRID";
    case FindMode::Values::End_: break;
    }
    return "<Unknown>";
}

std::ostream& operator<<(std::ostream& os, const FindMode::Values& v)
{
    return os << ToCString(v) << "(" << static_cast<int>(v) << ')';
}

template <class Variable>
std::optional<FindMode::Values> GetFindModeValueImpl2(Variable variable)
{
    auto str = env::value(variable);
    if(str.empty())
        return std::nullopt;
    for(auto& c : str)
        c = toupper(static_cast<unsigned char>(c));
    if(str == "NORMAL")
    {
        return FindMode::Values::Normal;
    }
    else if(str == "FAST")
    {
        return FindMode::Values::Fast;
    }
    else if(str == "HYBRID")
    {
        return FindMode::Values::Hybrid;
    }
    else if(str == "DYNAMIC_HYBRID")
    {
        return FindMode::Values::DynamicHybrid;
    }
    else
    { // Nop. Fall down & try numerics.
    }
    const auto val = static_cast<FindMode::Values>(stoul(str));
    if(FindMode::Values::Begin_ <= val && val < FindMode::Values::End_)
        return val;
    MIOPEN_LOG_NQE("Wrong " << variable.GetName() << ", using default.");
    return std::nullopt;
}

template <class Variable>
FindMode::Values GetFindModeValue(Variable variable, FindMode::Values defaultValue)
{
    static const FindMode::Values val = [&]() {
        auto rv = GetFindModeValueImpl2(variable).value_or(defaultValue);
        MIOPEN_LOG_NQI(variable.GetName() << " = " << rv);
        return rv;
    }();
    return val;
}

} // namespace

FindMode::FindMode(solver::Primitive primitive)
{
    switch(primitive)
    {
    case solver::Primitive::Fusion:
        value = GetFindModeValue(MIOPEN_FIND_MODE_FUSION, FindMode::Values::Fast);
        break;
    default: value = GetFindModeValue(MIOPEN_FIND_MODE, FindMode::Values::Default_); break;
    }
}

std::ostream& operator<<(std::ostream& os, const FindMode& obj) { return os << obj.value; }

static_assert(miopenConvolutionFindModeNormal ==
                  static_cast<miopenConvolutionFindMode_t>(FindMode::Values::Normal),
              "API is not in sync with the implementation.");
static_assert(miopenConvolutionFindModeFast ==
                  static_cast<miopenConvolutionFindMode_t>(FindMode::Values::Fast),
              "API is not in sync with the implementation.");
static_assert(miopenConvolutionFindModeHybrid ==
                  static_cast<miopenConvolutionFindMode_t>(FindMode::Values::Hybrid),
              "API is not in sync with the implementation.");
static_assert(miopenConvolutionFindModeDynamicHybrid ==
                  static_cast<miopenConvolutionFindMode_t>(FindMode::Values::DynamicHybrid),
              "API is not in sync with the implementation.");

} // namespace miopen
