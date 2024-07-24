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

#ifndef GUARD_MIOPEN_FIND_CONTROLS_HPP_
#define GUARD_MIOPEN_FIND_CONTROLS_HPP_

#include <miopen/logger.hpp>
#include <miopen/solver_id.hpp>

#include <boost/optional.hpp>

#include <ostream>

namespace miopen {

namespace debug {

/// Disable MIOPEN_FIND_ENFORCE. Intended for debugging/testing purposes.
/// Currently used during warm-up phase in MIOpenDriver.
/// WARNING: This switch is not intended for use in multi-threaded applications.
MIOPEN_EXPORT extern bool
    FindEnforceDisable; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)

} // namespace debug

enum class FindEnforceAction
{
    First_ = 1, // 0 is returned for non-numeric env.vars.
    None   = First_,
    DbUpdate,
    Search,
    SearchDbUpdate,
    DbClean,
    Last_          = DbClean,
    Default_       = None,
    EnforcedFirst_ = DbUpdate, // This range must be continuous.
    EnforcedLast_  = DbClean,
};

class MIOPEN_INTERNALS_EXPORT FindEnforce
{
    FindEnforceAction action;

private:
    template <class Context>
    bool IsEnabled(const Context& context) const
    {
        return !(context.disable_search_enforce || debug::FindEnforceDisable);
    }

public:
    FindEnforce();
    explicit FindEnforce(FindEnforceAction action_) : action(action_) {}

    template <class Context>
    bool IsDbClean(const Context& context) const
    {
        return IsEnabled(context) && action == FindEnforceAction::DbClean;
    }

    template <class Context>
    bool IsSearch(const Context& context) const
    {
        return IsEnabled(context) &&
               (action == FindEnforceAction::Search || action == FindEnforceAction::SearchDbUpdate);
    }

    template <class Context>
    bool IsDbUpdate(const Context& context) const
    {
        return IsEnabled(context) && (action == FindEnforceAction::DbUpdate ||
                                      action == FindEnforceAction::SearchDbUpdate);
    }

    template <class Context>
    bool IsSomethingEnforced(const Context& context) const
    {
        return IsEnabled(context) && (FindEnforceAction::EnforcedFirst_ <= action &&
                                      action <= FindEnforceAction::EnforcedLast_);
    }

    MIOPEN_INTERNALS_EXPORT friend std::ostream& operator<<(std::ostream&, const FindEnforce&);
};

MIOPEN_INTERNALS_EXPORT boost::optional<std::vector<solver::Id>> GetEnvFindOnlySolver();

class MIOPEN_INTERNALS_EXPORT FindMode
{
public:
    enum class Values
    {
        Begin_ = 1, // 0 is returned for non-numeric env.vars.
        Normal = Begin_,
        Fast,
        Hybrid,
        DeprecatedFastHybrid,
        DynamicHybrid,
        End_,
        Default_ = MIOPEN_DEFAULT_FIND_MODE,
    };

private:
    Values value;

    template <class Context>
    bool IsEnabled(const Context& context) const
    {
        if(FindEnforce{}.IsSomethingEnforced(context))
        {
            MIOPEN_LOG_NQI("MIOPEN_FIND_MODE is set to NORMAL due to MIOPEN_FIND_ENFORCE");
            return false;
        }
        return true;
    }

public:
    // Todo: remove default value of primitive
    FindMode(solver::Primitive primitive = solver::Primitive::Convolution);
    Values Get() const { return value; }
    void Set(Values const v) { value = v; }

    template <class Context>
    bool IsFast(const Context& context) const
    {
        return value == Values::Fast && IsEnabled(context);
    }

    template <class Context>
    bool IsHybrid(const Context& context) const
    {
        return (value == Values::Hybrid || value == Values::DynamicHybrid) && IsEnabled(context);
    }

    template <class Context>
    bool IsDynamicHybrid(const Context& context) const
    {
        return value == Values::DynamicHybrid && IsEnabled(context);
    }

    MIOPEN_INTERNALS_EXPORT friend std::ostream& operator<<(std::ostream&, const FindMode&);
};

} // namespace miopen

#endif // GUARD_MIOPEN_FIND_CONTROLS_HPP_
