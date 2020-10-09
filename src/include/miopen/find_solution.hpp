/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_GUARD_MLOPEN_FIND_SOLUTION_HPP
#define MIOPEN_GUARD_MLOPEN_FIND_SOLUTION_HPP

#include <miopen/env.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/find_controls.hpp>
#include <miopen/solver_id.hpp>

#include <limits>
#include <vector>

namespace miopen {

struct AnyInvokeParams;

namespace solver {

template <class Solver, class Context, class Db>
auto FindSolutionImpl(
    rank<1>, Solver s, const Context& context, Db& db, const AnyInvokeParams& invoke_ctx)
    -> decltype(s.GetSolution(context, s.Search(context, invoke_ctx)))
{
    const FindEnforce enforce;
    if(context.disable_perfdb_access)
    {
        MIOPEN_LOG_I(SolverDbId(s) << " (db access disabled)");
        return s.GetSolution(context, s.GetPerformanceConfig(context));
    }
    MIOPEN_LOG_I(SolverDbId(s));
    if(enforce.IsDbClean(context))
    {
        if(db.Remove(context, SolverDbId(s)))
            MIOPEN_LOG_W("Perf Db: record removed: " << SolverDbId(s) << ", enforce: " << enforce);
    }
    else
    {
        if((context.do_search || enforce.IsSearch(context)) && enforce.IsDbUpdate(context))
        {
            MIOPEN_LOG_W("Perf Db: load skipped: " << SolverDbId(s) << ", enforce: " << enforce);
        }
        else
        {
            using PerformanceConfig = decltype(s.GetPerformanceConfig(context));
            PerformanceConfig config{};
            if(db.Load(context, SolverDbId(s), config))
            {
                MIOPEN_LOG_I2("Perf Db: record loaded: " << SolverDbId(s));
                if(s.IsValidPerformanceConfig(context, config))
                {
                    return s.GetSolution(context, config);
                }
                MIOPEN_LOG_WE(
                    "Invalid config loaded from Perf Db: " << SolverDbId(s) << ": " << config
                                                           << ". Performance may degrade.");
            }
            else
            {
                MIOPEN_LOG_I("Perf Db: record not found for: " << SolverDbId(s));
            }
        }

        if(context.do_search || enforce.IsSearch(context)) // TODO: Make it a customization point
        {
            MIOPEN_LOG_I("Starting search: " << SolverDbId(s) << ", enforce: " << enforce);
            try
            {
                auto c = s.Search(context, invoke_ctx);
                db.Update(context, SolverDbId(s), c);
                return s.GetSolution(context, c);
            }
            catch(const miopen::Exception& ex)
            {
                MIOPEN_LOG_E("Search failed for: " << SolverDbId(s) << ": " << ex.what());
            }
        }
    }

    return s.GetSolution(context, s.GetPerformanceConfig(context));
}

template <class Solver, class Context, class Db>
auto FindSolutionImpl(rank<0>, Solver s, const Context& context, Db&, const AnyInvokeParams&)
    -> decltype(s.GetSolution(context))
{
    MIOPEN_LOG_I(SolverDbId(s) << " (not searchable)");
    return s.GetSolution(context);
}

/// Finds optimized Solution. Generic method.
///
/// Given the specific problem config, finds (hopefully) optimal
/// solution-specific parameters and returns the Solution object.
/// Could take long if an exhaustive search is requested/performed.
/// May read/write perfDb.
template <class Solver, class Context, class Db>
ConvSolution
FindSolution(Solver s, const Context& context, Db& db, const AnyInvokeParams& invoke_ctx)
{
    static_assert(std::is_empty<Solver>{} && std::is_trivially_constructible<Solver>{},
                  "Solver must be stateless");
    // TODO: This assumes all solutions are ConvSolution
    auto solution      = FindSolutionImpl(rank<1>{}, s, context, db, invoke_ctx);
    solution.solver_id = SolverDbId(s);
    return solution;
}

template <class... Solvers>
struct SolverContainer
{
    // Search for all applicable solutions among many solvers
    template <class Context, class Db, class Solution = miopen::solver::ConvSolution>
    std::vector<Solution>
    SearchForAllSolutions(const Context& search_params,
                          Db&& db,
                          const AnyInvokeParams& invoke_ctx,
                          std::size_t limit = std::numeric_limits<std::size_t>::max()) const
    {
        std::vector<Solution> ss;
        std::size_t count    = 0;
        const auto find_only = GetEnvFindOnlySolver();
        miopen::each_args(
            [&](auto solver) {
                if(count >= limit)
                    return;
                if(find_only.IsValid() && find_only != Id{SolverDbId(solver)})
                { // Do nothing (and keep silence for the sake of Tuna), just skip.
                }
                else if(!solver.IsApplicable(search_params))
                    MIOPEN_LOG_I2(SolverDbId(solver) << ": Not applicable");
                else if(search_params.use_dynamic_solutions_only && !solver.IsDynamic())
                    MIOPEN_LOG_I2(SolverDbId(solver) << ": Skipped (non-dynamic)");
                else
                {
                    const Solution s = FindSolution(solver, search_params, db, invoke_ctx);
                    if(s.Succeeded())
                    {
                        ++count;
                        ss.push_back(s);
                        MIOPEN_LOG_I2(SolverDbId(solver) << ": Success.");
                    }
                    else
                    {
                        /// \todo If Solver is applicable it must provide an appropriate Solution.
                        /// This is not the case for some 20x5 convolutions (and possibly others).
                        /// Normally we should not get here and message level should be Error.
                        /// For now, let's use Info (not Warning) level to avoid
                        /// flooding the console.
                        MIOPEN_LOG_I(SolverDbId(solver)
                                     << ": [Warning] Applicable Solver not succeeded.");
                    }
                }
            },
            Solvers{}...);
        return ss;
    }
    template <class Context>
    std::vector<std::pair<std::string, size_t>> GetWorkspaceSize(const Context& search_params) const
    {
        std::vector<std::pair<std::string, size_t>> res;
        const auto find_only = GetEnvFindOnlySolver();
        miopen::each_args(
            [&](auto solver) {
                if(find_only.IsValid() && find_only != Id{SolverDbId(solver)})
                { // Do nothing (and keep silence for the sake of Tuna), just skip.
                }
                else if(!solver.IsApplicable(search_params))
                    MIOPEN_LOG_I2(SolverDbId(solver) << ": Not applicable");
                else if(search_params.use_dynamic_solutions_only && !solver.IsDynamic())
                    MIOPEN_LOG_I2(SolverDbId(solver) << ": Skipped (non-dynamic)");
                else
                {
                    auto sz = solver.GetWorkspaceSize(search_params);
                    res.push_back(std::make_pair(SolverDbId(solver), sz));
                }
            },
            Solvers{}...);
        return res;
    }
};

} // namespace solver
} // namespace miopen

#endif
