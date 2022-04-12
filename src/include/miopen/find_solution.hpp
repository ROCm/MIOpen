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
#include <miopen/execution_context.hpp>
#include <miopen/find_controls.hpp>
#include <miopen/handle.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/solver.hpp>

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
        MIOPEN_LOG_I(s.SolverDbId() << " (db access disabled)");
        return s.GetSolution(context, s.GetPerformanceConfig(context));
    }
    MIOPEN_LOG_I(s.SolverDbId());
    if(enforce.IsDbClean(context))
    {
        if(db.Remove(context, s.SolverDbId()))
            MIOPEN_LOG_W("Perf Db: record removed: " << s.SolverDbId() << ", enforce: " << enforce);
    }
    else
    {
        if((context.do_search || enforce.IsSearch(context)) && enforce.IsDbUpdate(context))
        {
            MIOPEN_LOG_W("Perf Db: load skipped: " << s.SolverDbId() << ", enforce: " << enforce);
        }
        else
        {
            using PerformanceConfig = decltype(s.GetPerformanceConfig(context));
            PerformanceConfig config{};
            if(db.Load(context, s.SolverDbId(), config))
            {
                MIOPEN_LOG_I2("Perf Db: record loaded: " << s.SolverDbId());
                if(s.IsValidPerformanceConfig(context, config))
                {
                    return s.GetSolution(context, config);
                }
                MIOPEN_LOG_WE("Invalid config loaded from Perf Db: "
                              << s.SolverDbId() << ": " << config << ". Performance may degrade.");
            }
            else
            {
                MIOPEN_LOG_I("Perf Db: record not found for: " << s.SolverDbId());
            }
        }

        if(context.do_search || enforce.IsSearch(context)) // TODO: Make it a customization point
        {
            MIOPEN_LOG_I("Starting search: " << s.SolverDbId() << ", enforce: " << enforce);
            try
            {
                auto c = s.Search(context, invoke_ctx);
                db.Update(context, s.SolverDbId(), c);
                return s.GetSolution(context, c);
            }
            catch(const miopen::Exception& ex)
            {
                MIOPEN_LOG_E("Search failed for: " << s.SolverDbId() << ": " << ex.what());
            }
        }
    }

    return s.GetSolution(context, s.GetPerformanceConfig(context));
}

template <class Solver, class Context, class Db>
auto FindSolutionImpl(rank<0>, Solver s, const Context& context, Db&, const AnyInvokeParams&)
    -> decltype(s.GetSolution(context))
{
    MIOPEN_LOG_I(s.SolverDbId() << " (not searchable)");
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
    static_assert(sizeof(Solver) == sizeof(SolverBase), "Solver must be stateless");
    static_assert(std::is_base_of<SolverBase, Solver>{}, "Not derived class of SolverBase");
    // TODO: This assumes all solutions are ConvSolution
    auto solution      = FindSolutionImpl(rank<1>{}, s, context, db, invoke_ctx);
    solution.solver_id = s.SolverDbId();
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
                if(find_only &&
                   (std::find(find_only->begin(), find_only->end(), Id{solver.SolverDbId()}) ==
                    find_only->end()))
                { // Do nothing (and keep silence for the sake of Tuna), just skip.
                }
                // For better performance, check IsDynamic() first, because
                // it is much faster than IsApplicable().
                else if(search_params.use_dynamic_solutions_only && !solver.IsDynamic())
                    MIOPEN_LOG_I2(solver.SolverDbId() << ": Skipped (non-dynamic)");
                else if(!solver.IsApplicable(search_params))
                    MIOPEN_LOG_I2(solver.SolverDbId() << ": Not applicable");
                else
                {
                    const Solution s = FindSolution(solver, search_params, db, invoke_ctx);
                    if(s.Succeeded())
                    {
                        ++count;
                        ss.push_back(s);
                        MIOPEN_LOG_I2(solver.SolverDbId() << ": Success.");
                    }
                    else
                    {
                        /// \todo If Solver is applicable it must provide an appropriate Solution.
                        /// This is not the case for some 20x5 convolutions (and possibly others).
                        /// Normally we should not get here and message level should be Error.
                        /// For now, let's use Info (not Warning) level to avoid
                        /// flooding the console.
                        MIOPEN_LOG_I(solver.SolverDbId()
                                     << ": [Warning] Applicable Solver not succeeded.");
                    }
                }
            },
            Solvers{}...);
        return ss;
    }

    // Search for all applicable solutions among many solvers
    template <class Problem, class Solution = miopen::solver::ConvSolution>
    std::vector<Solution>
    SearchForSolutions(const ExecutionContext& ctx,
                       const Problem& problem,
                       std::size_t limit = std::numeric_limits<std::size_t>::max()) const
    {
        std::vector<Solution> ss;
        std::size_t count    = 0;
        const auto find_only = GetEnvFindOnlySolver();
        miopen::each_args(
            [&](auto solver) {
                if(count >= limit)
                    return;
                if(find_only &&
                   (std::find(find_only->begin(), find_only->end(), Id{solver.SolverDbId()}) ==
                    find_only->end()))
                { // Do nothing (and keep silence for the sake of Tuna), just skip.
                }
                // For better performance, check IsDynamic() first, because
                // it is much faster than IsApplicable().
                // else if(problem.use_dynamic_solutions_only && !solver.IsDynamic())
                //    MIOPEN_LOG_I2(solver.SolverDbId() << ": Skipped (non-dynamic)");
                else if(!solver.IsApplicable(ctx, problem))
                    MIOPEN_LOG_I2(solver.SolverDbId() << ": Not applicable");
                else
                {
                    auto s      = solver.GetSolution(ctx, problem);
                    s.solver_id = solver.SolverDbId();
                    if(s.Succeeded())
                    {
                        ++count;
                        ss.push_back(s);
                        MIOPEN_LOG_I2(solver.SolverDbId() << ": Success.");
                    }
                    else
                    {
                        MIOPEN_LOG_E(solver.SolverDbId() << ": Applicable Solver not succeeded.");
                    }
                }
            },
            Solvers{}...);
        return ss;
    }

    template <class Context>
    std::vector<std::pair<std::string, size_t>>
    GetWorkspaceSizes(const Context& search_params,
                      std::size_t limit = std::numeric_limits<std::size_t>::max()) const
    {
        std::vector<std::pair<std::string, size_t>> res;
        const auto find_only = GetEnvFindOnlySolver();
        std::size_t count    = 0;
        miopen::each_args(
            [&](auto solver) {
                if(count >= limit)
                    return;

                if(find_only &&
                   (std::find(find_only->begin(), find_only->end(), Id{solver.SolverDbId()}) ==
                    find_only->end()))
                { // Do nothing (and keep silence for the sake of Tuna), just skip.
                }
                else if(!solver.MayNeedWorkspace())
                    MIOPEN_LOG_I2(solver.SolverDbId() << ": Skipped (no workspace required)");
                // For better performance, check IsDynamic() first, because
                // it is much faster than IsApplicable().
                else if(search_params.use_dynamic_solutions_only && !solver.IsDynamic())
                    MIOPEN_LOG_I2(solver.SolverDbId() << ": Skipped (non-dynamic)");
                else if(!solver.IsApplicable(search_params))
                    MIOPEN_LOG_I2(solver.SolverDbId() << ": Not applicable");
                else
                {
                    ++count;
                    auto sz = solver.GetWorkspaceSize(search_params);
                    res.push_back(std::make_pair(solver.SolverDbId(), sz));
                    MIOPEN_LOG_I2(solver.SolverDbId() << ": " << sz);
                }
            },
            Solvers{}...);
        return res;
    }

    // Search for all applicable solutions among many solvers
    template <class Context>
    bool IsAnySolverApplicable(const Context& search_params) const
    {
        const auto find_only = GetEnvFindOnlySolver();
        auto found           = false;

        miopen::each_args(
            [&](auto solver) {
                if(found || (find_only && (std::find(find_only->begin(),
                                                     find_only->end(),
                                                     Id{solver.SolverDbId()}) == find_only->end())))
                    return;

                // For better performance, check IsDynamic() first, because
                // it is much faster than IsApplicable().
                if(search_params.use_dynamic_solutions_only && !solver.IsDynamic())
                {
                    MIOPEN_LOG_I2(solver.SolverDbId() << ": Skipped (non-dynamic)");
                    return;
                }

                if(solver.IsApplicable(search_params))
                {
                    found = true;
                    return;
                }

                MIOPEN_LOG_I2(solver.SolverDbId() << ": Not applicable");
            },
            Solvers{}...);

        return found;
    }

    template <class Problem>
    void ExecutePrimitive(Handle& handle,
                          const Problem& problem,
                          const AlgorithmName& algo,
                          const AnyInvokeParams& invoke_params) const
    {
        const auto network_config = problem.MakeNetworkConfig();

        if(const auto existingInvoker = handle.GetInvoker(network_config, boost::none, algo))
        {
            (*existingInvoker)(handle, invoke_params);
            return;
        }

        auto ctx = ExecutionContext{&handle};
        ctx.DetectRocm();
        const auto slns = SearchForSolutions(ctx, problem, 1);

        if(slns.empty())
            MIOPEN_THROW(miopenStatusNotImplemented, "No solver found.");

        const auto& sln = slns.front();
        if(!sln.invoker_factory)
            MIOPEN_THROW(miopenStatusInternalError, "Invoker missing in solver " + sln.solver_id);
        const auto invoker = handle.PrepareInvoker(*sln.invoker_factory, sln.construction_params);
        handle.RegisterInvoker(invoker, network_config, sln.solver_id, algo);
        invoker(handle, invoke_params);
    }
};

} // namespace solver
} // namespace miopen

#endif
