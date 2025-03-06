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

#include "miopen/miopen.h"
#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/find_controls.hpp>
#include <miopen/handle.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/search_options.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/solver.hpp>

#include <limits>
#include <type_traits>
#include <optional>
#include <vector>

namespace miopen {

struct AnyInvokeParams;

namespace solver {

template <class Solver, class Context, class Problem, class Db>
auto FindSolutionImpl(rank<1>,
                      const Solver& s,
                      const Context& context,
                      const Problem& problem,
                      Db&& db,
                      const AnyInvokeParams& invoke_ctx,
                      const std::string& perf_cfg,
                      const std::optional<FindOptions>& options)
    -> decltype(s.GetSolution(context, problem, s.Search(context, problem, invoke_ctx)))
{
    static_assert(std::is_invocable_v<Db>,
                  "db is meant to be a functor returning a reference to perfdb");

    const FindEnforce enforce =
        options && options->find_enforce ? *options->find_enforce : FindEnforce{};
    if(context.disable_perfdb_access)
    {
        MIOPEN_LOG_I(s.SolverDbId() << " (db access disabled)");
        return s.GetSolution(context, problem, s.GetDefaultPerformanceConfig(context, problem));
    }
    MIOPEN_LOG_I(s.SolverDbId());
    if(enforce.IsDbClean(context))
    {
        if(db().Remove(problem, s.SolverDbId()))
            MIOPEN_LOG_W("Perf Db: record removed: " << s.SolverDbId() << ", enforce: " << enforce);
    }
    else
    {
        if((context.do_search || enforce.IsSearch(context)) &&
           (context.db_update || enforce.IsDbUpdate(context)))
        {
            MIOPEN_LOG_W("Perf Db: load skipped: " << s.SolverDbId() << ", enforce: " << enforce);
        }
        else
        {
            using PerformanceConfig = decltype(s.GetDefaultPerformanceConfig(context, problem));
            PerformanceConfig config{};
            // The passes in string needs to have priority over the entry in the database
            if(!perf_cfg.empty())
            {
                config.Deserialize(perf_cfg);
                if(s.IsValidPerformanceConfig(context, problem, config))
                {
                    return s.GetSolution(context, problem, config);
                }
                MIOPEN_LOG_WE("Invalid config loaded from Perf Db: "
                              << s.SolverDbId() << ": " << config << ". Performance may degrade.");
            }
            else if(db().Load(problem, s.SolverDbId(), config))
            {
                MIOPEN_LOG_I2("Perf Db: record loaded: " << s.SolverDbId());
                if(s.IsValidPerformanceConfig(context, problem, config))
                {
                    return s.GetSolution(context, problem, config);
                }
                MIOPEN_LOG_WE("Invalid config loaded from Perf Db: "
                              << s.SolverDbId() << ": " << config << ". Performance may degrade.");
            }
            else if(!s.AltSolverDbId().empty() && db().Load(problem, s.AltSolverDbId(), config))
            {
                MIOPEN_LOG_I("Perf Db: alternate record loaded: " << s.AltSolverDbId());
                if(s.IsValidPerformanceConfig(context, problem, config))
                {
                    return s.GetSolution(context, problem, config);
                }
                MIOPEN_LOG_WE("Invalid alternate record loaded from Perf Db: "
                              << s.AltSolverDbId() << ": " << config
                              << ". Performance may degrade.");
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
                auto c = s.Search(context, problem, invoke_ctx);
                db().Update(problem, s.SolverDbId(), c);
                return s.GetSolution(context, problem, c);
            }
            catch(const miopen::Exception& ex)
            {
                MIOPEN_LOG_E("Search failed for: " << s.SolverDbId() << ": " << ex.what());
                return ConvSolution(miopenStatusInternalError);
            }
        }
    }

    return s.GetSolution(context, problem, s.GetDefaultPerformanceConfig(context, problem));
}

template <class Solver, class Context, class Problem, class Db>
auto FindSolutionImpl(rank<0>,
                      const Solver& s,
                      const Context& context,
                      const Problem& problem,
                      Db&&,
                      const AnyInvokeParams&,
                      const std::string&,
                      const std::optional<FindOptions>&)
    -> decltype(s.GetSolution(context, problem))
{
    MIOPEN_LOG_I(s.SolverDbId() << " (not searchable)");
    return s.GetSolution(context, problem);
}

template <class Solver, class Context, class Problem>
auto GetInvokeFactoryImpl(
    rank<1>, Solver s, const Context& context, const Problem& problem, const std::string& perf_cfg)
    -> decltype(s.GetInvokerFactory(context,
                                    problem,
                                    s.GetDefaultPerformanceConfig(context, problem)))
{
    if(!perf_cfg.empty())
    {
        using PerformanceConfig = decltype(s.GetDefaultPerformanceConfig(context, problem));
        PerformanceConfig config{};
        config.Deserialize(perf_cfg);
        if(s.IsValidPerformanceConfig(context, problem, config))
        {
            return s.GetInvokerFactory(context, problem, config);
        }
        MIOPEN_LOG_WE("Invalid config loaded from Perf Db: " << s.SolverDbId() << ": " << config
                                                             << ". Performance may degrade.");
    }

    return s.GetInvokerFactory(context, problem, s.GetDefaultPerformanceConfig(context, problem));
}

template <class Solver, class Context, class Problem>
auto GetInvokeFactoryImpl(
    rank<0>, Solver s, const Context& context, const Problem& problem, const std::string&)
    -> decltype(s.GetInvokerFactory(context, problem))
{
    MIOPEN_LOG_I(s.SolverDbId() << " (not searchable)");
    return s.GetInvokerFactory(context, problem);
}

/// Finds optimized Solution. Generic method.
///
/// Given the specific problem config, finds (hopefully) optimal
/// solution-specific parameters and returns the Solution object.
/// Could take long if an exhaustive search is requested/performed.
/// May read/write perfDb.
template <class Solver, class Context, class Problem, class Db>
ConvSolution FindSolution(const Solver& s,
                          const Context& context,
                          const Problem& problem,
                          Db&& db,
                          const AnyInvokeParams& invoke_ctx,
                          const std::string& perf_cfg               = "",
                          const std::optional<FindOptions>& options = std::nullopt)
{
    static_assert(sizeof(Solver) == sizeof(SolverBase), "Solver must be stateless");
    static_assert(std::is_base_of<SolverBase, Solver>{}, "Not derived class of SolverBase");

    decltype(auto) db_getter = [&]() -> decltype(auto) {
        if constexpr(std::is_invocable_v<Db>)
            return db;
        else
            return [&]() -> std::decay_t<Db>& { return db; };
    }();

    // TODO: This assumes all solutions are ConvSolution
    auto solution =
        FindSolutionImpl(rank<1>{}, s, context, problem, db_getter, invoke_ctx, perf_cfg, options);
    solution.solver_id = s.SolverDbId();
    return solution;
}

template <class Solver, class Context, class Problem>
InvokerFactory GetInvokeFactory(Solver s,
                                const Context& context,
                                const Problem& problem,
                                const std::string& perf_cfg)
{
    static_assert(sizeof(Solver) == sizeof(SolverBase), "Solver must be stateless");
    static_assert(std::is_base_of<SolverBase, Solver>{}, "Not derived class of SolverBase");
    // TODO: This assumes all solutions are ConvSolution
    return GetInvokeFactoryImpl(rank<1>{}, s, context, problem, perf_cfg);
}

template <class... Solvers>
struct SolverContainer
{
    template <class... SolversRight>
    auto operator+(SolverContainer<SolversRight...>) const
    {
        return SolverContainer<Solvers..., SolversRight...>{};
    }

    ///\todo: remove when AnySolver would be able to work with non-conv solvers
    template <class Functor>
    void FindById(solver::Id id, Functor&& receiver)
    {
        bool found = false;

        miopen::each_args(
            [&](auto solver) {
                if(found || id != solver::Id{solver.SolverDbId()})
                    return;

                found = true;
                receiver(solver);
            },
            Solvers{}...);
    }

    ///\todo: remove when AnySolver would be able to work with non-conv solvers
    template <class Functor>
    void Foreach(Functor&& receiver)
    {
        miopen::each_args([&](auto solver) { receiver(solver); }, Solvers{}...);
    }

    // Search for all applicable solutions among many solvers
    template <class Context, class Problem, class Db, class Solution = miopen::solver::ConvSolution>
    std::vector<Solution>
    SearchForAllSolutions(const Context& ctx,
                          const Problem& problem,
                          Db&& db,
                          const AnyInvokeParams& invoke_ctx,
                          std::size_t limit = std::numeric_limits<std::size_t>::max(),
                          const std::optional<FindOptions>& options = std::nullopt) const
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
                else if(ctx.use_dynamic_solutions_only && !solver.IsDynamic())
                {
                    MIOPEN_LOG_I2(solver.SolverDbId() << ": Skipped (non-dynamic)");
                }
                else if(!solver.IsApplicable(ctx, problem))
                {
                    MIOPEN_LOG_I2(solver.SolverDbId() << ": Not applicable");
                }
                else
                {
                    const Solution s =
                        FindSolution(solver, ctx, problem, db, invoke_ctx, "", options);
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
                       std::size_t limit = std::numeric_limits<std::size_t>::max(),
                       const AnyInvokeParams& invoke_params = {}) const
    {
        auto db_container = std::optional<PerformanceDb>{};
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
                {
                    MIOPEN_LOG_I2(solver.SolverDbId() << ": Not applicable");
                }
                else
                {
                    auto db = [&]() -> PerformanceDb& {
                        constexpr auto db_getter =
                            []([[maybe_unused]] const ExecutionContext& ctx,
                               [[maybe_unused]] const auto& problem) -> PerformanceDb {
                            if constexpr(IsTunable<decltype(solver)>())
                                return GetDb(ctx, problem);
                            else
                                MIOPEN_THROW(miopenStatusInternalError);
                        };

                        if(!db_container)
                            db_container.emplace(std::move(db_getter(ctx, problem)));

                        return *db_container;
                    };

                    auto s =
                        FindSolution(solver, ctx, problem, db, invoke_params, "", std::nullopt);

                    if(s.Succeeded())
                    {
                        ++count;
                        ss.emplace_back(std::move(s));
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

    template <class Context, class Problem>
    std::vector<std::pair<std::string, size_t>> GetWorkspaceSizes(
        const Context& ctx, const Problem& problem, const bool simple_primitive = false) const
    {
        std::vector<std::pair<std::string, size_t>> res;
        const auto find_only = GetEnvFindOnlySolver();
        miopen::each_args(
            [&](auto solver) {
                if(find_only &&
                   (std::find(find_only->begin(), find_only->end(), Id{solver.SolverDbId()}) ==
                    find_only->end()))
                { // Do nothing (and keep silence for the sake of Tuna), just skip.
                }
                // The following optimization is required to avoid checks
                // for solvers that have slow IsApplicable() and do not
                // require workspace (like MLIR convolutions). However we
                // do not want to use it for simple primitives, for example,
                // the ones that ExecutePrimitive() which uses the first applicable
                // solver:
                else if(!simple_primitive && !solver.MayNeedWorkspace())
                {
                    MIOPEN_LOG_I2(solver.SolverDbId() << ": Skipped (no workspace required)");
                }
                // For better performance, check IsDynamic() first, because
                // it is much faster than IsApplicable().
                else if(ctx.use_dynamic_solutions_only && !solver.IsDynamic())
                {
                    MIOPEN_LOG_I2(solver.SolverDbId() << ": Skipped (non-dynamic)");
                }
                else if(!solver.IsApplicable(ctx, problem))
                {
                    MIOPEN_LOG_I2(solver.SolverDbId() << ": Not applicable");
                }
                else
                {
                    auto sz = solver.GetWorkspaceSize(ctx, problem);
                    res.push_back(std::make_pair(solver.SolverDbId(), sz));
                    MIOPEN_LOG_I2(solver.SolverDbId() << ": " << sz);
                }
            },
            Solvers{}...);
        return res;
    }

    template <class Problem>
    void ExecutePrimitive(const ExecutionContext& ctx,
                          const Problem& problem,
                          const AlgorithmName& algo,
                          const AnyInvokeParams& invoke_params) const
    {
        const auto network_config = problem.MakeNetworkConfig();

        if(const auto existingInvoker =
               ctx.GetStream().GetInvoker(network_config, std::nullopt, algo))
        {
            (*existingInvoker)(ctx.GetStream(), invoke_params);
            return;
        }

        const auto slns = SearchForSolutions(ctx, problem, 1, invoke_params);

        if(slns.empty())
            MIOPEN_THROW(miopenStatusNotImplemented, "No solver found.");

        const auto& sln = slns.front();
        if(!sln.invoker_factory)
            MIOPEN_THROW(miopenStatusInternalError, "Invoker missing in solver " + sln.solver_id);
        const auto invoker =
            ctx.GetStream().PrepareInvoker(*sln.invoker_factory, sln.construction_params);
        ctx.GetStream().RegisterInvoker(invoker, network_config, sln.solver_id, algo);
        invoker(ctx.GetStream(), invoke_params);
    }

    template <class Problem>
    void ExecutePrimitive(Handle& handle,
                          const Problem& problem,
                          const AlgorithmName& algo,
                          const AnyInvokeParams& invoke_params) const
    {
        return ExecutePrimitive(&handle, problem, algo, invoke_params);
    }
};

} // namespace solver
} // namespace miopen

#endif
