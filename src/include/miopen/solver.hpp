/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_SOLVER_HPP_
#define GUARD_MIOPEN_SOLVER_HPP_

#include <miopen/config.hpp>

#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/invoker.hpp>
#include <miopen/performance_config.hpp>
#include <miopen/type_name.hpp>

#include <boost/any.hpp>

#include <string>
#include <type_traits>
#include <algorithm>

namespace miopen {

struct AnyInvokeParams;

namespace solver {

/// Base class for problem solvers.
///
/// Solvers are to be instantiated as const objects and shall not have any variable
/// internal state. Any non-const state information, if required, to be stored in the
/// solver-specific context objects.
///
/// There could be multiple solvers of the same algorithm for a problem config.
struct SolverBase
{
    virtual ~SolverBase() = default;

    /// This will retrieve the id of the solver to write to the database. By
    /// default it uses the class name. If the class is renamed, this function can
    /// overriden to keep the name to avoid DB corruption.
    virtual const std::string& SolverDbId() const = 0;

    /// In some instances (particularly fusions) the fused solver might like to
    /// fallback to the non-fused variant for performance parameters, this information
    /// is returned via AltSolverDbId
    virtual const std::string& AltSolverDbId() const
    {
        static const std::string null_id = "";
        return null_id;
    }

    /// Returns true if solution can work on given SW/HW platform (runtime/device)
    /// and provides correct result for the problem config.
    ///
    /// Every SolverBase which IsApplicable() for some problem config must be able to
    /// GetDefaultPerformanceConfig() so that GetSolution() would return valid
    /// solution for a problem (i.e. convolution). In other words, if a Solution
    /// says "I'm suitable" for a problem, it agrees to solve that problem correctly.
    virtual bool IsApplicable(const ExecutionContext& ctx, const boost::any& problem) const = 0;

    /// [Informative as of Sep 2020] The minimum requirement for Dynamic Solvers:
    /// Batch size and input picture size (N, W, H) must NOT be compiled into the
    /// kernel(s) that consist a Solution. These must go into the kernel as a
    /// run-time parameters.
    virtual bool IsDynamic() const { return false; }

    static constexpr float wti_approximate_worst = -2;

    /// [Informative as of Sep 2020] Returns an approximated value of the expected
    /// WTI or wti_approximate_worst when this value can't be computed. Tips:
    /// * Value 1.0 corresponds to the 100% utilization of HW capabilities as
    ///   if Direct computational algorithm is used.
    /// * [Notice] WTI may exceed 1.0 for highly optimized algorithms like Winograd.
    /// * @see https://github.com/ROCm/MIOpen/issues/410
    virtual float GetWti(const ExecutionContext& ctx, const boost::any& problem) const = 0;

    /// Returns the workspace size required by the solver for a given ExecutionContext
    virtual size_t GetWorkspaceSize(const ExecutionContext& ctx,
                                    const boost::any& problem) const = 0;

    /// Must return true if a Solver has its own implementation of GetWorkspaceSize().
    virtual bool MayNeedWorkspace() const { return false; }

protected:
    template <class Solver>
    static const std::string& GetSolverDbId()
    {
        static const auto result = ComputeSolverDbId(get_type_name<Solver>());
        return result;
    }
    SolverBase()                  = default;
    SolverBase(const SolverBase&) = default;

private:
    static std::string ComputeSolverDbId(const std::string& type_name)
    {
        auto idx  = type_name.find_last_of(':');
        auto name = type_name.substr(idx + 1);
        std::replace(name.begin(), name.end(), ',', '-');
        name.erase(std::remove(name.begin(), name.end(), ' '), name.end());

        return name;
    }
};

template <class Context, class Problem>
struct SolverMixin : SolverBase
{
    static_assert(std::is_base_of<ExecutionContext, Context>{},
                  "Context must be derived of ExecutionContext");

    virtual bool IsApplicable(const Context&, const Problem&) const = 0;
    virtual float GetWti(const Context&, const Problem&) const { return wti_approximate_worst; };
    virtual size_t GetWorkspaceSize(const Context&, const Problem&) const { return 0; };

    bool IsApplicable(const ExecutionContext& ctx, const boost::any& problem) const final
    {
        return IsApplicable(dynamic_cast<const Context&>(ctx),
                            boost::any_cast<const Problem&>(problem));
    }

    float GetWti(const ExecutionContext& ctx, const boost::any& problem) const final
    {
        return GetWti(dynamic_cast<const Context&>(ctx), boost::any_cast<const Problem&>(problem));
    }

    size_t GetWorkspaceSize(const ExecutionContext& ctx, const boost::any& problem) const final
    {
        return GetWorkspaceSize(dynamic_cast<const Context&>(ctx),
                                boost::any_cast<const Problem&>(problem));
    }
};

/// Base class for non tunable solvers
template <class Context, class Problem>
struct NonTunableSolverBase : SolverMixin<Context, Problem>
{
    /// Takes problem config, optimization parameters and other info
    /// and computes information required to build and run the kernel(s).
    virtual ConvSolution GetSolution(const Context&, const Problem&) const = 0;

    virtual InvokerFactory GetInvokerFactory(const Context& ctx, const Problem& problem) const
    {
        return *GetSolution(ctx, problem).invoker_factory;
    }
};

struct TunableSolverTrait
{
};

/// Base class for tunable solvers
template <class Context, class Problem>
struct TunableSolverBase : SolverMixin<Context, Problem>, TunableSolverTrait
{
    /// Initializes performance config to the default values.
    /// The function may involve some heuristic to guess the best solution
    /// configuration. It is assumed that the function takes constant time
    /// to finish and does not run kernels to measure performance etc.
    /// The function shall always return valid config.
    ///
    /// The int parameter is needed only to not change the name of the
    /// function in the derived class. Function declarations that differ
    /// only by its return type cannot be overloaded.
    virtual boost::any
    GetDefaultPerformanceConfig(const Context& ctx, const Problem& problem, int) const = 0;

    /// Should return false if performance config is wrong for a problem.
    /// Main use is validation of values read from the perf db.
    virtual bool IsValidPerformanceConfig(const Context& ctx,
                                          const Problem& problem,
                                          const PerfConfig& config) const = 0;

    /// Search
    ///
    /// The int parameter is needed only to not change the name of the
    /// function in the derived class. Function declarations that differ
    /// only by its return type cannot be overloaded.
    virtual boost::any Search(const Context& ctx,
                              const Problem& problem,
                              const AnyInvokeParams& invoke_ctx,
                              int) const = 0;

    /// Tunable solvers provide a GetSolution that takes a Context and PerformanceConfig
    virtual ConvSolution
    GetSolution(const Context& ctx, const Problem& problem, const PerfConfig& config) const = 0;

    virtual InvokerFactory
    GetInvokerFactory(const Context& ctx, const Problem& problem, const PerfConfig& config) const
    {
        return *GetSolution(ctx, problem, config).invoker_factory;
    }
};

template <class Context, class Problem, class PerformanceConfig>
struct TunableSolverMixin : TunableSolverBase<Context, Problem>
{
    static_assert(std::is_base_of<PerfConfig, PerformanceConfig>{},
                  "PerformanceConfig must be derived of PerfConfig");

    virtual PerformanceConfig GetDefaultPerformanceConfig(const Context&, const Problem&) const = 0;
    virtual bool
    IsValidPerformanceConfig(const Context&, const Problem&, const PerformanceConfig&) const = 0;
    virtual PerformanceConfig
    Search(const Context&, const Problem&, const AnyInvokeParams&) const = 0;
    virtual ConvSolution
    GetSolution(const Context&, const Problem&, const PerformanceConfig&) const = 0;

    boost::any
    GetDefaultPerformanceConfig(const Context& ctx, const Problem& problem, int) const final
    {
        return GetDefaultPerformanceConfig(ctx, problem);
    }

    bool IsValidPerformanceConfig(const Context& ctx,
                                  const Problem& problem,
                                  const PerfConfig& config) const final
    {
        return IsValidPerformanceConfig(
            ctx, problem, dynamic_cast<const PerformanceConfig&>(config));
    }

    boost::any Search(const Context& ctx,
                      const Problem& problem,
                      const AnyInvokeParams& invoke_ctx,
                      int) const final
    {
        return Search(ctx, problem, invoke_ctx);
    }

    ConvSolution
    GetSolution(const Context& ctx, const Problem& problem, const PerfConfig& config) const final
    {
        return GetSolution(ctx, problem, dynamic_cast<const PerformanceConfig&>(config));
    }
};

template <class Solver>
struct IsTunable : std::is_base_of<TunableSolverTrait, Solver>
{
    static_assert(!std::is_same_v<Solver, TunableSolverTrait>,
                  "Raw trait shouldn't be passed, explicit type is needed");
};

// Use struct as a syntactic sugar to make the intent as clear as possible.
struct ThisSolverIsDeprecatedStatic
{
    MIOPEN_INTERNALS_EXPORT static bool IsDisabled(const ExecutionContext& ctx);
};

} // namespace solver
} // namespace miopen

#endif // GUARD_MIOPEN_SOLVER_HPP_
