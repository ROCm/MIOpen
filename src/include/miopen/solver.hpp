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

#include <string>
#include <type_traits>
#include <algorithm>

namespace miopen {

namespace debug {

/// Enables deprecated solvers.
/// This variable is intended for use in unit tests.
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
MIOPEN_INTERNALS_EXPORT extern bool enable_deprecated_solvers;

} // namespace debug

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

    /// Returns true for tunable solvers
    virtual bool IsTunable() const = 0;

    /// [Informative as of Sep 2020] The minimum requirement for Dynamic Solvers:
    /// Batch size and input picture size (N, W, H) must NOT be compiled into the
    /// kernel(s) that consist a Solution. These must go into the kernel as a
    /// run-time parameters.
    virtual bool IsDynamic() const { return false; }

    /// Must return true if a Solver has its own implementation of GetWorkspaceSize().
    virtual bool MayNeedWorkspace() const { return false; }

protected:
    template <class Solver>
    static const std::string& GetSolverDbId()
    {
#if BUILD_SHARED_LIBS && MIOPEN_ENABLE_FIN_INTERFACE
        /// When using this function outside of the shared library, the static local variable is
        /// duplicated, both the library and the program using it have their own copy, but only one
        /// of them is initialized, depending on which entity calls the function first—the library
        /// or the program.
        /// \todo This needs to be removed when the interface matures, and internal class/function
        /// templates are no longer used by the fin.
        static std::string result;
        if(result.empty())
        {
            // The "new" operator is used here to avoid segmentation fault (since the variable is
            // not initialized).
            new(&result) std::string(ComputeSolverDbId(type_name_bare<Solver>()));
        }
#else  // !BUILD_SHARED_LIBS || !MIOPEN_ENABLE_FIN_INTERFACE
        static const auto result = ComputeSolverDbId(type_name_bare<Solver>());
#endif // !BUILD_SHARED_LIBS || !MIOPEN_ENABLE_FIN_INTERFACE
        return result;
    }
    SolverBase()                  = default;
    SolverBase(const SolverBase&) = default;

private:
    static std::string ComputeSolverDbId(std::string_view type_name)
    {
        auto name = std::string(type_name);
        if(name.back() == '>')
        {
            std::replace(name.begin(), name.end(), ',', '-');
            name.erase(std::remove(name.begin(), name.end(), ' '), name.end());
        }

        return name;
    }
};

/// Common interface for tunable and non-tunable solvers
template <class Context, class Problem>
struct SolverInterface : SolverBase
{
    static_assert(std::is_base_of<ExecutionContext, Context>{},
                  "Context must be derived of ExecutionContext");

    /// Returns true if a Solver can work on given SW/HW platform (runtime/device)
    /// and provides correct result for the problem config.
    virtual bool IsApplicable(const Context& ctx, const Problem& problem) const = 0;

    static constexpr float wti_approximate_worst = -2;

    /// [Informative as of Sep 2020] Returns an approximated value of the expected
    /// WTI or wti_approximate_worst when this value can't be computed. Tips:
    /// * Value 1.0 corresponds to the 100% utilization of HW capabilities as
    ///   if Direct computational algorithm is used.
    /// * [Notice] WTI may exceed 1.0 for highly optimized algorithms like Winograd.
    /// * @see https://github.com/ROCm/MIOpen/issues/410
    virtual float GetWti(const Context&, const Problem&) const { return wti_approximate_worst; };

    /// Returns the workspace size required by the solver for the given Problem
    virtual size_t GetWorkspaceSize(const Context&, const Problem&) const { return 0; };
};

/// Common interface for non-tunable solvers
template <class Context, class Problem>
struct SolverInterfaceNonTunable : SolverInterface<Context, Problem>
{
    /// Takes problem config, optimization parameters and other info
    /// and computes information required to build and run the kernel(s).
    virtual ConvSolution GetSolution(const Context& ctx, const Problem& problem) const = 0;
};

/// Common interface for tunable solvers
template <class Context, class Problem>
struct SolverInterfaceTunable : SolverInterface<Context, Problem>
{
    /// This function is a simplified version of FindSolution(), it does not obey search parameters
    /// from the Context and does not use the database. Intended to be used in unit tests.
    virtual ConvSolution FindSolutionSimple(const Context& ctx,
                                            const Problem& problem,
                                            const AnyInvokeParams& invoke_ctx) const = 0;
};

/// Base class for non-tunable solvers
template <class Context, class Problem>
struct SolverBaseNonTunable : SolverInterfaceNonTunable<Context, Problem>
{
    bool IsTunable() const final { return false; };

    InvokerFactory GetInvokerFactory(const Context& ctx, const Problem& problem) const
    {
        const auto solution = this->GetSolution(ctx, problem);
        return *solution.invoker_factory;
    }
};

struct TunableSolverTrait
{
};

/// Base class for tunable solvers
template <class Context, class Problem, class PerformanceConfig>
struct SolverBaseTunable : SolverInterfaceTunable<Context, Problem>, TunableSolverTrait
{
    bool IsTunable() const final { return true; };

    /// Initializes performance config to the default values.
    /// The function may involve some heuristic to guess the best solution
    /// configuration. It is assumed that the function takes constant time
    /// to finish and does not run kernels to measure performance etc.
    /// The function shall always return valid config.
    virtual PerformanceConfig GetDefaultPerformanceConfig(const Context& ctx,
                                                          const Problem& problem) const = 0;

    /// Should return false if performance config is wrong for a problem.
    /// Main use is validation of values read from the perf db.
    virtual bool IsValidPerformanceConfig(const Context& ctx,
                                          const Problem& problem,
                                          const PerformanceConfig& config) const = 0;

    /// Search
    virtual PerformanceConfig
    Search(const Context& ctx, const Problem& problem, const AnyInvokeParams& invoke_ctx) const = 0;

    /// Tunable solvers provide a GetSolution that takes a Context and PerformanceConfig
    virtual ConvSolution GetSolution(const Context& ctx,
                                     const Problem& problem,
                                     const PerformanceConfig& config) const = 0;

    ConvSolution FindSolutionSimple(const Context& ctx,
                                    const Problem& problem,
                                    const AnyInvokeParams& invoke_ctx) const final
    {
        const PerformanceConfig config = Search(ctx, problem, invoke_ctx);
        return GetSolution(ctx, problem, config);
    }

    InvokerFactory GetInvokerFactory(const Context& ctx,
                                     const Problem& problem,
                                     const PerformanceConfig& config) const
    {
        return *GetSolution(ctx, problem, config).invoker_factory;
    }
};

// \todo Should be removed
template <class Context, class Problem>
using NonTunableSolverBase = SolverBaseNonTunable<Context, Problem>;

// \todo Should be removed
template <class Context, class Problem, class PerformanceConfig>
using TunableSolverMixin = SolverBaseTunable<Context, Problem, PerformanceConfig>;

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
