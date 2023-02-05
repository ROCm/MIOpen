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

#ifndef GUARD_MIOPEN_GENERIC_SEARCH_HPP_
#define GUARD_MIOPEN_GENERIC_SEARCH_HPP_

#include <miopen/binary_cache.hpp>
#include <miopen/config.h>
#include <miopen/conv/context.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/logger.hpp>
#include <miopen/timer.hpp>
#include <miopen/type_traits.hpp>

#include <algorithm>
#include <vector>
#include <cstdlib>
#include <limits>
#include <iterator>
#include <chrono>
#include <cassert>

namespace miopen {
namespace solver {

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_COMPILE_ONLY)

/// This STL-like container together with corresponding iterator provide access
/// to a set of all available performance configs for the given problem config.
///
/// Implementation does not hold values themselves as these would take too much memory.
/// The container holds problem config information instead. This info
/// is required for advancing the iterator to the next valid configuration.
///
/// PerformanceConfig type requirements:
/// - (ctor)()
///     Constructs an instance with invalid value.
/// - (ctor)(bool)
///     Constructs an instance with minimal value.
/// - SetNextValue(const Context& c)
///     Advances instance value to the next available value and returns true.
///     If max value reached, returns false.
/// - IsValid(const Context& c) const
///     Checks if instance is valid for the given c.
///     For convolutions, Context represents a problem configuration.
/// - operator==(const PerformanceConfig&)
///     Ordinary semantics.
template <typename PerformanceConfig, typename Context>
class ComputedContainer;

template <typename PerformanceConfig, typename Context>
class ComputedIterator : public std::iterator<std::input_iterator_tag, PerformanceConfig>
{
    PerformanceConfig v;
    const Context* p; // For Next().

    ComputedIterator& Next()
    {
        if(p != nullptr)
        {
            do
            {
                if(!v.SetNextValue(*p))
                { // Wraparound, end reached. Iterator is useless from now.
                    p = nullptr;
                    break;
                }
            } while(!v.IsValid(*p));
        }
        return *this;
    }

    // Implements container's begin()
    ComputedIterator(const Context& problem, const bool spare) : v(spare), p(&problem)
    {
        if(!v.IsValid(*p))
            Next();
    }

public:
    // STL-like iterator shall be default contructible. Also implements container's end()
    ComputedIterator() : v(), p(nullptr) {}
    // STL-like iterator shall be copy contructible. The default copy ctor is ok.

    ComputedIterator& operator++() { return Next(); }
    const PerformanceConfig& operator*() const { return v; }
    bool operator!=(ComputedIterator const& other) const
    {
        if(p == other.p)
            if(p == nullptr // Ends are always equal.
               || v == other.v)
                return false;
        return true;
    }
    bool operator==(ComputedIterator const& other) const { return !(*this != other); }

    friend class ComputedContainer<PerformanceConfig, Context>;
};

template <typename PerformanceConfig, typename Context>
class ComputedContainer
{
    Context problem; // Hold a copy make the object independent of the environment.
    bool spare;      // Use spare set of perf configs. Those are usually slower than main set.
                     // Splitting the theoretically available set of perf configs to "main"
                     // and "spare" sets allows for acceleration of the auto-tune process:
                     // * If the "main" set is not empty, then skipping the "spare" set
                     //   avoids wasting time, because the latter is slower by definition.
                     // * Combining "spare" and "main" would lead to exponential growth of
                     //   the resulting container, and thus to exponential slowdown.
                     //
                     // Nevertheless, a Solver is free to either use or not use this capability
                     // (i.e. it is ok for PerformanceConfig(bool) to ignore its parameter).

    /// \note We do not add 'const' to keep the object assignable
    /// for the sake of flexibility. Nevertheless, all element accesses of
    /// the "computed container" shall be const.

public:
    using const_iterator = ComputedIterator<PerformanceConfig, Context>;

    ComputedContainer(const Context& problem_, const bool spare_ = false)
        : problem(problem_), spare(spare_)
    {
    }
    const_iterator begin() const { return {problem, spare}; }
    const_iterator end() const { return {}; }
};

template <typename PerformanceConfig>
class HeartBeat
{
    size_t n_within_beat;
    size_t n_best;
    float best_time; // within beat
    float elapsed_cumulative;
    Timer timer;
    PerformanceConfig best_config;

    void Continue()
    {
        best_time     = std::numeric_limits<float>::max();
        n_within_beat = 0;
        timer.start();
    }

public:
    HeartBeat() : n_within_beat(), n_best(), best_time(), elapsed_cumulative() {}

    void Start()
    {
        elapsed_cumulative = 0.0f;
        best_config        = PerformanceConfig();
        Continue();
    }

    void Monitor(const bool is_recent_failed,
                 const float recent_time,
                 const size_t n_recent,
                 const float total_best,
                 size_t n_failed,
                 size_t n_total,
                 const PerformanceConfig& recent_config)
    {
        ++n_within_beat;
        if(!is_recent_failed && (recent_time < best_time))
        {
            best_time   = recent_time;
            n_best      = n_recent;
            best_config = recent_config;
        }
        const float elapsed = timer.elapsed_ms();
        if(elapsed > 3000)
        {
            elapsed_cumulative += elapsed;
            const float eta_sec =
                n_recent != 0u ? (static_cast<float>(n_total - n_recent) *
                                  (elapsed_cumulative / static_cast<float>(n_recent)) / 1000.0f)
                               : 0.0f; // paraniod
            MIOPEN_LOG_W(n_recent << '/' << n_failed << '/' << n_total << ' ' << total_best
                                  << ", best within recent " << n_within_beat << ": " << best_time
                                  << " #" << n_best << ' ' << best_config << ", ETA:" << eta_sec
                                  << " sec.");
            Continue();
        }
    }
};

/// Solver member function requirements:
/// * GetDefaultPerformanceConfig shall be implemented.
///   - Its return type shall be suitable for instantiation of the ComputedContainer.
/// * GetSolution shall be implemented.
/// * Solution should provide invoker
/// * RunAndMeasureSolution must NOT be implemented. Invoker will be used instead.
///
/// clang-format-off
/// -----------------------------------------------
/// Dataflow:
///      Forward:
///          wei[] (w) --> +--------+
///                        | kernel | --> top[] (y)
///          bot[] (x) --> +--------+
///
///      Backward data:
///          wei[] (w) --> +--------+
///                        | kernel | --> top[] (dx)
///         bot[] (dy) --> +--------+
///
///      Backward WrW:
///         top[] (dx) --> +--------+
///                        | kernel | --> wei[] (dw)
///         bot[] (dy) --> +--------+
/// ------------------------------------------------
/// clang-format-on

template <class Solver, class Top, class Bottom>
using RunAndMeasure_t =
    decltype(std::declval<Solver>().RunAndMeasureSolution(std::declval<miopen::Handle&>(),
                                                          std::declval<Bottom>(),
                                                          std::declval<Top>(),
                                                          std::declval<ConstData_t>(),
                                                          std::declval<ConstData_t>(),
                                                          std::declval<ConvolutionContext>(),
                                                          std::declval<ConvSolution>(),
                                                          std::declval<float&>()));

template <class Solver, class Context>
auto GetAllConfigs(const Solver s, const Context& context)
    -> ComputedContainer<decltype(s.GetDefaultPerformanceConfig(context)), Context>
{
    using PerformanceConfig = decltype(s.GetDefaultPerformanceConfig(context));

    ComputedContainer<PerformanceConfig, Context> primary(context);
    const int primary_size = std::distance(primary.begin(), primary.end());
    ComputedContainer<PerformanceConfig, Context> spare(context, true);
    const int spare_size = std::distance(spare.begin(), spare.end());
    const bool useSpare  = (primary_size == 0);

    ComputedContainer<PerformanceConfig, Context> all_configs = useSpare ? spare : primary;
    const int n_runs_total = useSpare ? spare_size : primary_size;
    MIOPEN_LOG_W(s.SolverDbId() << ": Searching the best solution among " << n_runs_total
                                << (useSpare ? " (spare)" : "") << "...");

    return all_configs;
}

template <class Solver, class Context>
std::vector<ConvSolution> GetAllSolutions(const Solver s, const Context& context_)
{
    auto context                  = context_;
    context.is_for_generic_search = true;

    auto all_configs = GetAllConfigs(s, context);

    std::vector<ConvSolution> solutions;
    for(const auto& current_config : all_configs)
    {
        ConvSolution current_solution = s.GetSolution(context, current_config);
        solutions.push_back(current_solution);
    }
    return solutions;
}

template <class Solver, class Context, class Problem>
auto GenericSearch(const Solver s,
                   const Context& ctx,
                   const Problem& problem,
                   const AnyInvokeParams& invoke_ctx)
{
    std::ignore = problem;
    return GenericSearch(s, ctx, invoke_ctx);
}

std::size_t GetTuningIterationsMax();

template <class Solver, class Context>
auto GenericSearch(const Solver s, const Context& context_, const AnyInvokeParams& invoke_ctx_)
    -> decltype(s.GetDefaultPerformanceConfig(context_))
{
    static_assert(
        !(HasMember<RunAndMeasure_t, Solver, ConstData_t, Data_t>{} ||
          HasMember<RunAndMeasure_t, Solver, Data_t, ConstData_t>{}),
        "RunAndMeasure is obsolete. Solvers should implement auto-tune evaluation in invoker");

    auto context                  = context_;
    context.is_for_generic_search = true;

    using PerformanceConfig = decltype(s.GetDefaultPerformanceConfig(context));
    PerformanceConfig best_config;
    const auto default_solution = s.GetSolution(context, s.GetDefaultPerformanceConfig(context));
    const auto invoke_ctx       = [invoke_ctx_]() {
        auto copy = invoke_ctx_;
        copy.SetInvokeType(InvokeType::AutoTune);
        return copy;
    }();

    auto& profile_h = context.GetStream();
    AutoEnableProfiling enableProfiling{profile_h};

    auto all_configs = GetAllConfigs(s, context);
    const std::size_t n_runs_total =
        std::min(static_cast<std::size_t>(std::distance(all_configs.begin(), all_configs.end())),
                 GetTuningIterationsMax());

    bool is_passed  = false; // left false only if all iterations failed.
    float best_time = std::numeric_limits<float>::max();
    size_t n_failed = 0;
    size_t n_best   = 0;
    HeartBeat<PerformanceConfig> heartbeat;
    heartbeat.Start();

    if(!miopen::IsCacheDisabled()) // Otherwise precompilation is useless.
    {
        std::vector<KernelInfo> kernels;
        size_t n_current = 0;
        for(const auto& current_config : all_configs)
        {
            if(n_current >= n_runs_total)
                break;
            ConvSolution current_solution = s.GetSolution(context, current_config);
            for(auto&& kernel : current_solution.construction_params)
            {
                if(profile_h.HasProgram(kernel.kernel_file, kernel.comp_options))
                    continue;
                kernels.push_back(kernel);
            }
            ++n_current;
        }
        std::ignore = PrecompileKernels(profile_h, kernels);
    }

    if(!IsEnabled(MIOPEN_DEBUG_COMPILE_ONLY{}))
    {
        size_t n_current = 0;
        for(const auto& current_config : all_configs)
        {
            if(n_current >= n_runs_total)
                break;

            float elapsed_time = 0.0f;
            int ret            = 0;
            MIOPEN_LOG_I2('#' << n_current << '/' << n_failed << '/' << n_runs_total << ' '
                              << current_config);

            ConvSolution current_solution;
            Invoker invoker;

            try
            {
                current_solution = s.GetSolution(context, current_config);
                if(default_solution.workspace_sz != current_solution.workspace_sz)
                {
                    ret = -2;
                    MIOPEN_LOG_E('#' << n_current << " (" << n_runs_total << ") "
                                     << "Workspace size should not depend on PerformanceConfig: "
                                     << default_solution.workspace_sz
                                     << " != " << current_solution.workspace_sz);
                }

                invoker = profile_h.PrepareInvoker(*current_solution.invoker_factory,
                                                   current_solution.construction_params);
                invoker(profile_h, invoke_ctx);
                elapsed_time = profile_h.GetKernelTime();
            }
            catch(...)
            {
                ret = 1;
            }

            MIOPEN_LOG_T("##"
                         << "(n_current, n_failed, n_runs_total):  " << n_current << '/' << n_failed
                         << '/' << n_runs_total << " elapsed_time: " << elapsed_time
                         << ", best_time: " << best_time << ", " << current_config);

            if(ret == 0)
            {
                // Smooth the jitter of measurements:
                // If the 1st probe is NOT too bad (measured time <= 1.05 * best known time),
                // then re-run it 4 times more and compute average time,
                // and decide using average of all 5 attempts vs. the best.
                if(elapsed_time / best_time < 1.05f)
                {
                    MIOPEN_LOG_I2("Finding average for: " << elapsed_time << " / " << best_time
                                                          << " = " << (elapsed_time / best_time));

                    try
                    {
                        for(int i = 0; i < 4; ++i)
                        {
                            invoker(profile_h, invoke_ctx);
                            elapsed_time += profile_h.GetKernelTime();
                        }
                    }
                    catch(...)
                    {
                        ret = 1;
                    }

                    if(ret == 0)
                    {
                        is_passed = true;
                        elapsed_time /= 5;
                        if(elapsed_time < best_time)
                        {
                            MIOPEN_LOG_I('#' << n_current << '/' << n_failed << '/' << n_runs_total
                                             << ' ' << elapsed_time << " < " << best_time << ' '
                                             << current_config);
                            best_config = current_config;
                            best_time   = elapsed_time;
                            n_best      = n_current;
                        }
                        else
                        {
                            MIOPEN_LOG_I2("Average is not better: " << elapsed_time
                                                                    << " >= " << best_time);
                        }
                    }
                }
            }

            // Banchmarked kernels will not be used anymore.
            // Now we can delete Program objects that belong to OCL/HIP
            // runtime and free the associated resources (memory, file handles...)
            for(const auto& kernelInfo : current_solution.construction_params)
                profile_h.ClearProgram(kernelInfo.kernel_file, kernelInfo.comp_options);

            if(ret != 0)
            {
                MIOPEN_LOG_E('#' << n_current << " (" << n_runs_total << ") "
                                 << " Failed rc=" << ret);
                ++n_failed;
            }
            heartbeat.Monitor(ret != 0,
                              elapsed_time,
                              n_current,
                              best_time,
                              n_failed,
                              n_runs_total,
                              current_config);
            ++n_current;
        }
    }
    else
    {
        MIOPEN_THROW(miopenStatusGpuOperationsSkipped,
                     "Running kernels on GPU is disabled. Search skipped");
    }

    MIOPEN_LOG_W("Done: " << n_runs_total << '/' << n_failed << '/' << n_runs_total << ", best #"
                          << n_best << ' ' << best_time << ' ' << best_config);

    if(!is_passed)
        MIOPEN_THROW("Search failed");
    // Run once with the default config and show score.

    const auto& invoker = profile_h.PrepareInvoker(*default_solution.invoker_factory,
                                                   default_solution.construction_params);
    invoker(profile_h, invoke_ctx);
    const auto default_time = profile_h.GetKernelTime();
    const auto score        = (best_time > 0.0f) ? default_time / best_time : 0.0f;
    MIOPEN_LOG_W("...Score: " << score << " (default time " << default_time << ')');

    return best_config;
}

} // namespace solver
} // namespace miopen

#endif // GUARD_MIOPEN_GENERIC_SEARCH_HPP_
