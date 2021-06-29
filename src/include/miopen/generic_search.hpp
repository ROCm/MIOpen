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

#include <miopen/config.h>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/env.hpp>

#include <vector>
#include <cstdlib>
#include <limits>
#include <iterator>
#include <chrono>
#include <cassert>

#include <miopen/conv/context.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/timer.hpp>

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
/// - SetNextValue()
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
                if(!v.SetNextValue())
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
               ||
               v == other.v)
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
                                  << ", best within recent "
                                  << n_within_beat
                                  << ": "
                                  << best_time
                                  << " #"
                                  << n_best
                                  << ' '
                                  << best_config
                                  << ", ETA:"
                                  << eta_sec
                                  << " sec.");
            Continue();
        }
    }
};

inline void InitRandomly(std::vector<float>& vec, const double offset, const double factor)
{
    float* p = vec.data();
    for(unsigned long i = 0; i < vec.size(); ++i)
        *p++ = static_cast<float>(
            (rand() * (1.0 / RAND_MAX) + offset) * // NOLINT (concurrency-mt-unsafe)
            factor);
}

inline void InitRandomly(std::vector<float>& vec)
{
    float* p = vec.data();
    for(unsigned long i = 0; i < vec.size(); ++i)
        *p++ = static_cast<float>(rand() * (1.0 / RAND_MAX)); // NOLINT (concurrency-mt-unsafe)
}

inline size_t divide_round_plus_inf(const size_t x, const unsigned y)
{
    assert(y > 0);
    if(x % y != 0)
        return x / y + 1;
    return x / y;
}

/// Solver member function requirements:
/// * GetPerformanceConfig shall be implemented.
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

// This is to detect new solvers attempting to use obsolete functionality
namespace detail {
template <typename...>
using void_t = void;

template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
struct detector
{
    using value_t = std::false_type;
    using type    = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, void_t<Op<Args...>>, Op, Args...>
{
    using value_t = std::true_type;
    using type    = Op<Args...>;
};

} // namespace detail

template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<void, void, Op, Args...>::value_t;

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
auto GenericSearch(const Solver s, const Context& context_, const AnyInvokeParams& invoke_ctx_)
    -> decltype(s.GetPerformanceConfig(context_))
{
    static_assert(
        !(is_detected<RunAndMeasure_t, Solver, ConstData_t, Data_t>{} ||
          is_detected<RunAndMeasure_t, Solver, Data_t, ConstData_t>{}),
        "RunAndMeasure is obsolete. Solvers should implement auto-tune evaluation in invoker");

    auto context                  = context_;
    context.is_for_generic_search = true;

    using PerformanceConfig = decltype(s.GetPerformanceConfig(context));
    PerformanceConfig best_config;
    const auto default_solution = s.GetSolution(context, s.GetPerformanceConfig(context));
    const auto invoke_ctx       = [invoke_ctx_]() {
        auto copy = invoke_ctx_;
        copy.SetInvokeType(InvokeType::AutoTune);
        return copy;
    }();

    auto& profile_h = context.GetStream();
    AutoEnableProfiling enableProfiling{profile_h};

    const ComputedContainer<PerformanceConfig, Context> main(context);
    const int main_size = std::distance(main.begin(), main.end());
    const ComputedContainer<PerformanceConfig, Context> spare(context, true);
    const int spare_size = std::distance(spare.begin(), spare.end());
    const bool useSpare  = (main_size == 0);

    const ComputedContainer<PerformanceConfig, Context> all_configs = useSpare ? spare : main;
    const int n_runs_total = useSpare ? spare_size : main_size;
    MIOPEN_LOG_W(SolverDbId(s) << ": Searching the best solution among " << n_runs_total
                               << (useSpare ? " (spare)" : "")
                               << "...");

    bool is_passed  = false; // left false only if all iterations failed.
    float best_time = std::numeric_limits<float>::max();
    size_t n_failed = 0;
    size_t n_best   = 0;
    HeartBeat<PerformanceConfig> heartbeat;
    heartbeat.Start();

// PrecompileKernels call saves to binary_cache, this needs to be escaped if KERN_CACHE is not on.
#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
    std::vector<KernelInfo> kernels;
    for(const auto& current_config : all_configs)
    {
        ConvSolution current_solution = s.GetSolution(context, current_config, true);
        for(auto&& kernel : current_solution.construction_params)
        {
            if(profile_h.HasProgram(kernel.kernel_file, kernel.comp_options))
                continue;
            kernels.push_back(kernel);
        }
    }
    std::ignore = PrecompileKernels(profile_h, kernels);
#endif

    if(!IsEnabled(MIOPEN_DEBUG_COMPILE_ONLY{}))
    {
        size_t n_current = 0;
        for(const auto& current_config : all_configs)
        {
            float elapsed_time = 0.0f;
            int ret            = 0;
            MIOPEN_LOG_I2('#' << n_current << '/' << n_failed << '/' << n_runs_total << ' '
                              << current_config);

            ConvSolution current_solution;
            Invoker invoker;

            try
            {
                current_solution = s.GetSolution(context, current_config, true);
                if(default_solution.workspce_sz != current_solution.workspce_sz)
                {
                    ret = -2;
                    MIOPEN_LOG_E('#' << n_current << " (" << n_runs_total << ") "
                                     << "Workspace size should not depend on PerformanceConfig: "
                                     << default_solution.workspce_sz
                                     << " != "
                                     << current_solution.workspce_sz);
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
                         << "(n_current, n_failed, n_runs_total):  "
                         << n_current
                         << '/'
                         << n_failed
                         << '/'
                         << n_runs_total
                         << " elapsed_time: "
                         << elapsed_time
                         << ", best_time: "
                         << best_time
                         << ", "
                         << current_config);

            if(ret == 0)
            {
                // Smooth the jitter of measurements:
                // If the 1st probe is NOT too bad (measured time <= 1.05 * best known time),
                // then re-run it 4 times more and compute average time,
                // and decide using average of all 5 attempts vs. the best.
                if(elapsed_time / best_time < 1.05f)
                {
                    MIOPEN_LOG_I2("Finding average for: " << elapsed_time << " / " << best_time
                                                          << " = "
                                                          << (elapsed_time / best_time));

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
                                             << ' '
                                             << elapsed_time
                                             << " < "
                                             << best_time
                                             << ' '
                                             << current_config);
                            best_config = current_config;
                            best_time   = elapsed_time;
                            n_best      = n_current;
                        }
                        else
                        {
                            MIOPEN_LOG_I2(
                                "Average is not better: " << elapsed_time << " >= " << best_time);
                        }
                    }
                }
            }

            if(ret != 0)
            {
                MIOPEN_LOG_E('#' << n_current << " (" << n_runs_total << ") "
                                 << " Failed rc="
                                 << ret);
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
                          << n_best
                          << ' '
                          << best_time
                          << ' '
                          << best_config);
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
