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
#include <miopen/config.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/env.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/handle.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/logger.hpp>
#include <miopen/timer.hpp>
#include <miopen/mt_queue.hpp>
#include <miopen/generic_search_controls.hpp>

#include <algorithm>
#include <vector>
#include <cstdlib>
#include <limits>
#include <iterator>
#include <chrono>
#include <cassert>
#include <random>

namespace miopen {
namespace solver {

namespace debug {
// This struct is not MT-safe, meaning one should use it before starting threads, thus avoiding
// constructing it inside a worker thread.
/// \todo This class should be moved out of the library
struct MIOPEN_INTERNALS_EXPORT TuningIterationScopedLimiter
{
    TuningIterationScopedLimiter(std::size_t new_limit);
    ~TuningIterationScopedLimiter();

private:
    std::optional<std::size_t> old_limit;
};
} // namespace debug

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
/// - SetNextValue(const Problem& p)
///     Advances instance value to the next available value and returns true.
///     If max value reached, returns false.
/// - IsValid(const Context& c, const Problem& p) const
///     Checks if instance is valid for the given c.
///     For convolutions, Context represents a problem configuration.
/// - operator==(const PerformanceConfig&)
///     Ordinary semantics.
template <typename PerformanceConfig, typename Context, typename Problem>
class ComputedContainer;

template <typename PerformanceConfig, typename Context, typename Problem>
class ComputedIterator
{
    PerformanceConfig v;
    const Context* c; // For Next().
    const Problem* p; // For Next().

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
            } while(!v.IsValid(*c, *p));
        }
        return *this;
    }

    // Implements container's begin()
    ComputedIterator(const Context& context, const Problem& problem, const bool spare)
        : v(spare), c(&context), p(&problem)
    {
        if(!v.IsValid(*c, *p))
            Next();
    }

public:
    using iterator_category = std::input_iterator_tag;
    using value_type        = PerformanceConfig;
    using difference_type   = int;
    using pointer           = PerformanceConfig*;
    using reference         = PerformanceConfig&;
    // STL-like iterator shall be default contructible. Also implements container's end()
    ComputedIterator() : v(), c(nullptr), p(nullptr) {}
    // STL-like iterator shall be copy contructible. The default copy ctor is ok.

    ComputedIterator& operator++() { return Next(); }
    const PerformanceConfig& operator*() const { return v; }
    bool operator!=(ComputedIterator const& other) const
    {
        if(p == other.p)
        {
            if(p == nullptr // Ends are always equal.
               || v == other.v)
                return false;
        }
        return true;
    }
    bool operator==(ComputedIterator const& other) const { return !(*this != other); }

    friend class ComputedContainer<PerformanceConfig, Context, Problem>;
};

template <typename PerformanceConfig, typename Context, typename Problem>
class ComputedContainer
{
    Context context; // Hold a copy make the object independent of the environment.
    Problem problem; //
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
    using const_iterator = ComputedIterator<PerformanceConfig, Context, Problem>;

    ComputedContainer(const Context& context_, const Problem& problem_, const bool spare_ = false)
        : context(context_), problem(problem_), spare(spare_)
    {
    }
    const_iterator begin() const { return {context, problem, spare}; }
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

template <class Solver, class Context, class Problem>
auto GetAllConfigs(const Solver s, const Context& context, const Problem& problem)
    -> ComputedContainer<decltype(s.GetDefaultPerformanceConfig(context, problem)),
                         Context,
                         Problem>
{
    using PerformanceConfig = decltype(s.GetDefaultPerformanceConfig(context, problem));

    const ComputedContainer<PerformanceConfig, Context, Problem> primary(context, problem);
    const int primary_size = std::distance(primary.begin(), primary.end());
    const ComputedContainer<PerformanceConfig, Context, Problem> spare(context, problem, true);
    const int spare_size = std::distance(spare.begin(), spare.end());
    const bool useSpare  = (primary_size == 0);

    ComputedContainer<PerformanceConfig, Context, Problem> all_configs = useSpare ? spare : primary;
    const int n_runs_total = useSpare ? spare_size : primary_size;
    MIOPEN_LOG_W(s.SolverDbId() << ": Searching the best solution among " << n_runs_total
                                << (useSpare ? " (spare)" : "") << "...");

    return all_configs;
}

template <class Solver, class Context, class Problem>
std::vector<ConvSolution>
GetAllSolutions(const Solver s, const Context& context_, const Problem& problem)
{
    auto context                  = context_;
    context.is_for_generic_search = true;

    auto all_configs = GetAllConfigs(s, context, problem);

    std::vector<ConvSolution> solutions;
    for(const auto& current_config : all_configs)
    {
        ConvSolution current_solution = s.GetSolution(context, problem, current_config);
        solutions.push_back(current_solution);
    }
    return solutions;
}

std::size_t GetTuningIterationsMax();
std::chrono::milliseconds GetTuningTimeMax(); // returns the max allowed time in milliseconds
std::size_t GetTuningThreadsMax();

template <typename PerformanceConfig, typename Solver, typename Context, typename Problem>
void CompileAgent(size_t thread_index,
                  size_t total_threads,
                  const Solver& s,
                  const Context& context,
                  const Problem& problem,
                  std::vector<PerformanceConfig>& data,
                  ThreadSafeQueue<std::tuple<PerformanceConfig, ConvSolution, bool>>& comp_queue)
{
    const auto start_time =
        std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now());
    const auto data_size   = data.size();
    const auto time_budget = GetTuningTimeMax();
    const auto& profile_h  = context.GetStream();
    // start the counter
    for(auto idx = thread_index; idx < data_size; idx += total_threads)
    {
        // Check if we are out of time
        const auto current_time = std::chrono::time_point_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now());
        if(current_time - start_time > time_budget)
        {
            MIOPEN_LOG_I2("Thread: " << thread_index << " Done, exhausted time budget");
            auto tmp = std::make_tuple<PerformanceConfig, ConvSolution, bool>({}, {}, true);
            comp_queue.push(std::move(tmp));
            break;
        }
        auto& current_config          = data.at(idx);
        ConvSolution current_solution = s.GetSolution(context, problem, current_config);
        for(const auto& kernel : current_solution.construction_params)
        {
            if(profile_h.HasProgram(kernel.kernel_file, kernel.comp_options))
                continue;
            std::ignore = profile_h.LoadProgram(kernel.kernel_file, kernel.comp_options, "");
        }
        auto tup = std::make_tuple<PerformanceConfig, ConvSolution, bool>(
            std::move(current_config), std::move(current_solution), false);
        comp_queue.push(std::move(tup));
    }
    MIOPEN_LOG_I2("Thread: " << thread_index << " Done, completed tuning");
}

template <class Solver, class Context, class Problem>
auto GenericSearch(const Solver s,
                   const Context& context_,
                   const Problem& problem,
                   const AnyInvokeParams& invoke_ctx_)
    -> decltype(s.GetDefaultPerformanceConfig(context_, problem))
{
    auto context                  = context_;
    context.is_for_generic_search = true;

    using PerformanceConfig = decltype(s.GetDefaultPerformanceConfig(context, problem));
    PerformanceConfig best_config;
    const auto default_solution =
        s.GetSolution(context, problem, s.GetDefaultPerformanceConfig(context, problem));
    const auto invoke_ctx = [invoke_ctx_]() {
        auto copy = invoke_ctx_;
        copy.SetInvokeType(InvokeType::AutoTune);
        return copy;
    }();

    auto& profile_h = context.GetStream();
    const AutoEnableProfiling enableProfiling{profile_h};

    auto tmp_all_configs = GetAllConfigs(s, context, problem);
    // For random access
    std::vector<PerformanceConfig> all_configs;
    std::copy(tmp_all_configs.begin(), tmp_all_configs.end(), std::back_inserter(all_configs));
    // shuffle the configs
    std::random_device rd{};
    auto rng = std::default_random_engine{rd()};
    std::shuffle(all_configs.begin(), all_configs.end(), rng);
    std::size_t n_runs_total = std::min(all_configs.size(), GetTuningIterationsMax());
    all_configs.resize(n_runs_total);
    std::size_t patience = env::value(MIOPEN_TUNING_PATIENCE);

    if(all_configs.empty())
    {
        const auto default_config = s.GetDefaultPerformanceConfig(context, problem);

        if(default_config.IsValid(context, problem))
        {
            all_configs.emplace_back(default_config);
            n_runs_total += 1;
        }
        else
        {
            const auto id = s.SolverDbId();
            MIOPEN_THROW("Generic search has failed. Solver " + id +
                         " cannot produce any valid configuration.");
        }
    }

    bool is_passed  = false; // left false only if all iterations failed.
    float best_time = std::numeric_limits<float>::max();
    size_t n_failed = 0;
    size_t n_best   = 0;
    HeartBeat<PerformanceConfig> heartbeat;
    heartbeat.Start();

    const auto total_threads = GetTuningThreadsMax();

    ThreadSafeQueue<std::tuple<PerformanceConfig, ConvSolution, bool>> solution_queue;
    std::vector<std::thread> compile_agents;
    compile_agents.reserve(total_threads);
    for(auto idx = 0; idx < total_threads; ++idx)
    {
        compile_agents.emplace_back(CompileAgent<PerformanceConfig, Solver, Context, Problem>,
                                    idx,
                                    total_threads,
                                    std::cref(s),
                                    std::cref(context),
                                    std::cref(problem),
                                    std::ref(all_configs),
                                    std::ref(solution_queue));
    }

    if(!env::enabled(MIOPEN_DEBUG_COMPILE_ONLY))
    {
        size_t n_current       = 0;
        size_t last_imprv      = 0;
        auto threads_remaining = total_threads;
        while(true)
        {
            if(n_current >= n_runs_total)
            {
                MIOPEN_LOG_I2("Ending Search by total runs: " << n_runs_total);
                break;
            }
            if(last_imprv >= patience)
            {
                MIOPEN_LOG_I2("Ending Search by patience: " << patience);
                break;
            }

            last_imprv++;
            MIOPEN_LOG_I2("Waiting for item in queue");
            const auto kinder     = solution_queue.pop();
            auto current_config   = std::get<0>(kinder);
            auto current_solution = std::get<1>(kinder);

            if(std::get<2>(kinder))
            {
                threads_remaining--;
                if(threads_remaining == 0)
                {
                    break;
                }
                else
                {
                    continue;
                }
            }

            float elapsed_time = 0.0f;
            int ret            = 0;
            MIOPEN_LOG_I2('#' << n_current << '/' << n_failed << '/' << n_runs_total << ' '
                              << current_config);

            Invoker invoker;

            try
            {
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
            catch(const std::exception& e)
            {
                MIOPEN_LOG_E("Error: Exception encountered : " << e.what());
                ret = 1;
            }
            catch(...)
            {
                MIOPEN_LOG_E("Error: Unknown exception thrown.");
                ret = 1;
            }

            MIOPEN_LOG_T("##"
                         << "(n_current, n_failed, n_runs_total):  " << n_current << '/' << n_failed
                         << '/' << n_runs_total << " elapsed_time: " << elapsed_time
                         << ", best_time: " << best_time << ", " << current_config);

            if(ret == 0)
            {
                // Smooth the jitter of measurements:
                // If the 1st probe is NOT too bad (measured time <= 1.10 * best known time),
                // then re-run it 9 times more and compute average time,
                // and decide using average of all 10 attempts vs. the best.
                constexpr int N_RUNS = 10;
                if(elapsed_time / best_time < 1.10f)
                {
                    MIOPEN_LOG_I2("Finding average for: " << elapsed_time << " / " << best_time
                                                          << " = " << (elapsed_time / best_time));

                    try
                    {
                        for(int i = 1; i < N_RUNS; ++i)
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
                        elapsed_time /= N_RUNS;
                        if(elapsed_time < best_time)
                        {
                            MIOPEN_LOG_I('#' << n_current << '/' << n_failed << '/' << n_runs_total
                                             << ' ' << elapsed_time << " < " << best_time << ' '
                                             << current_config);
                            best_config = current_config;
                            best_time   = elapsed_time;
                            n_best      = n_current;
                            last_imprv  = 0;
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

    for(auto& agent : compile_agents)
        agent.join();

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
