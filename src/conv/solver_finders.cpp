/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/conv/solver_finders.hpp>

#include <miopen/conv_algo_name.hpp>
#include <miopen/config.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/perf_field.hpp>
#include <miopen/conv/problem_description.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEVICE_ARCH)

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_GEMM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_WINOGRAD)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_FFT)

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_FIND_CONV_INSUFFICIENT_WORKSPACE_ALLOW_FINDDB_UPDATE)

namespace miopen {

namespace conv {
namespace {

class DirectSolverFinder : public SolversFinderMixin<ProblemDescription, ConvFindParameters>
{
protected:
    AlgorithmName GetAlgorithmName(const ProblemDescription& problem) const override
    {
        return AlgorithmName{ConvolutionAlgoToDirectionalString(miopenConvolutionAlgoDirect,
                                                                problem.GetDirection())};
    }

    bool IsEnabled(const ExecutionContext& /*ctx*/,
                   const ProblemDescription& /*problem*/,
                   const ConvFindParameters& parameters) const override
    {
        return !parameters.use_winograd_only && !env::disabled(MIOPEN_DEBUG_CONV_DIRECT);
    }

    std::vector<solver::ConvSolution> FindImpl(const ExecutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx,
                                               const ConvFindParameters&,
                                               const std::optional<FindOptions>&) const override
    {
        /// \todo: actually use FindOptions
        return problem.GetDirection() != conv::Direction::BackwardWeights
                   ? FindAllDirectSolutions(ctx, problem, invoke_ctx)
                   : FindAllBwdWrW2DSolutions(ctx, problem, invoke_ctx);
    }
};

class ImplicitGemmSolverFinder : public SolversFinderMixin<ProblemDescription, ConvFindParameters>
{
protected:
    AlgorithmName GetAlgorithmName(const ProblemDescription& problem) const override
    {
        return AlgorithmName{ConvolutionAlgoToDirectionalString(miopenConvolutionAlgoImplicitGEMM,
                                                                problem.GetDirection())};
    }

    bool IsEnabled(const ExecutionContext& /*ctx*/,
                   const ProblemDescription& /*problem*/,
                   const ConvFindParameters& parameters) const override
    {
        return !parameters.use_winograd_only && !env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM);
    }

    std::vector<solver::ConvSolution> FindImpl(const ExecutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx,
                                               const ConvFindParameters&,
                                               const std::optional<FindOptions>&) const override
    {
        /// \todo: actually use FindOptions
        return problem.GetDirection() != conv::Direction::BackwardWeights
                   ? FindAllImplicitGemmSolutions(ctx, problem, invoke_ctx)
                   : FindImplicitGemmWrWAllSolutions(ctx, problem, invoke_ctx);
    }
};

class FftSolverFinder : public SolversFinderMixin<ProblemDescription, ConvFindParameters>
{
protected:
    AlgorithmName GetAlgorithmName(const ProblemDescription& problem) const override
    {
        return AlgorithmName{
            ConvolutionAlgoToDirectionalString(miopenConvolutionAlgoFFT, problem.GetDirection())};
    }

    bool IsEnabled(const ExecutionContext& /*ctx*/,
                   const ProblemDescription& problem,
                   const ConvFindParameters& parameters) const override
    {
        return !parameters.use_winograd_only &&
               problem.GetDirection() != conv::Direction::BackwardWeights &&
               !env::disabled(MIOPEN_DEBUG_CONV_FFT);
    }

    std::vector<solver::ConvSolution> FindImpl(const ExecutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx,
                                               const ConvFindParameters&,
                                               const std::optional<FindOptions>&) const override
    {
        /// \todo: actually use FindOptions
        return FindAllFFTSolutions(ctx, problem, invoke_ctx);
    }
};

class GemmSolverFinder : public SolversFinderMixin<ProblemDescription, ConvFindParameters>
{
protected:
    AlgorithmName GetAlgorithmName(const ProblemDescription& problem) const override
    {
        return AlgorithmName{
            ConvolutionAlgoToDirectionalString(miopenConvolutionAlgoGEMM, problem.GetDirection())};
    }

    bool IsEnabled(const ExecutionContext& /*ctx*/,
                   const ProblemDescription& /*problem*/,
                   const ConvFindParameters& parameters) const override
    {
        return !parameters.use_winograd_only && !env::disabled(MIOPEN_DEBUG_CONV_GEMM);
    }

    std::vector<solver::ConvSolution> FindImpl(const ExecutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx,
                                               const ConvFindParameters&,
                                               const std::optional<FindOptions>&) const override
    {
        /// \todo: actually use FindOptions
        return FindAllGemmSolutions(ctx, problem, invoke_ctx);
    }
};

class WinogradSolverFinder : public SolversFinderMixin<ProblemDescription, ConvFindParameters>
{
protected:
    AlgorithmName GetAlgorithmName(const ProblemDescription& problem) const override
    {
        return AlgorithmName{ConvolutionAlgoToDirectionalString(miopenConvolutionAlgoWinograd,
                                                                problem.GetDirection())};
    }

    bool IsEnabled(const ExecutionContext& /*ctx*/,
                   const ProblemDescription& /*problem*/,
                   const ConvFindParameters& /*parameters*/) const override
    {
        return !env::disabled(MIOPEN_DEBUG_CONV_WINOGRAD);
    }

    std::vector<solver::ConvSolution> FindImpl(const ExecutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx,
                                               const ConvFindParameters& parameters,
                                               const std::optional<FindOptions>&) const override
    {
        /// \todo: actually use FindOptions
        auto ctx_copy = ctx;
        if(parameters.use_winograd_only)
            ctx_copy.use_dynamic_solutions_only = true;
        return problem.GetDirection() != conv::Direction::BackwardWeights
                   ? FindAllWinogradSolutions(ctx_copy, problem, invoke_ctx)
                   : FindWinogradWrWAllSolutions(ctx_copy, problem, invoke_ctx);
    }
};

} // namespace

const std::vector<std::unique_ptr<ISolversFinder>>& GetConvSolverFinders()
{
    static const auto finders = []() {
        auto tmp = std::vector<std::unique_ptr<ISolversFinder>>{};
        tmp.emplace_back(std::make_unique<WinogradSolverFinder>());
        tmp.emplace_back(std::make_unique<DirectSolverFinder>());
        tmp.emplace_back(std::make_unique<ImplicitGemmSolverFinder>());
        tmp.emplace_back(std::make_unique<GemmSolverFinder>());
        tmp.emplace_back(std::make_unique<FftSolverFinder>());
        return tmp;
    }();

    return finders;
}

} // namespace conv

/// Register invoker only for the best solution within algorithm.
/// Add all solutions to the find-db record.
static void EvaluateInvokers(Handle& handle,
                             const std::vector<solver::ConvSolution>& solutions,
                             const AlgorithmName& algorithm_name,
                             const NetworkConfig& network_config,
                             const AnyInvokeParams& invoke_ctx,
                             DbRecord& record,
                             bool& is_result_optimal)
{
    const auto arch = env::value(MIOPEN_DEVICE_ARCH);
    if(!arch.empty())
        return;

    auto selected     = miopen::solver::ConvSolution{miopenStatusUnknownError};
    auto best         = std::numeric_limits<float>::max();
    auto best_invoker = Invoker{};

    for(const auto& sol : solutions)
    {
        if(!conv::IsEnoughWorkspace(
               "EvaluateInvokers", solver::Id{sol.solver_id}, sol.workspace_sz, &invoke_ctx))
        {
            // Providing smaller workspace may result in the selection of a slow convolution
            // algorithm, and therefore affect library performance. Moreover, sub-optimal data may
            // be cached in the user's find-db. This means that the performance drop will become
            // persistent, i.e. even providing sufficient workspace won't restore the performance.
            // To get rid of this problem, the user will need to either remove the user's find-db,
            // or repeat miopenFindConvolution*() with affected convolution configs in Normal Find
            // Mode (the latter will overwrite sub-optimal user's find-db records).
            //
            // That is why we do not write sub-optimal results into persistent find-db (on disk)
            // unless this is explicitly enabled via environment setting.
            if(!env::enabled(MIOPEN_FIND_CONV_INSUFFICIENT_WORKSPACE_ALLOW_FINDDB_UPDATE))
                is_result_optimal = false;
            continue;
        }

        if(!sol.invoker_factory)
            MIOPEN_THROW("Invoker is not provided by solver " + sol.solver_id);

        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
        try
        {
            invoker(handle, invoke_ctx); // Dry-run once to warm-up.

            constexpr int N_RUNS = 5;
            using elapsed_t      = decltype(handle.GetKernelTime());
            auto elapsed         = static_cast<elapsed_t>(0);
            for(int i = 0; i < N_RUNS; ++i)
            {
                invoker(handle, invoke_ctx);
                elapsed += handle.GetKernelTime();
            }
            elapsed /= static_cast<elapsed_t>(N_RUNS);

            record.SetValues(sol.solver_id, FindDbData{elapsed, sol.workspace_sz, algorithm_name});

            MIOPEN_LOG_I(sol << ": " << elapsed << (elapsed < best ? " < " : " >= ") << best);
            if(elapsed < best)
            {
                best         = elapsed;
                selected     = sol;
                best_invoker = invoker;
            }
        }
        catch(const miopen::Exception& ex)
        {
            MIOPEN_LOG_E(ex.what());
        }
    }

    if(selected.Succeeded())
    {
        handle.RegisterInvoker(best_invoker, network_config, selected.solver_id, algorithm_name);
        MIOPEN_LOG_I("Selected: " << selected << ": " << best
                                  << ", workspace_sz = " << selected.workspace_sz);
    }
}

static inline void AppendPointersToElements(const std::vector<miopen::solver::ConvSolution>& from,
                                            std::vector<const miopen::solver::ConvSolution*>& to)
{
    std::transform(from.begin(),
                   from.end(),
                   std::back_inserter(to),
                   [](const miopen::solver::ConvSolution& s) { return &s; });
}

bool FindCore(const AnyInvokeParams& invoke_ctx,
              DbRecord& record,
              const ExecutionContext& ctx,
              const ProblemDescriptionBase& problem,
              const PrimitiveFindParameters& parameters,
              const std::vector<std::unique_ptr<ISolversFinder>>& finders,
              const std::optional<FindOptions>& options)
{
    auto& handle = ctx.GetStream();

    // Find
    auto solutions = std::map<AlgorithmName, std::vector<solver::ConvSolution>>{};
    std::transform(
        finders.begin(), finders.end(), std::inserter(solutions, solutions.end()), [&](auto&& f) {
            return std::make_pair(f->GetAlgorithmName(problem),
                                  f->Find(ctx, problem, invoke_ctx, parameters, options));
        });

    // Precompile
    {
        auto all = std::vector<const miopen::solver::ConvSolution*>{};
        all.reserve(
            std::accumulate(solutions.begin(), solutions.end(), 0, [](auto&& before, auto&& ss) {
                return before + ss.second.size();
            }));
        for(const auto& ss : solutions)
            AppendPointersToElements(ss.second, all);
        PrecompileSolutions(handle, all);
    }

    // Evaluate Invokers
    AutoEnableProfiling enableProfiling{handle};
    const auto network_config = problem.MakeNetworkConfig();
    auto is_result_optimal    = true;

    for(const auto& ss : solutions)
    {
        if(!ss.second.empty())
            EvaluateInvokers(
                handle, ss.second, ss.first, network_config, invoke_ctx, record, is_result_optimal);
    }
    return is_result_optimal;
}

namespace conv {

bool IsAlgorithmDisabled(miopenConvAlgorithm_t algo)
{
    switch(algo)
    { // clang-format off
    case miopenConvolutionAlgoGEMM:
        return !MIOPEN_USE_GEMM || env::disabled(MIOPEN_DEBUG_CONV_GEMM);
    case miopenConvolutionAlgoDirect:
        return env::disabled(MIOPEN_DEBUG_CONV_DIRECT);
    case miopenConvolutionAlgoFFT:
        return env::disabled(MIOPEN_DEBUG_CONV_FFT);
    case miopenConvolutionAlgoWinograd:
        return env::disabled(MIOPEN_DEBUG_CONV_WINOGRAD);
    case miopenConvolutionAlgoImplicitGEMM:
        return env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM);
    default: // Disable future algos by default to enforce explicit handling:
        return true;
    } // clang-format on
}

bool IsEnoughWorkspace(std::string_view where,
                       const miopen::solver::Id& solver_id,
                       const std::size_t required_size,
                       const miopen::AnyInvokeParams* const invokeParams)
{
    if(invokeParams != nullptr && required_size > 0)
    {
        const auto provided_size = invokeParams->GetWorkspaceSize();
        const auto provided_ptr  = invokeParams->GetWorkspace();
        if(provided_ptr == nullptr || provided_size < required_size)
        {
            MIOPEN_LOG_W("[" << where << "] Solver <" << solver_id.ToString() << ">"
                             << ", workspace required: " << required_size
                             << ", provided ptr: " << provided_ptr << " size: " << provided_size);
            return false;
        }
    }
    return true;
}

} // namespace conv
} // namespace miopen
