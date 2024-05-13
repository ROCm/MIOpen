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
#include <miopen/solution.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_GEMM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_WINOGRAD)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_FFT)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEVICE_ARCH)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_COMPILE_ONLY)

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
        return !parameters.use_winograd_only && !IsDisabled(ENV(MIOPEN_DEBUG_CONV_DIRECT));
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
        return !parameters.use_winograd_only && !IsDisabled(ENV(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM));
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
               !IsDisabled(ENV(MIOPEN_DEBUG_CONV_FFT));
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
        return !parameters.use_winograd_only && !IsDisabled(ENV(MIOPEN_DEBUG_CONV_GEMM));
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
        return !IsDisabled(ENV(MIOPEN_DEBUG_CONV_WINOGRAD));
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
static std::vector<Solution> EvaluateInvokers(Handle& handle,
                                              const std::vector<solver::ConvSolution>& solutions,
                                              const AlgorithmName& algorithm_name,
                                              const NetworkConfig& network_config,
                                              const AnyInvokeParams& invoke_ctx,
                                              bool force_attach_binary)
{
    const auto& arch = miopen::GetStringEnv(ENV(MIOPEN_DEVICE_ARCH));
    if(!arch.empty())
        return {};

    auto selected      = miopen::solver::ConvSolution{miopenStatusUnknownError};
    auto best          = std::numeric_limits<float>::max();
    auto best_invoker  = Invoker{};
    auto best_programs = std::vector<Program>{};

    for(const auto& sol : solutions)
    {
        if(sol.workspace_sz > 0)
        {
            if(invoke_ctx.GetWorkspace() == nullptr)
            {
                MIOPEN_LOG_I("Warning: skipping solver <" << sol.solver_id
                                                          << "> due to no workspace provided ("
                                                          << sol.workspace_sz << " required)");
                continue;
            }
            if(invoke_ctx.GetWorkspaceSize() < sol.workspace_sz)
            {
                MIOPEN_LOG_I("Warning: skipping solver <"
                             << sol.solver_id << "> due to insufficient workspace ("
                             << invoke_ctx.GetWorkspaceSize() << " < " << sol.workspace_sz << ")");
                continue;
            }
        }

        if(!sol.invoker_factory)
            MIOPEN_THROW("Invoker is not provided by solver " + sol.solver_id);

        std::vector<Program> programs;
        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory,
                                                   sol.construction_params,
                                                   force_attach_binary ? &programs : nullptr);

        try
        {
            invoker(handle, invoke_ctx);
            const auto elapsed = handle.GetKernelTime();

            MIOPEN_LOG_I(sol << ": " << elapsed << (elapsed < best ? " < " : " >= ") << best);
            if(elapsed < best)
            {
                best          = elapsed;
                selected      = sol;
                best_invoker  = invoker;
                best_programs = programs;
            }
        }
        catch(const miopen::Exception& ex)
        {
            MIOPEN_LOG_E(ex.what());
        }
    }

    if(!selected.Succeeded())
        return {};

    handle.RegisterInvoker(best_invoker, network_config, selected.solver_id, algorithm_name);
    MIOPEN_LOG_I("Selected: " << selected << ": " << best
                              << ", workspace_sz = " << selected.workspace_sz);

    auto solution = Solution{solver::Id{selected.solver_id}, best, selected.workspace_sz};
    if(force_attach_binary)
        solution.SetInvoker(best_invoker, best_programs, selected.construction_params);
    else
        solution.SetInvoker(best_invoker, {}, {});
    return {{solution}};
}

std::vector<Solution> FindCore(const AnyInvokeParams& invoke_ctx,
                               const ExecutionContext& ctx,
                               const ProblemDescriptionBase& problem,
                               const PrimitiveFindParameters& parameters,
                               const std::vector<std::unique_ptr<ISolversFinder>>& finders,
                               const std::optional<FindOptions>& options,
                               bool force_attach_binary)
{
    auto& handle = ctx.GetStream();

    // Find
    auto solutions = std::map<AlgorithmName, std::vector<solver::ConvSolution>>{};
    std::transform(
        finders.begin(), finders.end(), std::inserter(solutions, solutions.end()), [&](auto&& f) {
            return std::make_pair(f->GetAlgorithmName(problem),
                                  f->Find(ctx, problem, invoke_ctx, parameters, options));
        });

    std::size_t total = 0;

    for(auto it = solutions.begin(); it != solutions.end();)
    {
        if(it->second.empty())
        {
            it = solutions.erase(it);
            continue;
        }

        total += it->second.size();
        ++it;
    }

    // Precompile
    {
        auto all = std::vector<const miopen::solver::ConvSolution*>{};
        all.reserve(total);
        for(const auto& ss : solutions)
            std::transform(ss.second.begin(),
                           ss.second.end(),
                           std::back_inserter(all),
                           [](auto&& s) { return &s; });
        PrecompileSolutions(handle, all, force_attach_binary);
    }

    if(IsEnabled(ENV(MIOPEN_DEBUG_COMPILE_ONLY)))
        MIOPEN_THROW(
            miopenStatusGpuOperationsSkipped,
            "MIOPEN_DEBUG_COMPILE_ONLY is enabled, escaping forward convolution. Search skipped.");

    // Evaluate Invokers
    AutoEnableProfiling enableProfiling{handle};
    const auto network_config = problem.MakeNetworkConfig();

    auto ret = std::vector<Solution>{};
    ret.reserve(total);

    for(const auto& ss : solutions)
    {
        if (ss.second.empty())
            continue;

        auto evaluated = EvaluateInvokers(
            handle, ss.second, ss.first, network_config, invoke_ctx, force_attach_binary);

        ret.insert(ret.end(),
                   std::make_move_iterator(evaluated.begin()),
                   std::make_move_iterator(evaluated.end()));
    }

    return ret;
}

namespace conv {

bool IsAlgorithmDisabled(miopenConvAlgorithm_t algo)
{
    switch(algo)
    { // clang-format off
    case miopenConvolutionAlgoGEMM:
        return !MIOPEN_USE_GEMM || miopen::IsDisabled(ENV(MIOPEN_DEBUG_CONV_GEMM));
    case miopenConvolutionAlgoDirect:
        return miopen::IsDisabled(ENV(MIOPEN_DEBUG_CONV_DIRECT));
    case miopenConvolutionAlgoFFT:
        return miopen::IsDisabled(ENV(MIOPEN_DEBUG_CONV_FFT));
    case miopenConvolutionAlgoWinograd:
        return miopen::IsDisabled(ENV(MIOPEN_DEBUG_CONV_WINOGRAD));
    case miopenConvolutionAlgoImplicitGEMM:
        return miopen::IsDisabled(ENV(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM));
    default: // Disable future algos by default to enforce explicit handling:
        return true;
    } // clang-format on
}

} // namespace conv
} // namespace miopen
