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
#include <miopen/solution.hpp>

namespace miopen {

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEVICE_ARCH)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_COMPILE_ONLY)

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_GEMM)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_WINOGRAD)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_FFT)

class DirectSolverFinder : public SolversFinder
{
public:
    AlgorithmName GetAlgorithmName(const conv::ProblemDescription& problem) const override
    {
        return AlgorithmName{ConvolutionAlgoToDirectionalString(miopenConvolutionAlgoDirect,
                                                                problem.GetDirection())};
    }

protected:
    bool IsEnabled(const ConvolutionContext& /*ctx*/,
                   const conv::ProblemDescription& /*problem*/,
                   bool use_winograd_only) const override
    {
        return !use_winograd_only && !IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{});
    }

    std::vector<solver::ConvSolution> FindImpl(const ConvolutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx,
                                               bool /*use_winograd_only*/) const override
    {
        return problem.conv_problem.GetDirection() != conv::Direction::BackwardWeights
                   ? FindAllDirectSolutions(ctx, problem, invoke_ctx)
                   : FindAllBwdWrW2DSolutions(ctx, problem, invoke_ctx);
    }
};

class ImplicitGemmSolverFinder : public SolversFinder
{
public:
    AlgorithmName GetAlgorithmName(const conv::ProblemDescription& problem) const override
    {
        return AlgorithmName{ConvolutionAlgoToDirectionalString(miopenConvolutionAlgoImplicitGEMM,
                                                                problem.GetDirection())};
    }

protected:
    bool IsEnabled(const ConvolutionContext& /*ctx*/,
                   const conv::ProblemDescription& /*problem*/,
                   bool use_winograd_only) const override
    {
        return !use_winograd_only && !IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM{});
    }

    std::vector<solver::ConvSolution> FindImpl(const ConvolutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx,
                                               bool /*use_winograd_only*/) const override
    {
        return problem.conv_problem.GetDirection() != conv::Direction::BackwardWeights
                   ? FindAllImplicitGemmSolutions(ctx, problem, invoke_ctx)
                   : FindImplicitGemmWrWAllSolutions(ctx, problem, invoke_ctx);
    }
};

class FftSolverFinder : public SolversFinder
{
public:
    AlgorithmName GetAlgorithmName(const conv::ProblemDescription& problem) const override
    {
        return AlgorithmName{
            ConvolutionAlgoToDirectionalString(miopenConvolutionAlgoFFT, problem.GetDirection())};
    }

protected:
    bool IsEnabled(const ConvolutionContext& /*ctx*/,
                   const conv::ProblemDescription& problem,
                   bool use_winograd_only) const override
    {
        return !use_winograd_only && problem.GetDirection() != conv::Direction::BackwardWeights &&
               !IsDisabled(MIOPEN_DEBUG_CONV_FFT{});
    }

    std::vector<solver::ConvSolution> FindImpl(const ConvolutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx,
                                               bool /*use_winograd_only*/) const override
    {
        return FindAllFFTSolutions(ctx, problem, invoke_ctx);
    }
};

class GemmSolverFinder : public SolversFinder
{
public:
    AlgorithmName GetAlgorithmName(const conv::ProblemDescription& problem) const override
    {
        return AlgorithmName{
            ConvolutionAlgoToDirectionalString(miopenConvolutionAlgoGEMM, problem.GetDirection())};
    }

protected:
    bool IsEnabled(const ConvolutionContext& /*ctx*/,
                   const conv::ProblemDescription& /*problem*/,
                   bool use_winograd_only) const override
    {
        return !use_winograd_only && !IsDisabled(MIOPEN_DEBUG_CONV_GEMM{});
    }

    std::vector<solver::ConvSolution> FindImpl(const ConvolutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx,
                                               bool /*use_winograd_only*/) const override
    {
        return FindAllGemmSolutions(ctx, problem, invoke_ctx);
    }
};

class WinogradSolverFinder : public SolversFinder
{
public:
    AlgorithmName GetAlgorithmName(const conv::ProblemDescription& problem) const override
    {
        return AlgorithmName{ConvolutionAlgoToDirectionalString(miopenConvolutionAlgoWinograd,
                                                                problem.GetDirection())};
    }

protected:
    bool IsEnabled(const ConvolutionContext& /*ctx*/,
                   const conv::ProblemDescription& /*problem*/,
                   bool /*use_winograd_only*/) const override
    {
        return !IsDisabled(MIOPEN_DEBUG_CONV_WINOGRAD{});
    }

    std::vector<solver::ConvSolution> FindImpl(const ConvolutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx,
                                               bool use_winograd_only) const override
    {
        auto ctx_copy = ctx;
        if(use_winograd_only)
            ctx_copy.use_dynamic_solutions_only = true;
        return problem.conv_problem.GetDirection() != conv::Direction::BackwardWeights
                   ? FindAllWinogradSolutions(ctx_copy, problem, invoke_ctx)
                   : FindWinogradWrWAllSolutions(ctx_copy, problem, invoke_ctx);
    }
};

const std::vector<std::unique_ptr<SolversFinder>>& GetConvSolverFinders()
{
    static const auto finders = []() {
        auto tmp = std::vector<std::unique_ptr<SolversFinder>>{};
        tmp.emplace_back(std::make_unique<WinogradSolverFinder>());
        tmp.emplace_back(std::make_unique<DirectSolverFinder>());
        tmp.emplace_back(std::make_unique<ImplicitGemmSolverFinder>());
        tmp.emplace_back(std::make_unique<GemmSolverFinder>());
        tmp.emplace_back(std::make_unique<FftSolverFinder>());
        return tmp;
    }();

    return finders;
}

static inline void AppendPointersToElements(const std::vector<miopen::solver::ConvSolution>& from,
                                            std::vector<const miopen::solver::ConvSolution*>& to)
{
    std::transform(from.begin(),
                   from.end(),
                   std::back_inserter(to),
                   [](const miopen::solver::ConvSolution& s) { return &s; });
}

/// Register invoker only for the best solution within algorithm.
template <class InvokeParams>
static std::vector<Solution> EvaluateInvokers(Handle& handle,
                                              const std::vector<solver::ConvSolution>& solutions,
                                              const AlgorithmName& algorithm_name,
                                              const NetworkConfig& network_config,
                                              const InvokeParams& invoke_ctx,
                                              bool force_attach_binary)
{
    const char* const arch = miopen::GetStringEnv(MIOPEN_DEVICE_ARCH{});
    if(arch != nullptr && strlen(arch) > 0)
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

std::vector<Solution> ConvFindCore(const AnyInvokeParams& invoke_ctx,
                                   const ConvolutionContext& ctx,
                                   const ProblemDescription& problem,
                                   bool use_winograd_only,
                                   const std::vector<std::unique_ptr<SolversFinder>>& finders,
                                   bool force_attach_binary)
{
    auto& handle = ctx.GetStream();

    // Find
    auto solutions = std::map<AlgorithmName, std::vector<solver::ConvSolution>>{};
    std::transform(
        finders.begin(), finders.end(), std::inserter(solutions, solutions.end()), [&](auto&& f) {
            return std::make_pair(f->GetAlgorithmName(problem.conv_problem),
                                  f->Find(ctx, problem, invoke_ctx, use_winograd_only));
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
            AppendPointersToElements(ss.second, all);
        PrecompileSolutions(handle, all, force_attach_binary);
    }

    if(IsEnabled(MIOPEN_DEBUG_COMPILE_ONLY{}))
        MIOPEN_THROW(
            miopenStatusGpuOperationsSkipped,
            "MIOPEN_DEBUG_COMPILE_ONLY is enabled, escaping forward convolution. Search skipped.");

    // Evaluate Invokers
    AutoEnableProfiling enableProfiling{handle};
    const auto network_config = problem.BuildConfKey();

    auto ret = std::vector<Solution>{};
    ret.reserve(total);

    for(const auto& ss : solutions)
    {
        auto evaluated = EvaluateInvokers(
            handle, ss.second, ss.first, network_config, invoke_ctx, force_attach_binary);

        ret.insert(ret.end(),
                   std::make_move_iterator(evaluated.begin()),
                   std::make_move_iterator(evaluated.end()));
    }

    return ret;
}

bool IsAlgorithmDisabled(miopenConvAlgorithm_t algo)
{
    switch(algo)
    { // clang-format off
    case miopenConvolutionAlgoGEMM:
        return !MIOPEN_USE_GEMM || miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{});
    case miopenConvolutionAlgoDirect:
        return miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{});
    case miopenConvolutionAlgoFFT:
        return miopen::IsDisabled(MIOPEN_DEBUG_CONV_FFT{});
    case miopenConvolutionAlgoWinograd:
        return miopen::IsDisabled(MIOPEN_DEBUG_CONV_WINOGRAD{});
    case miopenConvolutionAlgoImplicitGEMM:
        return miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM{});
    default: // Disable future algos by default to enforce explicit handling:
        return true;
    } // clang-format on
}

} // namespace miopen
