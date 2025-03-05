/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#pragma once

#include <miopen/cosineembeddingloss/problem_description.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/solver.hpp>

namespace miopen {

namespace solver {

const auto make_hip_kernel = [](std::vector<size_t> localsize,
                                std::vector<size_t> gridsize,
                                std::string kernel_file,
                                std::string kernel_name,
                                KernelBuildParameters build_params) {
    while(localsize.size() < 3)
        localsize.push_back(1);
    while(gridsize.size() < 3)
        gridsize.push_back(1);
    for(int i = 0; i < localsize.size(); ++i)
        gridsize[i] = AlignUp(gridsize[i], localsize[i]);
    return KernelInfo{
        build_params.GenerateFor(kbp::HIP{}), localsize, gridsize, kernel_file, kernel_name};
};

namespace cosineembeddingloss {

using CosineEmbeddingLossFwdUnreducedSolver =
    NonTunableSolverBase<ExecutionContext,
                         miopen::cosineembeddingloss::FwdUnreducedProblemDescription>;

using CosineEmbeddingLossFwdReducedSolver =
    NonTunableSolverBase<ExecutionContext,
                         miopen::cosineembeddingloss::FwdReducedProblemDescription>;

using CosineEmbeddingLossBwdUnreducedSolver =
    NonTunableSolverBase<ExecutionContext,
                         miopen::cosineembeddingloss::BwdUnreducedProblemDescription>;

using CosineEmbeddingLossBwdReducedSolver =
    NonTunableSolverBase<ExecutionContext,
                         miopen::cosineembeddingloss::BwdReducedProblemDescription>;

// FORWARD UNREDUCE
struct CosineEmbeddingLossUnreducedForward2d final : CosineEmbeddingLossFwdUnreducedSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<CosineEmbeddingLossUnreducedForward2d>();
    }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::FwdUnreducedProblemDescription& problem) const override;

    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::FwdUnreducedProblemDescription& problem) const override;

    std::size_t GetWorkspaceSize(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::FwdUnreducedProblemDescription& problem) const override;

    bool MayNeedWorkspace() const override { return true; }
};

// FORWARD UNREDUCE NON SUM
struct CosineEmbeddingLossUnreducedForward2dNonSum final : CosineEmbeddingLossFwdUnreducedSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<CosineEmbeddingLossUnreducedForward2dNonSum>();
    }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::FwdUnreducedProblemDescription& problem) const override;

    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::FwdUnreducedProblemDescription& problem) const override;

    bool MayNeedWorkspace() const override { return true; }
};

// FORWARD REDUCE
struct CosineEmbeddingLossReducedForward2d final : CosineEmbeddingLossFwdReducedSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<CosineEmbeddingLossReducedForward2d>();
    }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::FwdReducedProblemDescription& problem) const override;

    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::FwdReducedProblemDescription& problem) const override;

    std::size_t GetWorkspaceSize(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::FwdReducedProblemDescription& problem) const override;

    bool MayNeedWorkspace() const override { return true; }
};

// FORWARD REDUCE NON SUM
struct CosineEmbeddingLossReducedForward2dNonSum final : CosineEmbeddingLossFwdReducedSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<CosineEmbeddingLossReducedForward2dNonSum>();
    }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::FwdReducedProblemDescription& problem) const override;

    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::FwdReducedProblemDescription& problem) const override;

    std::size_t GetWorkspaceSize(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::FwdReducedProblemDescription& problem) const override;

    bool MayNeedWorkspace() const override { return true; }
};

// BACKWARD UNREDUCE
struct CosineEmbeddingLossUnreducedBackward2d final : CosineEmbeddingLossBwdUnreducedSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<CosineEmbeddingLossUnreducedBackward2d>();
    }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::BwdUnreducedProblemDescription& problem) const override;

    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::BwdUnreducedProblemDescription& problem) const override;

    std::size_t GetWorkspaceSize(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::BwdUnreducedProblemDescription& problem) const override;

    bool MayNeedWorkspace() const override { return true; }
};

// BACKWARD UNREDUCE NON SUM
struct CosineEmbeddingLossUnreducedBackward2dNonSum final : CosineEmbeddingLossBwdUnreducedSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<CosineEmbeddingLossUnreducedBackward2dNonSum>();
    }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::BwdUnreducedProblemDescription& problem) const override;

    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::BwdUnreducedProblemDescription& problem) const override;

    bool MayNeedWorkspace() const override { return true; }
};

// BACKWARD REDUCE
struct CosineEmbeddingLossReducedBackward2d final : CosineEmbeddingLossBwdReducedSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<CosineEmbeddingLossReducedBackward2d>();
    }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::BwdReducedProblemDescription& problem) const override;

    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::BwdReducedProblemDescription& problem) const override;

    std::size_t GetWorkspaceSize(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::BwdReducedProblemDescription& problem) const override;

    bool MayNeedWorkspace() const override { return true; }
};

// BACKWARD REDUCE NON SUM
struct CosineEmbeddingLossReducedBackward2dNonSum final : CosineEmbeddingLossBwdReducedSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<CosineEmbeddingLossReducedBackward2dNonSum>();
    }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::BwdReducedProblemDescription& problem) const override;

    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::cosineembeddingloss::BwdReducedProblemDescription& problem) const override;

    bool MayNeedWorkspace() const override { return true; }
};

} // namespace cosineembeddingloss

} // namespace solver

} // namespace miopen
