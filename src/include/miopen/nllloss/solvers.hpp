/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "miopen/conv_solution.hpp"
#include "miopen/execution_context.hpp"
#include <miopen/solver.hpp>
#include <miopen/nllloss/problem_description.hpp>
#include "miopen/kernel_build_params.hpp"
#include "miopen/kernel_info.hpp"

#include <utility>

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

namespace nllloss {

using NLLLossUnreduce =
    NonTunableSolverBase<ExecutionContext, miopen::nllloss::UnreduceProblemDescription>;

using NLLLossReduce =
    NonTunableSolverBase<ExecutionContext, miopen::nllloss::ReduceProblemDescription>;

struct NLLLossUnreduceSolver : NLLLossUnreduce
{
    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

struct NLLLossReduceSolver : NLLLossReduce
{
    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ReduceProblemDescription& problem) const override;
};

// FORWARD UNREDUCE
struct NLLLossUnreduceForwardContiguous4d final : NLLLossUnreduceSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceForwardContiguous4d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

struct NLLLossUnreduceForwardContiguous2d final : NLLLossUnreduceSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceForwardContiguous2d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

struct NLLLossUnreduceForward4d final : NLLLossUnreduceSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceForward4d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

struct NLLLossUnreduceForward2d final : NLLLossUnreduceSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceForward2d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

struct NLLLossUnreduceForward5d final : NLLLossUnreduceSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceForward5d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

// FORWARD REDUCE
struct NLLLossReduceForward5d final : NLLLossReduceSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossReduceForward5d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ReduceProblemDescription& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::ReduceProblemDescription& problem) const override;
    std::size_t
    GetWorkspaceSize(const ExecutionContext& context,
                     const miopen::nllloss::ReduceProblemDescription& problem) const override;
    bool MayNeedWorkspace() const override { return true; }
};

// BACKWARD UNREDUCE
struct NLLLossUnreduceBackwardContiguous2d final : NLLLossUnreduceSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceBackwardContiguous2d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

struct NLLLossUnreduceBackwardContiguous4d final : NLLLossUnreduceSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceBackwardContiguous4d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

struct NLLLossUnreduceBackward4d final : NLLLossUnreduceSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceBackward4d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

struct NLLLossUnreduceBackward2d final : NLLLossUnreduceSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceBackward2d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

struct NLLLossUnreduceBackward5d final : NLLLossUnreduceSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceBackward5d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

// BACKWARD REDUCE
struct NLLLossReduceBackward2d final : NLLLossReduceSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossReduceBackward2d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ReduceProblemDescription& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::ReduceProblemDescription& problem) const override;
};

struct NLLLossReduceBackward5d final : NLLLossReduceSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossReduceBackward5d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ReduceProblemDescription& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::ReduceProblemDescription& problem) const override;
};

} // namespace nllloss

} // namespace solver

} // namespace miopen
