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

#include <miopen/solver.hpp>
#include <miopen/nllloss/problem_description.hpp>
#include "miopen/kernel_build_params.hpp"
#include "miopen/kernel_info.hpp"
#include "miopen/mlo_internal.hpp"

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

struct NLLLossSolver : NonTunableSolverBase<ExecutionContext, miopen::nllloss::ProblemDescription>
{
    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ProblemDescription& problem) const override;
};

// FORWARD UNREDUCE
struct NLLLossUnreduceForwardContiguous4d final : NLLLossSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceForwardContiguous4d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ProblemDescription& problem) const override;

    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::nllloss::ProblemDescription& problem) const override;
};

struct NLLLossUnreduceForwardContiguous2d final : NLLLossSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceForwardContiguous2d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ProblemDescription& problem) const override;

    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::nllloss::ProblemDescription& problem) const override;
};

struct NLLLossUnreduceForward4d final : NLLLossSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceForward4d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ProblemDescription& problem) const override;

    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::nllloss::ProblemDescription& problem) const override;
};

struct NLLLossUnreduceForward2d final : NLLLossSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceForward2d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ProblemDescription& problem) const override;

    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::nllloss::ProblemDescription& problem) const override;
};

struct NLLLossUnreduceForward5d final : NLLLossSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceForward5d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ProblemDescription& problem) const override;

    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::nllloss::ProblemDescription& problem) const override;
};

// FORWARD REDUCE
struct NLLLossReduceForward5d final : NLLLossSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossReduceForward5d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::nllloss::ProblemDescription& problem) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::nllloss::ProblemDescription& problem) const override;
    bool MayNeedWorkspace() const override { return true; }
};

// BACKWARD UNREDUCE
struct NLLLossUnreduceBackwardContiguous2d final : NLLLossSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceBackwardContiguous2d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ProblemDescription& problem) const override;

    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::nllloss::ProblemDescription& problem) const override;
};

struct NLLLossUnreduceBackwardContiguous4d final : NLLLossSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceBackwardContiguous4d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ProblemDescription& problem) const override;

    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::nllloss::ProblemDescription& problem) const override;
};

struct NLLLossUnreduceBackward4d final : NLLLossSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceBackward4d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ProblemDescription& problem) const override;

    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::nllloss::ProblemDescription& problem) const override;
};

struct NLLLossUnreduceBackward2d final : NLLLossSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceBackward2d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ProblemDescription& problem) const override;

    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::nllloss::ProblemDescription& problem) const override;
};

struct NLLLossUnreduceBackward5d final : NLLLossSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceBackward5d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ProblemDescription& problem) const override;

    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::nllloss::ProblemDescription& problem) const override;
};

// BACKWARD REDUCE
struct NLLLossReduceBackward2d final : NLLLossSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossReduceBackward2d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::nllloss::ProblemDescription& problem) const override;
};

struct NLLLossReduceBackward5d final : NLLLossSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossReduceBackward5d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::nllloss::ProblemDescription& problem) const override;
};

} // namespace nllloss

} // namespace solver

} // namespace miopen
