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

#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/solver.hpp>
#include <miopen/adaptiveavgpool/problem_description.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/mlo_internal.hpp>

namespace miopen {

namespace solver {

namespace adaptiveavgpool {

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

using AdaptiveAvgPoolForward =
    NonTunableSolverBase<ExecutionContext, miopen::adaptiveavgpool::FwdProblemDescription>;

using AdaptiveAvgPoolBackward =
    NonTunableSolverBase<ExecutionContext, miopen::adaptiveavgpool::BwdProblemDescription>;

// FORWARD
struct AdaptiveAvgPoolForward1d final : AdaptiveAvgPoolForward
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<AdaptiveAvgPoolForward1d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::adaptiveavgpool::FwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::adaptiveavgpool::FwdProblemDescription& problem) const override;
};

struct AdaptiveAvgPoolForward2d final : AdaptiveAvgPoolForward
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<AdaptiveAvgPoolForward2d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::adaptiveavgpool::FwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::adaptiveavgpool::FwdProblemDescription& problem) const override;
};

struct AdaptiveAvgPoolForward3d final : AdaptiveAvgPoolForward
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<AdaptiveAvgPoolForward3d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::adaptiveavgpool::FwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::adaptiveavgpool::FwdProblemDescription& problem) const override;
};

// BACKWARD
struct AdaptiveAvgPoolBackward1d final : AdaptiveAvgPoolBackward
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<AdaptiveAvgPoolBackward1d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::adaptiveavgpool::BwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::adaptiveavgpool::BwdProblemDescription& problem) const override;
};

struct AdaptiveAvgPoolBackward2d final : AdaptiveAvgPoolBackward
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<AdaptiveAvgPoolBackward2d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::adaptiveavgpool::BwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::adaptiveavgpool::BwdProblemDescription& problem) const override;
};

struct AdaptiveAvgPoolBackward3d final : AdaptiveAvgPoolBackward
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<AdaptiveAvgPoolBackward3d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::adaptiveavgpool::BwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::adaptiveavgpool::BwdProblemDescription& problem) const override;
};

} // namespace adaptiveavgpool

} // namespace solver

} // namespace miopen
