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
#include <miopen/generate_random_bit_mask/problem_description.hpp>

namespace miopen {
namespace solver {
namespace generate_random_bit_mask {

using InitPRNGStateSolver =
    NonTunableSolverBase<ExecutionContext,
                         miopen::generate_random_bit_mask::PStateProblemDescription>;
using GenerateRandomBitMaskSolver =
    NonTunableSolverBase<ExecutionContext, miopen::generate_random_bit_mask::ProblemDescription>;

struct InitPRNGState : InitPRNGStateSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<InitPRNGStateSolver>(); }
    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::generate_random_bit_mask::PStateProblemDescription& problem) const override;
    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::generate_random_bit_mask::PStateProblemDescription& problem) const override;
    std::size_t GetWorkspaceSize(
        const ExecutionContext& context,
        const miopen::generate_random_bit_mask::PStateProblemDescription& problem) const override
    {
        return 0;
    }
    bool MayNeedWorkspace() const override { return false; }
};

struct GenerateRandomBitMask : GenerateRandomBitMaskSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<GenerateRandomBitMaskSolver>();
    }
    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::generate_random_bit_mask::ProblemDescription& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::generate_random_bit_mask::ProblemDescription& problem) const override;
    std::size_t GetWorkspaceSize(
        const ExecutionContext& context,
        const miopen::generate_random_bit_mask::ProblemDescription& problem) const override
    {
        return 0;
    }
    bool MayNeedWorkspace() const override { return false; }
};

} // namespace generate_random_bit_mask
} // namespace solver
} // namespace miopen
