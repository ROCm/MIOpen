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
#include <miopen/interpolate/problem_description.hpp>
#include "miopen/kernel_build_params.hpp"
#include "miopen/kernel_info.hpp"

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

namespace interpolate {

using InterpolateFwdSolver =
    NonTunableSolverBase<ExecutionContext, miopen::interpolate::FwdProblemDescription>;

using InterpolateBwdSolver =
    NonTunableSolverBase<ExecutionContext, miopen::interpolate::BwdProblemDescription>;

// FORWARD NEAREST
struct InterpolateNearestForward final : InterpolateFwdSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<InterpolateNearestForward>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::interpolate::FwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::interpolate::FwdProblemDescription& problem) const override;
};

// FORWARD LINEAR
struct InterpolateLinearForward final : InterpolateFwdSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<InterpolateLinearForward>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::interpolate::FwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::interpolate::FwdProblemDescription& problem) const override;
};

// FORWARD BILINEAR
struct InterpolateBilinearForward final : InterpolateFwdSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<InterpolateBilinearForward>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::interpolate::FwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::interpolate::FwdProblemDescription& problem) const override;
};

// FORWARD TRILINEAR
struct InterpolateTrilinearForward final : InterpolateFwdSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<InterpolateTrilinearForward>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::interpolate::FwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::interpolate::FwdProblemDescription& problem) const override;
};

// FORWARD BICUBIC
struct InterpolateBicubicForward final : InterpolateFwdSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<InterpolateBicubicForward>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::interpolate::FwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::interpolate::FwdProblemDescription& problem) const override;
};

// BACKWARD NEAREST
struct InterpolateNearestBackward final : InterpolateBwdSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<InterpolateNearestBackward>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::interpolate::BwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::interpolate::BwdProblemDescription& problem) const override;
};

// BACKWARD LINEAR
struct InterpolateLinearBackward final : InterpolateBwdSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<InterpolateLinearBackward>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::interpolate::BwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::interpolate::BwdProblemDescription& problem) const override;
};

// BACKWARD BILINEAR
struct InterpolateBilinearBackward final : InterpolateBwdSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<InterpolateBilinearBackward>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::interpolate::BwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::interpolate::BwdProblemDescription& problem) const override;
};

// BACKWARD TRILINEAR
struct InterpolateTrilinearBackward final : InterpolateBwdSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<InterpolateTrilinearBackward>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::interpolate::BwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::interpolate::BwdProblemDescription& problem) const override;
};

// BACKWARD BICUBIC
struct InterpolateBicubicBackward final : InterpolateBwdSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<InterpolateBicubicBackward>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::interpolate::BwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::interpolate::BwdProblemDescription& problem) const override;
};

} // namespace interpolate

} // namespace solver

} // namespace miopen
