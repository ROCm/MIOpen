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
#include <miopen/image_transform/adjust_brightness/problem_description.hpp>
#include <miopen/image_transform/adjust_hue/problem_description.hpp>
#include <miopen/image_transform/adjust_saturation/problem_description.hpp>
#include <miopen/image_transform/normalize/problem_description.hpp>
#include <miopen/solver.hpp>
#include <string>

namespace miopen {

namespace solver {

namespace image_transform {

namespace adjust_hue {

using ImageAdjustHueSolver =
    NonTunableSolverBase<ExecutionContext, miopen::image_transform::adjust_hue::ProblemDescription>;

struct ImageAdjustHue final : ImageAdjustHueSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ImageAdjustHue>(); }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::image_transform::adjust_hue::ProblemDescription& problem) const override;

    bool IsImprovementOverROCm(
        const ExecutionContext& context,
        const miopen::image_transform::adjust_hue::ProblemDescription& problem) const;

    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::image_transform::adjust_hue::ProblemDescription& problem) const override;
};

} // namespace adjust_hue

namespace adjust_brightness {

using ImageAdjustBrightnessSolver =
    NonTunableSolverBase<ExecutionContext,
                         miopen::image_transform::adjust_brightness::ProblemDescription>;

struct ImageAdjustBrightness final : ImageAdjustBrightnessSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ImageAdjustBrightness>();
    }

    bool IsImprovementOverROCm(
        const ExecutionContext& context,
        const miopen::image_transform::adjust_brightness::ProblemDescription& problem) const;

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::image_transform::adjust_brightness::ProblemDescription& problem)
        const override;

    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::image_transform::adjust_brightness::ProblemDescription&
                                 problem) const override;
};

} // namespace adjust_brightness

namespace adjust_saturation {

using ImageAdjustSaturationSolver =
    NonTunableSolverBase<ExecutionContext,
                         miopen::image_transform::adjust_saturation::ProblemDescription>;

struct ImageAdjustSaturation final : ImageAdjustSaturationSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ImageAdjustSaturation>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::image_transform::adjust_saturation::ProblemDescription& problem)
        const override;

    bool IsImprovementOverROCm(
        const ExecutionContext& context,
        const miopen::image_transform::adjust_saturation::ProblemDescription& problem) const;

    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::image_transform::adjust_saturation::ProblemDescription&
                                 problem) const override;

    bool MayNeedWorkspace() const override { return true; }

    size_t GetWorkspaceSize(const ExecutionContext& context,
                            const miopen::image_transform::adjust_saturation::ProblemDescription&
                                problem) const override;
};

} // namespace adjust_saturation

namespace normalize {

using ImageNormalizeSolver =
    NonTunableSolverBase<ExecutionContext, miopen::image_transform::normalize::ProblemDescription>;

struct ImageNormalize final : ImageNormalizeSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ImageNormalize>(); }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::image_transform::normalize::ProblemDescription& problem) const override;

    bool IsImprovementOverROCm(
        const ExecutionContext& context,
        const miopen::image_transform::normalize::ProblemDescription& problem) const;

    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::image_transform::normalize::ProblemDescription& problem) const override;
};

} // namespace normalize

} // namespace image_transform

} // namespace solver

} // namespace miopen
