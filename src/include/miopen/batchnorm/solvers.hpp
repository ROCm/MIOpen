/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <utility>

/// W/A for build error for OCL BN kernels when datatype is FP16 and MIO_BN_VARIANT=1. See:
/// https://github.com/ROCmSoftwarePlatform/MIOpen/issues/1549#issuecomment-1152644636
#define WORKAROUND_ISSUE_1549_FP16_BUILD_ERROR 1

namespace miopen {

namespace batchnorm {
struct ProblemDescription;
} // namespace batchnorm

namespace solver {

namespace batchnorm {

using OldStyleProblemDescription =
    std::tuple<const ExecutionContext*, const miopen::batchnorm::ProblemDescription*>;

struct BatchnormSolver : SolverMixin<OldStyleProblemDescription>
{
    // To suppress -Woverloaded-virtual
    using SolverMixin::IsApplicable;

    bool IsApplicable(const OldStyleProblemDescription& problem) const final
    {
        return IsApplicable(*std::get<0>(problem), *std::get<1>(problem));
    }

    ConvSolution GetSolution(const OldStyleProblemDescription& problem) const
    {
        return GetSolution(*std::get<0>(problem), *std::get<1>(problem));
    }

    virtual bool IsApplicable(const ExecutionContext& context,
                              const miopen::batchnorm::ProblemDescription& problem) const = 0;
    virtual ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::batchnorm::ProblemDescription& problem) const = 0;
};

struct BnFwdTrainingSpatialSingle final : BatchnormSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<BnFwdTrainingSpatialSingle>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::batchnorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::batchnorm::ProblemDescription& problem) const override;
};

struct BnFwdTrainingSpatialMultiple final : BatchnormSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<BnFwdTrainingSpatialMultiple>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::batchnorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::batchnorm::ProblemDescription& problem) const override;
};

struct BnFwdTrainingPerActivation final : BatchnormSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<BnFwdTrainingPerActivation>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::batchnorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::batchnorm::ProblemDescription& problem) const override;
};

struct BnBwdTrainingSpatialSingle final : BatchnormSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<BnBwdTrainingSpatialSingle>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::batchnorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::batchnorm::ProblemDescription& problem) const override;
};

struct BnBwdTrainingSpatialMultiple final : BatchnormSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<BnBwdTrainingSpatialMultiple>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::batchnorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::batchnorm::ProblemDescription& problem) const override;
};

struct BnBwdTrainingPerActivation final : BatchnormSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<BnBwdTrainingPerActivation>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::batchnorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::batchnorm::ProblemDescription& problem) const override;
};

struct BnFwdInference final : BatchnormSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<BnFwdInference>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::batchnorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::batchnorm::ProblemDescription& problem) const override;
};

} // namespace batchnorm

} // namespace solver

} // namespace miopen
