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
#include <miopen/batchnorm/problem_description.hpp>

#include <utility>

/// W/A for build error for OCL BN kernels when datatype is FP16 and MIO_BN_VARIANT=1. See:
/// https://github.com/ROCmSoftwarePlatform/MIOpen/issues/1549#issuecomment-1152644636
#define WORKAROUND_ISSUE_1549_FP16_BUILD_ERROR 1

namespace miopen {

namespace solver {

namespace batchnorm {

using BatchnormSolver =
    NonTunableSolverBase<ExecutionContext, miopen::batchnorm::ProblemDescription>;

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

struct BnCKFwdInference final : BatchnormSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<BnCKFwdInference>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::batchnorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::batchnorm::ProblemDescription& problem) const override;
};

struct BnCKBwdBackward final : BatchnormSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<BnCKBwdBackward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::batchnorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::batchnorm::ProblemDescription& problem) const override;
};

struct BnCKFwdTraining final : BatchnormSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<BnCKFwdTraining>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::batchnorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::batchnorm::ProblemDescription& problem) const override;
};

} // namespace batchnorm

} // namespace solver

} // namespace miopen
