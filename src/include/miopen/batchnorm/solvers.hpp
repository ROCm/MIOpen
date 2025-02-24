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
/// https://github.com/ROCm/MIOpen/issues/1549#issuecomment-1152644636
#define WORKAROUND_ISSUE_1549_FP16_BUILD_ERROR 1

namespace miopen {

namespace solver {

namespace batchnorm {

using BatchnormSolver =
    NonTunableSolverBase<ExecutionContext, miopen::batchnorm::ProblemDescription>;

template <class PerformanceConfig>
using BatchNormTunableSolver =
    TunableSolverMixin<ExecutionContext, miopen::batchnorm::ProblemDescription, PerformanceConfig>;
;

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
    bool IsDynamic() const override { return true; }
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::batchnorm::ProblemDescription& problem) const override;
};

struct PerformanceConfigBnCKFwdInference : PerfConfigBaseCK<PerformanceConfigBnCKFwdInference>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigBnCKFwdInference(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigBnCKFwdInference() : PerformanceConfigBnCKFwdInference(0, "") {}
    PerformanceConfigBnCKFwdInference(bool) : PerformanceConfigBnCKFwdInference(0, "") {}
    MIOPEN_INTERNALS_EXPORT void
    HeuristicInit(const miopen::batchnorm::ProblemDescription& problem);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::batchnorm::ProblemDescription& problem);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool
    IsValid(const ExecutionContext&, const miopen::batchnorm::ProblemDescription& problem) const;

    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.kernel_id, "kernel_id");
    }
    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceConfigBnCKFwdInference& other) const;

private:
    template <typename XDataType,
              typename YDataType,
              typename AccDataType,
              typename ScaleDataType,
              typename BiasDataType,
              typename MeanVarDataType>
    void Init(const miopen::batchnorm::ProblemDescription&);
    template <typename XDataType,
              typename YDataType,
              typename AccDataType,
              typename ScaleDataType,
              typename BiasDataType,
              typename MeanVarDataType>
    bool CheckIsSupportCKArgs(const miopen::batchnorm::ProblemDescription&) const;
};

struct BnCKFwdInference final : BatchNormTunableSolver<PerformanceConfigBnCKFwdInference>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<BnCKFwdInference>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext& ctx,
                 const miopen::batchnorm::ProblemDescription& problem) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT PerformanceConfigBnCKFwdInference GetDefaultPerformanceConfig(
        const ExecutionContext& ctx,
        const miopen::batchnorm::ProblemDescription& problem) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext& ctx,
                             const miopen::batchnorm::ProblemDescription& problem,
                             const PerformanceConfigBnCKFwdInference& config) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigBnCKFwdInference
    Search(const ExecutionContext& ctx,
           const miopen::batchnorm::ProblemDescription& problem,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext& ctx,
                const miopen::batchnorm::ProblemDescription& problem,
                const PerformanceConfigBnCKFwdInference& config) const override;
};

struct PerformanceConfigBnCKBwdBackward : PerfConfigBaseCK<PerformanceConfigBnCKBwdBackward>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigBnCKBwdBackward(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigBnCKBwdBackward() : PerformanceConfigBnCKBwdBackward(0, "") {}
    PerformanceConfigBnCKBwdBackward(bool) : PerformanceConfigBnCKBwdBackward(0, "") {}
    MIOPEN_INTERNALS_EXPORT void
    HeuristicInit(const miopen::batchnorm::ProblemDescription& problem);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::batchnorm::ProblemDescription& problem);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool
    IsValid(const ExecutionContext&, const miopen::batchnorm::ProblemDescription& problem) const;

    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.kernel_id, "kernel_id");
    }
    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceConfigBnCKBwdBackward& other) const;

private:
    template <typename XDataType,
              typename DxDataType,
              typename DyDataType,
              typename AccDataType,
              typename ScaleDataType,
              typename DscaleDbiasDataType,
              typename MeanVarDataType>
    void Init(const miopen::batchnorm::ProblemDescription&);
    template <typename XDataType,
              typename DxDataType,
              typename DyDataType,
              typename AccDataType,
              typename ScaleDataType,
              typename DscaleDbiasDataType,
              typename MeanVarDataType>
    bool CheckIsSupportCKArgs(const miopen::batchnorm::ProblemDescription&) const;
};

struct BnCKBwdBackward final : BatchNormTunableSolver<PerformanceConfigBnCKBwdBackward>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<BnCKBwdBackward>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext& ctx,
                 const miopen::batchnorm::ProblemDescription& problem) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT PerformanceConfigBnCKBwdBackward GetDefaultPerformanceConfig(
        const ExecutionContext& ctx,
        const miopen::batchnorm::ProblemDescription& problem) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext& ctx,
                             const miopen::batchnorm::ProblemDescription& problem,
                             const PerformanceConfigBnCKBwdBackward& config) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigBnCKBwdBackward
    Search(const ExecutionContext& ctx,
           const miopen::batchnorm::ProblemDescription& problem,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext& ctx,
                const miopen::batchnorm::ProblemDescription& problem,
                const PerformanceConfigBnCKBwdBackward& config) const override;
};

struct PerformanceConfigBnCKFwdTraining : PerfConfigBaseCK<PerformanceConfigBnCKFwdTraining>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigBnCKFwdTraining(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigBnCKFwdTraining() : PerformanceConfigBnCKFwdTraining(0, "") {}
    PerformanceConfigBnCKFwdTraining(bool) : PerformanceConfigBnCKFwdTraining(0, "") {}
    MIOPEN_INTERNALS_EXPORT void
    HeuristicInit(const miopen::batchnorm::ProblemDescription& problem);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::batchnorm::ProblemDescription& problem);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool
    IsValid(const ExecutionContext&, const miopen::batchnorm::ProblemDescription& problem) const;

    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.kernel_id, "kernel_id");
    }
    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceConfigBnCKFwdTraining& other) const;

private:
    template <typename XDataType,
              typename YDataType,
              typename AccDataType,
              typename ScaleDataType,
              typename BiasDataType,
              typename MeanVarDataType>
    void Init(const miopen::batchnorm::ProblemDescription&);
    template <typename XDataType,
              typename YDataType,
              typename AccDataType,
              typename ScaleDataType,
              typename BiasDataType,
              typename MeanVarDataType>
    bool CheckIsSupportCKArgs(const miopen::batchnorm::ProblemDescription&) const;
};

struct BnCKFwdTraining final : BatchNormTunableSolver<PerformanceConfigBnCKFwdTraining>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<BnCKFwdTraining>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext& ctx,
                 const miopen::batchnorm::ProblemDescription& problem) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT PerformanceConfigBnCKFwdTraining GetDefaultPerformanceConfig(
        const ExecutionContext& ctx,
        const miopen::batchnorm::ProblemDescription& problem) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext& ctx,
                             const miopen::batchnorm::ProblemDescription& problem,
                             const PerformanceConfigBnCKFwdTraining& config) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigBnCKFwdTraining
    Search(const ExecutionContext& ctx,
           const miopen::batchnorm::ProblemDescription& problem,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext& ctx,
                const miopen::batchnorm::ProblemDescription& problem,
                const PerformanceConfigBnCKFwdTraining& config) const override;
};

} // namespace batchnorm

} // namespace solver

} // namespace miopen
