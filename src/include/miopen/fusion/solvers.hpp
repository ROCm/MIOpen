/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include <miopen/conv/solvers.hpp>
#include <miopen/fusion.hpp>
#include <miopen/fusion_plan.hpp>
#include <utility>

#include <miopen/fusion/problem_description.hpp>
#include <miopen/fusion/context.hpp>

namespace miopen {
namespace solver {
namespace fusion {

using FusionSolverBase = NonTunableSolverBase<FusionContext, FusionDescription>;

template <class PerformanceConfig>
using FusionTunableSolver =
    TunableSolverMixin<FusionContext, miopen::FusionDescription, PerformanceConfig>;
;

struct PerformanceConfigConvBiasActivAsm1x1U : conv::PerformanceConfigConvAsm1x1U
{
    PerformanceConfigConvBiasActivAsm1x1U(const bool spare) : PerformanceConfigConvAsm1x1U(spare) {}
    PerformanceConfigConvBiasActivAsm1x1U()
        : PerformanceConfigConvAsm1x1U(-1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const FusionContext& ctx,
                                               const FusionDescription& problem);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const FusionDescription& problem);
    bool IsValid(const FusionContext&, const FusionDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const FusionDescription& problem) const;
};

struct ConvBiasActivAsm1x1U : FusionTunableSolver<PerformanceConfigConvBiasActivAsm1x1U>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvBiasActivAsm1x1U>(); }

    MIOPEN_INTERNALS_EXPORT bool IsApplicable(const FusionContext& context,
                                              const FusionDescription& problem) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const FusionContext& context,
                const FusionDescription& problem,
                const PerformanceConfigConvBiasActivAsm1x1U& /*config*/) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvBiasActivAsm1x1U
    GetDefaultPerformanceConfig(const FusionContext&, const FusionDescription&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvBiasActivAsm1x1U
    Search(const FusionContext& context,
           const FusionDescription& problem,
           const AnyInvokeParams& invoke_params) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const FusionContext&,
                             const FusionDescription&,
                             const PerformanceConfigConvBiasActivAsm1x1U&) const override;
    MIOPEN_INTERNALS_EXPORT float GetWti(const FusionContext&,
                                         const FusionDescription&) const override;
};

using PerformanceConfigConvOclDirectFwdFused = LegacyPerformanceConfig;
struct ConvOclDirectFwdFused final : FusionTunableSolver<LegacyPerformanceConfig>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvOclDirectFwdFused>();
    }

    MIOPEN_INTERNALS_EXPORT bool IsApplicable(const FusionContext& context,
                                              const FusionDescription& problem) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const FusionContext& context,
                const FusionDescription& problem,
                const PerformanceConfigConvOclDirectFwdFused&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvOclDirectFwdFused
    GetDefaultPerformanceConfig(const FusionContext&, const FusionDescription&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvOclDirectFwdFused
    Search(const FusionContext&,
           const FusionDescription&,
           const AnyInvokeParams& invoke_params) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const FusionContext&,
                             const FusionDescription&,
                             const PerformanceConfigConvOclDirectFwdFused&) const override;
    MIOPEN_INTERNALS_EXPORT float GetWti(const FusionContext&,
                                         const FusionDescription& problem) const override;
};

struct PerformanceConfigConvCKIgemmFwdBiasActivFused
    : PerfConfigBase<PerformanceConfigConvCKIgemmFwdBiasActivFused>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigConvCKIgemmFwdBiasActivFused(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigConvCKIgemmFwdBiasActivFused()
        : PerformanceConfigConvCKIgemmFwdBiasActivFused(0, "")
    {
    }
    PerformanceConfigConvCKIgemmFwdBiasActivFused(bool)
        : PerformanceConfigConvCKIgemmFwdBiasActivFused(0, "")
    {
    }
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const FusionDescription& fdesc_problem);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const FusionDescription& fdesc_problem);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool IsValid(const FusionContext&,
                                         const FusionDescription& fdesc_problem) const;

    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.kernel_id, "kernel_id");
    }
    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceConfigConvCKIgemmFwdBiasActivFused& other) const;

private:
    template <typename DataType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
};

struct ConvCKIgemmFwdBiasActivFused final
    : FusionTunableSolver<PerformanceConfigConvCKIgemmFwdBiasActivFused>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvCKIgemmFwdBiasActivFused>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvCKIgemmFwdBiasActivFused
    GetDefaultPerformanceConfig(const FusionContext& ctx,
                                const FusionDescription& fdesc_problem) const override;
    MIOPEN_INTERNALS_EXPORT bool IsValidPerformanceConfig(
        const FusionContext& ctx,
        const FusionDescription& fdesc_problem,
        const PerformanceConfigConvCKIgemmFwdBiasActivFused& config) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvCKIgemmFwdBiasActivFused
    Search(const FusionContext& ctx,
           const FusionDescription& fdesc_problem,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const FusionContext& ctx, const FusionDescription& fdesc_problem) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const FusionContext& ctx,
                const FusionDescription& fdesc_problem,
                const PerformanceConfigConvCKIgemmFwdBiasActivFused& config) const override;

private:
    template <typename DataType>
    bool CheckCKApplicability(const miopen::conv::ProblemDescription&) const;
};

struct PerfConfigConvCKIgemmFwdBiasResAddActivFused
    : PerfConfigBase<PerfConfigConvCKIgemmFwdBiasResAddActivFused>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerfConfigConvCKIgemmFwdBiasResAddActivFused(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerfConfigConvCKIgemmFwdBiasResAddActivFused()
        : PerfConfigConvCKIgemmFwdBiasResAddActivFused(0, "")
    {
    }
    PerfConfigConvCKIgemmFwdBiasResAddActivFused(bool)
        : PerfConfigConvCKIgemmFwdBiasResAddActivFused(0, "")
    {
    }
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const FusionDescription& fdesc_problem);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const FusionDescription& fdesc_problem);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool IsValid(const FusionContext&,
                                         const FusionDescription& fdesc_problem) const;

    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.kernel_id, "kernel_id");
    }
    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerfConfigConvCKIgemmFwdBiasResAddActivFused& other) const;

private:
    template <typename DataType, typename AccumDataType = DataType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType, typename AccumDataType = DataType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
};

struct ConvCKIgemmFwdBiasResAddActivFused final
    : FusionTunableSolver<PerfConfigConvCKIgemmFwdBiasResAddActivFused>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvCKIgemmFwdBiasResAddActivFused>();
    }

    MIOPEN_INTERNALS_EXPORT PerfConfigConvCKIgemmFwdBiasResAddActivFused
    GetDefaultPerformanceConfig(const FusionContext& ctx,
                                const FusionDescription& fdesc_problem) const override;
    MIOPEN_INTERNALS_EXPORT bool IsValidPerformanceConfig(
        const FusionContext& ctx,
        const FusionDescription& fdesc_problem,
        const PerfConfigConvCKIgemmFwdBiasResAddActivFused& config) const override;
    MIOPEN_INTERNALS_EXPORT PerfConfigConvCKIgemmFwdBiasResAddActivFused
    Search(const FusionContext& ctx,
           const FusionDescription& fdesc_problem,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const FusionContext& ctx, const FusionDescription& fdesc_problem) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const FusionContext& ctx,
                const FusionDescription& fdesc_problem,
                const PerfConfigConvCKIgemmFwdBiasResAddActivFused& config) const override;

private:
    template <typename DataType, typename AccumDataType = DataType>
    bool CheckCKApplicability(const miopen::conv::ProblemDescription&) const;
};

struct ConvBinWinogradRxSFused final : FusionSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvBinWinogradRxSFused>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const FusionContext& context,
                 const FusionDescription& fdesc_problem) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution GetSolution(
        const FusionContext& context, const FusionDescription& fdesc_problem) const override;
    MIOPEN_INTERNALS_EXPORT float GetWti(const FusionContext&,
                                         const FusionDescription&) const override;
};

struct ConvBinWinogradRxSf2x3g1Fused final : FusionSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvBinWinogradRxSf2x3g1Fused>();
    }

    MIOPEN_INTERNALS_EXPORT bool IsApplicable(const FusionContext& context,
                                              const FusionDescription& problem) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const FusionContext& context, const FusionDescription& problem) const override;
    MIOPEN_INTERNALS_EXPORT float GetWti(const FusionContext&,
                                         const FusionDescription&) const override;
};

template <uint32_t Winodata, uint32_t Winofilter>
struct ConvWinoFuryRxSFused final : FusionSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvWinoFuryRxSFused<Winodata, Winofilter>>();
    }

    bool IsApplicable(const FusionContext&, const FusionDescription&) const override;
    bool IsDynamic() const override { return true; }
    float GetWti(const FusionContext&, const FusionDescription&) const override;
    size_t GetWorkspaceSize(const FusionContext&, const FusionDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }

    ConvSolution GetSolution(const FusionContext&, const FusionDescription&) const override;
};

#ifndef CONV_WINO_FURY_RXS_CPP
extern template struct ConvWinoFuryRxSFused<2, 3>;
// extern template struct ConvWinoFuryRxSFused<3, 2>;
#endif

struct BnFwdInferActivationFused final : FusionSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<BnFwdInferActivationFused>();
    }

    MIOPEN_INTERNALS_EXPORT bool IsApplicable(const FusionContext& context,
                                              const FusionDescription& problem) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const FusionContext& context, const FusionDescription& problem) const override;
};

struct BnFwdTrgActivationFused final : FusionSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<BnFwdTrgActivationFused>();
    }

    MIOPEN_INTERNALS_EXPORT bool IsApplicable(const FusionContext& context,
                                              const FusionDescription& problem) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const FusionContext& context, const FusionDescription& problem) const override;
};

struct BnBwdTrgActivationFused final : FusionSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<BnBwdTrgActivationFused>();
    }

    MIOPEN_INTERNALS_EXPORT bool IsApplicable(const FusionContext& context,
                                              const FusionDescription& problem) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const FusionContext& context, const FusionDescription& problem) const override;
};

} // namespace fusion
} // namespace solver
} // namespace miopen
