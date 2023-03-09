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
#include <miopen/fusion.hpp>
#include <miopen/fusion_plan.hpp>
#include <utility>

#include <miopen/fusion/problem_description.hpp>
#include <miopen/fusion/context.hpp>

namespace miopen {
namespace solver {
namespace fusion {

using OldStyleFusionDesc = FusionContext;

using FusionSolverBase = SolverMixin<OldStyleFusionDesc>;

struct FusionTunableSolverBase : FusionSolverBase
{
    /// Initializes performance config to the default values.
    /// The function may involve some heuristic to guess the best solution
    /// configuration. It is assumed that the function takes constant time
    /// to finish and does not run kernels to measure performance etc.
    /// The function shall always return valid config.
    ///
    /// The int parameter is needed only to not change the name of the
    /// function in the derived class. Function declarations that differ
    /// only by its return type cannot be overloaded.
    virtual boost::any GetDefaultPerformanceConfig(const FusionContext& ctx, int) const = 0;

    /// Should return false if performance config is wrong for a problem.
    /// Main use is validation of values read from the perf db.
    virtual bool IsValidPerformanceConfig(const FusionContext& ctx,
                                          const boost::any& config) const = 0;

    /// Search
    ///
    /// The int parameter is needed only to not change the name of the
    /// function in the derived class. Function declarations that differ
    /// only by its return type cannot be overloaded.
    virtual boost::any
    Search(const OldStyleFusionDesc& ctx, const AnyInvokeParams& invoke_ctx, int) const = 0;

    /// Tunable solvers provide a GetSolution that takes a Context and PerformanceConfig
    virtual ConvSolution GetSolution(const OldStyleFusionDesc& ctx,
                                     const boost::any& config) const = 0;
};

template <class PerformanceConfig>
struct FusionTunableSolver : FusionTunableSolverBase
{
    virtual PerformanceConfig GetDefaultPerformanceConfig(const FusionContext&) const           = 0;
    virtual bool IsValidPerformanceConfig(const FusionContext&, const PerformanceConfig&) const = 0;
    virtual PerformanceConfig Search(const OldStyleFusionDesc&, const AnyInvokeParams&) const   = 0;
    virtual ConvSolution GetSolution(const OldStyleFusionDesc&, const PerformanceConfig&) const = 0;

    boost::any GetDefaultPerformanceConfig(const FusionContext& ctx, int) const final
    {
        return GetDefaultPerformanceConfig(ctx);
    }

    bool IsValidPerformanceConfig(const FusionContext& ctx, const boost::any& config) const final
    {
        return IsValidPerformanceConfig(ctx, boost::any_cast<const PerformanceConfig&>(config));
    }

    boost::any
    Search(const OldStyleFusionDesc& ctx, const AnyInvokeParams& invoke_ctx, int) const final
    {
        return Search(ctx, invoke_ctx);
    }

    ConvSolution GetSolution(const OldStyleFusionDesc& ctx, const boost::any& config) const final
    {
        return GetSolution(ctx, boost::any_cast<const PerformanceConfig&>(config));
    }
    bool IsDynamic() const override { return false; }
};

struct PerformanceConfigConvBiasActivAsm1x1U : PerformanceConfigConvAsm1x1U
{
    PerformanceConfigConvBiasActivAsm1x1U(const bool spare) : PerformanceConfigConvAsm1x1U(spare) {}
    PerformanceConfigConvBiasActivAsm1x1U()
        : PerformanceConfigConvAsm1x1U(-1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    void HeuristicInit(const FusionContext& context);
    bool SetNextValue(const FusionContext& context);
    bool IsValid(const FusionContext& context) const;
};

struct ConvBiasActivAsm1x1U : FusionTunableSolver<PerformanceConfigConvBiasActivAsm1x1U>
{
    using FusionTunableSolver::GetSolution;
    using FusionTunableSolver::IsApplicable;
    using FusionTunableSolver::Search;

    const std::string& SolverDbId() const override { return GetSolverDbId<ConvBiasActivAsm1x1U>(); }

    bool IsApplicable(const OldStyleFusionDesc& context) const override
    {
        return IsApplicable(context, context.problem);
    }

    bool IsApplicable(const FusionContext& fusion_ctx, const FusionDescription& problem) const;

    ConvSolution GetSolution(const OldStyleFusionDesc& context,
                             const PerformanceConfigConvBiasActivAsm1x1U& config) const override
    {
        return GetSolution(context, context.problem, config);
    }
    ConvSolution GetSolution(const FusionContext& ctx,
                             const FusionDescription& problem,
                             const PerformanceConfigConvBiasActivAsm1x1U& /*config*/) const;

    PerformanceConfigConvBiasActivAsm1x1U
    GetDefaultPerformanceConfig(const FusionContext&) const override;

    PerformanceConfigConvBiasActivAsm1x1U Search(const OldStyleFusionDesc& context,
                                                 const AnyInvokeParams& invoke_ctx) const override
    {
        return Search(context, context.problem, invoke_ctx);
    }

    PerformanceConfigConvBiasActivAsm1x1U Search(const FusionContext& context,
                                                 const FusionDescription& problem,
                                                 const AnyInvokeParams& invoke_ctx) const;
    bool IsValidPerformanceConfig(const FusionContext&,
                                  const PerformanceConfigConvBiasActivAsm1x1U&) const override;
};

using PerformanceConfigConvOclDirectFwdFused = LegacyPerformanceConfig;
struct ConvOclDirectFwdFused final : FusionTunableSolver<LegacyPerformanceConfig>
{
    using FusionTunableSolver::GetSolution;
    using FusionTunableSolver::IsApplicable;
    using FusionTunableSolver::Search;

    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvOclDirectFwdFused>();
    }

    bool IsApplicable(const OldStyleFusionDesc& context) const override
    {
        return IsApplicable(context, context.problem);
    }

    bool IsApplicable(const FusionContext& context, const FusionDescription& problem) const;

    ConvSolution GetSolution(const OldStyleFusionDesc& context,
                             const PerformanceConfigConvOclDirectFwdFused& config) const override
    {
        return GetSolution(context, context.problem, config);
    }

    ConvSolution GetSolution(const FusionContext& context,
                             const FusionDescription& problem,
                             const PerformanceConfigConvOclDirectFwdFused&) const;
    PerformanceConfigConvOclDirectFwdFused
    GetDefaultPerformanceConfig(const FusionContext&) const override;
    PerformanceConfigConvOclDirectFwdFused Search(const FusionContext&,
                                                  const FusionDescription&,
                                                  const AnyInvokeParams& invoke_params) const;
    PerformanceConfigConvOclDirectFwdFused
    Search(const OldStyleFusionDesc& context, const AnyInvokeParams& invoke_params) const override
    {
        return Search(context, context.problem, invoke_params);
    }
    bool IsValidPerformanceConfig(const FusionContext&,
                                  const PerformanceConfigConvOclDirectFwdFused&) const override;
};

struct ConvBinWinogradRxSFused final : FusionSolverBase
{
    using FusionSolverBase::IsApplicable;

    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvBinWinogradRxSFused>();
    }

    bool IsApplicable(const OldStyleFusionDesc& context) const override
    {
        return IsApplicable(context, context.problem);
    }

    bool IsApplicable(const FusionContext& context, const FusionDescription& problem) const;

    ConvSolution GetSolution(const OldStyleFusionDesc& context) const
    {
        return GetSolution(context, context.problem);
    }

    ConvSolution GetSolution(const FusionContext& context, const FusionDescription& problem) const;
};

struct ConvBinWinogradRxSf2x3g1Fused final : FusionSolverBase
{
    using FusionSolverBase::IsApplicable;

    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvBinWinogradRxSf2x3g1Fused>();
    }

    bool IsApplicable(const OldStyleFusionDesc& context) const override
    {
        return IsApplicable(context, context.problem);
    }

    bool IsApplicable(const FusionContext& context, const FusionDescription& problem) const;

    ConvSolution GetSolution(const OldStyleFusionDesc& context) const
    {
        return GetSolution(context, context.problem);
    }

    ConvSolution GetSolution(const FusionContext& context, const FusionDescription& problem) const;
};

struct BnFwdInferActivationFused final : FusionSolverBase
{
    using FusionSolverBase::IsApplicable;

    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<BnFwdInferActivationFused>();
    }

    bool IsApplicable(const OldStyleFusionDesc& context) const override
    {
        return IsApplicable(context, context.problem);
    }

    bool IsApplicable(const FusionContext& context, const FusionDescription& problem) const;

    ConvSolution GetSolution(const OldStyleFusionDesc& context) const
    {
        return GetSolution(context, context.problem);
    }

    ConvSolution GetSolution(const FusionContext& context, const FusionDescription& problem) const;
};

struct BnFwdTrgActivationFused final : FusionSolverBase
{
    using FusionSolverBase::IsApplicable;

    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<BnFwdTrgActivationFused>();
    }

    bool IsApplicable(const OldStyleFusionDesc& context) const override
    {
        return IsApplicable(context, context.problem);
    }
    bool IsApplicable(const FusionContext& context, const FusionDescription& problem) const;

    ConvSolution GetSolution(const OldStyleFusionDesc& context) const
    {
        return GetSolution(context, context.problem);
    }
    ConvSolution GetSolution(const FusionContext& context, const FusionDescription& problem) const;
};

struct BnBwdTrgActivationFused final : FusionSolverBase
{
    using FusionSolverBase::IsApplicable;

    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<BnBwdTrgActivationFused>();
    }

    bool IsApplicable(const OldStyleFusionDesc& context) const override
    {
        return IsApplicable(context, context.problem);
    }

    bool IsApplicable(const FusionContext& context, const FusionDescription& problem) const;

    ConvSolution GetSolution(const OldStyleFusionDesc& context) const
    {
        return GetSolution(context, context.problem);
    }

    ConvSolution GetSolution(const FusionContext& context, const FusionDescription& problem) const;
};

struct PerformanceConfigCKGEMM : PerfConfigBase<PerformanceConfigCKGEMM>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigCKGEMM(int idx, std::string kernl_id) : index(idx), kernel_id(kernl_id) {}
    PerformanceConfigCKGEMM() : PerformanceConfigCKGEMM(0, "") {}
    PerformanceConfigCKGEMM(bool) : PerformanceConfigCKGEMM(0, "") {}
    void HeuristicInit(const FusionContext& ctx);
    bool SetNextValue(const FusionContext& ctx);
    bool IsValidValue() const;
    bool IsValid(const FusionContext& ctx) const;

    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.kernel_id, "kernel_id");
    }
    bool operator==(const PerformanceConfigCKGEMM& other) const;

private:
    template <typename DataType>
    void Init(const gemm::ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const gemm::ProblemDescription&) const;
};

struct CKGEMM final : FusionTunableSolver<PerformanceConfigCKGEMM>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<CKGEMM>(); }

    PerformanceConfigCKGEMM GetDefaultPerformanceConfig(const FusionContext& ctx) const override;
    bool IsValidPerformanceConfig(const FusionContext& ctx,
                                  const PerformanceConfigCKGEMM& config) const override;
    PerformanceConfigCKGEMM Search(const FusionContext& ctx,
                                   const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const FusionContext& ctx) const override;
    ConvSolution GetSolution(const FusionContext& ctx,
                             const PerformanceConfigCKGEMM& config) const override;

private:
    template <typename DataType>
    bool CheckCKApplicability(const gemm::ProblemDescription&) const;
};

} // namespace fusion
} // namespace solver
} // namespace miopen
