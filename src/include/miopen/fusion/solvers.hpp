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
#include <miopen/fusion.hpp>
#include <miopen/fusion_plan.hpp>
#include <utility>

namespace miopen {

struct FusionDescription : SQLiteSerializable<FusionDescription>
{
    const miopen::FusionPlanDescriptor* fusion_plan_desc;
    FusionDescription(const miopen::FusionPlanDescriptor* ptr_desc) : fusion_plan_desc(ptr_desc) {}
    static std::string table_name() { return "config"; } // revisit this
    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        auto conv_prob = self.GetConvProblem(0, conv::Direction::Forward);
        ProblemDescription::Visit(conv_prob, f);
    }
    miopen::ProblemDescription GetConvProblem(size_t idx, conv::Direction dir) const
    {
        const auto& conv_op =
            dynamic_cast<ConvForwardOpDescriptor&>(*fusion_plan_desc->op_map[idx]);
        if(dir == conv::Direction::Forward)
        {
            TensorDescriptor out_desc;
            conv_op.GetOutputDesc(out_desc);
            return miopen::ProblemDescription{conv_op.input_desc,
                                              conv_op.filter_desc,
                                              out_desc,
                                              conv_op.base_desc /* conv desc */,
                                              dir};
        }
        else
        {
            MIOPEN_THROW(miopenStatusNotImplemented);
        }
        return {};
    }
};

struct FusionContext : miopen::ExecutionContext
{
    FusionDescription problem;
    FusionContext(FusionPlanDescriptor* ptr_desc, Handle& handle)
        : ExecutionContext(&handle), problem(ptr_desc)
    {
    }
    ConvolutionContext GetConvContext(size_t idx, conv::Direction dir) const
    {
        const auto conv_prob = problem.GetConvProblem(idx, dir);
        if(dir == conv::Direction::Forward)
        {
            auto ctx                       = ConvolutionContext{conv_prob};
            ctx.do_search                  = this->do_search;
            ctx.save_srch_req              = this->save_srch_req;
            ctx.use_asm_kernels            = this->use_asm_kernels;
            ctx.use_hip_kernels            = this->use_hip_kernels;
            ctx.use_opencl_convolutions    = this->use_opencl_convolutions;
            ctx.use_binaries               = this->use_binaries;
            ctx.disable_search_enforce     = this->disable_search_enforce;
            ctx.disable_perfdb_access      = this->disable_perfdb_access;
            ctx.use_dynamic_solutions_only = this->use_dynamic_solutions_only;
            ctx.general_compile_options    = "";

            ctx.SetStream(&this->GetStream());
            ctx.DetectRocm();
            ctx.SetupFloats();
            return ctx;
        }
        else
        {
            MIOPEN_THROW(miopenStatusNotImplemented);
        }
    }
    bool is_for_generic_search = false;
};

namespace solver {
namespace fusion {

using FusionSolverBase = SolverMixin<FusionContext>;

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
    Search(const FusionContext& ctx, const AnyInvokeParams& invoke_ctx, int) const = 0;

    /// Tunable solvers provide a GetSolution that takes a Context and PerformanceConfig
    virtual ConvSolution GetSolution(const FusionContext& ctx, const boost::any& config) const = 0;
};
template <class PerformanceConfig>
struct FusionTunableSolver : FusionTunableSolverBase
{
    virtual PerformanceConfig GetDefaultPerformanceConfig(const FusionContext&) const           = 0;
    virtual bool IsValidPerformanceConfig(const FusionContext&, const PerformanceConfig&) const = 0;
    virtual PerformanceConfig Search(const FusionContext&, const AnyInvokeParams&) const        = 0;
    virtual ConvSolution GetSolution(const FusionContext&, const PerformanceConfig&) const      = 0;

    boost::any GetDefaultPerformanceConfig(const FusionContext& ctx, int) const final
    {
        return GetDefaultPerformanceConfig(ctx);
    }

    bool IsValidPerformanceConfig(const FusionContext& ctx, const boost::any& config) const final
    {
        return IsValidPerformanceConfig(ctx, boost::any_cast<const PerformanceConfig&>(config));
    }

    boost::any Search(const FusionContext& ctx, const AnyInvokeParams& invoke_ctx, int) const final
    {
        return Search(ctx, invoke_ctx);
    }

    ConvSolution GetSolution(const FusionContext& ctx, const boost::any& config) const final
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
    void HeuristicInit(const FusionContext& config);
    bool SetNextValue(const FusionContext& config);
    bool IsValid(const FusionContext& config) const;
};
struct ConvBiasActivAsm1x1U : FusionTunableSolver<PerformanceConfigConvBiasActivAsm1x1U>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvBiasActivAsm1x1U>(); }

    bool IsApplicable(const FusionContext& desc) const override;
    ConvSolution
    GetSolution(const FusionContext& problem,
                const PerformanceConfigConvBiasActivAsm1x1U& /*config*/) const override;

    PerformanceConfigConvBiasActivAsm1x1U
    GetDefaultPerformanceConfig(const FusionContext&) const override;

    PerformanceConfigConvBiasActivAsm1x1U Search(const FusionContext&,
                                                 const AnyInvokeParams& invoke_ctx) const override;
    bool IsValidPerformanceConfig(const FusionContext&,
                                  const PerformanceConfigConvBiasActivAsm1x1U&) const override;
};

using PerformanceConfigConvOclDirectFwdFused = LegacyPerformanceConfig;
struct ConvOclDirectFwdFused final : FusionTunableSolver<LegacyPerformanceConfig>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvOclDirectFwdFused>();
    }

    bool IsApplicable(const FusionContext& problem) const override;
    ConvSolution GetSolution(const FusionContext& problem,
                             const PerformanceConfigConvOclDirectFwdFused&) const override;
    PerformanceConfigConvOclDirectFwdFused
    GetDefaultPerformanceConfig(const FusionContext&) const override;
    PerformanceConfigConvOclDirectFwdFused Search(const FusionContext&,
                                                  const AnyInvokeParams& invoke_ctx) const override;
    bool IsValidPerformanceConfig(const FusionContext&,
                                  const PerformanceConfigConvOclDirectFwdFused&) const override;
};

struct ConvBinWinogradRxSFused final : FusionSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvBinWinogradRxSFused>();
    }

    bool IsApplicable(const FusionContext& params) const override;
    ConvSolution GetSolution(const FusionContext& params) const;
};

} // namespace fusion
} // namespace solver
} // namespace miopen
