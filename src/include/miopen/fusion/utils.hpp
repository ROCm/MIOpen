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

#include <miopen/fusion/solvers.hpp>

namespace miopen {
namespace solver {
namespace fusion {

inline int GetOpIdx(const std::vector<std::shared_ptr<FusionOpDescriptor>>& op_map,
                    miopenFusionOp_t op)
{
    auto it = std::find_if(
        op_map.cbegin(), op_map.cend(), [op](auto&& item) { return item->kind() == op; });
    return it == op_map.cend() ? -1 : std::distance(op_map.cbegin(), it);
}

inline bool WinoCommonIsApplicable(const FusionContext& context, const FusionDescription& problem)
{
    const auto& desc = *problem.fusion_plan_desc;
    if(desc.op_map.empty())
    {
        MIOPEN_THROW("");
    }
    // check the sequence of prims
    if(desc.op_map.size() > 3)
        return false;
    if(desc.op_map[0]->kind() != miopenFusionOpConvForward)
        return false;
    if(desc.op_map.size() >= 2)
    {
        const auto prim = desc.op_map[1]->kind();
        if(!(prim == miopenFusionOpBiasForward || prim == miopenFusionOpActivForward))
            return false;
    }
    if(desc.op_map.size() == 3)
    {
        const auto prim = desc.op_map[2]->kind();
        if(prim != miopenFusionOpActivForward)
            return false;
    }
    const auto activ_idx = [&]() {
        const auto it = std::find_if(desc.op_map.cbegin(), desc.op_map.cend(), [](auto&& prim) {
            return prim->kind() == miopenFusionOpActivForward;
        });
        return it == desc.op_map.cend() ? -1 : std::distance(desc.op_map.cbegin(), it);
    }();
    if(activ_idx != -1)
    {
        const auto& activ_op  = dynamic_cast<ActivFwdFusionOpDescriptor&>(*desc.op_map[activ_idx]);
        const auto activ_mode = activ_op.activMode;
        if(!(activ_mode == miopenActivationRELU || activ_mode == miopenActivationLEAKYRELU))
            return false;
    }

    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto conv_ctx     = context.GetConvContext(conv_problem);

    if(!conv_problem.Is2d())
        return false;
    if(!conv_problem.IsFp32())
        return false;
    if(conv_problem.HasNonPackedTensors())
        return false;
    if(!conv_problem.AllTensorsDimsFitIntoInt())
        return false;
    if(!conv_problem.IsLayoutDefault())
        return false;
    if(!conv_problem.IsDirectionForward())
        return false;
    const auto& target = conv_ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    return true;
}

inline bool IsCKFusionSolverApplicable(const FusionContext& context,
                                       const FusionDescription& problem)
{
    const auto ck_ca_fusion_solver = miopen::solver::fusion::ConvCKIgemmGrpFwdActivFused{};
    if(ck_ca_fusion_solver.IsApplicable(context, problem))
    {
        MIOPEN_LOG_I("ConvCKIgemmGrpFwdActivFused is applicable, skipping current solver.");
        return true;
    }

    const auto ck_cba_fusion_solver = miopen::solver::fusion::ConvCKIgemmGrpFwdBiasActivFused{};
    if(ck_cba_fusion_solver.IsApplicable(context, problem))
    {
        MIOPEN_LOG_I("ConvCKIgemmGrpFwdBiasActivFused is applicable, skipping current solver.");
        return true;
    }

    return false;
}

} // namespace fusion
} // namespace solver
} // namespace miopen
