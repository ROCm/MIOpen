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
    int idx = 0;
    for(const auto& ptr : op_map)
    {
        if(ptr->kind() == op)
            return idx;
        ++idx;
    }
    return -1;
}

inline bool WinoCommonIsApplicable(const FusionContext& params)
{
    const auto& desc = *params.problem.fusion_plan_desc;
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
        int idx = 0;
        for(const auto& prim : desc.op_map)
        {
            if(prim->kind() == miopenFusionOpActivForward)
                return idx;
            ++idx;
        }
        return -1;
    }();
    if(activ_idx != -1)
    {
        const auto& activ_op  = dynamic_cast<ActivFwdFusionOpDescriptor&>(*desc.op_map[activ_idx]);
        const auto activ_mode = activ_op.activMode;
        if(!(activ_mode == miopenActivationRELU || activ_mode == miopenActivationLEAKYRELU))
            return false;
    }
    const miopen::ConvolutionContext conv_ctx =
        params.GetConvContext(0, miopen::conv::Direction::Forward);

    if(!conv_ctx.problem.Is2d())
        return false;
    if(!conv_ctx.problem.IsFp32())
        return false;
    if(!conv_ctx.problem.IsLayoutDefault())
        return false;
    if(!conv_ctx.problem.direction.IsForward())
        return false;
    const auto target = conv_ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;
    const auto c           = conv_ctx.problem.conv_problem.GetInChannels();
    const auto k           = conv_ctx.problem.conv_problem.GetOutChannels();
    const auto x           = conv_ctx.problem.conv_problem.GetWeightsWidth();
    const auto y           = conv_ctx.problem.conv_problem.GetWeightsHeight();
    const auto oH          = conv_ctx.problem.conv_problem.GetOutHeight();
    const auto oW          = conv_ctx.problem.conv_problem.GetOutWidth();
    const auto iH          = conv_ctx.problem.conv_problem.GetInHeight();
    const auto iW          = conv_ctx.problem.conv_problem.GetInWidth();
    const auto pad_h       = conv_ctx.problem.conv_problem.GetPadH();
    const auto pad_w       = conv_ctx.problem.conv_problem.GetPadW();
    const auto group_count = conv_ctx.problem.conv_problem.GetGroupCount();
    const auto N           = conv_ctx.problem.conv_problem.GetInBatchSize();

    return conv_ctx.problem.kernel_stride_h == conv_ctx.problem.kernel_stride_w &&
           conv_ctx.problem.kernel_dilation_h == 1 && conv_ctx.problem.kernel_dilation_w == 1 &&
           (c * x * y) <= std::pow(2, 28) && (k * x * y) <= std::pow(2, 28) &&
           (k * oH * oW) <= std::pow(2, 28) && (c * iH * iW) <= std::pow(2, 28) &&
           x <= std::pow(2, 16) && y <= std::pow(2, 16) && pad_h <= std::pow(2, 16) &&
           pad_w <= std::pow(2, 16) && oH <= std::pow(2, 16) && oW <= std::pow(2, 16) &&
           iH <= std::pow(2, 16) && oW <= std::pow(2, 16) && c <= std::pow(2, 16) &&
           k <= std::pow(2, 16) && N <= std::pow(2, 16) && group_count == 1;
}
} // namespace fusion
} // namespace solver
} // namespace miopen
