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

#include <miopen/fusion_plan.hpp>
#include <miopen/batchnorm/problem_description.hpp>

namespace miopen {

struct FusionDescription
#if MIOPEN_ENABLE_SQLITE
    : SQLiteSerializable<FusionDescription>
#endif
{
    const miopen::FusionPlanDescriptor* fusion_plan_desc;
    FusionDescription(const miopen::FusionPlanDescriptor* ptr_desc) : fusion_plan_desc(ptr_desc) {}

    void GetNetworkConfig(std::stringstream& net_config, Handle& handle) const
    {
        for(const auto& op : fusion_plan_desc->op_map)
        {
            if(op->kind() == miopenFusionOpConvForward)
            {
                const auto prob = GetConvProblem(op->GetIdx(), conv::Direction::Forward);
                net_config << prob.BuildConfKey().ToString();
            }
            else if(op->kind() == miopenFusionOpBatchNormInference)
            {
                const auto prob =
                    GetBnProblem(op->GetIdx(), miopen::batchnorm::Direction::ForwardInference);
                net_config << prob.MakeNetworkConfig().ToString();
            }
            else if(op->kind() == miopenFusionOpBatchNormFwdTrain)
            {
                const auto prob =
                    GetBnProblem(op->GetIdx(), miopen::batchnorm::Direction::ForwardTraining);
                net_config << prob.MakeNetworkConfig().ToString();
            }
            else if(op->kind() == miopenFusionOpBatchNormBwdTrain)
            {
                const auto prob =
                    GetBnProblem(op->GetIdx(), miopen::batchnorm::Direction::Backward);
                net_config << prob.MakeNetworkConfig().ToString();
            }
            else
            {
                op->GetNetworkConfig(net_config, handle);
            }
        }
        MIOPEN_LOG_I2(net_config.str());
    }

#if !MIOPEN_ENABLE_SQLITE
    /// \todo This function will be necessary for tuning of fusions
    void Serialize(std::ostream& stream) const
    {
        auto conv_problem = GetConvProblem(0, conv::Direction::Forward);
        conv_problem.Serialize(stream);
    }
#endif

#if MIOPEN_ENABLE_SQLITE
    static std::string table_name() { return "config"; } // revisit this

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        auto conv_prob = self.GetConvProblem(0, conv::Direction::Forward);
        ProblemDescription::Visit(conv_prob, f);
    }
#endif

    // This and the following method should be moved to the Ops once the return type can be unified
    miopen::ProblemDescription GetConvProblem(size_t idx, conv::Direction dir, int bias = 0) const
    {
        const auto& conv_op =
            dynamic_cast<ConvForwardOpDescriptor&>(*fusion_plan_desc->op_map[idx]);
        if(dir == conv::Direction::Forward)
        {
            TensorDescriptor out_desc;
            conv_op.GetOutputDesc(out_desc);
            return miopen::conv::ProblemDescription{conv_op.input_desc,
                                                    conv_op.filter_desc,
                                                    out_desc,
                                                    conv_op.base_desc /* conv desc */,
                                                    dir,
                                                    bias};
        }
        else
        {
            MIOPEN_THROW(miopenStatusNotImplemented);
        }
    }

    miopen::batchnorm::ProblemDescription GetBnProblem(size_t idx,
                                                       miopen::batchnorm::Direction dir) const
    {
        // epsilon is part of the BN ProblemDescription, which is incorrect since it should be part
        // of the invoke parameters epsilon is a runtime argument to the BN kernels, therefore here
        // we fill it with a dummy value and use it the value stored in the OperatorArgs (aka Invoke
        // Params) instead
        const double not_used = std::numeric_limits<double>::signaling_NaN(); // Temporary filler
        if(dir == miopen::batchnorm::Direction::ForwardInference)
        {
            const auto& bn_op =
                dynamic_cast<BatchNormInferenceFusionOpDescriptor&>(*fusion_plan_desc->op_map[idx]);
            miopen::TensorDescriptor out_desc;
            bn_op.GetOutputDesc(out_desc);
            return {bn_op.mode, bn_op.input_desc, out_desc, bn_op.base_desc, not_used};
        }
        else if(dir == miopen::batchnorm::Direction::ForwardTraining)
        {
            const auto& bn_op =
                dynamic_cast<BatchNormFwdTrainFusionOpDescriptor&>(*fusion_plan_desc->op_map[idx]);
            miopen::TensorDescriptor out_desc;
            bn_op.GetOutputDesc(out_desc);
            return {bn_op.mode,
                    bn_op.input_desc,
                    out_desc,
                    bn_op.base_desc,
                    not_used, // expAvgFactor filler
                    not_used,
                    true /* resultSave*/,
                    bn_op.runningMeanVar};
        }
        else if(dir == miopen::batchnorm::Direction::Backward)
        {
            const auto& bn_op =
                dynamic_cast<BatchNormBwdTrainFusionOpDescriptor&>(*fusion_plan_desc->op_map[idx]);
            miopen::TensorDescriptor out_desc;
            bn_op.GetOutputDesc(out_desc);
            return {bn_op.mode,
                    bn_op.input_desc,
                    out_desc,
                    bn_op.input_desc,
                    {} /*bn_op.base_desc*/,
                    not_used,
                    bn_op.useBatchStats /*useSaved*/};
        }
        else
            MIOPEN_THROW(miopenStatusNotImplemented);
    }
};

} // namespace miopen
