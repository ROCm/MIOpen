/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include "bn_test_base.hpp"
namespace test {
namespace FusionPlan {
template <typename DLModule>
void InitFusionPlan(miopen::FusionPlanDescriptor& fusePlanDesc, DLModule& dl_module)
{
    fusePlanDesc = miopen::FusionPlanDescriptor(miopenVerticalFusion, dl_module.GetInputDesc());
}

template <typename DLModule>
void AddBnInfer(miopen::FusionPlanDescriptor& fusePlanDesc,
                miopen::OperatorArgs& params,
                DLModule& dl_module)
{
    auto bnOp = std::make_shared<miopen::BatchNormInferenceFusionOpDescriptor>(dl_module.bn_mode,
                                                                               dl_module.bn_desc);
    EXPECT_EQ(fusePlanDesc.AddOp(bnOp), miopenStatusSuccess);
    bnOp->SetArgs(params,
                  &dl_module.alpha,
                  &dl_module.beta,
                  dl_module.scale_dev.get(),
                  dl_module.shift_dev.get(),
                  dl_module.estMean_dev.get(),
                  dl_module.estVariance_dev.get(),
                  dl_module.epsilon);
}

template <typename DLModule>
void AddBwdTrain(miopen::FusionPlanDescriptor& fusePlanDesc,
                miopen::OperatorArgs& params,
                DLModule& dl_module)
{
    auto bnOp = std::make_shared<miopen::BatchNormBwdTrainFusionOpDescriptor>(dl_module.bn_mode,
                                                                               dl_module.bn_desc);
    EXPECT_EQ(fusePlanDesc.AddOp(bnOp), miopenStatusSuccess);
    bnOp->SetArgs(params,
                  &dl_module.alpha,
                  &dl_module.beta,
                  dl_module.x.get(),
                  dl_module.bnScale.get(),
                  dl_module.bnBias.get(),
                  dl_module.resultBnScaleDiff.get(),
                  dl_module.resultBnBiasDiff.get(),
                  dl_module.savedMean.get(),
                  dl_module.savedInvVariance.get());
}


template <typename DLModule>
void AddActiv(miopen::FusionPlanDescriptor& fusePlanDesc,
              miopen::OperatorArgs& params,
              DLModule& dl_module,
              miopenActivationMode_t activ_mode)
{
    auto activOp = std::make_shared<miopen::ActivFwdFusionOpDescriptor>(activ_mode);
    EXPECT_EQ(fusePlanDesc.AddOp(activOp), miopenStatusSuccess);
    EXPECT_EQ(activOp->SetArgs(params,
                               &dl_module.alpha,
                               &dl_module.beta,
                               dl_module.activ_alpha,
                               dl_module.activ_beta,
                               dl_module.activ_gamma),
              miopenStatusSuccess);
}

bool Skip(miopen::Handle& handle)
{
    const std::string arch = handle.GetDeviceName();
    bool skip_test         = (arch != "gfx908" && arch != "gfx90a");

    return skip_test;
}

template <typename DLModule>
void ComputeRefBNInfer(DLModule& dl_module)
{
    if(dl_module.bn_mode == miopenBNPerActivation)
    {
        batchNormPerActivHostInference(dl_module.input,
                                       dl_module.ref_out,
                                       dl_module.scale,
                                       dl_module.shift,
                                       dl_module.epsilon,
                                       dl_module.estMean,
                                       dl_module.estVariance);
    }
    else
    {
        batchNormSpatialHostInference(dl_module.input,
                                      dl_module.ref_out,
                                      dl_module.scale,
                                      dl_module.shift,
                                      dl_module.epsilon,
                                      dl_module.estMean,
                                      dl_module.estVariance);
    }
}

//  tensor<T>& dx_out,
//                                   const tensor<U>& scale,
//                                   tensor<U>& dscale,
//                                   tensor<U>& dbias,
//                                   const tensor<U>& savedMean,
//                                   const tensor<U>& savedInvVar)

template <typename DLModule>
void ComputeRefBNBwdTrain(DLModule& dl_module)
{
    batchNormSpatialHostBwdTrain(dl_module.x_input,
                                 dl_module.input,
                                 dl_module.ref_out,
                                 dl_module.bnscale,
                                 dl_module.resBnScaleDiff,
                                 dl_module.dbeta,
                                 dl_module.resBnScaleDiff,
                                 dl_module.resBnBiasDiff,
                                 dl_module.savedMean,
                                 dl_module.savedInvVariance);

}


template <typename T>
void BnCmpare(const tensor<T>& output, const tensor<T>& ref_out)
{
    EXPECT_FALSE(miopen::range_zero(ref_out)) << "CPU data is all zeros";
    EXPECT_FALSE(miopen::range_zero(output)) << "GPU data is all zeros";
    EXPECT_FALSE(miopen::find_idx(output, miopen::not_finite) >= 0)
        << "Non finite number found in the GPU data";
    EXPECT_TRUE(miopen::range_distance(ref_out) == miopen::range_distance(output));
    const double tolerance = 80;
    double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
    auto error             = miopen::rms_range(ref_out, output);
    EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
        << "Non finite number found in the CPU data";
    EXPECT_TRUE(error < threshold)
        << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
}

} // namespace FusionPlan
} // namespace test
