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

namespace test {
template <typename DLModule>
void ComputeCPUBNInference(DLModule& dl_module)
{
    batchNormSpatialHostInference(dl_module.input,
                                  dl_module.ref_out,
                                  dl_module.scale,
                                  dl_module.shift,
                                  dl_module.epsilon,
                                  dl_module.estMean,
                                  dl_module.estVariance);
}

template <typename DLModule>
void ComputeCPUBNBwd(DLModule& dl_module)
{
    batchNormSpatialHostBwdTrain(dl_module.input,
                                 dl_module.dy,
                                 dl_module.ref_out,
                                 dl_module.bnScale,
                                 dl_module.dScale_ref,
                                 dl_module.dBias_ref,
                                 dl_module.savedMean,
                                 dl_module.savedInvVar);
}

template <typename DLModule>
void ComputeCPUBNFwdTrain(DLModule& dl_module)
{
    batchNormSpatialHostFwdTrain(dl_module.input,
                                 dl_module.ref_out,
                                 dl_module.scale,
                                 dl_module.shift,
                                 dl_module.epsilon,
                                 dl_module.averageFactor,
                                 dl_module.saveMean_ref,
                                 dl_module.saveVariance_ref,
                                 dl_module.runMean_ref,
                                 dl_module.runVariance_ref);
}

template <typename T>
void CompareTensor(const tensor<T>& output,
                   const tensor<T>& ref_out,
                   const double threshold = std::numeric_limits<T>::epsilon())
{
    EXPECT_FALSE(miopen::range_zero(ref_out)) << "CPU data is all zeros";
    EXPECT_FALSE(miopen::range_zero(output)) << "GPU data is all zeros";
    EXPECT_FALSE(miopen::find_idx(output, miopen::not_finite) >= 0)
        << "Non finite number found in the GPU data";
    EXPECT_TRUE(miopen::range_distance(ref_out) == miopen::range_distance(output));
    auto error = miopen::rms_range(ref_out, output);
    EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
        << "Non finite number found in the CPU data";
    EXPECT_TRUE(error < threshold)
        << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
}
} // namespace test

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

inline bool Skip(miopen::Handle& handle)
{
    const std::string arch = handle.GetDeviceName();
    bool skip_test         = (arch != "gfx908" && arch != "gfx90a");

    return skip_test;
}
} // namespace FusionPlan
} // namespace test
