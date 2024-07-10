/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "get_handle.hpp"
#include "random.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include <miopen/miopen.h>
#include <miopen/kernel_build_params.hpp>

#include <miopen/batchnorm/problem_description.hpp>

#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <miopen/solver/ck_utility_common.hpp>

#include "bn_test_data.hpp"
#include "test_operations.hpp"

void BatchNormForwardInferencGPU(miopen::Handle& handle,
                                 miopenBatchNormMode_t bn_mode,
                                 const void* alpha,
                                 const void* beta,
                                 const miopen::TensorDescriptor& xDesc,
                                 ConstData_t x,
                                 const miopen::TensorDescriptor& yDesc,
                                 Data_t y,
                                 const miopen::TensorDescriptor& bnScaleBiasMeanVarDesc,
                                 ConstData_t bnScale,
                                 ConstData_t bnBias,
                                 ConstData_t estimatedMean,
                                 ConstData_t estimatedVariance,
                                 double epsilon,
                                 bool use_hip = false)
{

    bool bfpmixparm = false;
    bool bfp16parm  = false;
    bool bfp32parm  = true;

    if(xDesc.GetType() == miopenHalf && bnScaleBiasMeanVarDesc.GetType() == miopenHalf)
    {
        bfp16parm = true;
        bfp32parm = false;
    }
    else if(xDesc.GetType() == miopenHalf && bnScaleBiasMeanVarDesc.GetType() == miopenFloat)
    {
        bfpmixparm = true;
        bfp32parm  = false;
    }

    int n, c, h, w;
    std::tie(n, c, h, w) = miopen::tien<4>(xDesc.GetLengths());

    unsigned int in_cstride = h * w;
    unsigned int in_nstride = c * in_cstride;

    size_t xlocalsize = 1;
    auto xgridsize    = c;
    size_t ylocalsize = 256;
    size_t ygridsize  = ylocalsize * ((in_cstride + ylocalsize - 1) / ylocalsize);
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    const auto build_params = miopen::KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(bfp16parm)},
        {"MIOPEN_USE_FP32", static_cast<int>(bfp32parm)},
        {"MIOPEN_USE_FPMIX", static_cast<int>(bfpmixparm)},
        {"MIO_BN_GRP0", xlocalsize},
        {"MIO_BN_GRP1", ylocalsize},
        {"MIO_BN_GRP2", zlocalsize},
        {"MIO_BN_GFX110X", (miopen::StartsWith(handle.GetDeviceName(), "gfx110") ? "1" : "0")},
        {"MIO_BN_GFX103X", (miopen::StartsWith(handle.GetDeviceName(), "gfx103") ? "1" : "0")},
    };

    std::string kernel_file = "MIOpenBatchNormFwdInfer";
    std::string kernel_name = "MIOpenBatchNormFwdInfer";

    std::string params = use_hip ? build_params.GenerateFor(miopen::kbp::OpenCL{})
                                 : build_params.GenerateFor(miopen::kbp::HIP{});

    if(bn_mode == miopenBNSpatial)
    {
        if(use_hip)
        {

            kernel_file += "SpatialHIP.cpp";
            kernel_name += "SpatialEstHIP";
        }
        else
        {

            kernel_file += "Spatial.cl";
            kernel_name += "SpatialEst";
        }
    }
    else
    {

        if(use_hip)
        {

            kernel_file += "PerActHIP.cpp";
            kernel_name += "PerActivationEstHIP";
        }
        else
        {

            kernel_file += "PerAct.cl";
            kernel_name += "PerActivationEst";
        }
    }

    const std::vector<size_t> vgd{xgridsize, ygridsize, zgridsize};
    const std::vector<size_t> vld{xlocalsize, ylocalsize, zlocalsize};

    std::ostringstream ss;

    ss << (use_hip ? "hip" : "ocl");
    ss << "fp16" << static_cast<int>(xDesc.GetType() == miopenHalf);
    ss << "fp32" << static_cast<int>(xDesc.GetType() == miopenFloat);
    ss << "fp64" << static_cast<int>(xDesc.GetType() == miopenDouble);
    ss << "fbf16" << static_cast<int>(xDesc.GetType() == miopenBFloat16);
    ss << "mode" << bn_mode;
    ss << "HWdims" << in_cstride;
    ss << "C" << c;
    ss << "layout" << xDesc.GetLayout_str();

    std::string network_config = ss.str();

    handle.AddKernel(kernel_name, network_config, kernel_file, kernel_name, vld, vgd, params)(
        x,
        y,
        estimatedMean,
        estimatedVariance,
        bnScale,
        bnBias,
        epsilon,
        n,
        in_cstride,
        in_nstride);
}

template <typename T>
void BNTensorCompare(const tensor<T>& output,
                     const tensor<T>& ref_out,
                     const double threshold = std::numeric_limits<T>::epsilon())
{
    EXPECT_FALSE(miopen::range_zero(ref_out)) << "OCL data is all zeros";
    EXPECT_FALSE(miopen::range_zero(output)) << "HIP data is all zeros";
    EXPECT_FALSE(miopen::find_idx(output, miopen::not_finite) >= 0)
        << "Non finite number found in the HIP data";
    EXPECT_TRUE(miopen::range_distance(ref_out) == miopen::range_distance(output));
    auto error = miopen::rms_range(ref_out, output);
    EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
        << "Non finite number found in the OCL data";

    EXPECT_TRUE(error < threshold)
        << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
}

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
struct BatchNormFwdInferTest
    : public ::testing::TestWithParam<std::tuple<BNTestCase, miopenTensorLayout_t>>
{
protected:
    void SetUp() override
    {
        std::tie(bn_config, tensor_layout) = GetParam();
        bn_infer_test_data.SetUpImpl(bn_config, tensor_layout);

    }

    void RunTestGPU(bool hip_en = false)
    {

        auto&& handle = get_handle();

        BatchNormForwardInferencGPU(handle,
                                    bn_config.mode,
                                    &bn_infer_test_data.alpha,
                                    &bn_infer_test_data.beta,
                                    bn_infer_test_data.input.desc,
                                    bn_infer_test_data.in_dev.get(),
                                    bn_infer_test_data.output.desc,
                                    bn_infer_test_data.out_dev.get(),
                                    bn_infer_test_data.scale.desc,
                                    bn_infer_test_data.scale_dev.get(),
                                    bn_infer_test_data.shift_dev.get(),
                                    bn_infer_test_data.estMean_dev.get(),
                                    bn_infer_test_data.estVariance_dev.get(),
                                    bn_infer_test_data.epsilon,
                                    hip_en);

        auto& output_ref = hip_en ? bn_infer_test_data.output.data : bn_infer_test_data.ref_out.data;
        
        std::fill(output_ref.begin(),
                  output_ref.end(),
                  std::numeric_limits<YDataType>::quiet_NaN());


        output_ref       = handle.Read<YDataType>(bn_infer_test_data.out_dev,
                                            bn_infer_test_data.output.data.size());
    }

    void ComputeCPU() { test::ComputeCPUBNInference(bn_infer_test_data); }

    void Verify() { BNTensorCompare<YDataType>(bn_infer_test_data.output, bn_infer_test_data.ref_out, 4e-3); }


    BNTestCase bn_config;
    bool test_skipped = false;
    BNInferTestData<XDataType, YDataType, ScaleDataType, BiasDataType, MeanVarDataType, BNTestCase>
        bn_infer_test_data;

    miopenTensorLayout_t tensor_layout;
};

namespace BatchNormFwdInfer {

struct BatchNormFwdInferTestFloat : BatchNormFwdInferTest<float, float, float, float, float>
{
};

} // namespace BatchNormFwdInfer
using namespace BatchNormFwdInfer;

TEST_P(BatchNormFwdInferTestFloat, BatchNormFwdInferTestFw)
{
    // Run the OpenCL reference
    RunTestGPU(false);

    // Optionally use the CPU output as reference 
    // ComputeCPU();

    // Run the HIP kernel
    RunTestGPU(true);

    // Compare the outputs.
    Verify();
};

INSTANTIATE_TEST_SUITE_P(BatchNormFwdInferTestSet,
                         BatchNormFwdInferTestFloat,
                         testing::Combine(testing::ValuesIn(Network1<BNTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW})));
