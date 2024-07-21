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

#include "na.hpp"

void BatchNormForwardInferencGPU(miopen::Handle& handle,
                                 miopenBatchNormMode_t bn_mode,
                                 miopenActivationMode_t activ_mode,
                                 const void* alpha,
                                 const void* beta,
                                 const float activ_alpha,
                                 const float activ_beta,
                                 const float activ_gamma,
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
                                 bool use_hip)
{
    int n, c, h, w;
    std::tie(n, c, h, w) = miopen::tien<4>(xDesc.GetLengths());

    size_t read_unit = 1;
    size_t read_len  = (bn_mode == miopenBNSpatial) ? h * w : c * h * w;

    // print the read_len, h, and w
    std::cout << "read_len: " << read_len << " chw: " << c * h * w << " w: " << w << std::endl;

    if(bn_mode == miopenBNSpatial && xDesc.GetType() != miopenHalf)
    {
        read_unit = (read_len % 4 == 0) ? 4 : (read_len % 2 == 0) ? 2 : 1;
    }
    std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(read_unit);

    size_t xgridsize = read_len / read_unit;
    size_t ygridsize = (bn_mode == miopenBNSpatial) ? size_t(c) : 1;
    size_t zgridsize = 1;

    size_t xlocalsize = 256;
    size_t ylocalsize = 1;
    size_t zlocalsize = 1;

    const std::vector<size_t> vgd{xgridsize, ygridsize, zgridsize};
    const std::vector<size_t> vld{xlocalsize, ylocalsize, zlocalsize};

    const auto build_params = miopen::KernelBuildParameters{
        {"MIO_BN_CHW", static_cast<unsigned>(c * h * w)},
        {"MIO_BN_HW", static_cast<unsigned>(h * w)},
        {"MIO_BN_N", static_cast<unsigned>(n)},
        {"MIO_BN_GRP0", xlocalsize},
        {"MIO_BN_GRP1", ylocalsize},
        {"MIO_BN_GRP2", zlocalsize},
        {"MIOPEN_READ_UNIT", static_cast<int>(read_unit)},
        {"MIOPEN_READ_TYPE", READ_TYPE},
        {"MIOPEN_YES_ACTIV", static_cast<int>(1)},
        {"MIOPEN_NRN_OP_ID", static_cast<int>(activ_mode)},
        {"MIOPEN_USE_FP16", static_cast<int>(xDesc.GetType() == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(xDesc.GetType() == miopenFloat)}
    };

    std::string kernel_file = (use_hip ? "MIOpenBatchNormActivInferHIP.cpp" : "MIOpenBatchNormActivInfer.cl");
    std::string kernel_name = "MIOpenBatchNormActivInfer";

    std::string params = use_hip ? build_params.GenerateFor(miopen::kbp::OpenCL{})
                                 : build_params.GenerateFor(miopen::kbp::HIP{});

    if(xDesc.GetType() == miopenHalf){ params += " -DMIOPEN_USE_FPMIX=1"; }

    params = "-g " + params;

    if(bn_mode == miopenBNSpatial)
    {
        params += " -DSPATIAL_BN";
        kernel_name += use_hip ? "SpatialEstHIP" : "SpatialEst";
    }
    else
    {
        params += " -DPERACT_BN";
        kernel_name += use_hip ? "PerActEstHIP" : "PerActEst";
    }

    std::ostringstream ss;

    ss << (use_hip ? "hip" : "ocl");
    ss << "fp16" << static_cast<int>(xDesc.GetType() == miopenHalf);
    ss << "fp32" << static_cast<int>(xDesc.GetType() == miopenFloat);
    ss << "fp64" << static_cast<int>(xDesc.GetType() == miopenDouble);
    ss << "fbf16" << static_cast<int>(xDesc.GetType() == miopenBFloat16);
    ss << "mode" << bn_mode;
    ss << "C" << c;
    ss << "layout" << xDesc.GetLayout_str();

    std::string network_config = ss.str();

    handle.AddKernel(kernel_name, network_config, kernel_file, kernel_name, vld, vgd, params)(
        activ_alpha,
        activ_beta,
        activ_gamma,
        epsilon,
        x,
        y,
        bnBias,
        bnScale,
        estimatedMean,
        estimatedVariance
        );
}

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
struct BatchNormInferFusedTest
    : public ::testing::TestWithParam<std::tuple<miopenActivationMode_t, BNTestCase>>
{
protected:
    void SetUp() override
    {
        std::tie(activ_mode, bn_config) = GetParam();
        bn_mode                         = bn_config.mode;
        input                           = tensor<XDataType>{bn_config.GetInput()};
        output        = tensor<YDataType>{bn_config.GetInput()};
        ref_out = tensor<YDataType>{bn_config.GetInput()};
        auto derivedBnDesc              = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, bn_mode);
        scale       = tensor<ScaleDataType>{derivedBnDesc.GetLengths()};
        shift       = tensor<BiasDataType>{derivedBnDesc.GetLengths()};
        estMean     = tensor<MeanVarDataType>{derivedBnDesc.GetLengths()};
        estVariance = tensor<MeanVarDataType>{derivedBnDesc.GetLengths()};

        auto gen_value = [](auto...) {
            return prng::gen_descreet_uniform_sign<XDataType>(1e-2, 100);
        };
        input.generate(gen_value);

        auto gen_scale = [](auto...) {
            return prng::gen_descreet_uniform_sign<ScaleDataType>(1e-2, 100);
        };
        scale.generate(gen_scale);
        shift.generate(gen_scale);
        estMean.generate(gen_scale);
        
        auto gen_var = [](auto...) {
            return static_cast<MeanVarDataType>(1e-2 * (prng::gen_0_to_B(100) + 1));
        };
        estVariance.generate(gen_var);
                
        auto&& handle = get_handle();
        std::fill(output.begin(), output.end(), std::numeric_limits<YDataType>::quiet_NaN());
        in_dev          = handle.Write(input.data);
        scale_dev       = handle.Write(scale.data);
        shift_dev       = handle.Write(shift.data);
        estMean_dev     = handle.Write(estMean.data);
        estVariance_dev = handle.Write(estVariance.data);
        out_dev         = handle.Write(output.data);
        out_dev2         = handle.Write(output.data);
    }

    void RunTestGPU(bool hip_en)
    {
        // // fill output with NaNs
        // if(hip_en){
        //     std::fill(output.begin(), output.end(), std::numeric_limits<YDataType>::quiet_NaN());
        //     // Write the output to the GPU
        //     out_dev = handle.Write(output.data);  
        // }
        // else{
        //     std::fill(ref_out.begin(), ref_out.end(), std::numeric_limits<YDataType>::quiet_NaN());
        //     // Write the output to the GPU
        //     out_dev = handle.Write(ref_out.data);
        // }



        auto&& handle = get_handle();
        BatchNormForwardInferencGPU(handle,
                                    bn_config.mode,
                                    activ_mode,
                                    &alpha,
                                    &beta,
                                    activ_alpha,
                                    activ_beta,
                                    activ_gamma,
                                    input.desc,
                                    in_dev.get(),
                                    output.desc,
                                    (hip_en ? out_dev.get(): out_dev2.get()),
                                    scale.desc,
                                    scale_dev.get(),
                                    shift_dev.get(),
                                    estMean_dev.get(),
                                    estVariance_dev.get(),
                                    epsilon,
                                    hip_en);

        // if use_hip is true read data into the output.data else read output into ref_out.data
        if(hip_en)
            output.data = handle.Read<YDataType>(out_dev, output.data.size());
        else
            ref_out.data = handle.Read<YDataType>(out_dev2, ref_out.data.size());

        // output.data   = handle.Read<YDataType>(out_dev, output.data.size());

    }

    void RunTestCPU()
    {
        if(bn_mode == miopenBNPerActivation)
        {
            batchNormPerActivHostInference(
                input, ref_out, scale, shift, epsilon, estMean, estVariance);
        }
        else
        {
            batchNormSpatialHostInference(
                input, ref_out, scale, shift, epsilon, estMean, estVariance);
        }
        activationHostInfer(
            activ_mode, activ_gamma, activ_beta, activ_alpha, ref_out.data, ref_out.data);
    }

    void Verify() 
    {
        EXPECT_FALSE(miopen::range_zero(ref_out)) << "CPU data is all zeros";
        EXPECT_FALSE(miopen::range_zero(output)) << "GPU data is all zeros";
        EXPECT_FALSE(miopen::find_idx(output, miopen::not_finite) >= 0)
            << "Non finite number found in the GPU data";
        // print output[3072] and ref_out[3072]
        // std::cout << "output[3072]: " << output[3072] << " ref_out[3072]: " << ref_out[3072] << std::endl;
        
        // print total tensor size
        // std::cout << "output size: " << output.data.size() << " ref_out size: " << ref_out.data.size() << std::endl;

        // print the first 3072 elements of output and ref_out
        // for(int i = 3071; i < 3200; i++){
        //     std::cout << "output[" << i << "]: " << output[i] << " ref_out[" << i << "]: " << ref_out[i] << std::endl;
        // }
        
        


        EXPECT_TRUE(miopen::range_distance(ref_out) == miopen::range_distance(output));
        const double tolerance = 80;
        double threshold       = std::numeric_limits<YDataType>::epsilon() * tolerance;
        auto error             = miopen::rms_range(ref_out, output);
        EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";
        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }

    BNTestCase bn_config;
    miopenBatchNormMode_t bn_mode;
    tensor<XDataType> input;
    tensor<YDataType> output;
    tensor<YDataType> ref_out;
    tensor<ScaleDataType> scale;
    tensor<BiasDataType> shift;
    tensor<MeanVarDataType> estMean;
    tensor<MeanVarDataType> estVariance;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopen::Allocator::ManageDataPtr out_dev2;


    miopen::Allocator::ManageDataPtr scale_dev;
    miopen::Allocator::ManageDataPtr shift_dev;
    miopen::Allocator::ManageDataPtr estMean_dev;
    miopen::Allocator::ManageDataPtr estVariance_dev;
    miopenActivationMode_t activ_mode;
    const float alpha       = static_cast<float>(1.0f);
    const float beta        = static_cast<float>(0);
    const float activ_alpha = static_cast<double>(0.5f);
    const float activ_beta  = static_cast<double>(0.5f);
    const float activ_gamma = static_cast<double>(0.5f);
    double epsilon          = 1.0e-5;
};

namespace BatchNormInferFused {

struct GPU_bn_infer_fused_spatial_FP32
    : BatchNormInferFusedTest<float, float, float, float, float>
{
};

struct GPU_bn_infer_fused_per_act_FP32
    : BatchNormInferFusedTest<float, float, float, float, float>
{
};

struct GPU_bn_infer_fused_spatial_FP16
    : BatchNormInferFusedTest<half_float::half, half_float::half, float, float, float>
{
};

struct GPU_bn_infer_fused_per_act_FP16
    : BatchNormInferFusedTest<half_float::half, half_float::half, float, float, float>
{
};

} // namespace BatchNormInferFused
using namespace BatchNormInferFused;


TEST_P(GPU_bn_infer_fused_spatial_FP32, PortTest){
    RunTestGPU(false);
    // Run the OpenCL reference
    RunTestGPU(true);
    // Optionally use the CPU output as reference
    // RunTestCPU();
    // Compare the outputs.
    Verify();
};

// TEST_P(GPU_bn_infer_fused_per_act_FP32, PortTest){
//     // Run the OpenCL reference
//     RunTestGPU(true);
//     // Optionally use the CPU output as reference
//     RunTestCPU();
//     // Compare the outputs.
//     Verify();
// };

// TEST_P(GPU_bn_infer_fused_spatial_FP16, PortTest){
//     // Run the OpenCL reference
//     RunTestGPU(false);
//     // Optionally use the CPU output as reference
//     RunTestCPU();
//     // Compare the outputs.
//     Verify();
// };

// TEST_P(GPU_bn_infer_fused_per_act_FP16, PortTest){
//     // Run the OpenCL reference
//     RunTestGPU(false);
//     // Optionally use the CPU output as reference
//     RunTestCPU();
//     // Compare the outputs.
//     Verify();
// };

// INSTANTIATE_TEST_SUITE_P(Smoke,
//                          GPU_bn_infer_fused_spatial_FP16,
//                          testing::Combine(testing::Values(miopenActivationRELU),
//                                           testing::ValuesIn(Networkna1())));
// INSTANTIATE_TEST_SUITE_P(Smoke,
//                          GPU_bn_infer_fused_per_act_FP16,
//                          testing::Combine(testing::Values(miopenActivationRELU),
//                                           testing::ValuesIn(Networkna1())));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_bn_infer_fused_spatial_FP32,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(Networkna1())));
// INSTANTIATE_TEST_SUITE_P(Smoke,
//                          GPU_bn_infer_fused_per_act_FP32,
//                          testing::Combine(testing::Values(miopenActivationRELU),
//                                           testing::ValuesIn(Networkna1())));
