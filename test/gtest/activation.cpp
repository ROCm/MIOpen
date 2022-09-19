/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include "test.hpp"
#include <array>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/activ.hpp>
#include <miopen/miopen.h>
#include <miopen/stringutils.hpp>
#include <miopen/tensor.hpp>
#include <utility>

#include <fusionHost.hpp>
#include "activ_driver.hpp"
#include "InputFlags.hpp"
#include "get_handle.hpp"
//#include "tensor_holder.hpp"
#include "verify.hpp"

#include "gtest/gtest.h"

struct ActivationConfig
{
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    miopenActivationMode_t activ_mode;
};

struct TestActivation : public ::testing::TestWithParam<ActivationConfig>
{
protected:
    void SetUp() override
    {
        double alpha = 0.95;
        double beta  = 2.3;
        double gamma = 3.4;
        activ_config = GetParam();
        input = tensor<float>{activ_config.N, activ_config.C, activ_config.H, activ_config.W};
        input.generate(tensor_elem_gen_integer{17});
        miopenCreateActivationDescriptor(&activ_desc);
        // TODO: same alpha beta gamma as below?
        miopenSetActivationDescriptor(activ_desc, activ_config.activ_mode, alpha, beta, gamma);

        gpu_output = tensor<float>{
            static_cast<size_t>(activ_config.N), // n from miopenGetConvolutionForwardOutputDim ?
            static_cast<size_t>(activ_config.C),
            static_cast<size_t>(activ_config.H),
            static_cast<size_t>(activ_config.W)};
        cpu_ref_out = tensor<float>{
            static_cast<size_t>(activ_config.N), // n from miopenGetConvolutionForwardOutputDim ?
            static_cast<size_t>(activ_config.C),
            static_cast<size_t>(activ_config.H),
            static_cast<size_t>(activ_config.W)};

        std::fill(gpu_output.begin(), gpu_output.end(), 0.0f);
        std::fill(cpu_ref_out.begin(), cpu_ref_out.end(), 0.0f);

        activationHostInfer(activ_config.activ_mode,
                            gamma,      // 0.0f?
                            beta,       // 0.0f?
                            alpha,      // 0.0f?
                            input.data, // TODO: cpu_ref_out.data?
                            cpu_ref_out.data);

        auto&& handle = get_handle();
        in_ptr        = handle.Write(input.data);
        out_ptr       = handle.Write(gpu_output.data);
    }

    void TearDown() override
    {
        auto&& handle   = get_handle();
        gpu_output.data = handle.Read<float>(out_ptr, gpu_output.data.size());
        EXPECT_FALSE(miopen::range_zero(cpu_ref_out)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(gpu_output)) << "Gpu data is all zeros";
        const auto maxDiff = miopen::max_diff(cpu_ref_out, gpu_output);
        std::ignore        = maxDiff;
        auto idx           = miopen::mismatch_idx(cpu_ref_out, gpu_output, miopen::float_equal);
        EXPECT_FALSE(miopen::find_idx(cpu_ref_out, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";
        EXPECT_FALSE(idx < miopen::range_distance(cpu_ref_out));
        miopenDestroyActivationDescriptor(activ_desc);
    }

    tensor<float> input;
    tensor<float> gpu_output;
    tensor<float> cpu_ref_out;
    ActivationConfig activ_config;
    miopenActivationDescriptor_t activ_desc;
    miopen::Allocator::ManageDataPtr in_ptr;
    miopen::Allocator::ManageDataPtr out_ptr;
};

#define MIOPEN_CHECK(x)          \
    if(x != miopenStatusSuccess) \
        return x;

miopenStatus_t RunActivation(miopenHandle_t handle,
                             const float* alpha1,
                             miopenTensorDescriptor_t xDesc,
                             ConstData_t x,

                             const miopenTensorDescriptor_t zDesc,
                             ConstData_t z,

                             const miopen::ActivationDescriptor& activationDesc,
                             const miopenTensorDescriptor_t yDesc,
                             Data_t y)
{
    if(alpha1 != nullptr)
    {
        const auto falpha1 = *(static_cast<const float*>(alpha1));
        if(falpha1 != 1.0f)
            MIOPEN_THROW(miopenStatusNotImplemented, "alpha1 can only be 1.0");
    }
    // if(z != nullptr || zDesc.GetSize() != 0)
    //    MIOPEN_THROW(miopenStatusNotImplemented, "The addition of z vector is not yet supported");

    miopen::OperatorArgs fusionArgs;

    auto activOp = std::make_shared<miopen::ActivFwdFusionOpDescriptor>(activationDesc.GetMode());

    float alpha       = static_cast<float>(1.0);
    float beta        = static_cast<float>(0);
    float activ_alpha = activationDesc.GetAlpha();
    float activ_beta  = activationDesc.GetBeta();
    float activ_gamma = activationDesc.GetGamma();

    // Set the Args
    MIOPEN_CHECK(activOp->SetArgs(fusionArgs, &alpha, &beta, activ_alpha, activ_beta, activ_gamma));
    // TODO: Execute?
    return miopenStatusSuccess;
}

INSTANTIATE_TEST_SUITE_P(ActivationTestSuite,
                         TestActivation,
                         ::testing::Values(ActivationConfig{16, 32, 8, 8, miopenActivationELU}));

TEST_P(TestActivation, ActivationFwdTest)
{
    tensor<float> z{};
    const float alpha     = 1.0f;
    miopenStatus_t status = miopenStatusUnsupportedOp;
    status                = RunActivation(&get_handle(),
                           &alpha,
                           &input.desc,
                           in_ptr.get(),
                           &z.desc,
                           nullptr,
                           miopen::deref(activ_desc),
                           &gpu_output.desc,
                           static_cast<Data_t>(out_ptr.get()));
    EXPECT_EQ(status, miopenStatusSuccess);
}
