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

        // TODO: use miopen API?
        miopenCreateActivationDescriptor(&activ_desc);
        miopenSetActivationDescriptor(activ_desc, activ_config.activ_mode, alpha, beta, gamma);

        // In TEST_P()
        // auto ptr_bwdfusionplan                = GetManagedFusionPlanDesc(&input.desc);
        // miopenCreateOpActivationForward(ptr_fwdfusionplan.get(), &activFwdOp, activ_mode);

        std::size_t n, c, h, w;
        std::tie(n, c, h, w) = miopen::tien<4>(input.desc.GetLengths());
        size_t total_mem     = 4 * input.desc.GetNumBytes(); // estimate based on backward pass
        volatile size_t device_mem = get_handle().GetGlobalMemorySize();

        ASSERT_GE(total_mem, device_mem) << "Tensor exceeds GPU memory size";

        gpu_output =
            tensor<float>{static_cast<size_t>(n), // n from miopenGetConvolutionForwardOutputDim ?
                          static_cast<size_t>(c),
                          static_cast<size_t>(h),
                          static_cast<size_t>(w)};
        cpu_ref_out =
            tensor<float>{static_cast<size_t>(n), // n from miopenGetConvolutionForwardOutputDim ?
                          static_cast<size_t>(c),
                          static_cast<size_t>(h),
                          static_cast<size_t>(w)};

        std::fill(gpu_output.begin(), gpu_output.end(), 0.0f);
        std::fill(cpu_ref_out.begin(), cpu_ref_out.end(), 0.0f);

        // Infer on CPU, forward
        activationHostInfer(activ_config.activ_mode,
                            gamma,      // 0.0f?
                            beta,       // 0.0f?
                            alpha,      // 0.0f?
                            input.data, // TODO: cpu_ref_out.data?
                            cpu_ref_out.data);

        // Infer on CPU, backward
        // activationHostBwd(...)

        auto&& handle = get_handle();
        in_dev        = handle.Write(input.data);
        out_dev       = handle.Write(gpu_output.data);
    }

    void TearDown() override
    {
        auto&& handle = get_handle();
        // Read data fro GPU
        gpu_output.data = handle.Read<float>(out_dev, gpu_output.data.size());
        EXPECT_FALSE(miopen::range_zero(cpu_ref_out)) << "CPU data is all zeros";
        EXPECT_FALSE(miopen::range_zero(gpu_output)) << "GPU data is all zeros";
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
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr out_dev;
};

miopenStatus_t RunActivation(miopen::Handle& handle,
                             miopenActivationDescriptor_t activationDesc,
                             const void* alpha,
                             const miopen::TensorDescriptor& xDesc,
                             ConstData_t x,
                             const void* beta,
                             const miopen::TensorDescriptor& yDesc,
                             Data_t y)
{
    // ASSERT_TRUE(alpha);
    // ASSERT_TRUE(beta != nullptr);

    if(alpha == nullptr || beta == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "alpha or beta is NULL");

    /*
    miopen::OperatorArgs fwdActivArgs;
    auto activOp = std::make_shared<miopen::ActivFwdFusionOpDescriptor>(activationDesc.GetMode());
    */

    // float alpha       = static_cast<float>(1.0);
    // float beta        = static_cast<float>(0);
    // float activ_alpha = activationDesc.GetAlpha();
    // float activ_beta  = activationDesc.GetBeta();
    // float activ_gamma = activationDesc.GetGamma();

    miopenStatus_t status =
        miopen::deref(activationDesc).Forward(handle, alpha, xDesc, x, beta, yDesc, y);

    /*
        // Set the Args
        miopenSetOpArgsActivForward(miopenOperatorArgs_t args,
                                const miopenFusionOpDescriptor_t activFwdOp,
                                const void* alpha,
                                const void* beta,
                                double activAlpha,
                                double activBeta,
                                double activGamma);


        MIOPEN_CHECK(activOp->SetArgs(fwdActivArgs, &alpha, &beta, activ_alpha, activ_beta,
       activ_gamma));
        */
    // TODO: Execute?
    return status;
}

INSTANTIATE_TEST_SUITE_P(ActivationTestSuite,
                         TestActivation,
                         ::testing::Values(ActivationConfig{16, 32, 8, 8, miopenActivationELU}));

TEST_P(TestActivation, ActivationFwdTest)
{
    const float alpha = 1.0f, beta = 0;
    miopenStatus_t status = RunActivation(get_handle(),
                                          activ_desc,
                                          &alpha,
                                          input.desc,
                                          in_dev.get(),
                                          &beta,
                                          gpu_output.desc,
                                          out_dev.get());

    EXPECT_EQ(status, miopenStatusSuccess);
}
