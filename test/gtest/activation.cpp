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

template <class T1, class T2>
void CompareTensors(T1&& t1, T2&& t2)
{
    EXPECT_FALSE(miopen::range_zero(t1)) << "CPU data is all zeros";
    EXPECT_FALSE(miopen::range_zero(t2)) << "GPU data is all zeros";
    const auto maxDiff = miopen::max_diff(t1, t2);
    std::ignore        = maxDiff;
    auto idx           = miopen::mismatch_idx(t1, t2, miopen::float_equal);
    EXPECT_FALSE(miopen::find_idx(t1, miopen::not_finite) >= 0)
        << "Non finite number found in the CPU data";
    EXPECT_FALSE(idx < miopen::range_distance(t1));
    return;
}

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
        dinput_cpu = input;
        dinput_gpu = input;

        // TODO: use miopen API?
        miopenCreateActivationDescriptor(&activ_desc);
        miopenSetActivationDescriptor(activ_desc, activ_config.activ_mode, alpha, beta, gamma);

        // In TEST_P()
        // auto ptr_bwdfusionplan                = GetManagedFusionPlanDesc(&input.desc);
        // miopenCreateOpActivationForward(ptr_fwdfusionplan.get(), &activFwdOp, activ_mode);

        std::size_t n, c, h, w;
        auto&& handle        = get_handle();
        std::tie(n, c, h, w) = miopen::tien<4>(input.desc.GetLengths());
        size_t total_mem     = 6 * input.desc.GetNumBytes(); // estimate based on both forward and backward passes
        size_t device_mem    = handle.GetGlobalMemorySize();

        ASSERT_LT(total_mem, device_mem) << "Tensor exceeds GPU memory size";

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
                            gamma,             // 0.0f?
                            beta,              // 0.0f?
                            alpha,             // 0.0f?
                            input.data,        // Input
                            cpu_ref_out.data); // Output

        // Infer on CPU, backward
        doutput  = cpu_ref_out;
        doutput.generate([&](int n1, int c1, int h1, int w1) {
            float x      = cpu_ref_out(n1, c1, h1, w1);
            double y = (877 * n1 + 547 * c1 + 701 * h1 + 1049 * w1 + static_cast<int>(769 * x)) % 2503;
            return ((x * y) / 1301.0);
        });

        activationHostBwd(activ_config.activ_mode,
                          gamma,
                          beta,
                          alpha,
                          doutput.data,     // dy
                          input.data,       // x
                          cpu_ref_out.data, // y
                          dinput_cpu.data); // dx

        in_dev   = handle.Write(input.data);
        out_dev  = handle.Write(gpu_output.data);
        din_dev  = handle.Write(dinput_gpu.data);
        dout_dev = handle.Write(doutput.data);
    }

    void TearDown() override
    {
        auto&& handle = get_handle();
        // Read data fro GPU
        gpu_output.data = handle.Read<float>(out_dev, gpu_output.data.size());

        CompareTensors(cpu_ref_out, gpu_output);

        dinput_gpu.data = handle.Read<float>(din_dev, dinput_gpu.data.size());

        CompareTensors(dinput_cpu, dinput_gpu);
        miopenDestroyActivationDescriptor(activ_desc);
    }

    tensor<float> input; // x
    tensor<float> gpu_output;
    tensor<float> cpu_ref_out; // y

    tensor<float> dinput_cpu; // dx
    tensor<float> dinput_gpu;
    tensor<float> doutput;

    ActivationConfig activ_config;
    miopenActivationDescriptor_t activ_desc;
    miopen::Allocator::ManageDataPtr in_dev;   // x
    miopen::Allocator::ManageDataPtr out_dev;  // y
    miopen::Allocator::ManageDataPtr din_dev;  // dx
    miopen::Allocator::ManageDataPtr dout_dev; // dy
};

miopenStatus_t RunActivation(miopen::Handle& handle,
                             miopenActivationDescriptor_t activationDesc,
                             const void* alpha,
                             const miopen::TensorDescriptor& xDesc,
                             ConstData_t x,
                             const void* beta,
                             const miopen::TensorDescriptor& yDesc,
                             Data_t y,
                             const miopen::TensorDescriptor& dyDesc,
                             ConstData_t dy,
                             const miopen::TensorDescriptor& dxDesc,
                             Data_t dx)
{
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

    miopen::ActivationDescriptor desc = miopen::deref(activationDesc);
    miopenStatus_t fwdStatus          = desc.Forward(handle,
                                            alpha,
                                            xDesc, // input.desc
                                            x,     // in_dev.get()
                                            beta,
                                            yDesc, // gpu_output.desc
                                            y);    // out_dev.get()

    miopenStatus_t bwdStatus = desc.Backward(handle,
                                             alpha,
                                             yDesc, // out.desc
                                             y,     // out_dev.get()
                                             dyDesc,
                                             dy,
                                             xDesc, // input.desc
                                             x,
                                             beta,
                                             dxDesc,
                                             dx);
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
    return ((fwdStatus == miopenStatusSuccess && bwdStatus == miopenStatusSuccess)
                ? miopenStatusSuccess
                : fwdStatus);
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
                                          input.desc, // x
                                          in_dev.get(),
                                          &beta,
                                          gpu_output.desc, // y
                                          out_dev.get(),
                                          doutput.desc, // dy
                                          dout_dev.get(),
                                          dinput_gpu.desc,
                                          din_dev.get());

    EXPECT_EQ(status, miopenStatusSuccess);
}
