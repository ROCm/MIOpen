/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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
#include "verify.hpp"
#include "gtest/gtest.h"
#include <half/half.hpp>

struct ActivationConfig
{
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    friend std::ostream& operator<<(std::ostream& outStream, const ActivationConfig& activConfig)
    {
        return outStream << "N:" << activConfig.N << " C:" << activConfig.C
                         << " H:" << activConfig.H << " W:" << activConfig.W;
    }
};

template <typename T>
struct GPU_TestActivationBase
    : public ::testing::TestWithParam<std::tuple<miopenActivationMode_t, ActivationConfig>>
{
protected:
    void SetUp() override
    {
        std::tie(activ_mode, activ_config) = GetParam();

        double alpha = 0.95;
        double beta  = 2.3;
        double gamma = 3.4;

        input = tensor<T>{activ_config.N, activ_config.C, activ_config.H, activ_config.W};

        input.generate(tensor_elem_gen_integer{17});
        dinput_cpu = input;
        dinput_gpu = input;

        miopenCreateActivationDescriptor(&activ_desc);
        miopenSetActivationDescriptor(activ_desc, activ_mode, alpha, beta, gamma);

        std::size_t n, c, h, w;
        auto&& handle        = get_handle();
        std::tie(n, c, h, w) = miopen::tien<4>(input.desc.GetLengths());
        size_t total_mem =
            4 * input.desc.GetNumBytes(); // estimate based on both forward and backward passes
        size_t device_mem = handle.GetGlobalMemorySize();

        ASSERT_LT(total_mem, device_mem) << "Tensor exceeds system memory size";

        output_gpu =
            tensor<T>{static_cast<size_t>(n), // n from miopenGetConvolutionForwardOutputDim ?
                      static_cast<size_t>(c),
                      static_cast<size_t>(h),
                      static_cast<size_t>(w)};
        output_cpu_ref =
            tensor<T>{static_cast<size_t>(n), // n from miopenGetConvolutionForwardOutputDim ?
                      static_cast<size_t>(c),
                      static_cast<size_t>(h),
                      static_cast<size_t>(w)};

        std::fill(output_gpu.begin(), output_gpu.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(
            output_cpu_ref.begin(), output_cpu_ref.end(), std::numeric_limits<T>::quiet_NaN());

        // Infer on CPU, forward
        activationHostInfer(activ_mode,
                            gamma,                // 0.0f?
                            beta,                 // 0.0f?
                            alpha,                // 0.0f?
                            input.data,           // Input
                            output_cpu_ref.data); // Output

        // Infer on CPU, backward
        doutput = output_cpu_ref;
        doutput.generate([&](int n1, int c1, int h1, int w1) {
            float x = output_cpu_ref(n1, c1, h1, w1);
            double y =
                (877 * n1 + 547 * c1 + 701 * h1 + 1049 * w1 + static_cast<int>(769 * x)) % 2503;
            return ((x * y) / 1301.0);
        });

        activationHostBwd(activ_mode,
                          gamma,
                          beta,
                          alpha,
                          doutput.data,        // dy
                          input.data,          // x
                          output_cpu_ref.data, // y
                          dinput_cpu.data);    // dx

        in_dev   = handle.Write(input.data);
        out_dev  = handle.Write(output_gpu.data);
        din_dev  = handle.Write(dinput_gpu.data);
        dout_dev = handle.Write(doutput.data);
    }

    void TearDown() override
    {
        auto&& handle = get_handle();
        // Read data from GPU
        output_gpu.data = handle.Read<T>(out_dev, output_gpu.data.size());

        CompareTensors(output_cpu_ref, output_gpu);

        if(isBwdActivation)
        {
            dinput_gpu.data = handle.Read<T>(din_dev, dinput_gpu.data.size());

            CompareTensors(dinput_cpu, dinput_gpu);
        }
        miopenDestroyActivationDescriptor(activ_desc);
    }

    template <class T1, class T2>
    void CompareTensors(T1&& t1, T2&& t2)
    {
        double tolerance = 80;
        EXPECT_FALSE(miopen::range_zero(t1)) << "CPU data is all zeros. Config: " << activ_config;
        EXPECT_FALSE(miopen::range_zero(t2)) << "GPU data is all zeros. Config: " << activ_config;
        EXPECT_EQ(miopen::range_distance(t1), miopen::range_distance(t2))
            << "range distance b/w CPU and GPU not equal. Config:" << activ_config;

        using value_type = miopen::range_value<decltype(t2)>;
        double threshold = std::numeric_limits<value_type>::epsilon() * tolerance;
        std::vector<double> error{miopen::rms_range(t1, t2)};
        EXPECT_FALSE(error.front() > threshold) << "Config:" << activ_config;
        return;
    }

    bool isBwdActivation;

    tensor<T> input;          // x
    tensor<T> output_gpu;     // share
    tensor<T> output_cpu_ref; // y share

    tensor<T> dinput_cpu; // dx
    tensor<T> dinput_gpu;
    tensor<T> doutput;

    ActivationConfig activ_config;
    miopenActivationMode_t activ_mode;
    miopenActivationDescriptor_t activ_desc;
    miopen::Allocator::ManageDataPtr in_dev;   // x
    miopen::Allocator::ManageDataPtr out_dev;  // y
    miopen::Allocator::ManageDataPtr din_dev;  // dx
    miopen::Allocator::ManageDataPtr dout_dev; // dy
};

struct GPU_TestActivation_FP32 : public GPU_TestActivationBase<float>
{
};
struct GPU_TestActivation_FP16 : public GPU_TestActivationBase<half_float::half>
{
};

miopenStatus_t RunFwdActivation(const miopen::Handle& handle,
                                miopenActivationDescriptor_t activationDesc,
                                const void* alpha,
                                const miopen::TensorDescriptor& xDesc,
                                ConstData_t x,
                                const void* beta,
                                const miopen::TensorDescriptor& yDesc,
                                Data_t y)
{
    if(alpha == nullptr || beta == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "alpha or beta is NULL");

    miopen::ActivationDescriptor desc = miopen::deref(activationDesc);
    miopenStatus_t status             = desc.Forward(handle,
                                         alpha,
                                         xDesc, // input.desc
                                         x,     // in_dev.get()
                                         beta,
                                         yDesc, // output_gpu.desc
                                         y);    // out_dev.get()

    return status;
}

miopenStatus_t RunFwdBwdActivation(const miopen::Handle& handle,
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

    miopen::ActivationDescriptor desc = miopen::deref(activationDesc);
    miopenStatus_t fwdStatus          = desc.Forward(handle,
                                            alpha,
                                            xDesc, // input.desc
                                            x,     // in_dev.get()
                                            beta,
                                            yDesc, // output_gpu.desc
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

    if(fwdStatus == miopenStatusSuccess && bwdStatus == miopenStatusSuccess)
        return miopenStatusSuccess;
    else if(fwdStatus != miopenStatusSuccess)
        return fwdStatus;
    else
        return bwdStatus;
}

TEST_P(GPU_TestActivation_FP32, ActivationFwdTest)
{
    const float alpha = 1.0f, beta = 0;
    isBwdActivation = false;

    miopenStatus_t status = RunFwdActivation(get_handle(),
                                             activ_desc,
                                             &alpha,
                                             input.desc, // x
                                             in_dev.get(),
                                             &beta,
                                             output_gpu.desc, // y
                                             out_dev.get());

    EXPECT_EQ(status, miopenStatusSuccess) << "Forward activation failed. Config: " << activ_config;
}

TEST_P(GPU_TestActivation_FP16, ActivationFwdTest)
{
    const float alpha = 1.0f, beta = 0;
    isBwdActivation = false;

    miopenStatus_t status = RunFwdActivation(get_handle(),
                                             activ_desc,
                                             &alpha,
                                             input.desc, // x
                                             in_dev.get(),
                                             &beta,
                                             output_gpu.desc, // y
                                             out_dev.get());

    EXPECT_EQ(status, miopenStatusSuccess) << "Forward activation failed. Config: " << activ_config;
}

TEST_P(GPU_TestActivation_FP32, ActivationBwdTest)
{
    const float alpha = 1.0f, beta = 0;
    isBwdActivation = true;

    miopenStatus_t status = RunFwdBwdActivation(get_handle(),
                                                activ_desc,
                                                &alpha,
                                                input.desc, // x
                                                in_dev.get(),
                                                &beta,
                                                output_gpu.desc, // y
                                                out_dev.get(),
                                                doutput.desc, // dy
                                                dout_dev.get(),
                                                dinput_gpu.desc, // dx
                                                din_dev.get());

    EXPECT_EQ(status, miopenStatusSuccess)
        << "Backward activation failed. Config: " << activ_config;
}

TEST_P(GPU_TestActivation_FP16, ActivationBwdTest)
{
    const float alpha = 1.0f, beta = 0;
    isBwdActivation = true;

    miopenStatus_t status = RunFwdBwdActivation(get_handle(),
                                                activ_desc,
                                                &alpha,
                                                input.desc, // x
                                                in_dev.get(),
                                                &beta,
                                                output_gpu.desc, // y
                                                out_dev.get(),
                                                doutput.desc, // dy
                                                dout_dev.get(),
                                                dinput_gpu.desc, // dx
                                                din_dev.get());

    EXPECT_EQ(status, miopenStatusSuccess)
        << "Backward activation failed. Config: " << activ_config;
}

std::string GetActivationModeName(miopenActivationMode_t mode)
{
    switch(mode)
    {
    case miopenActivationPASTHRU: return "PASSTHRU";
    case miopenActivationLOGISTIC: return "LOGISTIC";
    case miopenActivationTANH: return "TANH";
    case miopenActivationRELU: return "RELU";
    case miopenActivationSOFTRELU: return "SOFTRELU";
    case miopenActivationABS: return "ABS";
    case miopenActivationPOWER: return "POWER";
    case miopenActivationLEAKYRELU: return "LEAKYRELU";
    case miopenActivationELU: return "ELU";
    default: return "UNKNOWN";
    }
}

struct PrintToStringParamName
{
    std::string operator()(
        const testing::TestParamInfo<std::tuple<miopenActivationMode_t, ActivationConfig>>& info)
        const
    {
        auto activ_mode      = std::get<0>(info.param);
        std::string mode_str = GetActivationModeName(activ_mode);

        return std::to_string(info.index) + "_" + mode_str;
    }
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_TestActivation_FP32,
                         ::testing::Combine(::testing::Values(miopenActivationPASTHRU,
                                                              miopenActivationLOGISTIC,
                                                              miopenActivationTANH,
                                                              miopenActivationRELU,
                                                              miopenActivationSOFTRELU,
                                                              miopenActivationABS,
                                                              miopenActivationPOWER,
                                                              // miopenActivationCLIPPEDRELU,
                                                              miopenActivationLEAKYRELU,
                                                              miopenActivationELU),

                                            ::testing::Values(ActivationConfig{128, 128, 16, 16},
                                                              ActivationConfig{128, 16, 16, 16},
                                                              ActivationConfig{16, 128, 16, 16},
                                                              ActivationConfig{16, 32, 8, 8},
                                                              ActivationConfig{32, 16, 8, 8},
                                                              ActivationConfig{2, 16, 5, 4},
                                                              ActivationConfig{2, 2, 2, 2})),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_TestActivation_FP16,
                         ::testing::Combine(::testing::Values(miopenActivationPASTHRU,
                                                              miopenActivationLOGISTIC,
                                                              miopenActivationTANH,
                                                              miopenActivationRELU,
                                                              miopenActivationSOFTRELU,
                                                              miopenActivationABS,
                                                              miopenActivationPOWER,
                                                              // miopenActivationCLIPPEDRELU,
                                                              miopenActivationLEAKYRELU,
                                                              miopenActivationELU),

                                            ::testing::Values(ActivationConfig{128, 128, 16, 16},
                                                              ActivationConfig{128, 16, 16, 16},
                                                              ActivationConfig{16, 128, 16, 16},
                                                              ActivationConfig{16, 32, 8, 8},
                                                              ActivationConfig{32, 16, 8, 8},
                                                              ActivationConfig{2, 16, 5, 4},
                                                              ActivationConfig{2, 2, 2, 2})),
                         PrintToStringParamName());
