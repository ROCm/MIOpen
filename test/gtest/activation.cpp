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
#include "verify.hpp"
#include "gtest/gtest.h"

struct ActivationConfig
{
    size_t N;
    size_t C;
    size_t H;
    size_t W;
};

struct ActivationLOGISTIC
{
    inline static constexpr miopenActivationMode_t activMode = miopenActivationLOGISTIC;
};
struct ActivationRELU
{
    inline static constexpr miopenActivationMode_t activMode = miopenActivationRELU;
};

struct Config1
{
    inline static constexpr ActivationConfig config{128, 128, 16, 16};
};
struct Config2
{
    inline static constexpr ActivationConfig config{128, 16, 16, 16};
};

template <class T, class M, class C> // Type, Mode, Config
struct TestCase
{
    using type = T; // half_float::half int8_t  float bfloat16 double

    static constexpr miopenActivationMode_t getMode() { return M::activMode; }

    static constexpr ActivationConfig getConfig() { return C::config; }
};

using TestTypes = ::testing::Types<TestCase<float, ActivationLOGISTIC, Config1>,
                                   TestCase<float, ActivationLOGISTIC, Config2>,
                                   TestCase<float, ActivationRELU, Config1>,
                                   TestCase<float, ActivationRELU, Config2>,
                                   TestCase<double, ActivationLOGISTIC, Config1>,
                                   TestCase<double, ActivationLOGISTIC, Config2>,
                                   TestCase<double, ActivationRELU, Config1>,
                                   TestCase<double, ActivationRELU, Config2>>;

template <class T1, class T2>
void CompareTensors(T1&& t1, T2&& t2)
{
    double tolerance = 80;
    EXPECT_FALSE(miopen::range_zero(t1)) << "CPU data is all zeros";
    EXPECT_FALSE(miopen::range_zero(t2)) << "GPU data is all zeros";
    EXPECT_EQ(miopen::range_distance(t1), miopen::range_distance(t2))
        << "range distance b/w CPU and GPU not equal";

    using value_type = miopen::range_value<decltype(t2)>;
    double threshold = std::numeric_limits<value_type>::epsilon() * tolerance;
    std::vector<double> error{miopen::rms_range(t1, t2)};
    EXPECT_FALSE(error.front() > threshold);
    return;
}

template <class T>
struct TestActivation : public ::testing::Test
{
protected:
    using tensorType = typename T::type;

    void SetUp() override
    {
        double alpha = 0.95;
        double beta  = 2.3;
        double gamma = 3.4;
        activ_mode   = T::getMode();
        activ_config = T::getConfig();
        input = tensor<tensorType>{activ_config.N, activ_config.C, activ_config.H, activ_config.W};
        input.generate(tensor_elem_gen_integer{17});
        dinput_cpu = input;
        dinput_gpu = input;

        miopenCreateActivationDescriptor(&activ_desc);
        miopenSetActivationDescriptor(activ_desc, activ_mode, alpha, beta, gamma);

        std::size_t n, c, h, w;
        auto&& handle        = get_handle();
        std::tie(n, c, h, w) = miopen::tien<4>(input.desc.GetLengths());
        EXPECT_EQ(n, activ_config.N);
        EXPECT_EQ(c, activ_config.C);
        EXPECT_EQ(h, activ_config.H);
        EXPECT_EQ(w, activ_config.W);

        size_t total_mem =
            4 * input.desc.GetNumBytes(); // estimate based on both forward and backward passes
        size_t device_mem = handle.GetGlobalMemorySize();

        ASSERT_LT(total_mem, device_mem) << "Tensor exceeds system memory size";

        output_gpu = tensor<tensorType>{
            static_cast<size_t>(n), // n from miopenGetConvolutionForwardOutputDim ?
            static_cast<size_t>(c),
            static_cast<size_t>(h),
            static_cast<size_t>(w)};
        output_cpu_ref = tensor<tensorType>{
            static_cast<size_t>(n), // n from miopenGetConvolutionForwardOutputDim ?
            static_cast<size_t>(c),
            static_cast<size_t>(h),
            static_cast<size_t>(w)};

        std::fill(output_gpu.begin(), output_gpu.end(), NULL);
        std::fill(output_cpu_ref.begin(), output_cpu_ref.end(), NULL);

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
        output_gpu.data = handle.Read<tensorType>(out_dev, output_gpu.data.size());

        CompareTensors(output_cpu_ref, output_gpu);

        dinput_gpu.data = handle.Read<tensorType>(din_dev, dinput_gpu.data.size());

        CompareTensors(dinput_cpu, dinput_gpu);
        miopenDestroyActivationDescriptor(activ_desc);
    }

    tensor<tensorType> input; // x
    tensor<tensorType> output_gpu;
    tensor<tensorType> output_cpu_ref; // y

    tensor<tensorType> dinput_cpu; // dx
    tensor<tensorType> dinput_gpu;
    tensor<tensorType> doutput;

    ActivationConfig activ_config;
    miopenActivationMode_t activ_mode;
    miopenActivationDescriptor_t activ_desc;
    miopen::Allocator::ManageDataPtr in_dev;   // x
    miopen::Allocator::ManageDataPtr out_dev;  // y
    miopen::Allocator::ManageDataPtr din_dev;  // dx
    miopen::Allocator::ManageDataPtr dout_dev; // dy
};

miopenStatus_t RunFwdActivation(miopen::Handle& handle,
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

miopenStatus_t RunBwdActivation(miopen::Handle& handle,
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

    miopenStatus_t status = desc.Backward(handle,
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

    return status;
}

TYPED_TEST_SUITE_P(TestActivation);

TYPED_TEST_P(TestActivation, ActivationFwdTest)
{
    const float alpha = 1.0f, beta = 0;
    miopenStatus_t status = RunFwdActivation(get_handle(),
                                             this->activ_desc,
                                             &alpha,
                                             this->input.desc, // x
                                             this->in_dev.get(),
                                             &beta,
                                             this->output_gpu.desc, // y
                                             this->out_dev.get());

    EXPECT_EQ(status, miopenStatusSuccess);
}

TYPED_TEST_P(TestActivation, ActivationBwdTest)
{
    const float alpha = 1.0f, beta = 0;
    miopenStatus_t status = RunBwdActivation(get_handle(),
                                             this->activ_desc,
                                             &alpha,
                                             this->input.desc, // x
                                             this->in_dev.get(),
                                             &beta,
                                             this->output_gpu.desc, // y
                                             this->out_dev.get(),
                                             this->doutput.desc, // dy
                                             this->dout_dev.get(),
                                             this->dinput_gpu.desc, // dx
                                             this->din_dev.get());

    EXPECT_EQ(status, miopenStatusSuccess);
}

REGISTER_TYPED_TEST_SUITE_P(TestActivation, ActivationFwdTest, ActivationBwdTest);

INSTANTIATE_TYPED_TEST_CASE_P(Activation, TestActivation, TestTypes);
