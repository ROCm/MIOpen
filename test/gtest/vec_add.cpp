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

template <class T>
void cpu_vec_add(const tensor<T>& srcA, const tensor<T>& srcB, tensor<T>& dstC, size_t vec_size)
{

    for(size_t i = 0; i < vec_size; i++)
    {
        dstC[i] = srcA[i] + srcB[i];
    }
}

struct VecAddTestCase
{
    size_t vec_size;
    size_t threads_per_block;
};

std::vector<VecAddTestCase> VecAddTestConfigs()
{ // vector_size, threads_per_block
    // clang-format off
    return {{256, 256},
            {256, 32},
            {512, 256},
            {512, 32},
            {1024, 256},
            {1024, 32},
            {2048, 64},
            {32768, 128}};
    // clang-format on
}

template <typename T = float>
struct VecAddTest : public ::testing::TestWithParam<VecAddTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        vecadd_config  = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        // Allocate and initialize input tensors
        inputA = tensor<T>{vecadd_config.vec_size}.generate(gen_value);
        inputB = tensor<T>{vecadd_config.vec_size}.generate(gen_value);

        // Allocate output tensors
        outputC_ocl = tensor<T>{vecadd_config.vec_size};
        outputC_hip = tensor<T>{vecadd_config.vec_size};
        ref_outputC = tensor<T>{vecadd_config.vec_size};

        // Write the device input tensors
        inputA_dev = handle.Write(inputA.data);
        inputB_dev = handle.Write(inputB.data);

        // Clear the reference output tensor
        std::fill(ref_outputC.begin(), ref_outputC.end(), std::numeric_limits<T>::quiet_NaN());

        // Run the CPU implementation
        cpu_vec_add(inputA, inputB, ref_outputC, vecadd_config.vec_size);
    }

    void RunTestOCL()
    {
        auto&& handle = get_handle();

        // Clear the output tensor
        std::fill(outputC_ocl.begin(), outputC_ocl.end(), std::numeric_limits<T>::quiet_NaN());
        outputC_dev = handle.Write(outputC_ocl.data);

        // Setup the handle for OpenCL
        std::string program_name = "MIOpenVecAddOCL.cl";
        std::string kernel_name  = "vector_add_ocl";

        std::string network_config = "standalone_kernel_vector_add_ocl";

        miopen::KernelBuildParameters options{};

        std::string params = options.GenerateFor(miopen::kbp::OpenCL{});

        size_t totalElements   = vecadd_config.vec_size;
        size_t threadsPerBlock = vecadd_config.threads_per_block;
        size_t blocksPerGrid   = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

        const std::vector<size_t> vgd{blocksPerGrid * threadsPerBlock, 1, 1};
        const std::vector<size_t> vld{threadsPerBlock, 1, 1};

        handle.AddKernel(
            "vector_add_ocl", network_config, program_name, kernel_name, vld, vgd, params)(
            inputA_dev.get(),
            inputB_dev.get(),
            outputC_dev.get(),
            static_cast<unsigned long>(
                totalElements)); // OpenCL expects the totalElements as unsigned long

        // Read the device output tensor
        outputC_ocl.data = handle.Read<T>(outputC_dev, outputC_ocl.data.size());
    }

    void RunTestHIP()
    {
        auto&& handle = get_handle();

        // Clear the output tensor
        std::fill(outputC_hip.begin(), outputC_hip.end(), std::numeric_limits<T>::quiet_NaN());
        outputC_dev = handle.Write(outputC_hip.data);

        // Setup the handle for HIP
        std::string program_name = "MIOpenVecAdd.cpp";
        std::string kernel_name  = "vector_add_hip";

        std::string network_config = "standalone_kernel_vector_add_hip";

        miopen::KernelBuildParameters options{};

        std::string params = options.GenerateFor(miopen::kbp::HIP{});

        size_t totalElements   = vecadd_config.vec_size;
        size_t threadsPerBlock = vecadd_config.threads_per_block;
        size_t blocksPerGrid   = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

        const std::vector<size_t> vgd{blocksPerGrid * threadsPerBlock, 1, 1};
        const std::vector<size_t> vld{threadsPerBlock, 1, 1};

        handle.AddKernel(
            "vector_add_hip", network_config, program_name, kernel_name, vld, vgd, params)(
            inputA_dev.get(), inputB_dev.get(), outputC_dev.get(), totalElements);

        // Read the device output tensor
        outputC_hip.data = handle.Read<T>(outputC_dev, outputC_hip.data.size());
    }

    void VerifyOCL()
    {
        auto error = miopen::rms_range(ref_outputC, outputC_ocl);
        EXPECT_TRUE(miopen::range_distance(ref_outputC) == miopen::range_distance(outputC_ocl));
        EXPECT_TRUE(error == 0) << "Outputs do not match each other. Error:" << error;
    }

    void VerifyHIP()
    {
        auto error = miopen::rms_range(ref_outputC, outputC_hip);
        EXPECT_TRUE(miopen::range_distance(ref_outputC) == miopen::range_distance(outputC_hip));
        EXPECT_TRUE(error == 0) << "Outputs do not match each other. Error:" << error;
    }

    void VerifyGPU()
    {
        auto error = miopen::rms_range(outputC_ocl, outputC_hip);
        EXPECT_TRUE(miopen::range_distance(ref_outputC) == miopen::range_distance(outputC_hip));
        EXPECT_TRUE(error == 0) << "GPU outputs do not match each other. Error:" << error;
    }

    VecAddTestCase vecadd_config;

    tensor<T> inputA; // input tensor A
    tensor<T> inputB; // input tensor B

    tensor<T> outputC_ocl; // Output tensorC for OpenCL
    tensor<T> outputC_hip; // Output tensorC for HIP

    tensor<T> ref_outputC; // Output tensorC for CPU

    miopen::Allocator::ManageDataPtr inputA_dev; // input tensor A device
    miopen::Allocator::ManageDataPtr inputB_dev; // input tensor B device

    miopen::Allocator::ManageDataPtr outputC_dev; // output tensor C device
};

namespace vecadd {

struct GPU_VecAddTest_FP32 : VecAddTest<float>
{
};

} // namespace vecadd
using namespace vecadd;

TEST_P(GPU_VecAddTest_FP32, VecAddTestFw)
{
    RunTestOCL();
    // Verify OCL results against CPU reference
    VerifyOCL();

    RunTestHIP();
    // Verify HIP results against CPU reference
    VerifyHIP();

    // Verify OCL and HIP results against each other
    VerifyGPU();
};

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_VecAddTest_FP32, testing::ValuesIn(VecAddTestConfigs()));
