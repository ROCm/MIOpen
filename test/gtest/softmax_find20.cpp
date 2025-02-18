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

#include "random.hpp"
#include "test.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "../driver/mloSoftmaxHost.hpp"
#include "verify.hpp"

#include <miopen/softmax.hpp>

#include <miopen/miopen.h>

#include <miopen/solution.hpp>

#include <gtest/gtest.h>

#include <vector>

using namespace miopen;
template <typename T = float, typename Tref = double>
class SoftmaxFind20Test
{
public:
    SoftmaxFind20Test(bool forward,
                      miopenSoftmaxAlgorithm_t softmax_algo_arg,
                      miopenSoftmaxMode_t softmax_mode_arg)
        : problem(nullptr),
          softmax_algo(softmax_algo_arg),
          softmax_mode(softmax_mode_arg),
          isForward(forward)
    {
        Initialize();
    }

    void AddTensorDescriptors()
    {
        std::cerr << "Creating softmax tensor descriptors..." << std::endl;

        auto test_set_tensor_descriptor = [this](miopenTensorArgumentId_t name,
                                                 TensorDescriptor& desc) {
            EXPECT_EQUAL(miopenSetProblemTensorDescriptor(problem, name, &desc),
                         miopenStatusSuccess);
        };

        if(isForward)
        {
            test_set_tensor_descriptor(miopenTensorSoftmaxX, xTensor.desc);
            test_set_tensor_descriptor(miopenTensorSoftmaxY, yTensor.desc);
        }
        else
        {
            test_set_tensor_descriptor(miopenTensorSoftmaxY, yTensor.desc);
            test_set_tensor_descriptor(miopenTensorSoftmaxDY, dyTensor.desc);
            test_set_tensor_descriptor(miopenTensorSoftmaxDX, dxTensor.desc);
        }

        std::cerr << "Created softmax tensor descriptors." << std::endl;
    }

    std::vector<miopenSolution_t> TestFindSolutions(Handle& handle)
    {
        std::cerr << "Testing miopenFindSolutions..." << std::endl;

        auto solutions = std::vector<miopenSolution_t>{};
        std::size_t found;

        // We expect to get only 1 or 2 solutions for softmax for now. Hardcode value 16 as just big
        // enough value
        solutions.resize(16);

        EXPECT_EQUAL(miopenFindSolutions(
                         &handle, problem, nullptr, solutions.data(), &found, solutions.size()),
                     miopenStatusSuccess);
        EXPECT_TRUE(found > 0);

        solutions.resize(found);

        std::cerr << "Finished testing miopenFindSolutions." << std::endl;
        return solutions;
    }

    void TestSolutionAttributes(const std::vector<miopenSolution_t>& solutions)
    {
        std::cerr << "Testing miopenGetSolution<Attribute>..." << std::endl;

        for(const auto& solution : solutions)
        {
            float time;
            std::size_t workspace_size;
            uint64_t solver_id;

            EXPECT_EQUAL(miopenGetSolutionTime(solution, &time), miopenStatusSuccess);
            EXPECT_EQUAL(miopenGetSolutionWorkspaceSize(solution, &workspace_size),
                         miopenStatusSuccess);
            EXPECT_EQUAL(miopenGetSolutionSolverId(solution, &solver_id), miopenStatusSuccess);
        }

        std::cerr << "Finished testing miopenGetSolution<Attribute>." << std::endl;
    }

    void TestRunSolutionsForward(Handle& handle, const std::vector<miopenSolution_t>& solutions)
    {
        std::cerr << "Testing solution functions..." << std::endl;

        miopenTensorDescriptor_t x_desc = &xTensor.desc, y_desc = &yTensor.desc;

        const unsigned int numTensors = 2;

        for(const auto& solution : solutions)
        {
            auto arguments = std::make_unique<miopenTensorArgument_t[]>(numTensors);

            auto in_gpu  = handle.Write(xTensor.data);
            auto out_gpu = handle.Write(yTensor.data);

            miopenTensorArgumentId_t names[numTensors]       = {miopenTensorSoftmaxX,
                                                          miopenTensorSoftmaxY};
            void* buffers[numTensors]                        = {in_gpu.get(), out_gpu.get()};
            miopenTensorDescriptor_t descriptors[numTensors] = {x_desc, y_desc};

            for(auto i = 0; i < numTensors; ++i)
            {
                arguments[i].id         = names[i];
                arguments[i].descriptor = &descriptors[i];
                arguments[i].buffer     = buffers[i];
            }

            std::cerr << "Run a solution." << std::endl;
            EXPECT_EQUAL(
                miopenRunSolution(&handle, solution, numTensors, arguments.get(), nullptr, 0),
                miopenStatusSuccess);

            float alpha = softmax_descriptor.GetAlpha();
            float beta  = softmax_descriptor.GetBeta();

            // tensor<T> yTensorDup = yTensor;
            tensor<T> yTensorRef = tensor<T>{test_n, test_c, test_h, test_w};

            auto out_gpu_ref = handle.Write(yTensorRef.data);

            // Run softmax in a usual way (which is tested) and compare results
            EXPECT_EQUAL(miopenSoftmaxForward_V2(&handle,
                                                 &alpha,
                                                 x_desc,
                                                 in_gpu.get(),
                                                 &beta,
                                                 &yTensorRef.desc,
                                                 out_gpu_ref.get(),
                                                 softmax_descriptor.GetAlgorithm(),
                                                 softmax_descriptor.GetMode()),
                         miopenStatusSuccess);

            yTensor.data    = handle.Read<T>(out_gpu, yTensor.data.size());
            yTensorRef.data = handle.Read<T>(out_gpu_ref, yTensorRef.data.size());

            mloSoftmaxForwardRunHost<T, Tref>(&xTensor.desc,
                                              &yTensor.desc,
                                              xTensor.data.data(),
                                              outhost.data(),
                                              alpha,
                                              beta,
                                              softmax_descriptor.GetAlgorithm(),
                                              softmax_descriptor.GetMode());

            double error           = miopen::rms_range(yTensorRef.data, outhost);
            const double tolerance = 1e-3;

            EXPECT_TRUE(std::isfinite(error) && error <= tolerance)
                << "Outputs do not match each other. Error:" << error;
        }

        std::cerr << "Finished testing solution functions." << std::endl;
    }

    void TestRunSolutionsBackward(Handle& handle, const std::vector<miopenSolution_t>& solutions)
    {
        std::cerr << "Testing solution functions..." << std::endl;

        miopenTensorDescriptor_t y_desc  = &yTensor.desc;
        miopenTensorDescriptor_t dy_desc = &dyTensor.desc;
        miopenTensorDescriptor_t dx_desc = &dxTensor.desc;

        const unsigned int numTensors = 3;

        for(const auto& solution : solutions)
        {
            auto arguments = std::make_unique<miopenTensorArgument_t[]>(numTensors);

            auto in1_gpu = handle.Write(yTensor.data);
            auto in2_gpu = handle.Write(dyTensor.data);
            auto out_gpu = handle.Write(dxTensor.data);

            miopenTensorArgumentId_t names[numTensors] = {
                miopenTensorSoftmaxY, miopenTensorSoftmaxDY, miopenTensorSoftmaxDX};
            void* buffers[numTensors] = {in1_gpu.get(), in2_gpu.get(), out_gpu.get()};
            miopenTensorDescriptor_t descriptors[numTensors] = {y_desc, dy_desc, dx_desc};

            for(auto i = 0; i < numTensors; ++i)
            {
                arguments[i].id         = names[i];
                arguments[i].descriptor = &descriptors[i];
                arguments[i].buffer     = buffers[i];
            }

            std::cerr << "Run a solution." << std::endl;
            EXPECT_EQUAL(
                miopenRunSolution(&handle, solution, numTensors, arguments.get(), nullptr, 0),
                miopenStatusSuccess);

            float alpha = softmax_descriptor.GetAlpha();
            float beta  = softmax_descriptor.GetBeta();

            // tensor<T> yTensorDup = yTensor;
            tensor<T> dxTensorRef = tensor<T>{test_n, test_c, test_h, test_w};

            // this is dx
            auto out_gpu_ref = handle.Write(dxTensorRef.data);

            // Run softmax in a usual way (which is tested) and compare results
            EXPECT_EQUAL(miopenSoftmaxBackward_V2(&handle,
                                                  &alpha,
                                                  y_desc,
                                                  in1_gpu.get(),
                                                  dy_desc,
                                                  in2_gpu.get(),
                                                  &beta,
                                                  &dxTensorRef.desc,
                                                  out_gpu_ref.get(),
                                                  softmax_descriptor.GetAlgorithm(),
                                                  softmax_descriptor.GetMode()),
                         miopenStatusSuccess);

            dxTensorRef.data = handle.Read<T>(out_gpu_ref, dxTensorRef.data.size());

            // run softmax cpu
            mloSoftmaxBackwardRunHost<T, Tref>(&yTensor.desc,
                                               &dyTensor.desc,
                                               yTensor.data.data(),
                                               dyTensor.data.data(),
                                               dinhost.data(),
                                               alpha,
                                               beta,
                                               softmax_descriptor.GetAlgorithm(),
                                               softmax_descriptor.GetMode());

            double error           = miopen::rms_range(dxTensorRef.data, dinhost);
            const double tolerance = 1e-4;

            std::cout << "error =  " << error << std::endl;

            EXPECT_TRUE(std::isfinite(error) && error <= tolerance)
                << "Outputs do not match each other. Error:" << error;
        }

        std::cerr << "Finished testing solution functions." << std::endl;
    }

    void Finalize() { EXPECT_EQUAL(miopenDestroyProblem(problem), miopenStatusSuccess); }

private:
    void Initialize()
    {
        softmax_descriptor.SetParams(1.0f, 0.0f, softmax_algo, softmax_mode);

        if(isForward)
        {

            auto gen_value_fwd = [](auto...) {
                return prng::gen_descreet_uniform_sign<T>(0.1, 5.0);
            };

            xTensor = tensor<T>{test_n, test_c, test_h, test_w}.generate(gen_value_fwd);
            yTensor = tensor<T>{test_n, test_c, test_h, test_w};

            EXPECT_EQUAL(miopenCreateSoftmaxProblem(
                             &problem, &softmax_descriptor, miopenProblemDirectionForward),
                         miopenStatusSuccess);

            outhost = std::vector<Tref>(yTensor.data.size(), static_cast<Tref>(0));
        }
        else
        {
            yTensor  = tensor<T>{test_n, test_c, test_h, test_w};
            dyTensor = tensor<T>{test_n, test_c, test_h, test_w};

            const T Data_scale = static_cast<T>(0.1);
            for(int i = 0; i < dyTensor.data.size(); i++)
            {
                dyTensor.data[i] =
                    Data_scale * prng::gen_A_to_B(static_cast<T>(-0.5), static_cast<T>(0.5));
            }

            for(int i = 0; i < yTensor.data.size(); i++)
            {
                yTensor.data[i] = prng::gen_A_to_B(static_cast<T>(-0.6), static_cast<T>(0.6));
            }

            dxTensor = tensor<T>{test_n, test_c, test_h, test_w};

            dinhost = std::vector<Tref>(yTensor.data.size(), static_cast<Tref>(0));

            EXPECT_EQUAL(miopenCreateSoftmaxProblem(
                             &problem, &softmax_descriptor, miopenProblemDirectionBackward),
                         miopenStatusSuccess);
        }

        AddTensorDescriptors();
    }

private:
    tensor<T> xTensor;
    tensor<T> yTensor;
    std::vector<Tref> outhost;

    tensor<T> dxTensor;
    tensor<T> dyTensor;
    std::vector<Tref> dinhost;

    SoftmaxDescriptor softmax_descriptor;
    miopenProblem_t problem;
    miopenSoftmaxAlgorithm_t softmax_algo;
    miopenSoftmaxMode_t softmax_mode;

    bool isForward;

    const unsigned int test_n = 128;
    const unsigned int test_c = 1;
    const unsigned int test_h = 1;
    const unsigned int test_w = 1500;
};

TEST(GPU_SoftmaxFind20_FP32, softmaxForward)
{
    Handle& handle = get_handle();

    SoftmaxFind20Test<float> test(true, MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_CHANNEL);

    std::vector<miopenSolution_t> solutions = test.TestFindSolutions(handle);
    test.TestSolutionAttributes(solutions);

    test.TestRunSolutionsForward(handle, solutions);
    test.Finalize();
}

TEST(GPU_SoftmaxFind20_FP32, softmaxBackward_fp32)
{
    Handle& handle = get_handle();

    SoftmaxFind20Test<float> test(false, MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_CHANNEL);

    std::vector<miopenSolution_t> solutions = test.TestFindSolutions(handle);
    test.TestSolutionAttributes(solutions);

    test.TestRunSolutionsBackward(handle, solutions);
    test.Finalize();
}

TEST(GPU_SoftmaxFind20_FP16, softmaxBackward_log_instance_mode_fp16)
{
    Handle& handle = get_handle();

    SoftmaxFind20Test<half> test(false, MIOPEN_SOFTMAX_LOG, MIOPEN_SOFTMAX_MODE_INSTANCE);

    std::vector<miopenSolution_t> solutions = test.TestFindSolutions(handle);
    test.TestSolutionAttributes(solutions);

    test.TestRunSolutionsBackward(handle, solutions);
    test.Finalize();
}

TEST(GPU_SoftmaxFind20_FP16, softmaxBackward_log_channel_mode_fp16)
{
    Handle& handle = get_handle();

    SoftmaxFind20Test<half> test(false, MIOPEN_SOFTMAX_LOG, MIOPEN_SOFTMAX_MODE_CHANNEL);

    std::vector<miopenSolution_t> solutions = test.TestFindSolutions(handle);
    test.TestSolutionAttributes(solutions);

    test.TestRunSolutionsBackward(handle, solutions);
    test.Finalize();
}
