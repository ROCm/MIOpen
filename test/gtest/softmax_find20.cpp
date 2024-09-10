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

#include "test.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <miopen/softmax.hpp>

#include <miopen/miopen.h>

#include <miopen/solution.hpp>

#include <gtest/gtest.h>

#include <vector>

using namespace miopen;

class SoftmaxFind20Test
{
public:
    SoftmaxFind20Test(bool forward) : problem(nullptr), isForward(forward) { Initialize(); }

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

            // tensor<float> yTensorDup = yTensor;
            tensor<float> yTensorRef = tensor<float>{test_n, test_c, test_h, test_w};

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

            yTensor.data    = handle.Read<float>(out_gpu, yTensor.data.size());
            yTensorRef.data = handle.Read<float>(out_gpu_ref, yTensorRef.data.size());

            double error           = miopen::rms_range(yTensorRef.data, yTensor.data);
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

            // tensor<float> yTensorDup = yTensor;
            tensor<float> dxTensorRef = tensor<float>{test_n, test_c, test_h, test_w};

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

            yTensor.data     = handle.Read<float>(out_gpu, yTensor.data.size());
            dxTensorRef.data = handle.Read<float>(out_gpu_ref, dxTensorRef.data.size());

            double error           = miopen::rms_range(dxTensorRef.data, yTensor.data);
            const double tolerance = 1e-3;

            EXPECT_TRUE(std::isfinite(error) && error <= tolerance)
                << "Outputs do not match each other. Error:" << error;
        }

        std::cerr << "Finished testing solution functions." << std::endl;
    }

    void Finalize() { EXPECT_EQUAL(miopenDestroyProblem(problem), miopenStatusSuccess); }

private:
    void Initialize()
    {
        softmax_descriptor.SetParams(
            1.0f, 0.0f, MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_CHANNEL);

        if(isForward)
        {
            xTensor =
                tensor<float>{test_n, test_c, test_h, test_w}.generate(tensor_elem_gen_integer{17});
            yTensor = tensor<float>{test_n, test_c, test_h, test_w};

            EXPECT_EQUAL(miopenCreateSoftmaxProblem(
                             &problem, &softmax_descriptor, miopenProblemDirectionForward),
                         miopenStatusSuccess);
        }
        else
        {
            yTensor =
                tensor<float>{test_n, test_c, test_h, test_w}.generate(tensor_elem_gen_integer{17});
            dyTensor =
                tensor<float>{test_n, test_c, test_h, test_w}.generate(tensor_elem_gen_integer{17});
            dxTensor = tensor<float>{test_n, test_c, test_h, test_w};

            EXPECT_EQUAL(miopenCreateSoftmaxProblem(
                             &problem, &softmax_descriptor, miopenProblemDirectionBackward),
                         miopenStatusSuccess);
        }

        AddTensorDescriptors();
    }

private:
    tensor<float> xTensor;
    tensor<float> yTensor;

    tensor<float> dxTensor;
    tensor<float> dyTensor;

    SoftmaxDescriptor softmax_descriptor;
    miopenProblem_t problem;

    bool isForward;

    const unsigned int test_n = 100;
    const unsigned int test_c = 3;
    const unsigned int test_h = 32;
    const unsigned int test_w = 32;
};

TEST(GPU_SoftmaxFind20_FP32, softmaxForward)
{
    Handle& handle = get_handle();

    SoftmaxFind20Test test(true);

    std::vector<miopenSolution_t> solutions = test.TestFindSolutions(handle);
    test.TestSolutionAttributes(solutions);

    test.TestRunSolutionsForward(handle, solutions);
    test.Finalize();
}

TEST(GPU_SoftmaxFind20_FP32, softmaxBackward)
{
    Handle& handle = get_handle();

    SoftmaxFind20Test test(false);

    std::vector<miopenSolution_t> solutions = test.TestFindSolutions(handle);
    test.TestSolutionAttributes(solutions);

    test.TestRunSolutionsBackward(handle, solutions);
    test.Finalize();
}
