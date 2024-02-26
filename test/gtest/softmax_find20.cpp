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
    void Initialize(bool forward)
    {
        isForward = forward;

        if (isForward)
        {
            xTensor = tensor<float>{16, 32, 8, 8}.generate(tensor_elem_gen_integer{17});
            yTensor = tensor<float>{16, 32, 8, 8};

            softmax_descriptor.SetParams(1.0f, 0.0f, MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_CHANNEL);

            EXPECT_EQUAL(miopenCreateSoftmaxProblem(&problem, &softmax_descriptor, miopenProblemDirectionForward), miopenStatusSuccess);

            AddTensorDescriptors();
        }
    }

    void AddTensorDescriptors()
    {
        std::cerr << "Creating softmax tensor descriptors..." << std::endl;

        auto test_set_tensor_descriptor = [this](miopenTensorArgumentId_t name,
                                                    TensorDescriptor& desc) {
            EXPECT_EQUAL(miopenSetProblemTensorDescriptor(problem, name, &desc), miopenStatusSuccess);
        };

        test_set_tensor_descriptor(miopenTensorSoftmaxX, xTensor.desc);
        test_set_tensor_descriptor(miopenTensorSoftmaxY, yTensor.desc);

        std::cerr << "Created softmax tensor descriptos." << std::endl;
    }

    std::vector<miopenSolution_t> TestFindSolutions(miopenHandle_t handle)
    {
        std::cerr << "Testing miopenFindSolutions..." << std::endl;

        auto solutions = std::vector<miopenSolution_t>{};
        std::size_t found;

        // We expect to get only 2 solutions for softmax for now. Hardcode value 16 as just big enough value
        solutions.resize(16);

        EXPECT_EQUAL(miopenFindSolutions(handle, problem, nullptr, solutions.data(), &found, solutions.size()), miopenStatusSuccess);
        EXPECT_OP(found, >=, 0);

        solutions.resize(found);

        std::cerr << "Finished testing miopenFindSolutions." << std::endl;
        return solutions;
    }

    void TestSolutionAttributes(const std::vector<miopenSolution_t>& solutions)
    {
        std::cerr << "Testing miopenGetSolution<Attribute>..." << std::endl;

        for (const auto& solution : solutions)
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

    void TestRunSolutions(miopenHandle_t handle, const std::vector<miopenSolution_t>& solutions)
    {
        std::cerr << "Testing solution functions..." << std::endl;

        miopenTensorDescriptor_t x_desc = &xTensor.desc, y_desc = &yTensor.desc;

        const unsigned int numTensors = 2;

        for (const auto& solution : solutions)
        {
            auto arguments = std::make_unique<miopenTensorArgument_t[]>(numTensors);

            miopenTensorArgumentId_t names[numTensors] = {miopenTensorConvolutionX, miopenTensorConvolutionY};
            void* buffers[numTensors]                        = {xTensor.data.data(), yTensor.data.data()};
            miopenTensorDescriptor_t descriptors[numTensors] = {x_desc, y_desc};

            for (auto i = 0; i < numTensors; ++i)
            {
                arguments[i].id         = names[i];
                arguments[i].descriptor = &descriptors[i];
                arguments[i].buffer     = buffers[i];
            }

            std::cerr << "Run a solution." << std::endl;
            EXPECT_EQUAL(miopenRunSolution(handle, solution, numTensors, arguments.get(), nullptr, 0), miopenStatusSuccess);

            float alpha = softmax_descriptor.GetAlpha();
            float beta = softmax_descriptor.GetBeta();

            //tensor<float> yTensorDup = yTensor;
            tensor<float> yTensorDup = tensor<float>{16, 32, 8, 8};


            // Run softmax in a usual way (which is tested) and compare results
            EXPECT_EQUAL(miopenSoftmaxForward_V2(handle, &alpha, x_desc, xTensor.data.data(), &beta, &yTensorDup.desc, yTensorDup.data.data(),
                            softmax_descriptor.GetAlgorithm(), softmax_descriptor.GetMode()), miopenStatusSuccess);            

            auto error = miopen::rms_range(yTensorDup.data, yTensor.data);
            EXPECT_TRUE(miopen::range_distance(yTensorDup.data) == miopen::range_distance(yTensor.data));
            EXPECT_TRUE(error == 0) << "Outputs do not match each other. Error:" << error;
        }

        std::cerr << "Finished testing solution functions." << std::endl;
    }

    void Finalize()
    {
        EXPECT_EQUAL(miopenDestroyProblem(problem), miopenStatusSuccess);
    }

private:
    tensor<float> xTensor;
    tensor<float> yTensor;

    tensor<float> dxTensor;
    tensor<float> dyTensor;

    SoftmaxDescriptor softmax_descriptor;
    miopenProblem_t problem;

    bool isForward;
};

TEST (TestSoftmaxFind20, softmaxForwardFind20)
{
    miopenHandle_t handle = &get_handle();
    
    SoftmaxFind20Test test;

    test.Initialize(true);
    
    std::vector<miopenSolution_t> solutions = test.TestFindSolutions(handle);
    test.TestSolutionAttributes(solutions);

    test.TestRunSolutions(handle, solutions);
    test.Finalize();    
}

