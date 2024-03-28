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

#include <miopen/mha/mha_descriptor.hpp>

#include <miopen/miopen.h>

#include <miopen/solution.hpp>

#include <gtest/gtest.h>

#include <vector>

using namespace miopen;

struct TensorStruct
{
    TensorStruct(const std::string& name_, bool isFloat = true) : name(name_), isFloatTensor(isFloat) {}

    bool isFloatTensor;
    std::string name;
    tensor<float> tensorFloat;

    //
    //tensor<unit64_t> tensorUint64;
};

typedef std::shared_ptr<TensorStruct> TensorStructPtr;

class MHAFind20Test
{
public:
    MHAFind20Test(bool forward) : problem(nullptr), isForward(forward) { Initialize(); }

    void AddTensorDescriptors()
    {
        std::cerr << "Creating mha tensor descriptors..." << std::endl;

        auto test_set_tensor_descriptor = [this](miopenTensorArgumentId_t name,
                                                 TensorDescriptor& desc) {
            EXPECT_EQUAL(miopenSetProblemTensorDescriptor(problem, name, &desc),
                         miopenStatusSuccess);
        };

        if(isForward)
        {
            test_set_tensor_descriptor(miopenTensorSoftmaxX, xTensor.desc);
            //test_set_tensor_descriptor(miopenTensorSoftmaxY, yTensor.desc);
        }
        else
        {
            //test_set_tensor_descriptor(miopenTensorSoftmaxY, yTensor.desc);
            //test_set_tensor_descriptor(miopenTensorSoftmaxDY, dyTensor.desc);
            //test_set_tensor_descriptor(miopenTensorSoftmaxDX, dxTensor.desc);
        }

        std::cerr << "Created mha tensor descriptors." << std::endl;
    }

    std::vector<miopenSolution_t> TestFindSolutions(Handle& handle)
    {
        std::cerr << "Testing miopenFindSolutions..." << std::endl;

        auto solutions = std::vector<miopenSolution_t>{};
        std::size_t found;

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

        //miopenTensorDescriptor_t x_desc = &xTensor.desc, y_desc = &yTensor.desc;

        const unsigned int numTensors = 2;

        for(const auto& solution : solutions)
        {
            auto arguments = std::make_unique<miopenTensorArgument_t[]>(numTensors);

            /*auto in_gpu  = handle.Write(xTensor.data);
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

            float alpha = mha_descriptor.GetAlpha();
            float beta  = mha_descriptor.GetBeta();

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
                                                 mha_descriptor.GetAlgorithm(),
                                                 mha_descriptor.GetMode()),
                         miopenStatusSuccess);

            yTensor.data    = handle.Read<float>(out_gpu, yTensor.data.size());
            yTensorRef.data = handle.Read<float>(out_gpu_ref, yTensorRef.data.size());

            double error           = miopen::rms_range(yTensorRef.data, yTensor.data);
            const double tolerance = 1e-3;

            EXPECT_TRUE(std::isfinite(error) && error <= tolerance)
                << "Outputs do not match each other. Error:" << error;*/
        }

        std::cerr << "Finished testing solution functions." << std::endl;
    }

    void TestRunSolutionsBackward(Handle& handle, const std::vector<miopenSolution_t>& solutions)
    {
        std::cerr << "Testing solution functions..." << std::endl;

/*        miopenTensorDescriptor_t y_desc  = &yTensor.desc;
        miopenTensorDescriptor_t dy_desc = &dyTensor.desc;
        miopenTensorDescriptor_t dx_desc = &dxTensor.desc;*/

        const unsigned int numTensors = 3;

        for(const auto& solution : solutions)
        {
            auto arguments = std::make_unique<miopenTensorArgument_t[]>(numTensors);

            //auto in1_gpu = handle.Write(yTensor.data);
            //auto in2_gpu = handle.Write(dyTensor.data);
            //auto out_gpu = handle.Write(dxTensor.data);

            /*miopenTensorArgumentId_t names[numTensors] = {
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
                miopenStatusSuccess);*/

            // tensor<float> yTensorDup = yTensor;
            //tensor<float> dxTensorRef = tensor<float>{test_n, test_c, test_h, test_w};

            // this is dx
           /* auto out_gpu_ref = handle.Write(dxTensorRef.data);

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
                << "Outputs do not match each other. Error:" << error;*/
        }

        std::cerr << "Finished testing solution functions." << std::endl;
    }

    void Finalize() { EXPECT_EQUAL(miopenDestroyProblem(problem), miopenStatusSuccess); }

private:
    bool IsDropoutTensorId(miopenTensorArgumentId_t id) const 
    {
        return id == miopenTensorMHADropoutProbability ||
                id == miopenTensorMHADropoutSeed ||
                id == miopenTensorMHADropoutOffset;
    }

    void AddTensorMetadata(miopenTensorArgumentId_t id, const std::string& name)
    {
        bool floatFlag = !IsDropoutTensorId(id);
        tensors[id] = new TensorStruct(name, floatFlag);
    }

    void Initialize()
    {
        mha_descriptor.SetParams(1.0f);
            
        if(isForward)
        {
            AddTensorMetadata(miopenTensorMHAK, "miopenTensorMHAK");
            AddTensorMetadata(miopenTensorMHAQ, "miopenTensorMHAQ");
            AddTensorMetadata(miopenTensorMHAV, "miopenTensorMHAV");
            AddTensorMetadata(miopenTensorMHADescaleK, "miopenTensorMHADescaleK");

            AddTensorMetadata(miopenTensorMHADescaleQ, "miopenTensorMHADescaleQ");
            AddTensorMetadata(miopenTensorMHADescaleV, "miopenTensorMHADescaleV");
            AddTensorMetadata(miopenTensorMHADescaleS, "miopenTensorMHADescaleS");
            AddTensorMetadata(miopenTensorMHAScaleS, "miopenTensorMHAScaleS");
            AddTensorMetadata(miopenTensorMHAScaleO, "miopenTensorMHAScaleO");

            AddTensorMetadata(miopenTensorMHADropoutProbability, "miopenTensorMHADropoutProbability");
            AddTensorMetadata(miopenTensorMHADropoutSeed, "miopenTensorMHADropoutSeed");
            AddTensorMetadata(miopenTensorMHADropoutOffset, "miopenTensorMHADropoutOffset");

            AddTensorMetadata(miopenTensorMHAO, "miopenTensorMHAO");
            AddTensorMetadata(miopenTensorMHAAmaxO, "miopenTensorMHAAmaxO");
            AddTensorMetadata(miopenTensorMHAAmaxS, "miopenTensorMHAAmaxS");
            AddTensorMetadata(miopenTensorMHAM, "miopenTensorMHAM");
            AddTensorMetadata(miopenTensorMHAZInv, "miopenTensorMHAZInv");
              
          /*  xTensor =
                tensor<float>{test_n, test_c, test_h, test_w}.generate(tensor_elem_gen_integer{17});
            yTensor = tensor<float>{test_n, test_c, test_h, test_w};*/

            EXPECT_EQUAL(miopenCreateMHAProblem(
                             &problem, &mha_descriptor, miopenProblemDirectionForward),
                         miopenStatusSuccess);
        }
        else
        {
            /*yTensor =
                tensor<float>{test_n, test_c, test_h, test_w}.generate(tensor_elem_gen_integer{17});
            dyTensor =
                tensor<float>{test_n, test_c, test_h, test_w}.generate(tensor_elem_gen_integer{17});
            dxTensor = tensor<float>{test_n, test_c, test_h, test_w};

            EXPECT_EQUAL(miopenCreateSoftmaxProblem(
                             &problem, &softmax_descriptor, miopenProblemDirectionBackward),
                         miopenStatusSuccess);*/
        }

        AddTensorDescriptors();
    }

private:
    std::map<miopenTensorArgumentId_t, TensorStructPtr> tensors;

    MHADescriptor mha_descriptor;
    miopenProblem_t problem;

    bool isForward;

    const unsigned int test_n = 100;
    const unsigned int test_c = 3;
    const unsigned int test_h = 32;
    const unsigned int test_w = 32;

    const miopenTensorArgumentId_t
};

TEST(TestMHAFind20, MHAForward)
{
    Handle& handle = get_handle();

    MHAFind20Test test(true);

    std::vector<miopenSolution_t> solutions = test.TestFindSolutions(handle);
    test.TestSolutionAttributes(solutions);

    test.TestRunSolutionsForward(handle, solutions);
    test.Finalize();
}
