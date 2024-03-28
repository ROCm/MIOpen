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
#include "../workspace.hpp"

#include <miopen/mha/mha_descriptor.hpp>
#include <miopen/mha/solvers.hpp>
#include <miopen/mha/invoke_params.hpp>

#include <miopen/miopen.h>

#include <miopen/solution.hpp>

#include <gtest/gtest.h>

#include <vector>

using namespace miopen;

struct TensorStruct
{
    TensorStruct(bool isFloat = true) : isFloatTensor(isFloat) {}


    bool isFloatTensor;
    tensor<float> tensorFloat;

    Allocator::ManageDataPtr gpuBuffer;

    // TODO Unsued for now
    //tensor<unit64_t> tensorUint64;
};

typedef std::shared_ptr<TensorStruct> TensorStructPtr;

typedef std::map<miopenTensorArgumentId_t, TensorStructPtr> TensorStructMap;

class MHAFind20Test
{
public:
    MHAFind20Test(bool forward) : problem(nullptr), isForward(forward) { Initialize(); }
  
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
        std::cerr << "Testing forward solution functions..." << std::endl;

        const unsigned int numTensors = tensors.size();

        auto arguments = std::make_unique<miopenTensorArgument_t[]>(numTensors);

        std::vector<miopenTensorDescriptor_t>  descVector(numTensors);

        int i = 0;
        for (const auto& it : tensors)
        {
            descVector[i] = &it.second->tensorFloat.desc;

            it.second->gpuBuffer = handle.Write(it.second->tensorFloat.data);

            arguments[i].id         = it.first;
            arguments[i].descriptor = &descVector[i];
            arguments[i].buffer     = it.second->gpuBuffer.get();

            i++;
        }

        std::vector<miopenTensorArgumentId_t> output_ids = {miopenTensorMHAO, miopenTensorMHAAmaxO, miopenTensorMHAAmaxS, miopenTensorMHAM};

        std::size_t workspace_size = 0;
        uint64_t solver_id;

        Workspace workspace;

        for(const auto& solution : solutions)
        {    
            miopenGetSolutionWorkspaceSize(solution, &workspace_size);
            miopenGetSolutionSolverId(solution, &solver_id);

            workspace.resize(workspace_size);

            std::cerr << "Run a solution." << std::endl;
            EXPECT_EQUAL(
                miopenRunSolution(&handle, solution, numTensors, arguments.get(), nullptr, 0),
                miopenStatusSuccess);

            TensorStructMap outputTensorResults;

            // reading results from find 2.0 call and preparing structures for non find 2.0 outputs to compare with
            for (const auto& id : output_ids)
            {
                const TensorStructPtr& tensorStructPtr = tensors[id];
                tensorStructPtr->tensorFloat.data = handle.Read<float>(tensorStructPtr->gpuBuffer, tensorStructPtr->tensorFloat.data.size());

                outputTensorResults[id] = TensorStructPtr (new TensorStruct());
                outputTensorResults[id]->tensorFloat = tensorStructPtr->tensorFloat;
            }

            GetResultsWithoutFind20(handle, outputTensorResults, workspace, solver_id);

            /*double error           = miopen::rms_range(yTensorRef.data, yTensor.data);
            const double tolerance = 1e-3;

            EXPECT_TRUE(std::isfinite(error) && error <= tolerance)
                << "Outputs do not match each other. Error:" << error;*/
        }

        std::cerr << "Finished testing forward solution functions." << std::endl;
    }

    void TestRunSolutionsBackward(Handle& handle, const std::vector<miopenSolution_t>& solutions)
    {
        std::cerr << "Testing backward solution functions..." << std::endl;

        std::cerr << "Finished testing backward solution functions." << std::endl;
    }

    void Finalize() { EXPECT_EQUAL(miopenDestroyProblem(problem), miopenStatusSuccess); }

private:
    bool IsDropoutTensorId(miopenTensorArgumentId_t id) const 
    {
        return id == miopenTensorMHADropoutProbability ||
                id == miopenTensorMHADropoutSeed ||
                id == miopenTensorMHADropoutOffset;
    }

    void CreateTensor(miopenTensorArgumentId_t id, 
                        unsigned int n = 1,
                        unsigned int c = 1,
                        unsigned int h = 1,
                        unsigned int w = 1,
                        bool generate = true)
    {
        // TODO Unused for now
        bool floatFlag = !IsDropoutTensorId(id);
        tensors[id] = TensorStructPtr(new TensorStruct(floatFlag));

        tensors[id]->tensorFloat = tensor<float>{test_n, test_c, test_h, test_w};

        if (generate)
        {
            tensors[id]->tensorFloat.generate(tensor_elem_gen_integer{17});
        }

        EXPECT_EQUAL(miopenSetProblemTensorDescriptor(problem, id, &tensors[id]->tensorFloat.desc), miopenStatusSuccess);
    }

    void Initialize()
    {
        mha_descriptor.SetParams(1.0f);

        EXPECT_EQUAL(miopenCreateMHAProblem(&problem, &mha_descriptor, miopenProblemDirectionForward), miopenStatusSuccess);
            
        if(isForward)
        {
            CreateTensor(miopenTensorMHAK, test_n, test_c, test_h, test_w);
            CreateTensor(miopenTensorMHAQ, test_n, test_c, test_h, test_w);
            CreateTensor(miopenTensorMHAV, test_n, test_c, test_h, test_w);

            CreateTensor(miopenTensorMHADescaleK);
            CreateTensor(miopenTensorMHADescaleQ);
            CreateTensor(miopenTensorMHADescaleV);
            CreateTensor(miopenTensorMHADescaleS);
            CreateTensor(miopenTensorMHAScaleS);
            CreateTensor(miopenTensorMHAScaleO);

            CreateTensor(miopenTensorMHADropoutProbability);
            CreateTensor(miopenTensorMHADropoutSeed);
            CreateTensor(miopenTensorMHADropoutOffset);

            CreateTensor(miopenTensorMHAO, test_n, test_c, test_h, test_w);
            CreateTensor(miopenTensorMHAAmaxO, 1, 1, 1, 1, false);
            CreateTensor(miopenTensorMHAAmaxS, 1, 1, 1, 1, false);
            CreateTensor(miopenTensorMHAM, test_n, test_c, test_h, 1, false);
            CreateTensor(miopenTensorMHAZInv, test_n, test_c, test_h, 1, false);             
        }
        else
        {
            // todo add backward path test
        }
    }

    void GetResultsWithoutFind20(Handle& handle, TensorStructMap& outputResultsMap, Workspace& workspace, uint64_t solver_id)
    {
        // Get Problem object to use helper asMHA() function. Downcast is used in order to reuse some code
        ProblemContainer* pc = static_cast<ProblemContainer*>(problem);

        const Problem& problem_casted = boost::get<const Problem&>(pc->item);
        const mha::ProblemDescription problem_description = problem_casted.AsMHA();

        const auto invoke_ctx = [&]() -> AnyInvokeParams {
            const mha::MHAInputDescsForward& inputDescsForward = problem_description.GetDescs();

            mha::MHADataForward dataForward = {tensors[miopenTensorMHAK]->gpuBuffer.get(),
                                            tensors[miopenTensorMHAQ]->gpuBuffer.get(),
                                            tensors[miopenTensorMHAV]->gpuBuffer.get(),
                                            tensors[miopenTensorMHADescaleK]->gpuBuffer.get(),
                                            tensors[miopenTensorMHADescaleQ]->gpuBuffer.get(),
                                            tensors[miopenTensorMHADescaleV]->gpuBuffer.get(),
                                            tensors[miopenTensorMHADescaleS]->gpuBuffer.get(),
                                            tensors[miopenTensorMHAScaleS]->gpuBuffer.get(),
                                            tensors[miopenTensorMHAScaleO]->gpuBuffer.get(),

                                            tensors[miopenTensorMHADropoutProbability]->gpuBuffer.get(),
                                            tensors[miopenTensorMHADropoutSeed]->gpuBuffer.get(),
                                            tensors[miopenTensorMHADropoutOffset]->gpuBuffer.get(),

                                            outputResultsMap[miopenTensorMHAO]->gpuBuffer.get(),
                                            outputResultsMap[miopenTensorMHAAmaxO]->gpuBuffer.get(),
                                            outputResultsMap[miopenTensorMHAAmaxS]->gpuBuffer.get(),
                                            outputResultsMap[miopenTensorMHAM]->gpuBuffer.get(),
                                            outputResultsMap[miopenTensorMHAZInv]->gpuBuffer.get()};

            return mha::InvokeParams(inputDescsForward, dataForward, workspace.ptr(), workspace.size());
        }();

        const auto net_cfg       = problem_description.MakeNetworkConfig();
        const auto found_invoker = handle.GetInvoker(net_cfg, solver::Id(solver_id));

        if(found_invoker)
        {
            (*found_invoker)(handle, invoke_ctx);
        }
        else
        {
            auto ctx = ExecutionContext{&handle};

            solver::mha::MHA mha;

            const auto mha_solution = mha.GetSolution(ctx, problem_description);

            decltype(auto) invoker =
                handle.PrepareInvoker(*mha_solution.invoker_factory, mha_solution.construction_params);
            handle.RegisterInvoker(invoker, net_cfg, solver::Id(solver_id).ToString());
            invoker(handle, invoke_ctx);
        }

        for (const auto& it : outputResultsMap)
        {
            it.second->tensorFloat.data = handle.Read<float>(it.second->gpuBuffer, it.second->tensorFloat.data.size());
        }
    }    

private:
    TensorStructMap tensors;

    MHADescriptor mha_descriptor;

    miopenProblem_t problem;

    bool isForward;

    const unsigned int test_n = 2;
    const unsigned int test_c = 4;
    const unsigned int test_h = 8;
    const unsigned int test_w = 16;
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
