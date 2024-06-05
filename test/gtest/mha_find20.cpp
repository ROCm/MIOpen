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
    // tensor<unit64_t> tensorUint64;
};

typedef std::unique_ptr<TensorStruct> TensorStructPtr;

typedef std::map<miopenTensorArgumentId_t, TensorStructPtr> TensorStructMap;

class MhaFind20Test
{
public:
    MhaFind20Test(bool forward) : problem(nullptr), isForward(forward) { Initialize(); }

    std::vector<miopenSolution_t> TestFindSolutions(Handle& handle)
    {
        std::cerr << "Testing miopenFindSolutions..." << std::endl;

        auto solutions    = std::vector<miopenSolution_t>{};
        std::size_t found = 0;

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

    void TestRunSolutions(Handle& handle, const std::vector<miopenSolution_t>& solutions)
    {
        std::cerr << "Testing a solution..." << std::endl;

        const size_t numTensors = tensors.size();

        auto arguments = std::make_unique<miopenTensorArgument_t[]>(numTensors);

        std::vector<miopenTensorDescriptor_t> descVector(numTensors);

        int i = 0;
        for(const auto& it : tensors)
        {
            descVector[i] = &it.second->tensorFloat.desc;

            it.second->gpuBuffer = handle.Write(it.second->tensorFloat.data);

            arguments[i].id         = it.first;
            arguments[i].descriptor = &descVector[i];
            arguments[i].buffer     = it.second->gpuBuffer.get();

            ++i;
        }

        std::vector<miopenTensorArgumentId_t> output_ids;

        if(isForward)
        {
            output_ids = {miopenTensorMhaO,
                          miopenTensorMhaAmaxO,
                          miopenTensorMhaAmaxS,
                          miopenTensorMhaM,
                          miopenTensorMhaZInv};
        }
        else
        {
            output_ids = {miopenTensorMhaDQ,
                          miopenTensorMhaDK,
                          miopenTensorMhaDV,
                          miopenTensorMhaAmaxDQ,
                          miopenTensorMhaAmaxDK,
                          miopenTensorMhaAmaxDV,
                          miopenTensorMhaAmaxDS};
        }

        std::size_t workspace_size = 0;
        uint64_t solver_id;

        Workspace workspace;

        for(const auto& solution : solutions)
        {
            miopenGetSolutionWorkspaceSize(solution, &workspace_size);
            miopenGetSolutionSolverId(solution, &solver_id);

            workspace.resize(workspace_size);

            std::cerr << "Run a solution." << std::endl;
            EXPECT_EQUAL(miopenRunSolution(&handle,
                                           solution,
                                           numTensors,
                                           arguments.get(),
                                           workspace.ptr(),
                                           workspace.size()),
                         miopenStatusSuccess);

            std::cerr << "Solution executed." << std::endl;

            TensorStructMap outputTensorResults;

            // reading results from find 2.0 call and preparing structures for non find 2.0 outputs
            // to compare with
            for(const auto& id : output_ids)
            {
                const TensorStructPtr& tensorStructPtr = tensors[id];
                tensorStructPtr->tensorFloat.data      = handle.Read<float>(
                    tensorStructPtr->gpuBuffer, tensorStructPtr->tensorFloat.data.size());

                TensorStructPtr& ptr = outputTensorResults[id];

                ptr              = std::make_unique<TensorStruct>();
                ptr->tensorFloat = tensorStructPtr->tensorFloat;
                ptr->gpuBuffer   = handle.Write(ptr->tensorFloat.data);
            }

            std::cerr << "Run via solver infrastructure directly." << std::endl;
            if(isForward)
            {
                GetForwardResultsWithoutFind20(handle, outputTensorResults, workspace, solver_id);
            }
            else
            {
                GetBackwardResultsWithoutFind20(handle, outputTensorResults, workspace, solver_id);
            }
            std::cerr << "Run via solver infrastructure executed!" << std::endl;

            for(const auto& id : output_ids)
            {
                double error = miopen::rms_range(outputTensorResults[id]->tensorFloat.data,
                                                 tensors[id]->tensorFloat.data);
                const double tolerance = 1e-3;

                EXPECT_TRUE(std::isfinite(error)) << "Tensor id: " << id;
                EXPECT_LE(error, tolerance) << "Tensor id: " << id;
            }
        }

        std::cerr << "Finished testing solution functions." << std::endl;
    }

    void Finalize() { EXPECT_EQUAL(miopenDestroyProblem(problem), miopenStatusSuccess); }

private:
    bool IsInt64TensorId(miopenTensorArgumentId_t id) const
    {
        return id == miopenTensorMhaDropoutSeed || id == miopenTensorMhaDropoutOffset;
    }

    enum class GenerateType
    {
        DontGenerate,
        Generate_1_Always,
        Generate_0_Always,
        GenerateRandom
    };

    void CreateTensor(miopenTensorArgumentId_t id,
                      GenerateType generateType = GenerateType::Generate_1_Always,
                      unsigned int n            = 1,
                      unsigned int h            = 1,
                      unsigned int s            = 1,
                      unsigned int d            = 1)
    {
        // TODO Unused for now
        bool floatFlag = !IsInt64TensorId(id);
        tensors[id]    = std::make_unique<TensorStruct>(floatFlag);

        tensors[id]->tensorFloat = tensor<float>{n, h, s, d};

        switch(generateType)
        {
        case GenerateType::Generate_0_Always:
            tensors[id]->tensorFloat.generate([](auto n_, auto h_, auto s_, auto d_) { return 0; });
            break;

        case GenerateType::Generate_1_Always:
            tensors[id]->tensorFloat.generate([](auto n_, auto h_, auto s_, auto d_) { return 1; });
            break;

        case GenerateType::GenerateRandom:
            tensors[id]->tensorFloat.generate(tensor_elem_gen_integer{17});
            break;

        default: break;
        }

        EXPECT_EQUAL(miopenSetProblemTensorDescriptor(problem, id, &tensors[id]->tensorFloat.desc),
                     miopenStatusSuccess);
    }

    void Initialize()
    {
        mha_descriptor.SetParams(scale);

        EXPECT_EQUAL(miopenCreateMhaProblem(&problem,
                                            &mha_descriptor,
                                            isForward ? miopenProblemDirectionForward
                                                      : miopenProblemDirectionBackward),
                     miopenStatusSuccess);

        CreateTensor(
            miopenTensorMhaK, GenerateType::GenerateRandom, test_n, test_h, test_s, test_d);

        CreateTensor(
            miopenTensorMhaV, GenerateType::GenerateRandom, test_n, test_h, test_s, test_d);

        CreateTensor(miopenTensorMhaDescaleK);
        CreateTensor(miopenTensorMhaDescaleQ);
        CreateTensor(miopenTensorMhaDescaleV);
        CreateTensor(miopenTensorMhaDescaleS);

        CreateTensor(miopenTensorMhaScaleS);

        CreateTensor(miopenTensorMhaDropoutProbability, GenerateType::Generate_0_Always);
        CreateTensor(miopenTensorMhaDropoutSeed, GenerateType::GenerateRandom);
        CreateTensor(miopenTensorMhaDropoutOffset, GenerateType::GenerateRandom);

        if(isForward)
        {
            CreateTensor(
                miopenTensorMhaQ, GenerateType::GenerateRandom, test_n, test_h, test_s, test_d);

            CreateTensor(miopenTensorMhaScaleO);

            CreateTensor(
                miopenTensorMhaO, GenerateType::DontGenerate, test_n, test_h, test_s, test_d);
            CreateTensor(miopenTensorMhaAmaxO, GenerateType::DontGenerate);
            CreateTensor(miopenTensorMhaAmaxS, GenerateType::DontGenerate);
            CreateTensor(miopenTensorMhaM, GenerateType::DontGenerate, test_n, test_h, test_s, 1);
            CreateTensor(
                miopenTensorMhaZInv, GenerateType::DontGenerate, test_n, test_h, test_s, 1);
        }
        else
        {
            CreateTensor(
                miopenTensorMhaQ, GenerateType::Generate_0_Always, test_n, test_h, test_s, test_d);

            CreateTensor(
                miopenTensorMhaO, GenerateType::GenerateRandom, test_n, test_h, test_s, test_d);

            CreateTensor(
                miopenTensorMhaDO, GenerateType::GenerateRandom, test_n, test_h, test_s, test_d);

            CreateTensor(
                miopenTensorMhaM, GenerateType::Generate_0_Always, test_n, test_h, test_s, 1);
            CreateTensor(
                miopenTensorMhaZInv, GenerateType::Generate_1_Always, test_n, test_h, test_s, 1);

            CreateTensor(miopenTensorMhaDescaleO, GenerateType::GenerateRandom);
            CreateTensor(miopenTensorMhaDescaleDO, GenerateType::GenerateRandom);
            CreateTensor(miopenTensorMhaDescaleDS, GenerateType::GenerateRandom);
            CreateTensor(miopenTensorMhaScaleDS, GenerateType::GenerateRandom);
            CreateTensor(miopenTensorMhaScaleDQ, GenerateType::GenerateRandom);
            CreateTensor(miopenTensorMhaScaleDK, GenerateType::GenerateRandom);
            CreateTensor(miopenTensorMhaScaleDV, GenerateType::GenerateRandom);

            CreateTensor(
                miopenTensorMhaDQ, GenerateType::DontGenerate, test_n, test_h, test_s, test_d);
            CreateTensor(
                miopenTensorMhaDK, GenerateType::DontGenerate, test_n, test_h, test_s, test_d);
            CreateTensor(
                miopenTensorMhaDV, GenerateType::DontGenerate, test_n, test_h, test_s, test_d);
            CreateTensor(miopenTensorMhaAmaxDQ, GenerateType::DontGenerate);
            CreateTensor(miopenTensorMhaAmaxDK, GenerateType::DontGenerate);
            CreateTensor(miopenTensorMhaAmaxDV, GenerateType::DontGenerate);
            CreateTensor(miopenTensorMhaAmaxDS, GenerateType::DontGenerate);
        }
    }

    void GetForwardResultsWithoutFind20(Handle& handle,
                                        TensorStructMap& outputResultsMap,
                                        Workspace& workspace,
                                        uint64_t solver_id)
    {
        const auto& mhaK = tensors[miopenTensorMhaK];
        const auto& mhaQ = tensors[miopenTensorMhaQ];
        const auto& mhaV = tensors[miopenTensorMhaV];

        const auto& mhaDescaleK = tensors[miopenTensorMhaDescaleK];
        const auto& mhaDescaleQ = tensors[miopenTensorMhaDescaleQ];
        const auto& mhaDescaleV = tensors[miopenTensorMhaDescaleV];
        const auto& mhaDescaleS = tensors[miopenTensorMhaDescaleS];

        const auto& mhaScaleS = tensors[miopenTensorMhaScaleS];
        const auto& mhaScaleO = tensors[miopenTensorMhaScaleO];

        const auto& mhadp = tensors[miopenTensorMhaDropoutProbability];
        const auto& mhads = tensors[miopenTensorMhaDropoutSeed];
        const auto& mhado = tensors[miopenTensorMhaDropoutOffset];

        mha::MhaInputDescsForward inputDescs = {mhaK->tensorFloat.desc,
                                                mhaQ->tensorFloat.desc,
                                                mhaV->tensorFloat.desc,
                                                mhaDescaleK->tensorFloat.desc,
                                                mhaDescaleQ->tensorFloat.desc,
                                                mhaDescaleV->tensorFloat.desc,
                                                mhaDescaleS->tensorFloat.desc,
                                                mhaScaleS->tensorFloat.desc,
                                                mhaScaleO->tensorFloat.desc,
                                                scale,
                                                mhadp->tensorFloat.desc,
                                                mhads->tensorFloat.desc,
                                                mhado->tensorFloat.desc,
                                                tensors[miopenTensorMhaO]->tensorFloat.desc,
                                                tensors[miopenTensorMhaAmaxO]->tensorFloat.desc,
                                                tensors[miopenTensorMhaAmaxS]->tensorFloat.desc,
                                                tensors[miopenTensorMhaM]->tensorFloat.desc,
                                                tensors[miopenTensorMhaZInv]->tensorFloat.desc};

        const mha::ProblemDescription problem_description = {inputDescs};

        const auto invoke_ctx = [&]() -> AnyInvokeParams {
            mha::MhaDataForward dataForward = {
                mhaK->gpuBuffer.get(),
                mhaQ->gpuBuffer.get(),
                mhaV->gpuBuffer.get(),
                mhaDescaleK->gpuBuffer.get(),
                mhaDescaleQ->gpuBuffer.get(),
                mhaDescaleV->gpuBuffer.get(),
                mhaDescaleS->gpuBuffer.get(),
                mhaScaleS->gpuBuffer.get(),
                mhaScaleO->gpuBuffer.get(),
                mhadp->gpuBuffer.get(),
                mhads->gpuBuffer.get(),
                mhado->gpuBuffer.get(),
                outputResultsMap[miopenTensorMhaO]->gpuBuffer.get(),
                outputResultsMap[miopenTensorMhaAmaxO]->gpuBuffer.get(),
                outputResultsMap[miopenTensorMhaAmaxS]->gpuBuffer.get(),
                outputResultsMap[miopenTensorMhaM]->gpuBuffer.get(),
                outputResultsMap[miopenTensorMhaZInv]->gpuBuffer.get()};

            return mha::InvokeParams(dataForward, workspace.ptr(), workspace.size());
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

            solver::mha::MhaForward mhaForward;

            const auto mha_solution = mhaForward.GetSolution(ctx, problem_description);

            decltype(auto) invoker = handle.PrepareInvoker(*mha_solution.invoker_factory,
                                                           mha_solution.construction_params);
            handle.RegisterInvoker(invoker, net_cfg, solver::Id(solver_id).ToString());
            invoker(handle, invoke_ctx);
        }

        ReadData(handle, outputResultsMap);
    }

    void GetBackwardResultsWithoutFind20(Handle& handle,
                                         TensorStructMap& outputResultsMap,
                                         Workspace& workspace,
                                         uint64_t solver_id)
    {
        const auto& mhaK    = tensors[miopenTensorMhaK];
        const auto& mhaQ    = tensors[miopenTensorMhaQ];
        const auto& mhaV    = tensors[miopenTensorMhaV];
        const auto& mhaO    = tensors[miopenTensorMhaO];
        const auto& mhaDO   = tensors[miopenTensorMhaDO];
        const auto& mhaM    = tensors[miopenTensorMhaM];
        const auto& mhaZInv = tensors[miopenTensorMhaZInv];

        const auto& mhaDescaleK  = tensors[miopenTensorMhaDescaleK];
        const auto& mhaDescaleQ  = tensors[miopenTensorMhaDescaleQ];
        const auto& mhaDescaleV  = tensors[miopenTensorMhaDescaleV];
        const auto& mhaDescaleS  = tensors[miopenTensorMhaDescaleS];
        const auto& mhaDescaleO  = tensors[miopenTensorMhaDescaleO];
        const auto& mhaDescaleDO = tensors[miopenTensorMhaDescaleDO];
        const auto& mhaDescaleDS = tensors[miopenTensorMhaDescaleDS];

        const auto& mhaScaleS = tensors[miopenTensorMhaScaleS];

        const auto& mhaScaleDS = tensors[miopenTensorMhaScaleDS];
        const auto& mhaScaleDQ = tensors[miopenTensorMhaScaleDQ];
        const auto& mhaScaleDK = tensors[miopenTensorMhaScaleDK];
        const auto& mhaScaleDV = tensors[miopenTensorMhaScaleDV];

        const auto& mhadp = tensors[miopenTensorMhaDropoutProbability];
        const auto& mhads = tensors[miopenTensorMhaDropoutSeed];
        const auto& mhado = tensors[miopenTensorMhaDropoutOffset];

        mha::MhaInputDescsBackward inputDescs = {mhaK->tensorFloat.desc,
                                                 mhaQ->tensorFloat.desc,
                                                 mhaV->tensorFloat.desc,
                                                 mhaO->tensorFloat.desc,
                                                 mhaDO->tensorFloat.desc,
                                                 mhaM->tensorFloat.desc,
                                                 mhaZInv->tensorFloat.desc,
                                                 mhaDescaleK->tensorFloat.desc,
                                                 mhaDescaleQ->tensorFloat.desc,
                                                 mhaDescaleV->tensorFloat.desc,
                                                 mhaDescaleS->tensorFloat.desc,
                                                 mhaDescaleO->tensorFloat.desc,
                                                 mhaDescaleDO->tensorFloat.desc,
                                                 mhaDescaleDS->tensorFloat.desc,
                                                 mhaScaleS->tensorFloat.desc,
                                                 mhaScaleDS->tensorFloat.desc,
                                                 mhaScaleDQ->tensorFloat.desc,
                                                 mhaScaleDK->tensorFloat.desc,
                                                 mhaScaleDV->tensorFloat.desc,
                                                 scale,
                                                 mhadp->tensorFloat.desc,
                                                 mhads->tensorFloat.desc,
                                                 mhado->tensorFloat.desc,
                                                 tensors[miopenTensorMhaDQ]->tensorFloat.desc,
                                                 tensors[miopenTensorMhaDK]->tensorFloat.desc,
                                                 tensors[miopenTensorMhaDV]->tensorFloat.desc,
                                                 tensors[miopenTensorMhaAmaxDQ]->tensorFloat.desc,
                                                 tensors[miopenTensorMhaAmaxDK]->tensorFloat.desc,
                                                 tensors[miopenTensorMhaAmaxDV]->tensorFloat.desc,
                                                 tensors[miopenTensorMhaAmaxDS]->tensorFloat.desc};

        const mha::ProblemDescription problem_description = {inputDescs};

        const auto invoke_ctx = [&]() -> AnyInvokeParams {
            mha::MhaDataBackward dataBackward = {
                mhaK->gpuBuffer.get(),
                mhaQ->gpuBuffer.get(),
                mhaV->gpuBuffer.get(),
                mhaO->gpuBuffer.get(),
                mhaDO->gpuBuffer.get(),
                mhaM->gpuBuffer.get(),
                mhaZInv->gpuBuffer.get(),
                mhaDescaleK->gpuBuffer.get(),
                mhaDescaleQ->gpuBuffer.get(),
                mhaDescaleV->gpuBuffer.get(),
                mhaDescaleS->gpuBuffer.get(),
                mhaDescaleO->gpuBuffer.get(),
                mhaDescaleDO->gpuBuffer.get(),
                mhaDescaleDS->gpuBuffer.get(),
                mhaScaleS->gpuBuffer.get(),
                mhaScaleDS->gpuBuffer.get(),
                mhaScaleDQ->gpuBuffer.get(),
                mhaScaleDK->gpuBuffer.get(),
                mhaScaleDV->gpuBuffer.get(),
                mhadp->gpuBuffer.get(),
                mhads->gpuBuffer.get(),
                mhado->gpuBuffer.get(),
                outputResultsMap[miopenTensorMhaDQ]->gpuBuffer.get(),
                outputResultsMap[miopenTensorMhaDK]->gpuBuffer.get(),
                outputResultsMap[miopenTensorMhaDV]->gpuBuffer.get(),
                outputResultsMap[miopenTensorMhaAmaxDQ]->gpuBuffer.get(),
                outputResultsMap[miopenTensorMhaAmaxDK]->gpuBuffer.get(),
                outputResultsMap[miopenTensorMhaAmaxDV]->gpuBuffer.get(),
                outputResultsMap[miopenTensorMhaAmaxDS]->gpuBuffer.get()};

            return mha::InvokeParams(dataBackward, workspace.ptr(), workspace.size());
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

            solver::mha::MhaBackward mhaBackward;

            const auto mha_solution = mhaBackward.GetSolution(ctx, problem_description);

            decltype(auto) invoker = handle.PrepareInvoker(*mha_solution.invoker_factory,
                                                           mha_solution.construction_params);
            handle.RegisterInvoker(invoker, net_cfg, solver::Id(solver_id).ToString());
            invoker(handle, invoke_ctx);
        }

        ReadData(handle, outputResultsMap);
    }

    void ReadData(Handle& handle, TensorStructMap& outputResultsMap)
    {
        for(const auto& it : outputResultsMap)
        {
            it.second->tensorFloat.data =
                handle.Read<float>(it.second->gpuBuffer, it.second->tensorFloat.data.size());
        }
    }

private:
    TensorStructMap tensors;

    MhaDescriptor mha_descriptor;

    miopenProblem_t problem;

    bool isForward;

    const unsigned int test_n = 2;
    const unsigned int test_h = 4;
    const unsigned int test_s = 8;
    const unsigned int test_d = 16;

    float scale = 1.0f;
};

TEST(TestMhaFind20, MhaForward)
{
    Handle& handle = get_handle();

    MhaFind20Test test(true);

    std::vector<miopenSolution_t> solutions = test.TestFindSolutions(handle);
    test.TestSolutionAttributes(solutions);

    test.TestRunSolutions(handle, solutions);
    test.Finalize();
}

TEST(TestMhaFind20, MhaBackward)
{
    Handle& handle = get_handle();

    MhaFind20Test test(false);

    std::vector<miopenSolution_t> solutions = test.TestFindSolutions(handle);
    test.TestSolutionAttributes(solutions);

    test.TestRunSolutions(handle, solutions);
    test.Finalize();
}
