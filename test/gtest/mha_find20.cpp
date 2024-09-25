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
#include <variant>

using namespace miopen;

// support only two types for now
namespace {
enum TensorType
{
    Float = 0,
    Int64 = 1
};
}

using TensorVariant = std::variant<tensor<float>, tensor<int64_t>>;
struct TensorStruct
{
    // support only two types for now - float and int64.
    TensorStruct(TensorType type, unsigned int n, unsigned int h, unsigned int s, unsigned int d)
    {
        switch(type)
        {
        case TensorType::Float: tensorVariant = tensor<float>(n, h, s, d); break;
        case TensorType::Int64: tensorVariant = tensor<int64_t>(n, h, s, d); break;
        default: assert(false); // not supported
        }
    }

    TensorStruct(const TensorVariant& var) { tensorVariant = var; }

    void GpuRead(Handle& handle)
    {
        if(std::holds_alternative<tensor<float>>(tensorVariant))
        {
            auto t = std::get<tensor<float>>(tensorVariant);
            t.data = handle.Read<float>(gpuBuffer, t.data.size());
        }
        else if(std::holds_alternative<tensor<int64_t>>(tensorVariant))
        {
            auto t = std::get<tensor<int64_t>>(tensorVariant);
            t.data = handle.Read<int64_t>(gpuBuffer, t.data.size());
        }
        else
        {
            assert(false);
        }
    }

    void GpuWrite(Handle& handle)
    {
        if(std::holds_alternative<tensor<float>>(tensorVariant))
        {
            auto t    = std::get<tensor<float>>(tensorVariant);
            gpuBuffer = handle.Write(t.data);
        }
        else if(std::holds_alternative<tensor<int64_t>>(tensorVariant))
        {
            auto t    = std::get<tensor<int64_t>>(tensorVariant);
            gpuBuffer = handle.Write(t.data);
        }
        else
        {
            assert(false);
        }
    }

    void InitWithRandom()
    {
        if(std::holds_alternative<tensor<float>>(tensorVariant))
        {
            auto t = std::get<tensor<float>>(tensorVariant);
            t.generate(tensor_elem_gen_integer{17});
        }
        else if(std::holds_alternative<tensor<int64_t>>(tensorVariant))
        {
            auto t = std::get<tensor<int64_t>>(tensorVariant);
            t.generate(tensor_elem_gen_integer{17});
        }
        else
        {
            assert(false);
        }
    }

    void InitWithFloatValue(float val)
    {
        assert(std::holds_alternative<tensor<float>>(tensorVariant));
        std::get<tensor<float>>(tensorVariant).generate([=](auto...) { return val; });
    }

    void InitWithInt64Value(int64_t val)
    {
        assert(std::holds_alternative<tensor<int64_t>>(tensorVariant));
        std::get<tensor<int64_t>>(tensorVariant).generate([=](auto...) { return val; });
    }

    // helper function for cases when we know exactly that given tensor is float
    tensor<float>& GetFloatTensor() { return std::get<tensor<float>>(tensorVariant); }

    TensorDescriptor& GetTensorDescriptor()
    {
        if(std::holds_alternative<tensor<float>>(tensorVariant))
        {
            return std::get<tensor<float>>(tensorVariant).desc;
        }
        else
        {
            return std::get<tensor<int64_t>>(tensorVariant).desc;
        }
    }

    TensorVariant tensorVariant;

    Allocator::ManageDataPtr gpuBuffer;
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

        unsigned int numberOfBuffers = numTensors + 1; // +1 for passing a scalar for mhaMask

        auto arguments = std::make_unique<miopenTensorArgument_t[]>(numberOfBuffers);

        std::vector<miopenTensorDescriptor_t> descVector(numTensors);

        int i = 0;
        for(const auto& it : tensors)
        {
            descVector[i] = &it.second->GetTensorDescriptor();

            it.second->GpuWrite(handle);

            arguments[i].id         = it.first;
            arguments[i].descriptor = &descVector[i];
            arguments[i].buffer     = it.second->gpuBuffer.get();

            ++i;
        }

        // Passing a scalar is a special case for current Find 2.0 implementation
        arguments[i].id         = miopenTensorMhaMask;
        arguments[i].descriptor = nullptr;
        arguments[i].buffer     = &mhaMask;

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
                                           numberOfBuffers,
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
                tensorStructPtr->GpuRead(handle);

                TensorStructPtr& ptr = outputTensorResults[id];

                ptr = std::make_unique<TensorStruct>(tensorStructPtr->tensorVariant);
                ptr->GpuWrite(handle);
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
                double error = miopen::rms_range(outputTensorResults[id]->GetFloatTensor().data,
                                                 tensors[id]->GetFloatTensor().data);
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

    TensorStruct& CreateTensor(miopenTensorArgumentId_t id,
                               unsigned int n = 1,
                               unsigned int h = 1,
                               unsigned int s = 1,
                               unsigned int d = 1)
    {
        tensors[id] = std::make_unique<TensorStruct>(
            IsInt64TensorId(id) ? TensorType::Int64 : TensorType::Float, n, h, s, d);

        EXPECT_EQUAL(
            miopenSetProblemTensorDescriptor(problem, id, &tensors[id]->GetTensorDescriptor()),
            miopenStatusSuccess);

        return *tensors[id];
    }

    void Initialize()
    {
        mha_descriptor.SetParams(scale);

        EXPECT_EQUAL(miopenCreateMhaProblem(&problem,
                                            &mha_descriptor,
                                            isForward ? miopenProblemDirectionForward
                                                      : miopenProblemDirectionBackward),
                     miopenStatusSuccess);

        CreateTensor(miopenTensorMhaK, test_n, test_h, test_s, test_d).InitWithRandom();

        CreateTensor(miopenTensorMhaV, test_n, test_h, test_s, test_d).InitWithRandom();

        CreateTensor(miopenTensorMhaDescaleK).InitWithFloatValue(1.0f);
        CreateTensor(miopenTensorMhaDescaleQ).InitWithFloatValue(1.0f);
        CreateTensor(miopenTensorMhaDescaleV).InitWithFloatValue(1.0f);
        CreateTensor(miopenTensorMhaDescaleS).InitWithFloatValue(1.0f);

        CreateTensor(miopenTensorMhaScaleS).InitWithFloatValue(1.0f);

        CreateTensor(miopenTensorMhaDropoutProbability).InitWithFloatValue(0.5f);
        CreateTensor(miopenTensorMhaDropoutSeed).InitWithInt64Value(0);
        CreateTensor(miopenTensorMhaDropoutOffset).InitWithInt64Value(0);

        CreateTensor(miopenTensorMhaBias, test_n, test_h, test_s, test_s).InitWithRandom();

        if(isForward)
        {
            CreateTensor(miopenTensorMhaQ, test_n, test_h, test_s, test_d).InitWithRandom();

            CreateTensor(miopenTensorMhaScaleO).InitWithFloatValue(1.0f);
            CreateTensor(miopenTensorMhaO, test_n, test_h, test_s, test_d);
            CreateTensor(miopenTensorMhaAmaxO);
            CreateTensor(miopenTensorMhaAmaxS);
            CreateTensor(miopenTensorMhaM, test_n, test_h, test_s, 1);
            CreateTensor(miopenTensorMhaZInv, test_n, test_h, test_s, 1);
        }
        else
        {
            CreateTensor(miopenTensorMhaQ, test_n, test_h, test_s, test_d).InitWithFloatValue(0.0f);

            CreateTensor(miopenTensorMhaO, test_n, test_h, test_s, test_d).InitWithRandom();

            CreateTensor(miopenTensorMhaDO, test_n, test_h, test_s, test_d).InitWithRandom();

            CreateTensor(miopenTensorMhaM, test_n, test_h, test_s, 1).InitWithFloatValue(0.0f);
            CreateTensor(miopenTensorMhaZInv, test_n, test_h, test_s, 1).InitWithFloatValue(1.0f);

            CreateTensor(miopenTensorMhaDescaleO).InitWithRandom();
            CreateTensor(miopenTensorMhaDescaleDO).InitWithRandom();
            CreateTensor(miopenTensorMhaDescaleDS).InitWithRandom();
            CreateTensor(miopenTensorMhaScaleDS).InitWithRandom();
            CreateTensor(miopenTensorMhaScaleDQ).InitWithRandom();
            CreateTensor(miopenTensorMhaScaleDK).InitWithRandom();
            CreateTensor(miopenTensorMhaScaleDV).InitWithRandom();

            CreateTensor(miopenTensorMhaDQ, test_n, test_h, test_s, test_d);
            CreateTensor(miopenTensorMhaDK, test_n, test_h, test_s, test_d);
            CreateTensor(miopenTensorMhaDV, test_n, test_h, test_s, test_d);
            CreateTensor(miopenTensorMhaAmaxDQ);
            CreateTensor(miopenTensorMhaAmaxDK);
            CreateTensor(miopenTensorMhaAmaxDV);
            CreateTensor(miopenTensorMhaAmaxDS);
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

        const auto& mhabias = tensors[miopenTensorMhaBias];

        mha::MhaInputDescsForward inputDescs = {
            mhaK->GetTensorDescriptor(),
            mhaQ->GetTensorDescriptor(),
            mhaV->GetTensorDescriptor(),
            mhaDescaleK->GetTensorDescriptor(),
            mhaDescaleQ->GetTensorDescriptor(),
            mhaDescaleV->GetTensorDescriptor(),
            mhaDescaleS->GetTensorDescriptor(),
            mhaScaleS->GetTensorDescriptor(),
            mhaScaleO->GetTensorDescriptor(),
            scale,
            mhadp->GetTensorDescriptor(),
            mhads->GetTensorDescriptor(),
            mhado->GetTensorDescriptor(),
            mhabias->GetTensorDescriptor(),
            tensors[miopenTensorMhaO]->GetTensorDescriptor(),
            tensors[miopenTensorMhaAmaxO]->GetTensorDescriptor(),
            tensors[miopenTensorMhaAmaxS]->GetTensorDescriptor(),
            tensors[miopenTensorMhaM]->GetTensorDescriptor(),
            tensors[miopenTensorMhaZInv]->GetTensorDescriptor()};

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
                mhabias->gpuBuffer.get(),
                mhaMask,
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

        mha::MhaInputDescsBackward inputDescs = {
            mhaK->GetTensorDescriptor(),
            mhaQ->GetTensorDescriptor(),
            mhaV->GetTensorDescriptor(),
            mhaO->GetTensorDescriptor(),
            mhaDO->GetTensorDescriptor(),
            mhaM->GetTensorDescriptor(),
            mhaZInv->GetTensorDescriptor(),
            mhaDescaleK->GetTensorDescriptor(),
            mhaDescaleQ->GetTensorDescriptor(),
            mhaDescaleV->GetTensorDescriptor(),
            mhaDescaleS->GetTensorDescriptor(),
            mhaDescaleO->GetTensorDescriptor(),
            mhaDescaleDO->GetTensorDescriptor(),
            mhaDescaleDS->GetTensorDescriptor(),
            mhaScaleS->GetTensorDescriptor(),
            mhaScaleDS->GetTensorDescriptor(),
            mhaScaleDQ->GetTensorDescriptor(),
            mhaScaleDK->GetTensorDescriptor(),
            mhaScaleDV->GetTensorDescriptor(),
            scale,
            mhadp->GetTensorDescriptor(),
            mhads->GetTensorDescriptor(),
            mhado->GetTensorDescriptor(),
            tensors[miopenTensorMhaDQ]->GetTensorDescriptor(),
            tensors[miopenTensorMhaDK]->GetTensorDescriptor(),
            tensors[miopenTensorMhaDV]->GetTensorDescriptor(),
            tensors[miopenTensorMhaAmaxDQ]->GetTensorDescriptor(),
            tensors[miopenTensorMhaAmaxDK]->GetTensorDescriptor(),
            tensors[miopenTensorMhaAmaxDV]->GetTensorDescriptor(),
            tensors[miopenTensorMhaAmaxDS]->GetTensorDescriptor()};

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
            it.second->GpuRead(handle);
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

    float scale             = 1.0f;
    miopenMhaMask_t mhaMask = miopenMhaMaskNone;
};

TEST(GPU_TestMhaFind20_FP32, MhaForward)
{
    Handle& handle = get_handle();

    MhaFind20Test test(true);

    std::vector<miopenSolution_t> solutions = test.TestFindSolutions(handle);
    test.TestSolutionAttributes(solutions);

    test.TestRunSolutions(handle, solutions);
    test.Finalize();
}

TEST(GPU_TestMhaFind20_FP32, MhaBackward)
{
    Handle& handle = get_handle();

    MhaFind20Test test(false);

    std::vector<miopenSolution_t> solutions = test.TestFindSolutions(handle);
    test.TestSolutionAttributes(solutions);

    test.TestRunSolutions(handle, solutions);
    test.Finalize();
}
