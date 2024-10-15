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
#include <miopen/config.h>

#include "get_handle.hpp"
#include "mha_helper.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "gtest_common.hpp"
#include "../workspace.hpp"
#include "../tensor_util.hpp"

#include <miopen/miopen.h>
#include <miopen/solution.hpp>

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <variant>
#include <vector>

using namespace miopen;

namespace {

struct TensorStruct
{
    template <typename T>
    TensorStruct(tensor<T>&& val) : m_cpu_tensor(std::move(val))
    {
    }

    TensorStruct(const TensorStruct&) = delete;
    TensorStruct& operator=(const TensorStruct&) = delete;

    ~TensorStruct() = default;

    std::variant<tensor<float>, tensor<float8>, tensor<bfloat8>, tensor<int64_t>> m_cpu_tensor;
    Allocator::ManageDataPtr m_gpu_buffer;
};

struct TestCase
{
    size_t n;
    size_t h;
    size_t s;
    size_t d;
    float dropout;
};

inline std::vector<TestCase> GetSmokeCases()
{
    return {
        {9, 8, 8, 8, 0.0f},
        {1, 2, 4, 5, 0.0f},
        {2, 1, 5, 4, 0.0f},
        {4, 2, 1, 3, 0.0f},
        {5, 3, 4, 1, 0.0f},
        {1, 2, 65, 5, 0.0f},
        {2, 1, 67, 4, 0.0f},
        {8, 7, 68, 1, 0.0f},
        {1, 2, 257, 5, 0.0f},
        {2, 1, 259, 4, 0.0f},
        {8, 7, 270, 1, 0.0f},
        {1, 1, 1, 1, 0.0f},
        {3, 5, 32, 7, 0.8f},
        {2, 2, 64, 128, 0.8f},
        {2, 1, 128, 4, 0.8f},
        {2, 7, 256, 31, 0.8f},
    };
}

inline std::vector<TestCase> GetFullTestCases()
{
    return {
        {3, 15, 2047, 15, 0.0f},
        {2049, 17, 7, 7, 0.0f},
        {3, 3, 257, 91, 0.0f},
        {11, 150, 255, 31, 0.0f},
        {9, 3, 129, 1023, 0.0f},
        {3, 15, 31, 2047, 0.0f},
        {2049, 17, 32, 7, 0.2f},
        {11, 150, 256, 31, 0.4f},
    };
}
} // namespace

template <typename T>
class Test_Bwd_Mha : public testing::TestWithParam<TestCase>
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, float8>);

protected:
    using dO_T = std::conditional_t<std::is_same_v<T, float>, float, bfloat8>;

    void SetUp() override
    {
        prng::reset_seed();
        auto [n, h, s, d, drop] = GetParam();
        Handle& handle          = get_handle();

        if((drop > 0.0f))
        {
            GTEST_SKIP() << "CPU Dropout for backward pass currently is not supprorted";
        }

        if((drop > 0.0f) && (s % handle.GetWavefrontWidth() != 0))
        {
            GTEST_SKIP() << "CPU Dropout currently supprorts only fully occupied warps";
        }

        mha_descriptor.SetParams(1);
        ASSERT_EQ(miopenCreateMhaProblem(&problem, &mha_descriptor, miopenProblemDirectionBackward),
                  miopenStatusSuccess);

        auto InitTensor = [this, &handle](miopenTensorArgumentId_t id, auto&& tensor) {
            auto tmp = std::make_unique<TensorStruct>(std::move(tensor));
            std::visit(
                [this, id, &handle, &gpu_buff = tmp->m_gpu_buffer](auto&& cpu_tensor) {
                    ASSERT_EQ(miopenSetProblemTensorDescriptor(problem, id, &cpu_tensor.desc),
                              miopenStatusSuccess);

                    gpu_buff = handle.Write(cpu_tensor.data);
                    descVector.push_back(&(cpu_tensor.desc));
                },
                tmp->m_cpu_tensor);

            args.emplace_back();
            args.back().id = id;
            // args.back().descriptor will be filled later
            args.back().buffer = tmp->m_gpu_buffer.get();

            // check that we don't try to create duplicates
            ASSERT_EQ(tensors.count(id), 0);

            tensors[id] = std::move(tmp);
        };

        auto q = test::cpu::GenScaledTensorBackward<T>(n, h, s, d);
        InitTensor(miopenTensorMhaQ, std::move(q.mTensor));

        auto k = test::cpu::GenScaledTensorBackward<T>(n, h, s, d);
        InitTensor(miopenTensorMhaK, std::move(k.mTensor));

        auto v = test::cpu::GenScaledTensorBackward<T>(n, h, s, d);
        InitTensor(miopenTensorMhaV, std::move(v.mTensor));

        float s_scale = 1.f;
        // clang-tidy complains about the same expression on both sides of "/": 1.f / 1.f
        float s_descale = 1.f; // / s_scale;

        float o_scale = 1.f;
        // clang-tidy complains about the same expression on both sides of "/": 1.f / 1.f
        float o_descale = 1.f; // / o_scale;

        InitTensor(miopenTensorMhaDescaleQ,
                   tensor<float>{1, 1, 1, 1}.generate([&q](auto...) { return q.mDescale; }));
        InitTensor(miopenTensorMhaDescaleK,
                   tensor<float>{1, 1, 1, 1}.generate([&k](auto...) { return k.mDescale; }));
        InitTensor(miopenTensorMhaDescaleV,
                   tensor<float>{1, 1, 1, 1}.generate([&v](auto...) { return v.mDescale; }));
        InitTensor(miopenTensorMhaDescaleS,
                   tensor<float>{1, 1, 1, 1}.generate([=](auto...) { return s_descale; }));
        InitTensor(miopenTensorMhaScaleS,
                   tensor<float>{1, 1, 1, 1}.generate([=](auto...) { return s_scale; }));

        InitTensor(miopenTensorMhaDropoutProbability,
                   tensor<float>{1, 1, 1, 1}.generate([rate = drop](auto...) { return rate; }));
        InitTensor(miopenTensorMhaDropoutSeed,
                   tensor<int64_t>{1, 1, 1, 1}.generate([](auto...) { return 0xAAFFFFFFFFull; }));
        InitTensor(miopenTensorMhaDropoutOffset,
                   tensor<int64_t>{1, 1, 1, 1}.generate([](auto...) { return 1; }));

        tensor<float> softmax  = tensor<float>{n, h, s, s};
        tensor<T> oDesc        = tensor<T>{n, h, s, d};
        tensor<float> mDesc    = tensor<float>{n, h, s, 1};
        tensor<float> zInvDesc = tensor<float>{n, h, s, 1};
        float amaxS;
        float amaxO;

        // proper O, M and zInv tensors are required for backward pass.
        // randomly generated M and zInv may lead to nan\inf values
        test::cpu::MultiHeadAttentionfp8(
            std::get<tensor<T>>(tensors[miopenTensorMhaQ]->m_cpu_tensor),
            std::get<tensor<T>>(tensors[miopenTensorMhaK]->m_cpu_tensor),
            std::get<tensor<T>>(tensors[miopenTensorMhaV]->m_cpu_tensor),
            softmax,
            mDesc,
            zInvDesc,
            q.mDescale,
            k.mDescale,
            v.mDescale,
            s_descale,
            s_scale,
            o_scale,
            drop,
            std::get<tensor<int64_t>>(tensors[miopenTensorMhaDropoutSeed]->m_cpu_tensor)
                .data.front(),
            std::get<tensor<int64_t>>(tensors[miopenTensorMhaDropoutOffset]->m_cpu_tensor)
                .data.front(),
            amaxS,
            amaxO,
            oDesc);

        auto dO = test::cpu::GenScaledTensorBackward<dO_T>(n, h, s, d);
        InitTensor(miopenTensorMhaDO, std::move(dO.mTensor));

        InitTensor(miopenTensorMhaO, std::move(oDesc));
        InitTensor(miopenTensorMhaM, std::move(mDesc));
        InitTensor(miopenTensorMhaZInv, std::move(zInvDesc));

        float dS_scale = 1.f;
        // clang-tidy complains about the same expression on both sides of "/": 1.f / 1.f
        float dS_descale = 1.f; // / dS_scale;

        float dQ_scale = 1.f;
        float dK_scale = 1.f;
        float dV_scale = 1.f;

        InitTensor(miopenTensorMhaDescaleO,
                   tensor<float>{1, 1, 1, 1}.generate([=](auto...) { return o_descale; }));
        InitTensor(miopenTensorMhaDescaleDO,
                   tensor<float>{1, 1, 1, 1}.generate([&dO](auto...) { return dO.mDescale; }));
        InitTensor(miopenTensorMhaDescaleDS,
                   tensor<float>{1, 1, 1, 1}.generate([=](auto...) { return dS_descale; }));
        InitTensor(miopenTensorMhaScaleDS,
                   tensor<float>{1, 1, 1, 1}.generate([=](auto...) { return dS_scale; }));
        InitTensor(miopenTensorMhaScaleDQ,
                   tensor<float>{1, 1, 1, 1}.generate([=](auto...) { return dQ_scale; }));
        InitTensor(miopenTensorMhaScaleDK,
                   tensor<float>{1, 1, 1, 1}.generate([=](auto...) { return dK_scale; }));
        InitTensor(miopenTensorMhaScaleDV,
                   tensor<float>{1, 1, 1, 1}.generate([=](auto...) { return dV_scale; }));

        InitTensor(miopenTensorMhaDQ, tensor<T>{n, h, s, d});
        InitTensor(miopenTensorMhaDK, tensor<T>{n, h, s, d});
        InitTensor(miopenTensorMhaDV, tensor<T>{n, h, s, d});
        InitTensor(miopenTensorMhaAmaxDQ, tensor<float>{1, 1, 1, 1});
        InitTensor(miopenTensorMhaAmaxDK, tensor<float>{1, 1, 1, 1});
        InitTensor(miopenTensorMhaAmaxDV, tensor<float>{1, 1, 1, 1});
        InitTensor(miopenTensorMhaAmaxDS, tensor<float>{1, 1, 1, 1});

        for(size_t i = 0; i < descVector.size(); ++i)
        {
            args[i].descriptor = &descVector[i];
        }

        dQDesc_ref = tensor<T>{n, h, s, d};
        dKDesc_ref = tensor<T>{n, h, s, d};
        dVDesc_ref = tensor<T>{n, h, s, d};

        test::cpu::MultiHeadAttentionBackwardDataf8(
            std::get<tensor<T>>(tensors[miopenTensorMhaQ]->m_cpu_tensor),
            std::get<tensor<T>>(tensors[miopenTensorMhaK]->m_cpu_tensor),
            std::get<tensor<T>>(tensors[miopenTensorMhaV]->m_cpu_tensor),
            std::get<tensor<T>>(tensors[miopenTensorMhaO]->m_cpu_tensor),
            std::get<tensor<dO_T>>(tensors[miopenTensorMhaDO]->m_cpu_tensor),
            softmax,
            q.mDescale,
            k.mDescale,
            v.mDescale,
            dQ_scale,
            dK_scale,
            dV_scale,
            s_scale,
            s_descale,
            dS_scale,
            dS_descale,
            o_descale,
            dO.mDescale,
            amax_dS_ref,
            amax_dQ_ref,
            amax_dK_ref,
            amax_dV_ref,
            dQDesc_ref,
            dKDesc_ref,
            dVDesc_ref);
    }

    void TestBody() override
    {
        Handle& handle = get_handle();

        auto FindSolutions = [&handle](miopenProblem_t problem_) {
            std::size_t found;
            std::vector<miopenSolution_t> solutions(16);
            if(miopenFindSolutions(
                   &handle, problem_, nullptr, solutions.data(), &found, solutions.size()) !=
               miopenStatusSuccess)
            {
                found = 0;
            }

            solutions.resize(found);
            return solutions;
        };

        std::vector<miopenSolution_t> solutions = FindSolutions(problem);
        ASSERT_GT(solutions.size(), 0);

        size_t workspace_size = 0;
        Workspace workspace;

        auto GetResult = [this, &handle](miopenTensorArgumentId_t id, auto type) {
            using ResultT         = std::decay_t<decltype(type)>;
            auto& tensorStructPtr = tensors[id];
            auto& cpu_tensor      = std::get<tensor<ResultT>>(tensorStructPtr->m_cpu_tensor);

            cpu_tensor.data =
                handle.Read<ResultT>(tensorStructPtr->m_gpu_buffer, cpu_tensor.data.size());

            return cpu_tensor;
        };

        const double error_threshold     = 5e-5;
        const double fp8_error_threshold = (std::is_same_v<T, float8>) ? 3e-3 : error_threshold;

        auto checkAmax = [GetResult, error_threshold](
                             miopenTensorArgumentId_t id, std::string_view name, float refAmax) {
            const auto& resAmax = GetResult(id, float{});
            float amax_rel_diff = std::abs(refAmax - resAmax[0]);
            float divisor       = std::min(refAmax, resAmax[0]);
            amax_rel_diff /= divisor > std::numeric_limits<float>::min() ? divisor : 1.0f;
            EXPECT_LT(amax_rel_diff, error_threshold)
                << name << " ref: " << refAmax << " result: " << resAmax[0];
        };

        auto checkOutput = [GetResult, fp8_error_threshold](miopenTensorArgumentId_t id,
                                                            std::string_view name,
                                                            const auto& ref) {
            EXPECT_LT(miopen::rms_range(ref, GetResult(id, T{})), fp8_error_threshold) << name;
        };

        for(const auto& solution : solutions)
        {
            miopenGetSolutionWorkspaceSize(solution, &workspace_size);
            workspace.resize(workspace_size);

            ASSERT_EQ(
                miopenRunSolution(
                    &handle, solution, args.size(), args.data(), workspace.ptr(), workspace.size()),
                miopenStatusSuccess);

            checkAmax(miopenTensorMhaAmaxDQ, "amax dQ", amax_dQ_ref);
            checkAmax(miopenTensorMhaAmaxDK, "amax dK", amax_dK_ref);
            checkAmax(miopenTensorMhaAmaxDV, "amax dV", amax_dV_ref);
            checkAmax(miopenTensorMhaAmaxDS, "amax dS", amax_dS_ref);

            checkOutput(miopenTensorMhaDQ, "tensor dQ", dQDesc_ref);
            checkOutput(miopenTensorMhaDK, "tensor dK", dKDesc_ref);
            checkOutput(miopenTensorMhaDV, "tensor dV", dVDesc_ref);
        }
    }

    void TearDown() override
    {
        if(problem)
        {
            ASSERT_EQ(miopenDestroyProblem(problem), miopenStatusSuccess);
        }
    }

    std::map<miopenTensorArgumentId_t, std::unique_ptr<TensorStruct>> tensors;
    std::vector<miopenTensorDescriptor_t> descVector;
    std::vector<miopenTensorArgument_t> args;

    tensor<T> dQDesc_ref;
    tensor<T> dKDesc_ref;
    tensor<T> dVDesc_ref;
    float amax_dQ_ref;
    float amax_dK_ref;
    float amax_dV_ref;
    float amax_dS_ref;

    MhaDescriptor mha_descriptor;
    miopenProblem_t problem = nullptr;
};

class GPU_Bwd_Mha_FP32 : public Test_Bwd_Mha<float>
{
};

class GPU_Bwd_Mha_FP8 : public Test_Bwd_Mha<float8>
{
    void SetUp() override
    {
        using e_mask = enabled<Gpu::gfx94X>;
        using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx90A>;
        if(!IsTestSupportedForDevMask<d_mask, e_mask>() || MIOPEN_FP8_IEEE_EXPONENT_BIAS != 0)
        {
            GTEST_SKIP() << "FP8 is unsupported on this HW";
        }

        Test_Bwd_Mha<float8>::SetUp();
    }
};

TEST_P(GPU_Bwd_Mha_FP32, Test_float) { return Test_Bwd_Mha<float>::TestBody(); };

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Bwd_Mha_FP32, testing::ValuesIn(GetSmokeCases()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_Bwd_Mha_FP32, testing::ValuesIn(GetFullTestCases()));
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GPU_Bwd_Mha_FP32);

TEST_P(GPU_Bwd_Mha_FP8, Test_float) { return Test_Bwd_Mha<float8>::TestBody(); };

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Bwd_Mha_FP8, testing::ValuesIn(GetSmokeCases()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_Bwd_Mha_FP8, testing::ValuesIn(GetFullTestCases()));
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GPU_Bwd_Mha_FP8);
