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

    std::variant<tensor<float>, tensor<float8>, tensor<int64_t>> m_cpu_tensor;
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
class Test_Fwd_Mha : public testing::TestWithParam<TestCase>
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, float8>);

protected:
    void SetUp() override
    {
        prng::reset_seed();
        auto [n, h, s, d, drop] = GetParam();
        Handle& handle          = get_handle();

        if((drop > 0.0f) && (s % handle.GetWavefrontWidth() != 0))
        {
            GTEST_SKIP() << "CPU Dropout currently supports only fully occupied warps";
        }

        mha_descriptor.SetParams(1);
        ASSERT_EQ(miopenCreateMhaProblem(&problem, &mha_descriptor, miopenProblemDirectionForward),
                  miopenStatusSuccess);

        auto InitTensor = [this, &handle](miopenTensorArgumentId_t id, auto&& tensor) {
            auto tmp = std::make_unique<TensorStruct>(std::move(tensor));
            std::visit(
                [this, id, &handle, &gpu_buff = tmp->m_gpu_buffer](auto&& cpu_tensor) {
                    EXPECT_EQ(miopenSetProblemTensorDescriptor(problem, id, &cpu_tensor.desc),
                              miopenStatusSuccess);

                    gpu_buff = handle.Write(cpu_tensor.data);
                    descVector.push_back(&(cpu_tensor.desc));
                },
                tmp->m_cpu_tensor);

            args.emplace_back();
            args.back().id = id;
            // args.back().descriptor will be filled later
            args.back().buffer = tmp->m_gpu_buffer.get();

            tensors[id] = std::move(tmp);
        };

        auto q = test::cpu::GenScaledTensor<T>(n, h, s, d);
        InitTensor(miopenTensorMhaQ, std::move(q.mTensor));

        auto k = test::cpu::GenScaledTensor<T>(n, h, s, d);
        InitTensor(miopenTensorMhaK, std::move(k.mTensor));

        auto v = test::cpu::GenScaledTensor<T>(n, h, s, d);
        InitTensor(miopenTensorMhaV, std::move(v.mTensor));

        float s_scale = 1.f;
        // clang-tidy complains about the same expression on both sides of "/": 1.f / 1.f
        float s_descale = 1.f; // / s_scale;

        float o_scale = 1.f;
        // clang-tidy complains about the same expression on both sides of "/": 1.f / 1.f

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
        InitTensor(miopenTensorMhaScaleO,
                   tensor<float>{1, 1, 1, 1}.generate([=](auto...) { return o_scale; }));

        InitTensor(miopenTensorMhaDropoutProbability,
                   tensor<float>{1, 1, 1, 1}.generate([rate = drop](auto...) { return rate; }));
        InitTensor(miopenTensorMhaDropoutSeed,
                   tensor<int64_t>{1, 1, 1, 1}.generate([](auto...) { return 0xAAFFFFFFFFull; }));
        InitTensor(miopenTensorMhaDropoutOffset,
                   tensor<int64_t>{1, 1, 1, 1}.generate([](auto...) { return 1; }));

        InitTensor(miopenTensorMhaO, tensor<T>{n, h, s, d});
        InitTensor(miopenTensorMhaAmaxO, tensor<float>{1, 1, 1, 1});
        InitTensor(miopenTensorMhaAmaxS, tensor<float>{1, 1, 1, 1});
        InitTensor(miopenTensorMhaM, tensor<float>{n, h, s, 1});
        InitTensor(miopenTensorMhaZInv, tensor<float>{n, h, s, 1});

        for(size_t i = 0; i < descVector.size(); ++i)
        {
            args[i].descriptor = &descVector[i];
        }

        softmax_ref  = tensor<float>{n, h, s, s};
        oDesc_ref    = tensor<T>{n, h, s, d};
        mDesc_ref    = tensor<float>{n, h, s, 1};
        zInvDesc_ref = tensor<float>{n, h, s, 1};

        test::cpu::MultiHeadAttentionfp8(
            std::get<tensor<T>>(tensors[miopenTensorMhaQ]->m_cpu_tensor),
            std::get<tensor<T>>(tensors[miopenTensorMhaK]->m_cpu_tensor),
            std::get<tensor<T>>(tensors[miopenTensorMhaV]->m_cpu_tensor),
            softmax_ref,
            mDesc_ref,
            zInvDesc_ref,
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
            amaxS_ref,
            amaxO_ref,
            oDesc_ref);
    }

    void TestBody() override
    {
        Handle& handle = get_handle();

        std::vector<miopenSolution_t> solutions(16);
        std::size_t found;

        ASSERT_EQ(miopenFindSolutions(
                      &handle, problem, nullptr, solutions.data(), &found, solutions.size()),
                  miopenStatusSuccess);
        ASSERT_GT(found, 0);
        solutions.resize(found);

        size_t workspace_size = 0;
        Workspace workspace;

        for(const auto& solution : solutions)
        {
            miopenGetSolutionWorkspaceSize(solution, &workspace_size);
            workspace.resize(workspace_size);

            ASSERT_EQ(
                miopenRunSolution(
                    &handle, solution, args.size(), args.data(), workspace.ptr(), workspace.size()),
                miopenStatusSuccess);

            auto GetResult = [this, &handle](miopenTensorArgumentId_t id, auto type) {
                using ResultT         = std::decay_t<decltype(type)>;
                auto& tensorStructPtr = tensors[id];
                auto& cpu_tensor      = std::get<tensor<ResultT>>(tensorStructPtr->m_cpu_tensor);

                cpu_tensor.data =
                    handle.Read<ResultT>(tensorStructPtr->m_gpu_buffer, cpu_tensor.data.size());

                return cpu_tensor;
            };

            const double error_threshold     = 5e-6;
            const double fp8_error_threshold = (std::is_same_v<T, float8>) ? 2e-4 : error_threshold;

            const auto& resAmaxS = GetResult(miopenTensorMhaAmaxS, float{});
            auto amaxS_abs_diff  = std::abs(amaxS_ref - resAmaxS[0]);
            EXPECT_LT(amaxS_abs_diff, error_threshold)
                << " ref: " << amaxS_ref << " result: " << resAmaxS[0];

            const auto& resAmaxO = GetResult(miopenTensorMhaAmaxO, float{});
            auto amaxO_abs_diff  = std::abs(amaxO_ref - resAmaxO[0]);
            EXPECT_LT(amaxO_abs_diff, error_threshold)
                << " ref: " << amaxO_ref << " result: " << resAmaxO[0];

            double M_error = miopen::rms_range(mDesc_ref, GetResult(miopenTensorMhaM, float{}));
            EXPECT_LT(M_error, error_threshold);

            double ZInv_error =
                miopen::rms_range(zInvDesc_ref, GetResult(miopenTensorMhaZInv, float{}));
            EXPECT_LT(ZInv_error, error_threshold);

            double O_error = miopen::rms_range(oDesc_ref, GetResult(miopenTensorMhaO, T{}));
            EXPECT_LT(O_error, fp8_error_threshold);
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

    // ref data
    tensor<float> softmax_ref;
    tensor<T> oDesc_ref;
    tensor<float> mDesc_ref;
    tensor<float> zInvDesc_ref;
    float amaxS_ref;
    float amaxO_ref;

    MhaDescriptor mha_descriptor;
    miopenProblem_t problem = nullptr;
};

class GPU_Fwd_Mha_FP32 : public Test_Fwd_Mha<float>
{
};

class GPU_Fwd_Mha_FP8 : public Test_Fwd_Mha<float8>
{
    void SetUp() override
    {
        using e_mask = enabled<Gpu::gfx94X>;
        using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx90A>;
        if(!IsTestSupportedForDevMask<d_mask, e_mask>() || MIOPEN_FP8_IEEE_EXPONENT_BIAS != 0)
        {
            GTEST_SKIP() << "FP8 is unsupported on this HW";
        }

        Test_Fwd_Mha<float8>::SetUp();
    }
};

TEST_P(GPU_Fwd_Mha_FP32, Test_float) { return Test_Fwd_Mha<float>::TestBody(); };

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Fwd_Mha_FP32, testing::ValuesIn(GetSmokeCases()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_Fwd_Mha_FP32, testing::ValuesIn(GetFullTestCases()));
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GPU_Fwd_Mha_FP32);

TEST_P(GPU_Fwd_Mha_FP8, Test_float) { return Test_Fwd_Mha<float8>::TestBody(); };

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Fwd_Mha_FP8, testing::ValuesIn(GetSmokeCases()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_Fwd_Mha_FP8, testing::ValuesIn(GetFullTestCases()));
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GPU_Fwd_Mha_FP8);
