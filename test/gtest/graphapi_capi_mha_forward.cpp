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

#include "graphapi_capi_mha_common.hpp"

using namespace capi_test_mha_common;

template <typename T>
class MhaForwardTest : public MhaCommonTest
{
protected:
    virtual void MakeRealTensorsAndFillData(miopen::Handle& handle) override
    {
        // We use identifiers from Find 2.0 enum to have sopmething unique for the test purposes
        MakeAndAddRealTensorDescriptor(
            miopenTensorMhaQ, m_testN, m_testH, m_testS, m_testD, GetMainType<T>());
        MakeAndAddRealTensorDescriptor(miopenTensorMhaK,
                                       m_testN,
                                       m_testH,
                                       m_testS,
                                       m_testD,
                                       GetMainType<T>(),
                                       false); // no transpose for now

        MakeAndAddRealTensorDescriptor(
            miopenTensorMhaV, m_testN, m_testH, m_testS, m_testD, GetMainType<T>());
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleK);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleQ);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleV);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleS);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaScaleS);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaScaleO);

        MakeAndAddRealTensorDescriptor(miopenTensorMhaDropoutProbability);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDropoutSeed, 1, 1, 1, 1, miopenInt64, false);
        MakeAndAddRealTensorDescriptor(
            miopenTensorMhaDropoutOffset, 1, 1, 1, 1, miopenInt64, false);

        // output real tensors
        MakeAndAddRealTensorDescriptor(
            miopenTensorMhaO, m_testN, m_testH, m_testS, m_testD, GetMainType<T>());
        MakeAndAddRealTensorDescriptor(miopenTensorMhaAmaxO);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaAmaxS);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaM, m_testN, m_testH, m_testS, 1);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaZInv, m_testN, m_testH, m_testS, 1);

        InitTensorValues(handle);

        // get next value for the rest of the tensors (which don't have any particular enum value)
        m_nextTensorId++;
    }

    void InitTensorValues(miopen::Handle& handle)
    {
        using namespace test::cpu;

        auto Q = GenScaledTensor<T>(m_testN, m_testH, m_testS, m_testD);
        auto K = GenScaledTensor<T>(m_testN, m_testH, m_testS, m_testD);
        auto V = GenScaledTensor<T>(m_testN, m_testH, m_testS, m_testD);

        for(auto& [k, v] : m_realTensorMap)
        {
            if(k == miopenTensorMhaQ)
            {
                v->InitAndWriteToGPU(handle, std::move(Q.mTensor));
            }
            else if(k == miopenTensorMhaDescaleQ)
            {
                v->InitAndWriteToGPU(handle, Q.mDescale);
            }
            else if(k == miopenTensorMhaK)
            {
                v->InitAndWriteToGPU(handle, std::move(K.mTensor));
            }
            else if(k == miopenTensorMhaDescaleK)
            {
                v->InitAndWriteToGPU(handle, K.mDescale);
            }
            else if(k == miopenTensorMhaV)
            {
                v->InitAndWriteToGPU(handle, std::move(V.mTensor));
            }
            else if(k == miopenTensorMhaDescaleV)
            {
                v->InitAndWriteToGPU(handle, V.mDescale);
            }
            else if(k == miopenTensorMhaScaleO || k == miopenTensorMhaScaleS ||
                    k == miopenTensorMhaDescaleS)
            {
                v->InitAndWriteToGPU(handle, 1.0f);
            }
            else if(k == miopenTensorMhaDropoutProbability)
            {
                v->InitAndWriteToGPU(handle, m_bernulliProbability);
            }
            else if(k == miopenTensorMhaDropoutSeed)
            {
                v->InitAndWriteToGPU(handle, static_cast<int64_t>(0));
            }
            else if(k == miopenTensorMhaDropoutOffset)
            {
                v->InitAndWriteToGPU(handle, static_cast<int64_t>(0));
            }
            else if(k == miopenTensorMhaM || k == miopenTensorMhaZInv ||
                    k == miopenTensorMhaAmaxO || k == miopenTensorMhaAmaxS)
            {
                // these are outputs
                v->InitAndWriteToGPU(handle, 0.0f);
            }
            else if(k == miopenTensorMhaO)
            {
                v->InitAndWriteToGPU(handle, static_cast<T>(0.0f));
            }
            else
            {
                FAIL() << "Uninitialized input or output: " << k;
            }
        }
    }

    virtual void MakeVirtualTensorsAndNodes() override
    {
        // virtual tensors
        auto tMM0 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);
        auto pwS0 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);
        auto pwS1 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);
        auto pwS2 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);

        auto tSub   = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);
        auto tExp   = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);
        auto tSum   = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, 1);
        auto tMult0 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);
        auto tRnd   = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);
        auto tMult1 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);
        auto pwS3   = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);
        auto pwS4 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS, GetMainType<T>());

        auto tMM1 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testD);
        auto pwS5 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testD);
        auto pwS6 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testD);

        // Node creation
        MakeMatmul(m_realTensorMap[miopenTensorMhaQ]->m_gapiDesc,
                   m_realTensorMap[miopenTensorMhaK]->m_gapiDesc,
                   tMM0);

        MakePointwise(MIOPEN_POINTWISE_IDENTITY, tMM0, nullptr, pwS0, true, m_attentionScale);
        MakePointwise(
            MIOPEN_POINTWISE_MUL, pwS0, m_realTensorMap[miopenTensorMhaDescaleQ]->m_gapiDesc, pwS1);
        MakePointwise(
            MIOPEN_POINTWISE_MUL, pwS1, m_realTensorMap[miopenTensorMhaDescaleK]->m_gapiDesc, pwS2);

        MakeReduction(
            MIOPEN_REDUCE_TENSOR_MAX, pwS2, m_realTensorMap[miopenTensorMhaM]->m_gapiDesc);
        MakePointwise(
            MIOPEN_POINTWISE_SUB, pwS2, m_realTensorMap[miopenTensorMhaM]->m_gapiDesc, tSub);
        MakePointwise(MIOPEN_POINTWISE_EXP, tSub, DescriptorWrapperPtr(), tExp);
        MakeReduction(MIOPEN_REDUCE_TENSOR_ADD, tExp, tSum);
        MakePointwise(MIOPEN_POINTWISE_RECIPROCAL,
                      tSum,
                      DescriptorWrapperPtr(),
                      m_realTensorMap[miopenTensorMhaZInv]->m_gapiDesc);
        MakePointwise(
            MIOPEN_POINTWISE_MUL, tExp, m_realTensorMap[miopenTensorMhaZInv]->m_gapiDesc, tMult0);

        MakeReduction(
            MIOPEN_REDUCE_TENSOR_MAX, tMult0, m_realTensorMap[miopenTensorMhaAmaxS]->m_gapiDesc);

        MakeRNG(m_bernulliProbability,
                m_realTensorMap[miopenTensorMhaDropoutSeed]->m_gapiDesc,
                m_realTensorMap[miopenTensorMhaDropoutOffset]->m_gapiDesc,
                tRnd);

        MakePointwise(MIOPEN_POINTWISE_MUL, tMult0, tRnd, tMult1);
        MakePointwise(MIOPEN_POINTWISE_MUL,
                      tMult1,
                      m_realTensorMap[miopenTensorMhaDropoutProbability]->m_gapiDesc,
                      pwS3);
        MakePointwise(
            MIOPEN_POINTWISE_MUL, pwS3, m_realTensorMap[miopenTensorMhaScaleS]->m_gapiDesc, pwS4);

        MakeMatmul(pwS4, m_realTensorMap[miopenTensorMhaV]->m_gapiDesc, tMM1);
        MakePointwise(
            MIOPEN_POINTWISE_MUL, tMM1, m_realTensorMap[miopenTensorMhaDescaleS]->m_gapiDesc, pwS5);
        MakePointwise(
            MIOPEN_POINTWISE_MUL, pwS5, m_realTensorMap[miopenTensorMhaDescaleV]->m_gapiDesc, pwS6);
        MakeReduction(
            MIOPEN_REDUCE_TENSOR_MAX, pwS6, m_realTensorMap[miopenTensorMhaAmaxO]->m_gapiDesc);
        MakePointwise(MIOPEN_POINTWISE_MUL,
                      pwS6,
                      m_realTensorMap[miopenTensorMhaScaleO]->m_gapiDesc,
                      m_realTensorMap[miopenTensorMhaO]->m_gapiDesc);
    }

    virtual void RunCPUverify(miopen::Handle& handle) override
    {
        auto softmaxRef  = tensor<float>{m_testN, m_testH, m_testS, m_testS};
        auto oDescRef    = tensor<T>{m_testN, m_testH, m_testS, m_testD};
        auto mDescRef    = tensor<float>{m_testN, m_testH, m_testS, 1};
        auto zInvDescRef = tensor<float>{m_testN, m_testH, m_testS, 1};
        float amaxSRef   = 0;
        float amaxORef   = 0;

        auto lookup = [this](const int64_t id) -> TensorVariant& {
            auto it = m_realTensorMap.find(id);
            assert(it != m_realTensorMap.cend());
            return it->second->m_tensorVariant;
        };

        test::cpu::MultiHeadAttentionForwardfp8(
            GetTensor<T>(lookup(miopenTensorMhaQ)),
            GetTensor<T>(lookup(miopenTensorMhaK)),
            GetTensor<T>(lookup(miopenTensorMhaV)),
            softmaxRef,
            mDescRef,
            zInvDescRef,
            GetTensor<float>(lookup(miopenTensorMhaDescaleQ))[0],
            GetTensor<float>(lookup(miopenTensorMhaDescaleK))[0],
            GetTensor<float>(lookup(miopenTensorMhaDescaleV))[0],
            GetTensor<float>(lookup(miopenTensorMhaDescaleS))[0],
            GetTensor<float>(lookup(miopenTensorMhaScaleS))[0],
            GetTensor<float>(lookup(miopenTensorMhaScaleO))[0],
            GetTensor<float>(lookup(miopenTensorMhaDropoutProbability))[0],
            GetTensor<int64_t>(lookup(miopenTensorMhaDropoutSeed))[0],
            GetTensor<int64_t>(lookup(miopenTensorMhaDropoutOffset))[0],
            amaxSRef,
            amaxORef,
            oDescRef);

        const double errorThreshold      = 5e-6;
        const double typedErrorThreshold = (std::is_same_v<T, float8>) ? 2e-4 : errorThreshold;

        const auto& resAmaxS = GetResult<float>(miopenTensorMhaAmaxS, handle);
        auto amaxSAbsDiff    = std::abs(amaxSRef - resAmaxS[0]);
        EXPECT_LT(amaxSAbsDiff, errorThreshold)
            << " ref: " << amaxSRef << " result: " << resAmaxS[0];

        const auto& resAmaxO = GetResult<float>(miopenTensorMhaAmaxO, handle);
        auto amaxOAbsDiff    = std::abs(amaxORef - resAmaxO[0]);
        EXPECT_LT(amaxOAbsDiff, errorThreshold)
            << " ref: " << amaxORef << " result: " << resAmaxO[0];

        double mError = miopen::rms_range(mDescRef, GetResult<float>(miopenTensorMhaM, handle));
        EXPECT_LT(mError, errorThreshold);

        double zInvError =
            miopen::rms_range(zInvDescRef, GetResult<float>(miopenTensorMhaZInv, handle));
        EXPECT_LT(zInvError, errorThreshold);

        auto oRes = GetResult<T>(miopenTensorMhaO, handle);

        double oError = miopen::rms_range(oDescRef, oRes);
        EXPECT_LT(oError, typedErrorThreshold);
    }
};

class GPU_MhaForward_FP32 : public MhaForwardTest<float>
{
};

class GPU_MhaForward_FP8 : public MhaForwardTest<float8>
{
    void SetUp() override
    {
        using e_mask = enabled<Gpu::gfx94X>;
        using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx90A>;
        if(!IsTestSupportedForDevMask<d_mask, e_mask>() || MIOPEN_FP8_IEEE_EXPONENT_BIAS != 0)
        {
            GTEST_SKIP() << "FP8 is unsupported on this HW";
        }

        MhaForwardTest<float8>::SetUp();
    }
};

class GPU_MhaForward_FP16 : public MhaForwardTest<half_float::half>
{
    void SetUp() override
    {
        if(!IsTestSupportedByDevice(Gpu::gfx90A | Gpu::gfx94X))
        {
            GTEST_SKIP() << "FP16 is unsupported on this HW";
        }

        MhaForwardTest<half_float::half>::SetUp();

        if(m_bernulliProbability != 0.0f)
        {
            GTEST_SKIP() << "Dropout not currently supported for FP16";
        }
    }

    void RunCPUverify(miopen::Handle& handle) override
    {
        auto softmaxRef  = tensor<float>{m_testN, m_testH, m_testS, m_testS};
        auto oDescRef    = tensor<half_float::half>{m_testN, m_testH, m_testS, m_testD};
        auto mDescRef    = tensor<float>{m_testN, m_testH, m_testS, 1};
        auto zInvDescRef = tensor<float>{m_testN, m_testH, m_testS, 1};

        auto lookup = [this](const int64_t id) -> TensorVariant& {
            auto it = m_realTensorMap.find(id);
            assert(it != m_realTensorMap.cend());
            return it->second->m_tensorVariant;
        };

        test::cpu::MultiHeadAttentionForwardfp16(
            GetTensor<half_float::half>(lookup(miopenTensorMhaQ)),
            GetTensor<half_float::half>(lookup(miopenTensorMhaK)),
            GetTensor<half_float::half>(lookup(miopenTensorMhaV)),
            softmaxRef,
            mDescRef,
            zInvDescRef,
            oDescRef);

        const double errorThreshold = 4e-4;

        auto oRes     = GetResult<half_float::half>(miopenTensorMhaO, handle);
        double oError = miopen::rms_range(oDescRef, oRes);
        EXPECT_LT(oError, errorThreshold);
    }
};

TEST_P(GPU_MhaForward_FP32, TestFloat) { Run(); }
TEST_P(GPU_MhaForward_FP16, TestFloat16) { Run(); }
TEST_P(GPU_MhaForward_FP8, TestFloat8) { Run(); }

inline auto GetCases()
{
    return testing::Combine(testing::ValuesIn({2}),           // n
                            testing::ValuesIn({4}),           // h
                            testing::ValuesIn({64}),          // s
                            testing::ValuesIn({16}),          // d
                            testing::ValuesIn({0.0f, 0.5f})); // bernulli probability
}

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_MhaForward_FP32, GetCases());

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_MhaForward_FP16, GetCases());

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_MhaForward_FP8, GetCases());
