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
class MhaBackwardTest : public MhaCommonTest
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, float8>);
    using dO_T = std::conditional_t<std::is_same_v<T, float>, float, bfloat8>;

protected:
    void SetUp() override
    {
        MhaCommonTest::SetUp();

        if((m_bernulliProbability > 0.0f))
        {
            GTEST_SKIP() << "CPU Dropout for backward pass currently is not supported";
        }
    }

    virtual void MakeRealTensorsAndFillData(miopen::Handle& handle) override
    {
        auto q = test::cpu::GenScaledTensorBackward<T>(m_testN, m_testH, m_testS, m_testD);
        auto k = test::cpu::GenScaledTensorBackward<T>(m_testN, m_testH, m_testS, m_testD);
        auto v = test::cpu::GenScaledTensorBackward<T>(m_testN, m_testH, m_testS, m_testD);

        MakeAndAddRealTensorDescriptor(
            miopenTensorMhaQ, m_testN, m_testH, m_testS, m_testD, GetMainType<T>())
            .InitAndWriteToGPU(handle, std::move(q.mTensor));

        MakeAndAddRealTensorDescriptor(
            miopenTensorMhaK, m_testN, m_testH, m_testS, m_testD, GetMainType<T>())
            .InitAndWriteToGPU(handle, std::move(k.mTensor));

        MakeAndAddRealTensorDescriptor(
            miopenTensorMhaV, m_testN, m_testH, m_testS, m_testD, GetMainType<T>())
            .InitAndWriteToGPU(handle, std::move(v.mTensor));

        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleQ)
            .InitAndWriteToGPU(handle, q.mDescale);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleK)
            .InitAndWriteToGPU(handle, k.mDescale);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleV)
            .InitAndWriteToGPU(handle, v.mDescale);

        float sScale   = 1.0f;
        float sDescale = 1.0f;

        float oScale   = 1.0f;
        float oDescale = 1.0f;

        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleS).InitAndWriteToGPU(handle, sDescale);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaScaleS).InitAndWriteToGPU(handle, sScale);

        MakeAndAddRealTensorDescriptor(miopenTensorMhaDropoutProbability)
            .InitAndWriteToGPU(handle, m_bernulliProbability);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDropoutSeed, 1, 1, 1, 1, miopenInt64)
            .InitAndWriteToGPU(handle, static_cast<int64_t>(0xAAFFFFFFFFUL));
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDropoutOffset, 1, 1, 1, 1, miopenInt64)
            .InitAndWriteToGPU(handle, static_cast<int64_t>(1));

        tensor<float> softmax  = tensor<float>{m_testN, m_testH, m_testS, m_testS};
        tensor<T> oDesc        = tensor<T>{m_testN, m_testH, m_testS, m_testD};
        tensor<float> mDesc    = tensor<float>{m_testN, m_testH, m_testS, 1};
        tensor<float> zInvDesc = tensor<float>{m_testN, m_testH, m_testS, 1};
        float amaxS;
        float amaxO;

        // proper O, M and zInv tensors are required for backward pass.
        // randomly generated M and zInv may lead to nan\inf values
        test::cpu::MultiHeadAttentionForwardfp8(
            GetTensor<T>(m_realTensorMap[miopenTensorMhaQ]->m_tensorVariant),
            GetTensor<T>(m_realTensorMap[miopenTensorMhaK]->m_tensorVariant),
            GetTensor<T>(m_realTensorMap[miopenTensorMhaV]->m_tensorVariant),
            softmax,
            mDesc,
            zInvDesc,
            q.mDescale,
            k.mDescale,
            v.mDescale,
            sDescale,
            sScale,
            oScale,
            m_bernulliProbability,
            GetTensor<int64_t>(m_realTensorMap[miopenTensorMhaDropoutSeed]->m_tensorVariant)
                .data.front(),
            GetTensor<int64_t>(m_realTensorMap[miopenTensorMhaDropoutOffset]->m_tensorVariant)
                .data.front(),
            amaxS,
            amaxO,
            oDesc);

        auto dO = test::cpu::GenScaledTensorBackward<dO_T>(m_testN, m_testH, m_testS, m_testD);
        MakeAndAddRealTensorDescriptor(
            miopenTensorMhaDO, m_testN, m_testH, m_testS, m_testD, GetMainType<dO_T>())
            .InitAndWriteToGPU(handle, std::move(dO.mTensor));

        MakeAndAddRealTensorDescriptor(
            miopenTensorMhaO, m_testN, m_testH, m_testS, m_testD, GetMainType<T>())
            .InitAndWriteToGPU(handle, std::move(oDesc));

        MakeAndAddRealTensorDescriptor(miopenTensorMhaM, m_testN, m_testH, m_testS, 1)
            .InitAndWriteToGPU(handle, std::move(mDesc));

        MakeAndAddRealTensorDescriptor(miopenTensorMhaZInv, m_testN, m_testH, m_testS, 1)
            .InitAndWriteToGPU(handle, std::move(zInvDesc));

        float dsScale   = 1.0f;
        float dsDescale = 1.0f;

        float dqScale = 1.0f;
        float dkScale = 1.0f;
        float dvScale = 1.0f;

        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleO).InitAndWriteToGPU(handle, oDescale);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleDO)
            .InitAndWriteToGPU(handle, dO.mDescale);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleDS)
            .InitAndWriteToGPU(handle, dsDescale);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaScaleDS).InitAndWriteToGPU(handle, dsScale);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaScaleDQ).InitAndWriteToGPU(handle, dqScale);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaScaleDK).InitAndWriteToGPU(handle, dkScale);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaScaleDV).InitAndWriteToGPU(handle, dvScale);

        MakeAndAddRealTensorDescriptor(
            miopenTensorMhaDQ, m_testN, m_testH, m_testS, m_testD, GetMainType<T>())
            .InitAndWriteToGPU(handle, static_cast<T>(0.0f));
        MakeAndAddRealTensorDescriptor(
            miopenTensorMhaDK, m_testN, m_testH, m_testS, m_testD, GetMainType<T>())
            .InitAndWriteToGPU(handle, static_cast<T>(0.0f));
        MakeAndAddRealTensorDescriptor(
            miopenTensorMhaDV, m_testN, m_testH, m_testS, m_testD, GetMainType<T>())
            .InitAndWriteToGPU(handle, static_cast<T>(0.0f));
        MakeAndAddRealTensorDescriptor(miopenTensorMhaAmaxDQ).InitAndWriteToGPU(handle, 0.0f);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaAmaxDK).InitAndWriteToGPU(handle, 0.0f);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaAmaxDV).InitAndWriteToGPU(handle, 0.0f);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaAmaxDS).InitAndWriteToGPU(handle, 0.0f);

        m_dQDescRef = tensor<T>{m_testN, m_testH, m_testS, m_testD};
        m_dKDescRef = tensor<T>{m_testN, m_testH, m_testS, m_testD};
        m_dVDescRef = tensor<T>{m_testN, m_testH, m_testS, m_testD};

        test::cpu::MultiHeadAttentionBackwardDataf8(
            GetTensor<T>(m_realTensorMap[miopenTensorMhaQ]->m_tensorVariant),
            GetTensor<T>(m_realTensorMap[miopenTensorMhaK]->m_tensorVariant),
            GetTensor<T>(m_realTensorMap[miopenTensorMhaV]->m_tensorVariant),
            GetTensor<T>(m_realTensorMap[miopenTensorMhaO]->m_tensorVariant),
            GetTensor<dO_T>(m_realTensorMap[miopenTensorMhaDO]->m_tensorVariant),
            softmax,
            q.mDescale,
            k.mDescale,
            v.mDescale,
            dqScale,
            dkScale,
            dvScale,
            sScale,
            sDescale,
            dsScale,
            dsDescale,
            oDescale,
            dO.mDescale,
            m_amaxDSRef,
            m_amaxDQRef,
            m_amaxDKRef,
            m_amaxDVRef,
            m_dQDescRef,
            m_dKDescRef,
            m_dVDescRef);

        // get next value for the rest of the tensors (which don't have any particular enum value)
        GetNextId();
    }

    virtual void MakeVirtualTensorsAndNodes() override
    {
        ////////////// Left part (column) of the Mha backward graph ///////////////
        ///////////////////////////////////////////////////////////////////////////

        auto mm0 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);

        auto kT = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testD, m_testS);
        MakeReshapeTranspose(m_realTensorMap[miopenTensorMhaK]->m_gapiDesc, kT);

        MakeMatmul(m_realTensorMap[miopenTensorMhaQ]->m_gapiDesc, kT, mm0);

        // MakePointwise function will automatically create a tensor for output,
        // if output is nullptr
        DescriptorWrapperPtr pwS0;
        MakePointwise(MIOPEN_POINTWISE_IDENTITY, mm0, nullptr, pwS0, true, m_attentionScale);

        DescriptorWrapperPtr pwS1;
        DescriptorWrapperPtr pwS2;
        MakePointwise(
            MIOPEN_POINTWISE_MUL, pwS0, m_realTensorMap[miopenTensorMhaDescaleQ]->m_gapiDesc, pwS1);
        MakePointwise(
            MIOPEN_POINTWISE_MUL, pwS1, m_realTensorMap[miopenTensorMhaDescaleK]->m_gapiDesc, pwS2);

        DescriptorWrapperPtr sub0;
        DescriptorWrapperPtr exp0;
        DescriptorWrapperPtr mult0;
        MakePointwise(
            MIOPEN_POINTWISE_SUB, pwS2, m_realTensorMap[miopenTensorMhaM]->m_gapiDesc, sub0);
        MakePointwise(MIOPEN_POINTWISE_EXP, sub0, DescriptorWrapperPtr(), exp0);
        MakePointwise(
            MIOPEN_POINTWISE_MUL, exp0, m_realTensorMap[miopenTensorMhaZInv]->m_gapiDesc, mult0);

        auto rnd = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);
        MakeRNG(m_bernulliProbability,
                m_realTensorMap[miopenTensorMhaDropoutSeed]->m_gapiDesc,
                m_realTensorMap[miopenTensorMhaDropoutOffset]->m_gapiDesc,
                rnd);

        DescriptorWrapperPtr mult1;
        DescriptorWrapperPtr pwS3, pwS4;
        MakePointwise(MIOPEN_POINTWISE_MUL, mult0, rnd, mult1);

        // _TODO there is a 1/(1-p) probability on the picture
        MakePointwise(MIOPEN_POINTWISE_MUL,
                      mult1,
                      m_realTensorMap[miopenTensorMhaDropoutProbability]->m_gapiDesc,
                      pwS3);
        MakePointwise(
            MIOPEN_POINTWISE_MUL, pwS3, m_realTensorMap[miopenTensorMhaScaleS]->m_gapiDesc, pwS4);

        auto pwS4T = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);
        MakeReshapeTranspose(pwS4, pwS4T);

        auto mm1 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testD);
        MakeMatmul(pwS4T, m_realTensorMap[miopenTensorMhaDO]->m_gapiDesc, mm1);
        DescriptorWrapperPtr pwS5, pwS6;
        MakePointwise(
            MIOPEN_POINTWISE_MUL, mm1, m_realTensorMap[miopenTensorMhaDescaleS]->m_gapiDesc, pwS5);
        MakePointwise(MIOPEN_POINTWISE_MUL,
                      pwS5,
                      m_realTensorMap[miopenTensorMhaDescaleDO]->m_gapiDesc,
                      pwS6);

        MakeReduction(
            MIOPEN_REDUCE_TENSOR_MAX, pwS6, m_realTensorMap[miopenTensorMhaAmaxDV]->m_gapiDesc);
        MakePointwise(MIOPEN_POINTWISE_MUL,
                      pwS6,
                      m_realTensorMap[miopenTensorMhaScaleDV]->m_gapiDesc,
                      m_realTensorMap[miopenTensorMhaDV]->m_gapiDesc);

        ////////////////// Center-top, Right-top //////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////

        auto vT = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testD, m_testS);
        MakeReshapeTranspose(m_realTensorMap[miopenTensorMhaV]->m_gapiDesc, vT);

        auto mm2 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);
        MakeMatmul(m_realTensorMap[miopenTensorMhaDO]->m_gapiDesc, vT, mm2);

        DescriptorWrapperPtr pwS7, pwS8, pwS9, pwS10;
        MakePointwise(
            MIOPEN_POINTWISE_MUL, mm2, m_realTensorMap[miopenTensorMhaDescaleDO]->m_gapiDesc, pwS7);
        MakePointwise(
            MIOPEN_POINTWISE_MUL, pwS7, m_realTensorMap[miopenTensorMhaDescaleV]->m_gapiDesc, pwS8);
        MakePointwise(MIOPEN_POINTWISE_MUL, pwS8, rnd, pwS9);

        MakePointwise(MIOPEN_POINTWISE_MUL,
                      pwS9,
                      m_realTensorMap[miopenTensorMhaDropoutProbability]->m_gapiDesc,
                      pwS10);

        ////////////////
        DescriptorWrapperPtr pwS11, pwS12, pwS13, mult2;
        MakePointwise(MIOPEN_POINTWISE_MUL,
                      m_realTensorMap[miopenTensorMhaDO]->m_gapiDesc,
                      m_realTensorMap[miopenTensorMhaDescaleDO]->m_gapiDesc,
                      pwS11);
        MakePointwise(MIOPEN_POINTWISE_MUL,
                      m_realTensorMap[miopenTensorMhaO]->m_gapiDesc,
                      m_realTensorMap[miopenTensorMhaDescaleO]->m_gapiDesc,
                      pwS12);

        MakePointwise(MIOPEN_POINTWISE_MUL, pwS11, pwS12, mult2);

        MakePointwise(MIOPEN_POINTWISE_MUL,
                      mult2,
                      m_realTensorMap[miopenTensorMhaDropoutProbability]->m_gapiDesc,
                      pwS13);

        auto sum0 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, 1);
        MakeReduction(MIOPEN_REDUCE_TENSOR_ADD, pwS13, sum0);

        ////////////////// Center Part ////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////
        DescriptorWrapperPtr sub1;
        MakePointwise(MIOPEN_POINTWISE_SUB, pwS10, sum0, sub1);

        DescriptorWrapperPtr pwS14, pwS15, mult3;
        MakePointwise(MIOPEN_POINTWISE_IDENTITY, sub1, nullptr, pwS14, true, m_attentionScale);
        MakePointwise(MIOPEN_POINTWISE_MUL, pwS14, pwS3 /*output from left column*/, mult3);
        MakeReduction(
            MIOPEN_REDUCE_TENSOR_MAX, mult3, m_realTensorMap[miopenTensorMhaAmaxDS]->m_gapiDesc);
        MakePointwise(MIOPEN_POINTWISE_MUL,
                      mult3,
                      m_realTensorMap[miopenTensorMhaScaleDS]->m_gapiDesc,
                      pwS15);

        ////////////////// Center-bottom, Right-bottom ////////////////////////////
        ///////////////////////////////////////////////////////////////////////////
        auto mm3 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testD);
        MakeMatmul(pwS15, m_realTensorMap[miopenTensorMhaK]->m_gapiDesc, mm3);

        DescriptorWrapperPtr pwS16, pwS17;
        MakePointwise(MIOPEN_POINTWISE_MUL,
                      mm3,
                      m_realTensorMap[miopenTensorMhaDescaleDS]->m_gapiDesc,
                      pwS16);
        MakePointwise(MIOPEN_POINTWISE_MUL,
                      pwS16,
                      m_realTensorMap[miopenTensorMhaDescaleK]->m_gapiDesc,
                      pwS17);
        MakeReduction(
            MIOPEN_REDUCE_TENSOR_MAX, pwS17, m_realTensorMap[miopenTensorMhaAmaxDQ]->m_gapiDesc);
        MakePointwise(MIOPEN_POINTWISE_MUL,
                      pwS17,
                      m_realTensorMap[miopenTensorMhaScaleDQ]->m_gapiDesc,
                      m_realTensorMap[miopenTensorMhaDQ]->m_gapiDesc);

        ///////////////////
        auto mm4 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testD);

        // Reshape transpose happens here for pwS15
        auto pwS15T = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);
        MakeReshapeTranspose(pwS15, pwS15T);

        MakeMatmul(pwS15T, m_realTensorMap[miopenTensorMhaQ]->m_gapiDesc, mm4);

        DescriptorWrapperPtr pwS18, pwS19;
        MakePointwise(MIOPEN_POINTWISE_MUL,
                      mm4,
                      m_realTensorMap[miopenTensorMhaDescaleDS]->m_gapiDesc,
                      pwS18);
        MakePointwise(MIOPEN_POINTWISE_MUL,
                      pwS18,
                      m_realTensorMap[miopenTensorMhaDescaleQ]->m_gapiDesc,
                      pwS19);

        MakeReduction(
            MIOPEN_REDUCE_TENSOR_MAX, pwS19, m_realTensorMap[miopenTensorMhaAmaxDK]->m_gapiDesc);
        MakePointwise(MIOPEN_POINTWISE_MUL,
                      pwS19,
                      m_realTensorMap[miopenTensorMhaScaleDK]->m_gapiDesc,
                      m_realTensorMap[miopenTensorMhaDK]->m_gapiDesc);
    }

    virtual void RunCPUverify(miopen::Handle& handle) override
    {
        const double errorThreshold    = 5e-5;
        const double fp8ErrorThreshold = (std::is_same_v<T, float8>) ? 3e-3 : errorThreshold;

        auto checkAmax = [this, errorThreshold, &handle](
                             miopenTensorArgumentId_t id, std::string_view name, float refAmax) {
            const auto& resAmax = GetResult<float>(id, handle);
            float amaxRelDiff   = std::abs(refAmax - resAmax[0]);
            float divisor       = std::min(refAmax, resAmax[0]);
            amaxRelDiff /= divisor > std::numeric_limits<float>::min() ? divisor : 1.0f;
            EXPECT_LT(amaxRelDiff, errorThreshold)
                << name << " ref: " << refAmax << " result: " << resAmax[0];
        };

        auto checkOutput = [this, fp8ErrorThreshold, &handle](miopenTensorArgumentId_t id,
                                                              std::string_view name,
                                                              const auto& ref) {
            EXPECT_LT(miopen::rms_range(ref, GetResult<T>(id, handle)), fp8ErrorThreshold) << name;
        };

        checkAmax(miopenTensorMhaAmaxDQ, "amax dQ", m_amaxDQRef);
        checkAmax(miopenTensorMhaAmaxDK, "amax dK", m_amaxDKRef);
        checkAmax(miopenTensorMhaAmaxDV, "amax dV", m_amaxDVRef);
        checkAmax(miopenTensorMhaAmaxDS, "amax dS", m_amaxDSRef);

        checkOutput(miopenTensorMhaDQ, "tensor dQ", m_dQDescRef);
        checkOutput(miopenTensorMhaDK, "tensor dK", m_dKDescRef);
        checkOutput(miopenTensorMhaDV, "tensor dV", m_dVDescRef);
    }

private:
    tensor<T> m_dQDescRef;
    tensor<T> m_dKDescRef;
    tensor<T> m_dVDescRef;

    float m_amaxDQRef = 0.0f; // values will be set later. Initializetion is reqired for tidy-checks
    float m_amaxDKRef = 0.0f;
    float m_amaxDVRef = 0.0f;
    float m_amaxDSRef = 0.0f;
};

class GPU_MhaBackward_FP32 : public MhaBackwardTest<float>
{
};

class GPU_MhaBackward_FP8 : public MhaBackwardTest<float8>
{
    void SetUp() override
    {
        using e_mask = enabled<Gpu::gfx94X>;
        using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx90A>;
        if(!IsTestSupportedForDevMask<d_mask, e_mask>() || MIOPEN_FP8_IEEE_EXPONENT_BIAS != 0)
        {
            GTEST_SKIP() << "FP8 is unsupported on this HW";
        }

        MhaBackwardTest<float8>::SetUp();
    }
};

TEST_P(GPU_MhaBackward_FP32, TestFloat) { Run(); }
TEST_P(GPU_MhaBackward_FP8, TestFloat8) { Run(); }

inline auto GetCases()
{
    return testing::Combine(testing::ValuesIn({2}),           // n
                            testing::ValuesIn({4}),           // h
                            testing::ValuesIn({64}),          // s
                            testing::ValuesIn({16}),          // d
                            testing::ValuesIn({0.0f, 0.5f})); // bernulli probability
}

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_MhaBackward_FP32, GetCases());
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_MhaBackward_FP8, GetCases());
