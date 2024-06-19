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
    virtual void MakeRealTensorsAndFillData(miopen::Handle& handle) override
    {
        auto q = test::cpu::GenScaledTensorBackward<T>(m_testN, m_testH, m_testS, m_testD);
        auto k = test::cpu::GenScaledTensorBackward<T>(m_testN, m_testH, m_testS, m_testD);
        auto v = test::cpu::GenScaledTensorBackward<T>(m_testN, m_testH, m_testS, m_testD);

        MakeAndAddRealTensorDescriptor(miopenTensorMhaQ, false, m_testN, m_testH, m_testS, m_testD)
            .InitAndWriteToGPU(handle, std::move(q.mTensor));

        MakeAndAddRealTensorDescriptor(miopenTensorMhaK, false, m_testN, m_testH, m_testS, m_testD)
            .InitAndWriteToGPU(handle, std::move(k.mTensor));

        MakeAndAddRealTensorDescriptor(miopenTensorMhaV, false, m_testN, m_testH, m_testS, m_testD)
            .InitAndWriteToGPU(handle, std::move(v.mTensor));

        float sScale   = 1.0f;
        float sDescale = 1.0f;

        float oScale   = 1.0f;
        float oDescale = 1.0f;

        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleQ)
            .InitAndWriteToGPU(handle, q.mDescale);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleK)
            .InitAndWriteToGPU(handle, k.mDescale);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleV)
            .InitAndWriteToGPU(handle, v.mDescale);

        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleS).InitAndWriteToGPU(handle, sDescale);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaScaleS).InitAndWriteToGPU(handle, sScale);

        MakeAndAddRealTensorDescriptor(miopenTensorMhaDropoutProbability)
            .InitAndWriteToGPU(handle, m_bernulliProbability);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDropoutSeed)
            .InitAndWriteToGPU(handle, static_cast<int64_t>(0xAAFFFFFFFFul));
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDropoutOffset)
            .InitAndWriteToGPU(handle, static_cast<int64_t>(1));

        tensor<float> softmax  = tensor<float>{m_testN, m_testH, m_testS, m_testS};
        tensor<T> oDesc        = tensor<T>{m_testN, m_testH, m_testS, m_testD};
        tensor<float> mDesc    = tensor<float>{m_testN, m_testH, m_testS, 1};
        tensor<float> zInvDesc = tensor<float>{m_testN, m_testH, m_testS, 1};
        float amaxS;
        float amaxO;

        // proper O, M and zInv tensors are required for backward pass.
        // randomly generated M and zInv may lead to nan\inf values
        test::cpu::MultiHeadAttentionfp8(
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
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDO, false, m_testN, m_testH, m_testS, m_testD)
            .InitAndWriteToGPU(handle, std::move(dO.mTensor));

        MakeAndAddRealTensorDescriptor(miopenTensorMhaO, false, m_testN, m_testH, m_testS, m_testD)
            .InitAndWriteToGPU(handle, std::move(oDesc));

        MakeAndAddRealTensorDescriptor(miopenTensorMhaM, false, m_testN, m_testH, m_testS, 1)
            .InitAndWriteToGPU(handle, std::move(mDesc));

        MakeAndAddRealTensorDescriptor(miopenTensorMhaZInv, false, m_testN, m_testH, m_testS, 1)
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
            miopenTensorMhaDQ, false, m_testN, m_testH, m_testS, m_testD);
        MakeAndAddRealTensorDescriptor(
            miopenTensorMhaDK, false, m_testN, m_testH, m_testS, m_testD);
        MakeAndAddRealTensorDescriptor(
            miopenTensorMhaDV, false, m_testN, m_testH, m_testS, m_testD);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaAmaxDQ);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaAmaxDK);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaAmaxDV);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaAmaxDS);

        // get next value for the rest of the tensors (which don't have any particular enum value)
        GetNextId();
    }

    virtual void MakeVirtualTensorsAndNodes() override 
    {
        ////////////// Left part (column) of the Mha backward graph ///////////////
        ///////////////////////////////////////////////////////////////////////////

        auto mm0 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);

        MakeMatmul(m_realTensorMap[miopenTensorMhaQ]->m_gapiDesc,
                   m_realTensorMap[miopenTensorMhaK]->m_gapiDesc,
                   mm0);

        // MakePointwise function will automatically create a tensor for output,
        // if output is nullptr    
        DescriptorWrapperPtr pwS0;
        MakePointwise(MIOPEN_POINTWISE_IDENTITY, mm0, nullptr, pwS0, false, m_attentionScale);

        DescriptorWrapperPtr pwS1;
        DescriptorWrapperPtr pwS2;
        MakePointwise(
            MIOPEN_POINTWISE_MUL, pwS0, m_realTensorMap[miopenTensorMhaDescaleQ]->m_gapiDesc, pwS1);
        MakePointwise(
            MIOPEN_POINTWISE_MUL, pwS1, m_realTensorMap[miopenTensorMhaDescaleK]->m_gapiDesc, pwS2);

        DescriptorWrapperPtr sub0;
        DescriptorWrapperPtr exp0;
        DescriptorWrapperPtr mult0;
        MakePointwise(MIOPEN_POINTWISE_SUB, pwS2, m_realTensorMap[miopenTensorMhaM]->m_gapiDesc, sub0);
        MakePointwise(MIOPEN_POINTWISE_EXP, sub0, DescriptorWrapperPtr(), exp0);
        MakePointwise(MIOPEN_POINTWISE_MUL, exp0, m_realTensorMap[miopenTensorMhaZInv]->m_gapiDesc, mult0);

        auto rnd   = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testS);
        MakeRNG(m_bernulliProbability,
                m_realTensorMap[miopenTensorMhaDropoutSeed]->m_gapiDesc,
                m_realTensorMap[miopenTensorMhaDropoutOffset]->m_gapiDesc,
                rnd);

        DescriptorWrapperPtr mult1;
        DescriptorWrapperPtr pwS3, pwS4;
        MakePointwise(MIOPEN_POINTWISE_MUL, mult0, rnd, mult1);
        MakePointwise(MIOPEN_POINTWISE_MUL, mult1, m_realTensorMap[miopenTensorMhaDropoutProbability]->m_gapiDesc, pwS3);
        MakePointwise(MIOPEN_POINTWISE_MUL, pwS3, m_realTensorMap[miopenTensorMhaScaleS]->m_gapiDesc, pwS4);

        //////reshape transpose on a scheme here////////

        auto mm1 = MakeGapiVirtualTensorDesc(m_testN, m_testH, m_testS, m_testD);
        MakeMatmul(pwS4, m_realTensorMap[miopenTensorMhaDO]->m_gapiDesc, mm1);
        DescriptorWrapperPtr pwS5, pwS6;
        MakePointwise(MIOPEN_POINTWISE_MUL, mm1, m_realTensorMap[miopenTensorMhaDescaleS]->m_gapiDesc, pwS5);
        MakePointwise(MIOPEN_POINTWISE_MUL, pwS5, m_realTensorMap[miopenTensorMhaDescaleDO]->m_gapiDesc, pwS6);

        MakeReduction(MIOPEN_REDUCE_TENSOR_MAX, pwS6, m_realTensorMap[miopenTensorMhaAmaxDV]->m_gapiDesc);
        MakePointwise(MIOPEN_POINTWISE_MUL,
                      pwS6,
                      m_realTensorMap[miopenTensorMhaScaleDV]->m_gapiDesc,
                      m_realTensorMap[miopenTensorMhaDV]->m_gapiDesc);

        ////////////////// Viddle-top, right-top //////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////

        
    }

    virtual void RunCPUverify(miopen::Handle& handle) override {}
};

class MhaBackwardTestFp32 : public MhaBackwardTest<float>
{
};

class MhaBackwardTestFp8 : public MhaBackwardTest<float8>
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

TEST_P(MhaBackwardTestFp32, TestFloat) { Run(); }
// TEST_P(MhaBackwardTestFp8, TestFloat) { Run(); }

inline auto GetCases()
{
    return testing::Combine(testing::ValuesIn({2}),           // n
                            testing::ValuesIn({4}),           // h
                            testing::ValuesIn({64}),          // s
                            testing::ValuesIn({16}),          // d
                            testing::ValuesIn({0.0f, 0.5f})); // bernulli probability
}

INSTANTIATE_TEST_SUITE_P(MhaBwdSuiteFp32, MhaBackwardTestFp32, GetCases());

INSTANTIATE_TEST_SUITE_P(MhaBwdSuiteFp8, MhaBackwardTestFp8, GetCases());
