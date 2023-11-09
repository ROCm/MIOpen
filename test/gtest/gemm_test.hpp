/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#pragma once

#include <random>

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>
#include <serialize.hpp>
#include <fusionHost.hpp>
#include <miopen/check_numerics.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"
#include "conv_common.hpp"
#include "gemm.hpp"

template <typename T>
miopenDataType_t GetDataType();

template <>
miopenDataType_t GetDataType<float>()
{
    return miopenFloat;
}

template <>
miopenDataType_t GetDataType<half_float::half>()
{
    return miopenHalf;
}

// a[m, k] * b[k,n] = c[m, n]

struct GemmTestCase
{
    size_t M;
    size_t N;
    size_t K;

    int stride_a; // leading dimension
    int stride_b;
    int stride_c;

    friend std::ostream& operator<<(std::ostream& os, const GemmTestCase& tc)
    {
        return os << "(M: " << tc.M << " N:" << tc.N << " K:" << tc.K << ", A(" << tc.M << ","
                  << tc.K << ")"
                  << ", B(" << tc.K << "," << tc.N << ")"
                  << ", C(" << tc.M << "," << tc.N << ")"
                  << " stride_a: " << tc.stride_a << " stride_b: " << tc.stride_b
                  << " stride_c: " << tc.stride_c << "\n";
    }
    std::vector<size_t> GetA() { return {M, K}; }
    std::vector<size_t> GetB() { return {K, N}; }
    std::vector<size_t> GetC() { return {M, N}; }
};

std::vector<GemmTestCase> GetTestData()
{
    return {
        // A(M, K)  B(K, N), C(M, N)

        // M, N, K, ldA (K), ldB (N), ldC (N)
        {96, 204, 451, 451, 204, 204},
        {45, 24, 651, 651, 24, 24},
        {16, 108, 104, 104, 108, 108},
        {36, 18, 623, 623, 18, 18},
        {36, 36, 36, 36, 36, 36},
        {36, 36, 36, 43, 36, 36},
    };
}

// Fast GeLU
// https://paperswithcode.com/method/gelu
// y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
template <typename T>
void RunHostFastGeLU(tensor<T>& ref_out)
{
    T one        = T(1);
    T two        = T(2);
    T point_five = T(0.5);
    T const_1    = T(0.035677);
    T const_2    = T(0.797885);
    for(auto& val : ref_out.data)
    {
        const T u   = two * val * (const_1 * val * val + const_2);
        const T emu = exp(-u);
        const T cdf = point_five + point_five * (two / (one + emu) - one);
        val         = val * cdf;
    }
}

std::vector<size_t> GetStrideForLayout(const miopenTensorLayout_t& layout,
                                       const std::vector<size_t>& lens)
{
    assert(lens.size() == 2);

    if(layout == miopenTensorRowMajor)
    {
        return {1, lens[0]};
    }
    else if(layout == miopenTensorColumnMajor)
    {
        return {lens[0], 1};
    }
    else
    {
        MIOPEN_THROW("layout not supported");
    }
}

template <typename T = half_float::half>
struct GemmTest : public ::testing::TestWithParam<
                      std::tuple<miopenActivationMode_t, GemmTestCase, miopenTensorLayout_t>>
{
protected:
    void SetUp() override
    {
        //  we need stride too.
        test_skipped                                     = false;
        std::tie(activ_mode, gemm_config, tensor_layout) = GetParam();

        A_tensor = tensor<T>(tensor_layout,
                             gemm_config.GetA(),
                             GetStrideForLayout(tensor_layout, gemm_config.GetA()));
        B_tensor = tensor<T>(tensor_layout,
                             gemm_config.GetB(),
                             GetStrideForLayout(tensor_layout, gemm_config.GetB()));

        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<> d{-3, 3};
        auto gen_value = [&](auto...) { return d(gen); };
        A_tensor.generate(gen_value);
        B_tensor.generate(gen_value);

        miopenInitGemmDescriptor(&gemm_desc_t,
                                 gemm_config.M,
                                 gemm_config.N,
                                 gemm_config.K,
                                 gemm_config.stride_a,
                                 gemm_config.stride_b,
                                 gemm_config.stride_c,
                                 true,
                                 false,
                                 false);
        miopenCreateActivationDescriptor(&activ_desc);
        miopenSetActivationDescriptor(activ_desc, activ_mode, activ_alpha, activ_beta, activ_gamma);
        C_tensor      = miopen::deref(gemm_desc_t).GetOutputTensor(A_tensor.desc, B_tensor.desc);
        auto&& handle = get_handle();
        std::fill(C_tensor.begin(), C_tensor.end(), std::numeric_limits<double>::quiet_NaN());

        a_dev = handle.Write(A_tensor.data);
        b_dev = handle.Write(B_tensor.data);
        c_dev = handle.Write(C_tensor.data);

        fusePlanDesc = miopen::FusionPlanDescriptor(
            miopenVerticalFusion, A_tensor.desc); // todo : change miopenVerticalFusion

        // Create GEMM Operation
        auto gemmOp = std::make_shared<miopen::GemmForwardInferenceOpDescriptor>(
            miopen::deref(gemm_desc_t), B_tensor.desc, C_tensor.desc);
        // Create Activation Operation
        auto activOp = std::make_shared<miopen::ActivFwdFusionOpDescriptor>(
            miopen::deref(activ_desc).GetMode());

        // Add Gemm Operation as part of fusion plan.
        EXPECT_EQ(fusePlanDesc.AddOp(gemmOp), miopenStatusSuccess);
        // Here for fusion we set up the B matrix space (b_dev). The A (in) and C (out) matrix was
        // prepared when we call RunTunableSolver.
        gemmOp->SetArgs(params, b_dev.get(), c_dev.get());
        // activation
        EXPECT_EQ(fusePlanDesc.AddOp(activOp), miopenStatusSuccess);
        activOp->SetArgs(params, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
    }
    void TearDown() override
    {
        if(test_skipped)
            return;
        ref_out = C_tensor;
        gemm(gemm_config.N, gemm_config.M, gemm_config.K, A_tensor, B_tensor, ref_out);
        RunHostFastGeLU(ref_out);
        auto&& handle = get_handle();

        C_tensor.data = handle.Read<T>(c_dev, C_tensor.data.size());

        EXPECT_FALSE(miopen::range_zero(ref_out)) << "CPU data is all zeros";
        EXPECT_FALSE(miopen::range_zero(C_tensor)) << "GPU data is all zeros";
        EXPECT_TRUE(miopen::range_distance(ref_out) == miopen::range_distance(C_tensor));
        const double tolerance = 80;
        double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
        auto error             = miopen::rms_range(ref_out, C_tensor);
        EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";
        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
        miopenDestroyGemmDescriptor(gemm_desc_t);
        miopenDestroyActivationDescriptor(activ_desc);
    }

    GemmTestCase gemm_config;
    miopenActivationDescriptor_t activ_desc;
    miopenGemmDescriptor_t gemm_desc_t;
    tensor<T> A_tensor;
    tensor<T> B_tensor;
    tensor<T> C_tensor;
    tensor<T> ref_out;
    miopen::Allocator::ManageDataPtr a_dev;
    miopen::Allocator::ManageDataPtr b_dev;
    miopen::Allocator::ManageDataPtr c_dev; // output
    bool test_skipped = false;
    miopen::FusionPlanDescriptor fusePlanDesc;
    miopen::OperatorArgs params;
    const float alpha       = static_cast<float>(1.0f);
    const float beta        = static_cast<float>(0);
    const float activ_alpha = static_cast<double>(0.5f);
    const float activ_beta  = static_cast<double>(0.5f);
    const float activ_gamma = static_cast<double>(0.5f);

    miopenActivationMode_t activ_mode;

    miopenTensorLayout_t tensor_layout;
};
