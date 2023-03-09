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

#include "../driver/tensor_driver.hpp"

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
    int M;
    int N;
    int K;

    long long int StrideA;
    long long int StrideB;
    long long int StrideC;

    miopenDataType_t dataType;

    friend std::ostream& operator<<(std::ostream& os, const GemmTestCase& tc)
    {
        return os << "(M: " << tc.M << " N:" << tc.N << " K:" << tc.K << ", A(" << tc.M << ","
                  << tc.K << ")"
                  << ", B(" << tc.K << "," << tc.N << ")"
                  << ", C(" << tc.M << "," << tc.N << ")"
                  << " StrideA: " << tc.StrideA << " StrideB: " << tc.StrideB
                  << " StrideC: " << tc.StrideC << " dataType: " << tc.dataType << " )";
    }
    std::vector<int> GetA() { return {M, K}; }
    std::vector<int> GetB() { return {K, N}; }
    std::vector<int> GetC() { return {M, N}; }

    miopen::GemmNewDescriptor GetGemm()
    {
        return miopen::GemmNewDescriptor{M, N, K, StrideA, StrideB, StrideC, dataType};
    }
};

inline int SetTensorLayout(miopen::TensorDescriptor& desc)
{
    // get layout string names
    std::string layout_str = desc.GetLayout_str();

    std::vector<std::size_t> lens = desc.GetLengths();
    std::vector<int> int_lens(lens.begin(), lens.end());

    // set the strides for the tensor
    return SetTensorNd(&desc, int_lens, layout_str, desc.GetType());
}

std::vector<GemmTestCase> GetTestData()
{
    // A(M, K)  B(K, N), C(M, N)

    return {
        // M,    N,    K,   StrideA (K), StrideB (N), StrideC (N)
        {960, 2048, 1024, 1024, 2048, 2048, miopenHalf}
        // { 1024, 1024, 1024,   1088,        1088,        1088, miopenHalf} /////
        /*
        { 960, 2048, 1024, 1024, 2048, 2048, miopenHalf},
        { 1024, 1024, 1024, 1024, 1024, 1024, miopenHalf},
        { 960, 2048, 2048, 2048, 2048, 2048, miopenHalf},
        { 1024, 1024, 1024, 1088, 1088, 1088, miopenHalf},*/
    };
}

template <typename T = half_float::half>
struct GemmTest : public ::testing::TestWithParam<std::tuple<GemmTestCase, miopenTensorLayout_t>>
{
protected:
    void SetUp() override
    {
        //  we need stride too.
        test_skipped                         = false;
        std::tie(gemm_config, tensor_layout) = GetParam();
        A_tensor                             = tensor<T>(gemm_config.GetA());
        B_tensor                             = tensor<T>(gemm_config.GetB());
        C_tensor                             = tensor<T>(gemm_config.GetC());
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<> d{-3, 3};
        auto gen_value = [&](auto...) { return d(gen); };
        A_tensor.generate(gen_value);
        B_tensor.generate(gen_value);

        gemm_desc = gemm_config.GetGemm();

        auto&& handle = get_handle();
        std::fill(C_tensor.begin(), C_tensor.end(), std::numeric_limits<double>::quiet_NaN());

        a_dev = handle.Write(A_tensor.data);
        b_dev = handle.Write(B_tensor.data);
        c_dev = handle.Write(C_tensor.data);

        fusePlanDesc = miopen::FusionPlanDescriptor(
            miopenVerticalFusion, A_tensor.desc); // todo : change miopenVerticalFusion
        // Create gemm Operation. This operation will be part of fusion plan.
        auto gemmOp = std::make_shared<miopen::GemmOpDescriptor>(gemm_desc, B_tensor.desc);
        // Add Operation Gemm as part of fusion plan.
        EXPECT_EQ(fusePlanDesc.AddOp(gemmOp), miopenStatusSuccess);
        // Here for fusion we set up the B matrix space (b_dev). The A (in) and C (out) matrix was
        // prepared when we call RunTunableSolver.
        gemmOp->SetArgs(params, b_dev.get());
    }
    void TearDown() override
    {
        if(test_skipped)
            return;
        ref_out = tensor<T>(gemm_config.GetC());
        gemm<T>(gemm_config.N, gemm_config.M, gemm_config.K, A_tensor, B_tensor, ref_out);
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
    }

    GemmTestCase gemm_config;
    miopen::GemmNewDescriptor gemm_desc;
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
    const float alpha = static_cast<float>(1.0f);
    const float beta  = static_cast<float>(0);

    miopenTensorLayout_t tensor_layout;
};
