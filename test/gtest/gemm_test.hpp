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

    friend std::ostream& operator<<(std::ostream& os, const GemmTestCase& tc)
    {
        return os << "(M: " << tc.M << " N:" << tc.N << " K:" << tc.K << ", A(" << tc.M << ","
                  << tc.K << ")"
                  << ", B(" << tc.K << "," << tc.N << ")"
                  << ", C(" << tc.M << "," << tc.N << ")\n";
    }
    std::vector<size_t> GetA() { return {M, K}; }
    std::vector<size_t> GetB() { return {K, N}; }
    std::vector<size_t> GetD() { return {M, N}; }
    std::vector<size_t> GetC() { return {M, N}; }
    std::vector<size_t> GetE() { return {M, N}; }
    std::vector<size_t> GetATensorStrides(miopenTensorLayout_t layout)
    {
        return GetTensorKStrides(layout);
    }
    std::vector<size_t> GetBTensorStrides(miopenTensorLayout_t layout)
    {
        return GetTensorNStrides(layout);
    }
    std::vector<size_t> GetDTensorStrides(miopenTensorLayout_t layout)
    {
        return GetTensorNStrides(layout);
    }
    std::vector<size_t> GetETensorStrides(miopenTensorLayout_t layout)
    {
        return GetTensorNStrides(layout);
    }

    std::vector<size_t> GetTensorKStrides(miopenTensorLayout_t layout)
    {
        if(layout == miopenTensorRowMajor)
        {
            return {K, 1};
        }
        else if(layout == miopenTensorColumnMajor)
        {
            return {1, K};
        }
        else
        {
            MIOPEN_THROW("layout not supported");
        }
    }
    std::vector<size_t> GetTensorNStrides(miopenTensorLayout_t layout)
    {
        if(layout == miopenTensorRowMajor)
        {
            return {N, 1};
        }
        else if(layout == miopenTensorColumnMajor)
        {
            return {1, N};
        }
        else
        {
            MIOPEN_THROW("layout not supported");
        }
    }

    size_t GetMatAStride() { return K; }
    size_t GetMatBStride() { return N; }
    size_t GetMatCStride() { return N; }

    size_t GetAddCStride() { return M; }
    size_t GetAddDStride() { return N; }
    size_t GetAddEStride() { return N; }
};

std::vector<GemmTestCase> GetTestData()
{
    return {
        // A(M, K)  B(K, N), C(M, N)

        // M, N, K, stride_a(K), stride_b, stride_c, stride_d0, stride_e,
        // {12, 12, 12,  12, 12, 12,12, 12},
        //  {45, 24, 651,  651, 651, 651, 651, 651},
        {45, 24, 651},
        // {16, 108, 104, 108, 108, 108},
        // {36, 18, 623, 623, 623, 623},
        // {36, 36, 36, 36, 36, 36}
    };
}

// Fast GeLU
// https://paperswithcode.com/method/gelu
// y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
template <typename T>
void RunHostAddFastGeLU(const tensor<T>& C_tensor,
                        const tensor<T>& D_tensor,
                        size_t rows,
                        size_t cols,
                        tensor<T>& E_tensor)
{
    // E = C + D
    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            E_tensor(i, j) = C_tensor(i, j) + D_tensor(i, j);
        }
    }
    // E = FastGELU(E)
    T one        = T(1);
    T two        = T(2);
    T point_five = T(0.5);
    T const_1    = T(0.035677);
    T const_2    = T(0.797885);
    for(auto& val : E_tensor.data)
    {
        const T u   = two * val * (const_1 * val * val + const_2);
        const T emu = exp(-u);
        const T cdf = point_five + point_five * (two / (one + emu) - one);
        val         = val * cdf;
    }
}

// E = A*B + D
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

        A_tensor = tensor<T>(
            tensor_layout, gemm_config.GetA(), gemm_config.GetATensorStrides(tensor_layout));
        B_tensor = tensor<T>(
            tensor_layout, gemm_config.GetB(), gemm_config.GetBTensorStrides(tensor_layout));
        D_tensor = tensor<T>(
            tensor_layout, gemm_config.GetD(), gemm_config.GetDTensorStrides(tensor_layout));

        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<> d{-3, 3};
        auto gen_value = [&](auto...) { return d(gen); };
        A_tensor.generate(gen_value);
        B_tensor.generate(gen_value);
        D_tensor.generate(gen_value);

        miopenInitGemmDescriptor(&gemm_desc_t,
                                 gemm_config.M,
                                 gemm_config.N,
                                 gemm_config.K,
                                 gemm_config.GetMatAStride(),
                                 gemm_config.GetMatBStride(),
                                 gemm_config.GetMatCStride(),
                                 true,
                                 false,
                                 false);
        miopenInitMatrixAdditionDescriptor(&mat_add_desc_t,
                                           gemm_config.M,
                                           gemm_config.N,
                                           gemm_config.K,
                                           gemm_config.GetAddCStride(),
                                           gemm_config.GetAddDStride(),
                                           gemm_config.GetAddEStride(),
                                           true,
                                           false,
                                           false);
        miopenCreateActivationDescriptor(&activ_desc);
        miopenSetActivationDescriptor(activ_desc, activ_mode, activ_alpha, activ_beta, activ_gamma);
        C_tensor      = miopen::deref(gemm_desc_t).GetOutputTensor(A_tensor.desc, B_tensor.desc);
        E_tensor      = C_tensor;
        auto&& handle = get_handle();

        a_dev = handle.Write(A_tensor.data);
        b_dev = handle.Write(B_tensor.data);
        c_dev = handle.Write(C_tensor.data);
        d_dev = handle.Write(D_tensor.data);
        e_dev = handle.Write(E_tensor.data);

        fusePlanDesc = miopen::FusionPlanDescriptor(miopenVerticalFusion, A_tensor.desc);

        // Create GEMM Operation C = A*B
        auto gemmOp = std::make_shared<miopen::GemmForwardInferenceOpDescriptor>(
            miopen::deref(gemm_desc_t), B_tensor.desc, C_tensor.desc);
        // E = C + D
        auto matAddOp = std::make_shared<miopen::MatrixAddOpDescriptor>(
            miopen::deref(mat_add_desc_t), D_tensor.desc, E_tensor.desc, 0);
        // Create Activation Operation
        auto activOp = std::make_shared<miopen::ActivFwdFusionOpDescriptor>(
            miopen::deref(activ_desc).GetMode());

        // Add Gemm Operation as part of fusion plan.
        EXPECT_EQ(fusePlanDesc.AddOp(gemmOp), miopenStatusSuccess);
        EXPECT_EQ(fusePlanDesc.AddOp(matAddOp), miopenStatusSuccess);
        // Here for fusion we set up the B matrix space (b_dev). The A (in) and C (out) matrix was
        // prepared when we call RunTunableSolver.
        gemmOp->SetArgs(params, b_dev.get(), c_dev.get());
        matAddOp->SetArgs(params, d_dev.get(), e_dev.get());
        // activation
        EXPECT_EQ(fusePlanDesc.AddOp(activOp), miopenStatusSuccess);
        activOp->SetArgs(params, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
    }
    void TearDown() override
    {
        if(test_skipped)
            return;
        ref_out       = E_tensor;
        E_tensor_host = E_tensor;
        // C = A*B
        gemm(gemm_config.N, gemm_config.M, gemm_config.K, A_tensor, B_tensor, ref_out);
        RunHostAddFastGeLU(ref_out, D_tensor, gemm_config.M, gemm_config.N, E_tensor_host);
        //
        // std::cout << "\nCPU E after relu : \n";
        // for(const auto& it : E_tensor_host.data)
        // {
        //     std::cout << it << ",";
        // }

        auto&& handle = get_handle();
        std::fill(E_tensor.begin(), E_tensor.end(), std::numeric_limits<double>::quiet_NaN());
        // std::cout << "\n\nnefore\n";
        // for(const auto& it : E_tensor.data)
        //    {
        //     std::cout << it << ",";
        // }
        // std::cout << "\n======\n";
        E_tensor.data = handle.Read<T>(e_dev, E_tensor.data.size());
        // std::cout << "\nGPU\n";
        // for(const auto& it : E_tensor.data)
        // {
        //     std::cout << it << ",";
        // }
        // std::cout << "\n======\n";
        EXPECT_FALSE(miopen::range_zero(E_tensor_host)) << "CPU data is all zeros";
        EXPECT_FALSE(miopen::range_zero(E_tensor)) << "GPU data is all zeros";
        EXPECT_TRUE(miopen::range_distance(E_tensor_host) == miopen::range_distance(E_tensor));
        const double tolerance = 80;
        double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
        auto error             = miopen::rms_range(E_tensor_host, E_tensor);
        EXPECT_FALSE(miopen::find_idx(E_tensor_host, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";
        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
        miopenDestroyGemmDescriptor(gemm_desc_t);
        miopenDestroyMatrixAdditionDescriptor(mat_add_desc_t);
        miopenDestroyActivationDescriptor(activ_desc);
    }
    // E = A*B + D // A*B = C
    GemmTestCase gemm_config;
    miopenActivationDescriptor_t activ_desc;
    miopenGemmDescriptor_t gemm_desc_t;
    miopenMatrixAdditionDescriptor_t mat_add_desc_t;
    tensor<T> A_tensor;
    tensor<T> B_tensor;
    tensor<T> D_tensor;
    tensor<T> E_tensor;
    tensor<T> E_tensor_host;
    tensor<T> C_tensor;
    tensor<T> ref_out;
    miopen::Allocator::ManageDataPtr a_dev;
    miopen::Allocator::ManageDataPtr b_dev;
    miopen::Allocator::ManageDataPtr c_dev;
    miopen::Allocator::ManageDataPtr d_dev;
    miopen::Allocator::ManageDataPtr e_dev; // output
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
