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

    long long int ldA;
    long long int ldB;
    long long int ldC;

    friend std::ostream& operator<<(std::ostream& os, const GemmTestCase& tc)
    {
        return os << "(M: " << tc.M << " N:" << tc.N << " K:" << tc.K << ", A(" << tc.M << ","
                  << tc.K << ")"
                  << ", B(" << tc.K << "," << tc.N << ")"
                  << ", C(" << tc.M << "," << tc.N << ")"
                  << " ldA: " << tc.ldA << " ldB: " << tc.ldB
                  << " ldC: " << tc.ldC << " )";
    }
    std::vector<int> GetA() { return {M, K}; }
    std::vector<int> GetB() { return {K, N}; }
    std::vector<int> GetC() { return {M, N}; }
};

// Fast GeLU
// https://paperswithcode.com/method/gelu
// y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
template<typename T>
void RunHostFastGeLU(tensor<T>& ref_out)
{
    T one = T(1);
    T two = T(2);
    T point_five = T(0.5);
    T const_1  = T(0.035677);
    T const_2  = T(0.797885);
    for(auto& val : ref_out.data){
        const T u   = two * val * (const_1 * val * val + const_2);
        const T emu = exp(-u);
        const T cdf = point_five + point_five * (two / (one + emu) - one);
        T tmp_val = val * cdf;
        val = tmp_val;
    }
    
}

template <typename T = half_float::half>
struct GemmAPIFusionTest : public ::testing::TestWithParam<std::tuple<GemmTestCase>>
{
protected:
    void SetUp() override
    {
        std::tie(gemm_config) = GetParam();
        A_tensor              = tensor<T>(gemm_config.GetA());
        B_tensor              = tensor<T>(gemm_config.GetB());
        C_tensor              = tensor<T>(gemm_config.GetC());
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<> d{-3, 3};
        auto gen_value = [&](auto...) { return d(gen); };
        A_tensor.generate(gen_value);
        B_tensor.generate(gen_value);

        miopenCreateGemmDescriptor(&gemm_desc);
        miopenInitGemmDescriptor(gemm_desc,
                                 gemm_config.M,
                                 gemm_config.N,
                                 gemm_config.K,
                                 gemm_config.ldA,
                                 gemm_config.ldB,
                                 gemm_config.ldC,
                                 GetDataType<T>());

        auto&& handle = get_handle();
        std::fill(C_tensor.begin(), C_tensor.end(), std::numeric_limits<double>::quiet_NaN());

        a_dev = handle.Write(A_tensor.data);
        b_dev = handle.Write(B_tensor.data);
        c_dev = handle.Write(C_tensor.data);
    }
    void TearDown() override
    {
        ref_out = tensor<T>(gemm_config.GetC());
        gemm<T>(gemm_config.N, gemm_config.M, gemm_config.K, A_tensor, B_tensor, ref_out);
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
        miopenDestroyGemmDescriptor(gemm_desc);
    }

    GemmTestCase gemm_config;
    miopenGemmDescriptor_t gemm_desc;
    tensor<T> A_tensor;
    tensor<T> B_tensor;
    tensor<T> C_tensor;
    tensor<T> ref_out;
    miopen::Allocator::ManageDataPtr a_dev;
    miopen::Allocator::ManageDataPtr b_dev;
    miopen::Allocator::ManageDataPtr c_dev;
};

struct GemmAPIFusionTestHalf : GemmAPIFusionTest<half_float::half>
{
};

TEST_P(GemmAPIFusionTestHalf, GEMMAPI)
{
    const auto status = miopenGemmActivFusion(&get_handle(),
                                         gemm_desc,
                                         &A_tensor.desc,
                                         a_dev.get(),
                                         &B_tensor.desc,
                                         b_dev.get(),
                                         &C_tensor.desc,
                                         c_dev.get());
    EXPECT_EQ(status, miopenStatusSuccess);
}

void GatherGemmTestCase(std::vector<GemmTestCase>& cba_test_cases)
{
    std::string arch = get_handle().GetDeviceName();
    if(miopen::StartsWith(arch, "gfx908") || miopen::StartsWith(arch, "gfx90a"))
    {
        cba_test_cases.push_back(GemmTestCase{960, 2048, 1024, 1024, 2048, 2048});
    }
    else
    {
        GTEST_SKIP() << " Skipping fusion test on unsupported ASIC";
    }
}

// Extra layer of indirection introduced since GTEST_SKIP() cannot be called from non-void function.
std::vector<GemmTestCase> GetTestValues()
{
    std::vector<GemmTestCase> cba_test_cases;
    GatherGemmTestCase(cba_test_cases);
    return cba_test_cases;
}

INSTANTIATE_TEST_SUITE_P(GEMMAPITest,
                         GemmAPIFusionTestHalf,
                         testing::Combine(testing::ValuesIn(GetTestValues())));
