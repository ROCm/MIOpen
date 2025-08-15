/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <miopen/tensor_ops.hpp>
#include <tensor_util.hpp>
#include "gtest_common.hpp"

namespace {
std::vector<std::vector<size_t>> tensorALensArr    = {{1, 1, 4}, {1, 1, 8}, {1, 1, 32}}; // tensor A
std::vector<std::vector<size_t>> tensorAStridesArr = {{4, 4, 1}, {8, 8, 1}, {32, 32, 1}};

std::vector<std::vector<size_t>> tensorBLensArr = {{1, 16, 4}, {1, 32, 8}, {1, 8, 32}}; // tensor B
std::vector<std::vector<size_t>> tensorBStridesArr = {
    {16 * 4, 4 * 1, 1}, {32 * 8, 8 * 1, 1}, {8 * 32, 32 * 1, 1}};

std::vector<std::vector<int64_t>> offsetsArr = {
    {0, 0, 0}, {64, 32, 16}, {32, 16, 32}, {32, 16, 32}};

std::vector<std::vector<float>> alphabetaArr = {{1, 1, 0}, {-1, 1, 1}, {1.0, 0.5, 0.3}};

std::vector<bool> packedArr = {true, false};

std::vector<miopenTensorOp_t> operationArr = {
    miopenTensorOpAdd, miopenTensorOpMul, miopenTensorOpMin, miopenTensorOpMax};
} // namespace

struct TestCase
{
    std::vector<size_t> tensorlens_ac;
    std::vector<size_t> tensorlens_b;
    std::vector<int64_t> offsets;
    std::vector<size_t> stride_a;
    std::vector<size_t> stride_b;
    std::vector<size_t> stride_c;
    std::vector<float> alphabeta;
    bool packed;
    miopenTensorOp_t operation;

    friend std::ostream& operator<<(std::ostream& os, const TestCase& tc)
    {
        return os << "TensorOp: " << tc.operation << std::endl
                  << "A tensor: " << tc.tensorlens_ac[0] << "," << tc.tensorlens_ac[1] << ","
                  << tc.tensorlens_ac[2] << std::endl
                  << "B tensor: " << tc.tensorlens_b[0] << "," << tc.tensorlens_b[1] << ","
                  << tc.tensorlens_b[2] << std::endl
                  << "IsPacked: " << tc.packed << std::endl
                  << "Offsets: " << tc.offsets[0] << "," << tc.offsets[1] << "," << tc.offsets[2]
                  << std::endl;
    }
};

template <typename T>
struct TensorOpsCommon : public testing::TestWithParam<TestCase>
{
    void SetUp() override { prng::reset_seed(); }

    void Run()
    {
        CreateTensors();

        std::vector<T> tensorGPUData = CalculateOnGPU();
        std::vector<T> tensorCPUData = CalculateOnCPU();

        CompareResults(tensorGPUData, tensorCPUData);
    }

private:
    void CreateTensors()
    {
        const TestCase& testCase = GetParam();

        tensorA = CreateTensor(
            testCase.tensorlens_ac, testCase.stride_a, testCase.offsets[0], testCase.packed);
        tensorB = CreateTensor(
            testCase.tensorlens_b, testCase.stride_b, testCase.offsets[1], testCase.packed);
        tensorC = CreateTensor(
            testCase.tensorlens_ac, testCase.stride_c, testCase.offsets[2], testCase.packed);
    }

    tensor<T> CreateTensor(const std::vector<size_t>& lens,
                           const std::vector<size_t>& strides,
                           int64_t offset,
                           bool isPacked)
    {
        uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

        if(!isPacked)
        {
            std::vector<size_t> real_strides(strides.begin() + (strides.size() - lens.size()),
                                             strides.end());
            auto r = tensor<T>{lens, real_strides}.generate(tensor_elem_gen_integer{max_value});
            r.data.insert(r.data.begin(), offset, T());
            return r;
        }
        else
        {
            auto r = tensor<T>{lens}.generate(tensor_elem_gen_integer{max_value});
            r.data.insert(r.data.begin(), offset, T());
            return r;
        }
    }

    std::vector<T> CalculateOnGPU() const
    {
        const TestCase& testCase = GetParam();

        auto&& handle = get_handle();

        auto a_dev = handle.Write(tensorA.data);
        auto b_dev = handle.Write(tensorB.data);
        auto c_dev = handle.Write(tensorC.data);

        miopen::OpTensor(handle,
                         testCase.operation,
                         &testCase.alphabeta[0],
                         tensorA.desc,
                         a_dev.get(),
                         &testCase.alphabeta[1],
                         tensorB.desc,
                         b_dev.get(),
                         &testCase.alphabeta[2],
                         tensorC.desc,
                         c_dev.get(),
                         testCase.offsets[0],
                         testCase.offsets[1],
                         testCase.offsets[2],
                         true);

        return handle.Read<T>(c_dev, tensorC.data.size());
    }

    std::vector<T> CalculateOnCPU()
    {
        const TestCase& testCase = GetParam();

        float alpha2 = testCase.alphabeta[1];

        if(testCase.operation == miopenTensorOpAdd)
        {
            return CalculateOnCPUDataOp([alpha2](auto A, auto B) { return A + B * alpha2; });
        }
        else if(testCase.operation == miopenTensorOpMul)
        {
            return CalculateOnCPUDataOp([alpha2](auto A, auto B) { return A * B * alpha2; });
        }
        else if(testCase.operation == miopenTensorOpMin)
        {
            return CalculateOnCPUDataOp(
                [alpha2](auto A, auto B) { return std::min(A, B * alpha2); });
        }
        else
        {
            return CalculateOnCPUDataOp(
                [alpha2](auto A, auto B) { return std::max(A, B * alpha2); });
        }
    }

    template <typename DataOp>
    std::vector<T> CalculateOnCPUDataOp(DataOp&& dataOp)
    {
        const TestCase& testCase = GetParam();

        auto a = tensorA;
        auto b = tensorB;
        auto c = tensorC;

        auto a_idx = testCase.offsets[0];
        auto b_idx = testCase.offsets[1];
        auto c_idx = testCase.offsets[2];

        // Currently non standard squashed operation is supported only on 3D tensors
        // A/C tensors are 1x1xH, tensor B is 1xCxH (C > 1)
        // This is simulated 3D non standard squashed tensor operation
        for(auto i = 0; i < testCase.tensorlens_ac[2]; i++)
        {
            auto a_val = a.data[a_idx] * testCase.alphabeta[0];
            auto c_val = c.data[c_idx] * testCase.alphabeta[2];
            for(auto j = 0; j < testCase.tensorlens_b[1]; j++)
            {
                c_val += dataOp(a_val, b.data[b_idx]);
                b_idx += testCase.stride_b[1];
            }
            c.data[c_idx] = c_val;
            a_idx += testCase.stride_a[2];
            c_idx += testCase.stride_c[2];
            b_idx = testCase.offsets[1] + (i + 1) * testCase.stride_b[2];
        }

        return c.data;
    }

    void CompareResults(const std::vector<T>& tensorGPUData, const std::vector<T>& tensorCPUData)
    {
        const TestCase& testCase = GetParam();

        double tolerance = 1;
        if(std::is_same_v<T, half_float::half>)
        {
            // taken from original c-test
            tolerance = 80;
        }

        double threshold = std::numeric_limits<T>::epsilon() * tolerance;
        double error     = miopen::rms_range(tensorCPUData, tensorGPUData);

        ASSERT_LE(error, threshold) << testCase;
    }

private:
    tensor<T> tensorA;
    tensor<T> tensorB;
    tensor<T> tensorC;
};

using GPU_TernaryNonStandardSquashedTensorOps_FP32 = TensorOpsCommon<float>;
using GPU_TernaryNonStandardSquashedTensorOps_FP16 = TensorOpsCommon<half_float::half>;
using GPU_TernaryNonStandardSquashedTensorOps_FP64 = TensorOpsCommon<double>;

namespace {

void AddTestCases(std::vector<TestCase>& testCases,
                  const std::vector<size_t>& tensorALens,
                  const std::vector<size_t>& tensorBLens,
                  const std::vector<size_t> stride_a,
                  const std::vector<size_t> stride_b,
                  const std::vector<size_t> stride_c)
{
    for(bool packed : packedArr)
        for(const auto& offsets : offsetsArr)
        {
            std::vector<int64_t> final_offsets{0, 0, 0};
            if(!packed)
            {
                if(std::any_of(offsets.begin(), offsets.end(), [](int64_t o) { return o < 0; }))
                    continue;

                final_offsets = offsets;
            }

            auto checkStride = [p = packed](const std::vector<size_t>& lens,
                                            const std::vector<size_t>& strides) {
                if(p)
                    return true;

                if(lens.size() > strides.size())
                    return false;

                // only sparsed case allowed, since all the kernels do not support the last
                // dimension strides
                if(strides.back() == 1)
                {
                    // we use float here for all types because strides are independent to type
                    auto packedStrides =
                        miopen::TensorDescriptor(miopen_type<float>{}, lens).GetStrides();

                    return std::equal(packedStrides.rbegin(),
                                      packedStrides.rend(),
                                      strides.rbegin(),
                                      [](size_t ps, size_t s) { return s >= ps; });
                }

                // currently tensor operations do not support non-one stride in the last dimention.
                return false;
            };

            if(!checkStride(tensorALens, stride_a))
                continue;
            if(!checkStride(tensorBLens, stride_b))
                continue;
            if(!checkStride(tensorALens, stride_c))
                continue;

            for(const auto& alphabeta : alphabetaArr)
                for(const auto& operation : operationArr)
                {
                    TestCase& testCase = testCases.emplace_back();

                    testCase.tensorlens_ac = tensorALens;
                    testCase.tensorlens_b  = tensorBLens;
                    testCase.alphabeta     = alphabeta;
                    testCase.offsets       = final_offsets;
                    testCase.packed        = packed;
                    testCase.operation     = operation;
                    testCase.stride_a      = stride_a;
                    testCase.stride_b      = stride_b;
                    testCase.stride_c      = stride_c;
                }
        }
}

std::vector<TestCase> GenCases()
{
    std::vector<TestCase> testCases;

    for(int i = 0; i < tensorALensArr.size(); i++)
    {
        AddTestCases(testCases,
                     tensorALensArr[i],
                     tensorBLensArr[i],
                     tensorAStridesArr[i],
                     tensorBStridesArr[i],
                     tensorAStridesArr[i]);
    }

    return testCases;
}

inline auto GetCases()
{
    static const auto cases = testing::ValuesIn(GenCases());
    return cases;
}
} // namespace

TEST_P(GPU_TernaryNonStandardSquashedTensorOps_FP32, TestFloat) { this->Run(); }

TEST_P(GPU_TernaryNonStandardSquashedTensorOps_FP16, TestFloat16) { this->Run(); }

TEST_P(GPU_TernaryNonStandardSquashedTensorOps_FP64, TestDouble) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_TernaryNonStandardSquashedTensorOps_FP32, GetCases());
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_TernaryNonStandardSquashedTensorOps_FP64, GetCases());
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_TernaryNonStandardSquashedTensorOps_FP16, GetCases());
