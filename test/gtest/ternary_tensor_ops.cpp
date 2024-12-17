/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
std::vector<std::vector<size_t>> tensorALensArr = {{32, 16, 8, 4, 4}, // tensor A
                                                   {16, 20, 16, 8},
                                                   {20, 16, 8},
                                                   {1, 16, 8},
                                                   {16, 8},
                                                   {8}};

std::vector<std::vector<size_t>> tensorBLensArr = {{32, 16, 8, 4, 4}, // tensor B
                                                   {32, 16, 1, 1, 1},
                                                   {1, 16, 8, 1, 1},
                                                   {1, 1, 8, 4, 1},
                                                   {16, 20, 16, 8},
                                                   {16, 20, 16, 1},
                                                   {16, 20, 1, 1},
                                                   {16, 1, 1, 1},
                                                   {1, 20, 16, 8},
                                                   {1, 20, 16, 1},
                                                   {1, 20, 1, 1},
                                                   {1, 1, 16, 8},
                                                   {1, 1, 1, 8},
                                                   {20, 16, 8},
                                                   {20, 16, 1},
                                                   {1, 16, 8},
                                                   {1, 16, 1},
                                                   {20, 1, 1},
                                                   {16, 8},
                                                   {16, 1},
                                                   {1, 8},
                                                   {8},
                                                   {1}};

std::vector<std::vector<int64_t>> offsetsArr = {
    {0, 0, 0}, {64, 32, 16}, {32, 16, 32}, {32, 16, 32}};

std::vector<std::vector<float>> alphabetaArr = {{1, 1, 0}, {-1, 1, 1}, {1.0, 0.5, 0.3}};

std::vector<std::vector<size_t>> stridesArr = {{8 * 16 * 20 * 16, 8 * 16 * 20, 8 * 16, 8, 1}};

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
            r.data.resize(r.data.size() + offset);
            return r;
        }
        else
        {
            return tensor<T>{lens}.generate(tensor_elem_gen_integer{max_value});
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
                         false); // it does not verify non-standard behaviour

        return handle.Read<T>(c_dev, tensorC.data.size());
    }

    std::vector<T> CalculateOnCPU()
    {
        const TestCase& testCase = GetParam();

        float alpha1 = testCase.alphabeta[0];
        float alpha2 = testCase.alphabeta[1];
        float beta   = testCase.alphabeta[2];

        if(testCase.operation == miopenTensorOpAdd)
        {
            return CalculateOnCPUDataOp([alpha1, alpha2, beta](auto& C, auto A, auto B) {
                C = A * alpha1 + B * alpha2 + C * beta;
            });
        }
        else if(testCase.operation == miopenTensorOpMul)
        {
            return CalculateOnCPUDataOp([alpha1, alpha2, beta](auto& C, auto A, auto B) {
                C = A * alpha1 * B * alpha2 + C * beta;
            });
        }
        else if(testCase.operation == miopenTensorOpMin)
        {
            return CalculateOnCPUDataOp([alpha1, alpha2, beta](auto& C, auto A, auto B) {
                C = std::min(A * alpha1, B * alpha2) + C * beta;
            });
        }
        else
        {
            return CalculateOnCPUDataOp([alpha1, alpha2, beta](auto& C, auto A, auto B) {
                C = std::max(A * alpha1, B * alpha2) + C * beta;
            });
        }
    }

    template <typename DataOp>
    std::vector<T> CalculateOnCPUDataOp(DataOp&& dataOp)
    {
        const TestCase& testCase = GetParam();

        auto r = tensorC;

        operate_over_subtensor<>(dataOp,
                                 r.data,
                                 tensorA.data,
                                 tensorB.data,
                                 r.desc,
                                 tensorA.desc,
                                 tensorB.desc,
                                 testCase.offsets[2],
                                 testCase.offsets[0],
                                 testCase.offsets[1]);

        return r.data;
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

        ASSERT_LE(error, threshold)
            << "TensorOp: " << testCase.operation << std::endl
            << "A tensor: " << tensorA.desc.ToString() << std::endl
            << "B tensor: " << tensorB.desc.ToString() << std::endl
            << "IsPacked: " << testCase.packed << std::endl
            << "Offsets: " << testCase.offsets[0] << "," << testCase.offsets[1] << ","
            << testCase.offsets[2] << std::endl;
    }

private:
    tensor<T> tensorA;
    tensor<T> tensorB;
    tensor<T> tensorC;
};

using GPU_TensorOps_FP32 = TensorOpsCommon<float>;
using GPU_TensorOps_FP16 = TensorOpsCommon<half_float::half>;
using GPU_TensorOps_FP64 = TensorOpsCommon<double>;

namespace {
bool checkTensorsCompatibility(const std::vector<size_t>& tensorALens,
                               const std::vector<size_t>& tensorBLens)
{
    if(tensorALens.size() != tensorBLens.size())
    {
        return false;
    }

    for(size_t idx = 0; idx < tensorBLens.size(); ++idx)
    {
        if((tensorBLens[idx] != 1) && (tensorALens[idx] != tensorBLens[idx]))
        {
            return false;
        }
    }

    return true;
}

void AddTestCases(std::vector<TestCase>& testCases,
                  const std::vector<size_t>& tensorALens,
                  const std::vector<size_t>& tensorBLens)
{
    const auto& stride_a = stridesArr[0];
    const auto& stride_b = stridesArr[0];
    const auto& stride_c = stridesArr[0];

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

    for(const auto& tensorALens : tensorALensArr)
        for(const auto& tensorBLens : tensorBLensArr)
        {
            if(!checkTensorsCompatibility(tensorALens, tensorBLens))
            {
                continue;
            }

            AddTestCases(testCases, tensorALens, tensorBLens);
        }

    return testCases;
}

inline auto GetCases()
{
    static const auto cases = testing::ValuesIn(GenCases());
    return cases;
}
} // namespace

TEST_P(GPU_TensorOps_FP32, TestFloat) { this->Run(); }

TEST_P(GPU_TensorOps_FP16, TestFloat16) { this->Run(); }

TEST_P(GPU_TensorOps_FP64, TestDouble) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_TensorOps_FP32, GetCases());
INSTANTIATE_TEST_SUITE_P(Full, GPU_TensorOps_FP64, GetCases());
INSTANTIATE_TEST_SUITE_P(Full, GPU_TensorOps_FP16, GetCases());
