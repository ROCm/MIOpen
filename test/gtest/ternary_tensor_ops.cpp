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
#include "gtest_common.hpp"

#define MIO_OPS_DEBUG 0

static std::vector<std::vector<int>> tensorALensArr = {{32, 16, 8, 4, 4}, // tensor A
                                                       {16, 20, 16, 8},
                                                       {20, 16, 8},
                                                       {1, 16, 8},
                                                       {16, 8},
                                                       {8}};

static std::vector<std::vector<int>> tensorBLensArr = {{32, 16, 8, 4, 4}, // tensor B
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

static std::vector<std::vector<int64_t>> offsetsArr = {
    {0, 0, 0}, {64, 32, 16}, {32, 16, 32}, {32, 16, 32}};

static std::vector<std::vector<float>> alphabetaArr = {{1, 1, 0}, {-1, 1, 1}, {1.0, 0.5, 0.3}};

static std::vector<std::vector<int>> stridesArr = {{8 * 16 * 20 * 16, 8 * 16 * 20, 8 * 16, 8, 1}};

static std::vector<bool> packedArr = {true, false};

static std::vector<miopenTensorOp_t> operationArr = {
    miopenTensorOpAdd, miopenTensorOpMul, miopenTensorOpMin, miopenTensorOpMax};

struct TestCase
{
    std::vector<int> tensorlens_ac;
    std::vector<int> tensorlens_b;
    std::vector<int64_t> offsets;
    std::vector<int> stride_a;
    std::vector<int> stride_b;
    std::vector<int> stride_c;
    std::vector<float> alphabeta;
    bool packed;
    miopenTensorOp_t operation;
};

template <typename T>
struct TensorOpsCommon : public testing::TestWithParam<TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();

        CreateTensors();

        tensor<T> tensorGPU = std::move(CalculateOnGPU());
        tensor<T> tensorCPU = std::move(CalculateOnCPU());

        CompareResults(tensorGPU, tensorCPU);
    }

private:
    void CreateTensors()
    {
        const TestCase& testCase = GetParam();

        tensorA = std::move(CreateTensor(
            testCase.tensorlens_ac, testCase.stride_a, testCase.offsets[0], testCase.packed));
        tensorB = std::move(CreateTensor(
            testCase.tensorlens_b, testCase.stride_b, testCase.offsets[1], testCase.packed));
        tensorC = std::move(CreateTensor(
            testCase.tensorlens_ac, testCase.stride_c, testCase.offsets[2], testCase.packed));
    }

    tensor<T> CreateTensor(const std::vector<int>& lens,
                           const std::vector<int>& strides,
                           int offset,
                           bool isPacked)
    {
        uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

        if(!isPacked)
        {
            std::vector<int> real_strides(strides.begin() + (strides.size() - lens.size()),
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

    tensor<T> CalculateOnGPU() const
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

        auto r = tensorC;
        r.data = handle.Read<T>(c_dev, r.data.size());
#if(MIO_OPS_DEBUG)
        handle.Finish();
        auto clens    = r.desc.GetLengths();
        auto cstrides = r.desc.GetStrides();
        for(int i = 0; i < r.desc.GetElementSize(); i++)
            printf("GPU_C[%d]: %f\n", i, c.data[i + Coffset]);
#endif
        return r;
    }

    tensor<T> CalculateOnCPU()
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
                C = ((A * alpha1) < B * alpha2 ? A * alpha1 : B * alpha2) + C * beta;
            });
        }
        else
        {
            return CalculateOnCPUDataOp([alpha1, alpha2, beta](auto& C, auto A, auto B) {
                C = ((A * alpha1) > B * alpha2 ? A * alpha1 : B * alpha2) + C * beta;
            });
        }
    }

    template <typename DataOp>
    tensor<T> CalculateOnCPUDataOp(DataOp&& dataOp)
    {
        const TestCase& testCase = GetParam();

        auto r     = tensorC;
        auto clens = r.desc.GetLengths();
        auto blens = tensorB.desc.GetLengths();

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

#if(MIO_OPS_DEBUG)
        for(int i = 0; i < r.desc.GetElementSize(); i++)
            printf("CPU_C[%d]: %f\n", i, r.data[i + Coffset]);
#endif
        return r;
    }

    template <typename DataOp, typename Container>
    void operate_over_subtensor(DataOp&& dataOp,
                                Container& dstSuperTensor,
                                const Container& src1SuperTensor,
                                const Container& src2SuperTensor,
                                const miopen::TensorDescriptor& dstSubDesc,
                                const miopen::TensorDescriptor& src1SubDesc,
                                const miopen::TensorDescriptor& src2SubDesc,
                                const int64_t dstOffset,
                                const int64_t src1Offset,
                                const int64_t src2Offset)
    {
        const auto& dstStrides  = dstSubDesc.GetStrides();
        const auto& src1Strides = src1SubDesc.GetStrides();
        const auto& src2Strides = src2SubDesc.GetStrides();

        const auto& src1Lens = src1SubDesc.GetLengths();
        const auto& src2Lens = src2SubDesc.GetLengths();

        auto operate_over_subtensor_impl =
            [&, dataOp, max_dim = src1Lens.size() - 1](auto&& self,
                                                       const size_t current_dim,
                                                       const int64_t dstOff,
                                                       const int64_t src1Off,
                                                       const int64_t src2Off) -> void {
            const auto dstStride  = dstStrides[current_dim];
            const auto src1Stride = src1Strides[current_dim];
            const auto src2Stride = src2Strides[current_dim];
            const bool squashed   = src1Lens[current_dim] != src2Lens[current_dim];

            int64_t dstIdx  = dstOff;
            int64_t src1Idx = src1Off;
            int64_t src2Idx = src2Off;

            for(size_t i = 0; i < src1Lens[current_dim]; ++i)
            {
                if(current_dim < max_dim)
                {
                    self(self, current_dim + 1, dstIdx, src1Idx, src2Idx);
                }
                else
                {
                    dataOp(
                        dstSuperTensor[dstIdx], src1SuperTensor[src1Idx], src2SuperTensor[src2Idx]);
                }
                dstIdx += dstStride;
                src1Idx += src1Stride;
                src2Idx += squashed ? 0 : src2Stride;
            }
        };
        operate_over_subtensor_impl(
            operate_over_subtensor_impl, 0, dstOffset, src1Offset, src2Offset);
    }

    void CompareResults(const tensor<T>& tensorGPU, const tensor<T>& tensorCPU)
    {
        const TestCase& testCase = GetParam();

        double tolerance = 1;

        if(std::is_same_v<T, half_float::half>)
        {
            // taken from original c-test
            tolerance = 80;
        }

        double threshold = std::numeric_limits<T>::epsilon() * tolerance;
        double error     = miopen::rms_range(tensorCPU.data, tensorGPU.data);

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

struct GPU_TensorOps_FP32 : public TensorOpsCommon<float>
{
};

struct GPU_TensorOps_FP16 : public TensorOpsCommon<half_float::half>
{
};

struct GPU_TensorOps_FP64 : public TensorOpsCommon<double>
{
};

bool checkTensorsCompatibility(const std::vector<int>& tensorALens,
                               const std::vector<int>& tensorBLens)
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

template <typename T>
void AddTestCases(std::vector<TestCase>& testCases,
                  const std::vector<int> tensorALens,
                  const std::vector<int> tensorBLens)
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

            auto checkStride = [p = packed](const std::vector<int>& lens,
                                            const std::vector<int>& strides) {
                if(p)
                    return true;

                if(lens.size() > strides.size())
                    return false;

                // only sparsed case allowed, since all the kernels do not support the last
                // dimension strides
                if(strides.back() == 1)
                {
                    auto packedStrides =
                        miopen::TensorDescriptor(miopen_type<T>{}, lens).GetStrides();
                    return std::equal(packedStrides.rbegin(),
                                      packedStrides.rend(),
                                      strides.rbegin(),
                                      [](int ps, int s) { return s >= ps; });
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

template <typename T>
inline auto GetCases()
{
    static std::vector<TestCase> testCases;

    if(!testCases.empty())
    {
        return testing::ValuesIn(testCases);
    }

    for(const auto& tensorALens : tensorALensArr)
        for(const auto& tensorBLens : tensorBLensArr)
        {
            if(!checkTensorsCompatibility(tensorALens, tensorBLens))
            {
                continue;
            }

            AddTestCases<T>(testCases, tensorALens, tensorBLens);
        }

    return testing::ValuesIn(testCases);
}

TEST_P(GPU_TensorOps_FP32, TestFloat) {}

TEST_P(GPU_TensorOps_FP16, TestFloat16) {}

TEST_P(GPU_TensorOps_FP64, TestDouble) {}

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_TensorOps_FP32, GetCases<float>());
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_TensorOps_FP64, GetCases<double>());
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_TensorOps_FP16, GetCases<half_float::half>());
