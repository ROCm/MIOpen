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

#include <gtest/gtest.h>

#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>

#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "tensor_util.hpp"
#include "verify.hpp"

namespace {
using UnaryTensorOpsCase = std::tuple<std::vector<size_t>, int>;

template <typename T>
class GPU_unaryTensorOps : public ::testing::TestWithParam<UnaryTensorOpsCase>
{
public:
    static tensor<T> superTensor;

    static void SetUpTestSuite()
    {
        uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;
        superTensor        = tensor<T>{std::vector<size_t>{32, 32, 16, 16, 16}}.generate(
            tensor_elem_gen_integer{max_value});
    }

protected:
    const T alpha = static_cast<T>(2.048);
    size_t dataSize;
    miopen::TensorDescriptor subDesc;

    void SetUp() override
    {
        const auto& [lens, offset] = GetParam();
        ASSERT_GE(superTensor.desc.GetNumDims(), lens.size());

        const std::vector<size_t>& superStrides = superTensor.desc.GetStrides();
        std::vector<size_t> strides(superStrides.begin() +
                                        (superTensor.desc.GetNumDims() - lens.size()),
                                    superStrides.end());

        subDesc  = miopen::TensorDescriptor(miopen_type<T>{}, lens, strides);
        dataSize = subDesc.GetElementSpace() + offset;
        ASSERT_GE(superTensor.desc.GetElementSpace(), dataSize);
    }

    template <typename DataOp, typename GpuOp>
    void Run(DataOp&& dataOp, GpuOp&& gpuOp)
    {
        std::vector<T> superCpu(superTensor.begin(), superTensor.begin() + dataSize);
        auto offset = std::get<int>(GetParam());

        auto&& handle  = get_handle();
        auto super_dev = handle.Write(superCpu);
        gpuOp(handle, subDesc, super_dev.get(), &alpha, offset);
        auto result = handle.Read<T>(super_dev, dataSize);

        operate_over_subtensor(dataOp, superCpu, subDesc, offset);

        auto mismatch_index = miopen::mismatch_idx(superCpu, result, miopen::float_equal);

        ASSERT_EQ(result.size(), mismatch_index)
            << "The first mismatched elements are:"                           //
            << " GPU[" << mismatch_index << "] " << result[mismatch_index]    //
            << " Ref[" << mismatch_index << "] " << superCpu[mismatch_index]; //
    }

    void RunScale()
    {
        Run([a = alpha](auto& val) { val *= a; },
            [](auto&&... params) { miopen::ScaleTensor(params...); });
    }

    void RunSet()
    {
        Run([a = alpha](auto& val) { val = a; },
            [](auto&&... params) { miopen::SetTensor(params...); });
    }

    void TearDown() override {}
};

template <typename T>
tensor<T> GPU_unaryTensorOps<T>::superTensor;

} // namespace

using float16 = half_float::half;

#define X_CONCAT_FIRST_SECOND_(first, second) first##second

#define X_INSTANTIATE(TEST_TYPE, REAL_TYPE, ...)                                                 \
    using GPU_unaryTensorOps_##TEST_TYPE = GPU_unaryTensorOps<REAL_TYPE>;                        \
    TEST_P(GPU_unaryTensorOps_##TEST_TYPE, X_CONCAT_FIRST_SECOND_(__VA_ARGS__, TestTensorScale)) \
    {                                                                                            \
        RunScale();                                                                              \
    };                                                                                           \
    TEST_P(GPU_unaryTensorOps_##TEST_TYPE, TestTensorSet) { RunSet(); };                         \
                                                                                                 \
    INSTANTIATE_TEST_SUITE_P(                                                                    \
        Smoke,                                                                                   \
        GPU_unaryTensorOps_##TEST_TYPE,                                                          \
        testing::Combine(testing::Values(std::vector<size_t>{32, 8, 10}), testing::Values(7)));  \
                                                                                                 \
    INSTANTIATE_TEST_SUITE_P(Full,                                                               \
                             GPU_unaryTensorOps_##TEST_TYPE,                                     \
                             testing::Combine(testing::ValuesIn(get_sub_tensor<size_t>()),       \
                                              testing::ValuesIn(get_tensor_offset())));

X_INSTANTIATE(FP32, float);
X_INSTANTIATE(FP16, float16);
X_INSTANTIATE(I32, int);
X_INSTANTIATE(I8, int8_t, DISABLED_);      // disable Scale for int8
X_INSTANTIATE(BFP16, bfloat16, DISABLED_); // disable Scale for bfloat16

#undef X_INSTANTIATE
#undef X_CONCAT_FIRST_SECOND_
