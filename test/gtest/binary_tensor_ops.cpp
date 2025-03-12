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
using BinaryTensorOpsCase = std::tuple<std::vector<size_t>, std::vector<int>, float, bool>;

template <typename DstType, typename SrcType = DstType>
class GPU_binaryTensorOps : public ::testing::TestWithParam<BinaryTensorOpsCase>
{
public:
    static tensor<DstType> dstSuperTensor;
    static tensor<SrcType> srcSuperTensor;

    static void SetUpTestSuite()
    {
        static constexpr auto dstType = miopen_type<DstType>{};
        static constexpr auto srcType = miopen_type<SrcType>{};

        uint64_t dstMaxValue = dstType == miopenHalf ? 5 : (dstType == miopenInt8 ? 126 : 32767);
        uint64_t srcMaxValue = srcType == miopenHalf ? 5 : (srcType == miopenInt8 ? 126 : 32767);

        uint64_t maxValue = std::min(dstMaxValue, srcMaxValue);

        dstSuperTensor = tensor<DstType>{std::vector<size_t>{32, 32, 16, 16, 16}}.generate(
            tensor_elem_gen_integer{maxValue});

        srcSuperTensor = tensor<SrcType>{std::vector<size_t>{32, 16, 32, 16, 16}}.generate(
            tensor_elem_gen_integer{maxValue});
    }

protected:
    size_t dstDataSize;
    size_t srcDataSize;
    miopen::TensorDescriptor dstDesc;
    miopen::TensorDescriptor srcDesc;

    void SetUp() override
    {
        const auto& [lens, offsets, alpha, clamp] = GetParam();

        ASSERT_GE(dstSuperTensor.desc.GetNumDims(), lens.size());

        const std::vector<size_t>& dstSuperStrides = dstSuperTensor.desc.GetStrides();
        std::vector<size_t> dstStrides(dstSuperStrides.begin() +
                                           (dstSuperTensor.desc.GetNumDims() - lens.size()),
                                       dstSuperStrides.end());

        dstDesc     = miopen::TensorDescriptor(miopen_type<DstType>{}, lens, dstStrides);
        dstDataSize = dstDesc.GetElementSpace() + offsets[1];

        ASSERT_GE(srcSuperTensor.desc.GetElementSpace(), dstDataSize);

        ASSERT_GE(srcSuperTensor.desc.GetNumDims(), lens.size());

        const std::vector<size_t>& srcSuperStrides = srcSuperTensor.desc.GetStrides();
        std::vector<size_t> srcStrides(srcSuperStrides.begin() +
                                           (srcSuperTensor.desc.GetNumDims() - lens.size()),
                                       srcSuperStrides.end());

        srcDesc     = miopen::TensorDescriptor(miopen_type<SrcType>{}, lens, srcStrides);
        srcDataSize = srcDesc.GetElementSpace() + offsets[0];

        ASSERT_GE(srcSuperTensor.desc.GetElementSpace(), srcDataSize);
    }

    void RunCast()
    {
        std::vector<DstType> dstSuperCpu(dstSuperTensor.begin(),
                                         dstSuperTensor.begin() + dstDataSize);
        std::vector<SrcType> srcSuperCpu(srcSuperTensor.begin(),
                                         srcSuperTensor.begin() + srcDataSize);

        const auto [srcOffset, dstOffset] = miopen::tien<2>(std::get<std::vector<int>>(GetParam()));
        const auto alpha                  = std::get<float>(GetParam());
        const auto clamp                  = std::get<bool>(GetParam());

        auto&& handle     = get_handle();
        auto dstSuper_dev = handle.Write(dstSuperCpu);
        auto srcSuper_dev = handle.Write(srcSuperCpu);

        miopen::CastTensor(handle,
                           &alpha,
                           clamp,
                           srcDesc,
                           srcSuper_dev.get(),
                           dstDesc,
                           dstSuper_dev.get(),
                           srcOffset,
                           dstOffset);

        auto result = handle.Read<DstType>(dstSuper_dev, dstDataSize);

        if(clamp)
        {
            operate_over_subtensor(
                [alpha, clampVal = static_cast<float>(std::numeric_limits<DstType>::max())](
                    auto& dst, auto src) {
                    dst = std::min(static_cast<float>(src) * alpha, clampVal);
                },
                dstSuperCpu,
                srcSuperCpu,
                dstDesc,
                srcDesc,
                dstOffset,
                srcOffset);
        }
        else
        {
            operate_over_subtensor(
                [alpha](auto& dst, auto src) { dst = static_cast<float>(src) * alpha; },
                dstSuperCpu,
                srcSuperCpu,
                dstDesc,
                srcDesc,
                dstOffset,
                srcOffset);
        }

        auto mismatch_index     = miopen::mismatch_idx(dstSuperCpu, result, miopen::float_equal);
        auto mismatch_src_index = mismatch_index - dstOffset + srcOffset;

        ASSERT_EQ(result.size(), mismatch_index)
            << "The first mismatched elements are:"                                     //
            << " Src[" << mismatch_src_index << "] " << srcSuperCpu[mismatch_src_index] //
            << " GPU[" << mismatch_index << "] " << result[mismatch_index]              //
            << " Ref[" << mismatch_index << "] " << dstSuperCpu[mismatch_index];        //
    }

    void RunCopy()
    {
        std::vector<DstType> dstSuperCpu(dstSuperTensor.begin(),
                                         dstSuperTensor.begin() + dstDataSize);
        std::vector<SrcType> srcSuperCpu(srcSuperTensor.begin(),
                                         srcSuperTensor.begin() + srcDataSize);

        const auto [srcOffset, dstOffset] = miopen::tien<2>(std::get<std::vector<int>>(GetParam()));

        auto&& handle     = get_handle();
        auto dstSuper_dev = handle.Write(dstSuperCpu);
        auto srcSuper_dev = handle.Write(srcSuperCpu);

        miopen::CopyTensor(
            handle, srcDesc, srcSuper_dev.get(), dstDesc, dstSuper_dev.get(), srcOffset, dstOffset);

        auto result = handle.Read<DstType>(dstSuper_dev, dstDataSize);

        operate_over_subtensor([](auto& dst, auto src) { dst = src; },
                               dstSuperCpu,
                               srcSuperCpu,
                               dstDesc,
                               srcDesc,
                               dstOffset,
                               srcOffset);

        auto mismatch_index     = miopen::mismatch_idx(dstSuperCpu, result, miopen::float_equal);
        auto mismatch_src_index = mismatch_index - dstOffset + srcOffset;

        ASSERT_EQ(result.size(), mismatch_index)
            << "The first mismatched elements are:"                                     //
            << " Src[" << mismatch_src_index << "] " << srcSuperCpu[mismatch_src_index] //
            << " GPU[" << mismatch_index << "] " << result[mismatch_index]              //
            << " Ref[" << mismatch_index << "] " << dstSuperCpu[mismatch_index];        //
    }

    void TearDown() override {}
};

template <typename DstType, typename SrcType>
tensor<DstType> GPU_binaryTensorOps<DstType, SrcType>::dstSuperTensor;

template <typename DstType, typename SrcType>
tensor<SrcType> GPU_binaryTensorOps<DstType, SrcType>::srcSuperTensor;
} // namespace

using float16 = half_float::half;

#define X_CONCAT_FIRST_SECOND_(first, second) first##second

#define X_INSTANTIATE_CAST(TEST_TYPE, DST_TYPE, SRC_TYPE, ...)                            \
    using GPU_binaryTensorOps_cast_##SRC_TYPE##_##TEST_TYPE =                             \
        GPU_binaryTensorOps<DST_TYPE, SRC_TYPE>;                                          \
    TEST_P(GPU_binaryTensorOps_cast_##SRC_TYPE##_##TEST_TYPE,                             \
           X_CONCAT_FIRST_SECOND_(__VA_ARGS__, TestTensorCast))                           \
    {                                                                                     \
        RunCast();                                                                        \
    };                                                                                    \
                                                                                          \
    INSTANTIATE_TEST_SUITE_P(                                                             \
        Smoke,                                                                            \
        GPU_binaryTensorOps_cast_##SRC_TYPE##_##TEST_TYPE,                                \
        testing::Combine(testing::Values(std::vector<size_t>{32, 8, 10}),                 \
                         testing::Values(std::vector<int>{7, 11}),                        \
                         testing::ValuesIn({1.0f / 127 / 127, 1.0f / 127, 127.0f, 1.0f}), \
                         testing::Values(true, false)));                                  \
                                                                                          \
    INSTANTIATE_TEST_SUITE_P(                                                             \
        Full,                                                                             \
        GPU_binaryTensorOps_cast_##SRC_TYPE##_##TEST_TYPE,                                \
        testing::Combine(testing::ValuesIn(get_sub_tensor<size_t>()),                     \
                         testing::ValuesIn(get_tensor_offsets()),                         \
                         testing::ValuesIn({1.0f / 127 / 127, 1.0f / 127, 127.0f, 1.0f}), \
                         testing::Values(true, false)));

X_INSTANTIATE_CAST(FP32, float, float);
X_INSTANTIATE_CAST(FP16, float16, float);
X_INSTANTIATE_CAST(BFP16, bfloat16, float);
X_INSTANTIATE_CAST(I32, int, float);
X_INSTANTIATE_CAST(I8, int8_t, float);

X_INSTANTIATE_CAST(FP32, float, float16);
X_INSTANTIATE_CAST(FP16, float16, float16);
X_INSTANTIATE_CAST(BFP16, bfloat16, float16, DISABLED_);
X_INSTANTIATE_CAST(I32, int, float16);
X_INSTANTIATE_CAST(I8, int8_t, float16);

X_INSTANTIATE_CAST(FP32,
                   float,
                   bfloat16,
                   DISABLED_); // bfp16 is just broken except float->bfp16 case
X_INSTANTIATE_CAST(FP16, float16, bfloat16, DISABLED_);
X_INSTANTIATE_CAST(BFP16, bfloat16, bfloat16, DISABLED_);
X_INSTANTIATE_CAST(I32, int, bfloat16, DISABLED_);
X_INSTANTIATE_CAST(I8, int8_t, bfloat16, DISABLED_);

X_INSTANTIATE_CAST(FP32, float, int);
X_INSTANTIATE_CAST(FP16, float16, int);
X_INSTANTIATE_CAST(BFP16, bfloat16, int, DISABLED_);
X_INSTANTIATE_CAST(I32, int, int);
X_INSTANTIATE_CAST(I8, int8_t, int);

X_INSTANTIATE_CAST(FP32, float, int8_t);
X_INSTANTIATE_CAST(FP16, float16, int8_t);
X_INSTANTIATE_CAST(BFP16, bfloat16, int8_t, DISABLED_);
X_INSTANTIATE_CAST(I32, int, int8_t);
X_INSTANTIATE_CAST(I8, int8_t, int8_t);

#undef X_INSTANTIATE_CAST
#undef X_CONCAT_FIRST_SECOND_

#define X_INSTANTIATE_COPY(TEST_TYPE, REAL_TYPE)                                               \
    using GPU_binaryTensorOps_copy_##TEST_TYPE = GPU_binaryTensorOps<REAL_TYPE>;               \
    TEST_P(GPU_binaryTensorOps_copy_##TEST_TYPE, TestTensorCopy) { RunCopy(); };               \
                                                                                               \
    INSTANTIATE_TEST_SUITE_P(Smoke,                                                            \
                             GPU_binaryTensorOps_copy_##TEST_TYPE,                             \
                             testing::Combine(testing::Values(std::vector<size_t>{32, 8, 10}), \
                                              testing::Values(std::vector<int>{7, 11}),        \
                                              testing::Values(0.0f),                           \
                                              testing::Values(false)));                        \
                                                                                               \
    INSTANTIATE_TEST_SUITE_P(Full,                                                             \
                             GPU_binaryTensorOps_copy_##TEST_TYPE,                             \
                             testing::Combine(testing::ValuesIn(get_sub_tensor<size_t>()),     \
                                              testing::ValuesIn(get_tensor_offsets()),         \
                                              testing::Values(0.0f),                           \
                                              testing::Values(false)));

X_INSTANTIATE_COPY(FP32, float);
X_INSTANTIATE_COPY(FP16, float16);
X_INSTANTIATE_COPY(BFP16, bfloat16);

#undef X_INSTANTIATE_COPY
