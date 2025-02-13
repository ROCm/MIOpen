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

#include "cpu_matrix_diag.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/matrix_diag.hpp>

#include <vector>

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

struct MatrixDiagPartTestcase
{
    std::vector<size_t> inputSize;
    int diagOffset0;
    int diagOffset1;
    miopenMatrixDiagAlignMode_t align = MIOPEN_MATRIX_ALIGN_RIGHT_LEFT;

    friend std::ostream& operator<<(std::ostream& os, const MatrixDiagPartTestcase& tc)
    {
        return os << " inputSize:" << tc.inputSize << " offset0:" << tc.diagOffset0
                  << " offset1:" << tc.diagOffset1 << " Align:" << tc.align;
    }
};

inline std::vector<MatrixDiagPartTestcase>
MatrixDiagPartConfigs(const std::vector<MatrixDiagPartTestcase> configs)
{
    std::vector<MatrixDiagPartTestcase> tcs;
    const auto all_mode = {MIOPEN_MATRIX_ALIGN_LEFT_LEFT,
                           MIOPEN_MATRIX_ALIGN_LEFT_RIGHT,
                           MIOPEN_MATRIX_ALIGN_RIGHT_LEFT,
                           MIOPEN_MATRIX_ALIGN_RIGHT_RIGHT};
    for(auto config : configs)
        for(auto align : all_mode)
        {
            config.align = align;
            tcs.push_back(config);
        }
    return tcs;
}

inline std::vector<MatrixDiagPartTestcase> MatrixDiagPartFwdSmokeTestConfigs()
{
    return MatrixDiagPartConfigs({{{2, 3, 4}, -1, 2}, {{2, 3, 4}, -2, -1}, {{2, 3, 4}, 1, 3}});
}

inline std::vector<MatrixDiagPartTestcase> MatrixDiagPartFwdPerfTestConfigs()
{
    return MatrixDiagPartConfigs({{{1024, 1024, 1024}, -100, 100}});
}

inline std::vector<MatrixDiagPartTestcase> MatrixDiagPartFwdFullTestConfigs()
{
    std::vector<MatrixDiagPartTestcase> tcs;
    auto smoke_test = MatrixDiagPartFwdSmokeTestConfigs();
    auto perf_test  = MatrixDiagPartFwdPerfTestConfigs();
    tcs.insert(tcs.end(), smoke_test.begin(), smoke_test.end());
    tcs.insert(tcs.end(), perf_test.begin(), perf_test.end());
    return tcs;
}

inline std::vector<MatrixDiagPartTestcase> MatrixDiagPartBwdSmokeTestConfigs()
{
    return MatrixDiagPartConfigs({{{2, 3, 4}, 0, 0},
                                  {{2, 3, 4}, 1, 1},
                                  {{2, 3, 4}, -1, 2},
                                  {{2, 3, 4}, -2, -1},
                                  {{2, 3, 4}, 1, 3}});
}

inline std::vector<MatrixDiagPartTestcase> MatrixDiagPartBwdPerfTestConfigs()
{
    return MatrixDiagPartConfigs({{{1024, 1024, 1024}, -100, 100}});
}

inline std::vector<MatrixDiagPartTestcase> MatrixDiagPartBwdFullTestConfigs()
{
    std::vector<MatrixDiagPartTestcase> tcs;
    auto smoke_test = MatrixDiagPartBwdSmokeTestConfigs();
    auto perf_test  = MatrixDiagPartBwdPerfTestConfigs();
    tcs.insert(tcs.end(), smoke_test.begin(), smoke_test.end());
    tcs.insert(tcs.end(), perf_test.begin(), perf_test.end());
    return tcs;
}

template <typename TIO = float>
struct MatrixDiagPartTestForward : public ::testing::TestWithParam<MatrixDiagPartTestcase>
{
protected:
    void SetUp() override
    {
        auto&& handle          = get_handle();
        matrix_set_diag_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(1e-2, 100); };

        k0                = matrix_set_diag_config.diagOffset0;
        k1                = matrix_set_diag_config.diagOffset1;
        align             = matrix_set_diag_config.align;
        auto inputSize    = matrix_set_diag_config.inputSize;
        auto max_diag_len = std::min(inputSize[inputSize.size() - 2] + std::min(k1, 0L),
                                     inputSize[inputSize.size() - 1] + std::min(-k0, 0L));
        auto outputSize   = inputSize;
        if(k0 == k1)
            outputSize.pop_back();
        else
            outputSize[outputSize.size() - 2] = k1 - k0 + 1;
        outputSize.back() = max_diag_len;

        input = tensor<TIO>{inputSize}.generate(gen_value);

        pad    = tensor<TIO>{{1}};
        pad[0] = -1;

        output = tensor<TIO>{outputSize};
        std::fill(output.begin(), output.end(), std::numeric_limits<TIO>::quiet_NaN());

        ref_output = tensor<TIO>{outputSize};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<TIO>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        pad_dev    = handle.Write(pad.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_matrix_diag_part_forward(input, pad, ref_output, k0, k1, align);
        miopenStatus_t status = miopen::MatrixDiagPartForward(handle,
                                                              input.desc,
                                                              input_dev.get(),
                                                              pad.desc,
                                                              pad_dev.get(),
                                                              output.desc,
                                                              output_dev.get(),
                                                              k0,
                                                              k1,
                                                              align);
        ASSERT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<TIO>(output_dev, output.data.size());
    }

    void Verify()
    {
        auto error = miopen::rms_range(ref_output, output);

        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_EQ(error, 0) << "Error! Incorrect output!";
    }
    MatrixDiagPartTestcase matrix_set_diag_config;

    tensor<TIO> input;
    tensor<TIO> pad;
    tensor<TIO> output;

    tensor<TIO> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr pad_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    int64_t k0, k1;

    miopenMatrixDiagAlignMode_t align;
};

template <typename TIO = float>
struct MatrixDiagPartTestBackward : public ::testing::TestWithParam<MatrixDiagPartTestcase>
{
protected:
    void SetUp() override
    {
        auto&& handle          = get_handle();
        matrix_set_diag_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(1e-2, 100); };

        k0                = matrix_set_diag_config.diagOffset0;
        k1                = matrix_set_diag_config.diagOffset1;
        align             = matrix_set_diag_config.align;
        auto inputSize    = matrix_set_diag_config.inputSize;
        auto max_diag_len = std::min(inputSize[inputSize.size() - 2] + std::min(k1, 0L),
                                     inputSize[inputSize.size() - 1] + std::min(-k0, 0L));
        auto outputSize   = inputSize;
        if(k0 == k1)
            outputSize.pop_back();
        else
            outputSize[outputSize.size() - 2] = k1 - k0 + 1;
        outputSize.back() = max_diag_len;

        output_grad = tensor<TIO>{outputSize}.generate(gen_value);

        input_grad = tensor<TIO>{inputSize};
        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<TIO>::quiet_NaN());

        ref_input_grad = tensor<TIO>{inputSize};
        std::fill(
            ref_input_grad.begin(), ref_input_grad.end(), std::numeric_limits<TIO>::quiet_NaN());

        output_grad_dev = handle.Write(output_grad.data);
        input_grad_dev  = handle.Write(input_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        auto fake_pad = tensor<TIO>{{1}};
        fake_pad[0]   = 0;
        cpu_matrix_set_diag_forward(fake_pad, output_grad, ref_input_grad, k0, k1, true, align);
        miopenStatus_t status = miopen::MatrixDiagPartBackward(handle,
                                                               output_grad.desc,
                                                               output_grad_dev.get(),
                                                               input_grad.desc,
                                                               input_grad_dev.get(),
                                                               k0,
                                                               k1,
                                                               align);
        ASSERT_EQ(status, miopenStatusSuccess);

        input_grad.data = handle.Read<TIO>(input_grad_dev, input_grad.data.size());
    }

    void Verify()
    {
        auto error = miopen::rms_range(ref_input_grad, input_grad);

        ASSERT_EQ(miopen::range_distance(ref_input_grad), miopen::range_distance(input_grad));
        EXPECT_EQ(error, 0) << "Error! Incorrect input gradient!";
    }
    MatrixDiagPartTestcase matrix_set_diag_config;

    tensor<TIO> output_grad;
    tensor<TIO> input_grad;

    tensor<TIO> ref_input_grad;

    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;

    int64_t k0, k1;

    miopenMatrixDiagAlignMode_t align;
};
