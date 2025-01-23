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

struct MatrixSetDiagTestcase
{
    std::vector<size_t> inputSize;
    std::vector<size_t> diagSize;
    int diagOffset0;
    int diagOffset1;
    miopenMatrixDiagAlignMode_t align = MIOPEN_MATRIX_ALIGN_RIGHT_LEFT;

    friend std::ostream& operator<<(std::ostream& os, const MatrixSetDiagTestcase& tc)
    {
        return os << " inputSize:" << tc.inputSize << " DiagSize:" << tc.diagSize
                  << " offset0:" << tc.diagOffset0 << " offset1:" << tc.diagOffset1
                  << " Align:" << tc.align;
    }
};

inline std::vector<MatrixSetDiagTestcase>
MatrixSetDiagConfigs(const std::vector<MatrixSetDiagTestcase> configs)
{
    std::vector<MatrixSetDiagTestcase> tcs;
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

inline std::vector<MatrixSetDiagTestcase> MatrixSetDiagFwdSmokeTestConfigs()
{
    return MatrixSetDiagConfigs(
        {{{2, 3, 4}, {2, 3}, 0, 0}, {{2, 3, 4}, {2, 3}, 1, 1}, {{2, 3, 4}, {2, 4, 3}, -1, 2}});
}

inline std::vector<MatrixSetDiagTestcase> MatrixSetDiagFwdPerfTestConfigs()
{
    return MatrixSetDiagConfigs({});
}

inline std::vector<MatrixSetDiagTestcase> MatrixSetDiagFwdFullTestConfigs()
{
    std::vector<MatrixSetDiagTestcase> tcs;
    auto smoke_test = MatrixSetDiagFwdSmokeTestConfigs();
    auto perf_test  = MatrixSetDiagFwdPerfTestConfigs();
    tcs.insert(tcs.end(), smoke_test.begin(), smoke_test.end());
    tcs.insert(tcs.end(), perf_test.begin(), perf_test.end());
    return tcs;
}

inline std::vector<MatrixSetDiagTestcase> MatrixSetDiagBwdSmokeTestConfigs()
{
    return MatrixSetDiagConfigs(
        {{{2, 3, 4}, {2, 3}, 0, 0}, {{2, 3, 4}, {2, 3}, 1, 1}, {{2, 3, 4}, {2, 4, 3}, -1, 2}});
}

inline std::vector<MatrixSetDiagTestcase> MatrixSetDiagBwdPerfTestConfigs()
{
    return MatrixSetDiagConfigs({});
}

inline std::vector<MatrixSetDiagTestcase> MatrixSetDiagBwdFullTestConfigs()
{
    std::vector<MatrixSetDiagTestcase> tcs;
    auto smoke_test = MatrixSetDiagBwdSmokeTestConfigs();
    auto perf_test  = MatrixSetDiagBwdPerfTestConfigs();
    tcs.insert(tcs.end(), smoke_test.begin(), smoke_test.end());
    tcs.insert(tcs.end(), perf_test.begin(), perf_test.end());
    return tcs;
}

template <typename TIO = float>
struct MatrixSetDiagTestForward : public ::testing::TestWithParam<MatrixSetDiagTestcase>
{
protected:
    void SetUp() override
    {
        auto&& handle          = get_handle();
        matrix_set_diag_config = GetParam();
        auto gen_value1 = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(1e-2, 100); };
        auto gen_value2 = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(1e-2, 100); };

        k0              = matrix_set_diag_config.diagOffset0;
        k1              = matrix_set_diag_config.diagOffset1;
        align           = matrix_set_diag_config.align;
        auto inputSize  = matrix_set_diag_config.inputSize;
        auto diagSize   = matrix_set_diag_config.diagSize;
        auto outputSize = inputSize;

        input = tensor<TIO>{inputSize}.generate(gen_value1);
        diag  = tensor<TIO>{diagSize}.generate(gen_value2);

        output = tensor<TIO>{outputSize};
        std::fill(output.begin(), output.end(), std::numeric_limits<TIO>::quiet_NaN());

        ref_output = tensor<TIO>{outputSize};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<TIO>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        diag_dev   = handle.Write(diag.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_matrix_set_diag_forward(input, diag, ref_output, k0, k1, true, align);
        miopenStatus_t status = miopen::MatrixSetDiagForward(handle,
                                                             input.desc,
                                                             input_dev.get(),
                                                             diag.desc,
                                                             diag_dev.get(),
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
    MatrixSetDiagTestcase matrix_set_diag_config;

    tensor<TIO> input;
    tensor<TIO> diag;
    tensor<TIO> output;

    tensor<TIO> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr diag_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    int64_t k0, k1;

    miopenMatrixDiagAlignMode_t align;
};

template <typename TIO = float>
struct MatrixSetDiagTestBackward : public ::testing::TestWithParam<MatrixSetDiagTestcase>
{
protected:
    void SetUp() override
    {
        auto&& handle          = get_handle();
        matrix_set_diag_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(1e-2, 100); };

        k0              = matrix_set_diag_config.diagOffset0;
        k1              = matrix_set_diag_config.diagOffset1;
        align           = matrix_set_diag_config.align;
        auto inputSize  = matrix_set_diag_config.inputSize;
        auto diagSize   = matrix_set_diag_config.diagSize;
        auto outputSize = inputSize;

        output_grad = tensor<TIO>{outputSize}.generate(gen_value);

        input_grad = tensor<TIO>{inputSize};
        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<TIO>::quiet_NaN());

        diag_grad = tensor<TIO>{diagSize};
        std::fill(diag_grad.begin(), diag_grad.end(), std::numeric_limits<TIO>::quiet_NaN());

        ref_input_grad = tensor<TIO>{inputSize};
        std::fill(
            ref_input_grad.begin(), ref_input_grad.end(), std::numeric_limits<TIO>::quiet_NaN());

        ref_diag_grad = tensor<TIO>{diagSize};
        std::fill(
            ref_diag_grad.begin(), ref_diag_grad.end(), std::numeric_limits<TIO>::quiet_NaN());

        output_grad_dev = handle.Write(output_grad.data);
        input_grad_dev  = handle.Write(input_grad.data);
        diag_grad_dev   = handle.Write(diag_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_matrix_set_diag_backward(output_grad, ref_input_grad, ref_diag_grad, k0, k1, align);
        miopenStatus_t status = miopen::MatrixSetDiagBackward(handle,
                                                              output_grad.desc,
                                                              output_grad_dev.get(),
                                                              input_grad.desc,
                                                              input_grad_dev.get(),
                                                              diag_grad.desc,
                                                              diag_grad_dev.get(),
                                                              k0,
                                                              k1,
                                                              align);
        ASSERT_EQ(status, miopenStatusSuccess);

        input_grad.data = handle.Read<TIO>(input_grad_dev, input_grad.data.size());
        diag_grad.data  = handle.Read<TIO>(diag_grad_dev, diag_grad.data.size());
    }

    void Verify()
    {
        auto error_input_grad = miopen::rms_range(ref_input_grad, input_grad);
        ASSERT_EQ(miopen::range_distance(ref_input_grad), miopen::range_distance(input_grad));
        EXPECT_EQ(error_input_grad, 0) << "Error! Incorrect input gradient!";

        auto error_diag_grad = miopen::rms_range(ref_diag_grad, diag_grad);
        ASSERT_EQ(miopen::range_distance(ref_diag_grad), miopen::range_distance(diag_grad));
        EXPECT_EQ(error_diag_grad, 0) << "Error! Incorrect diagonal gradient!";
    }
    MatrixSetDiagTestcase matrix_set_diag_config;

    tensor<TIO> input_grad;
    tensor<TIO> diag_grad;
    tensor<TIO> output_grad;

    tensor<TIO> ref_input_grad;
    tensor<TIO> ref_diag_grad;

    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr diag_grad_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;

    int64_t k0, k1;

    miopenMatrixDiagAlignMode_t align;
};
