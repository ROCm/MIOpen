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

struct MatrixDiagTestcase
{
    std::vector<size_t> diagSize;
    int diagOffset0;
    int diagOffset1;
    int num_rows;
    int num_cols;
    miopenMatrixDiagAlignMode_t align = MIOPEN_MATRIX_ALIGN_RIGHT_LEFT;

    friend std::ostream& operator<<(std::ostream& os, const MatrixDiagTestcase& tc)
    {
        return os << " DiagSize:" << tc.diagSize << " offset0:" << tc.diagOffset0
                  << " offset1:" << tc.diagOffset1 << " NRow:" << tc.num_rows
                  << " NCol:" << tc.num_cols << " Align:" << tc.align;
    }
};

inline std::vector<MatrixDiagTestcase>
MatrixDiagConfigs(const std::vector<MatrixDiagTestcase> configs)
{
    std::vector<MatrixDiagTestcase> tcs;
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

inline std::vector<MatrixDiagTestcase> MatrixDiagFwdSmokeTestConfigs()
{
    return MatrixDiagConfigs({{{2, 4}, 0, 0, 4, 4},
                              {{2, 3}, 1, 1, 4, 4},
                              {{2, 3, 3}, -1, 1, 3, 3},
                              {{2}, -1, -1, 3, 4},
                              {{2}, -1, -1, 3, 2},
                              {{2, 2}, 0, 0, 2, 2},
                              {{4, 2}, 0, 0, 2, 2}});
}

inline std::vector<MatrixDiagTestcase> MatrixDiagFwdPerfTestConfigs()
{
    return MatrixDiagConfigs({});
}

inline std::vector<MatrixDiagTestcase> MatrixDiagFwdFullTestConfigs()
{
    std::vector<MatrixDiagTestcase> tcs;
    auto smoke_test = MatrixDiagFwdSmokeTestConfigs();
    auto perf_test  = MatrixDiagFwdPerfTestConfigs();
    tcs.insert(tcs.end(), smoke_test.begin(), smoke_test.end());
    tcs.insert(tcs.end(), perf_test.begin(), perf_test.end());
    return tcs;
}

inline std::vector<MatrixDiagTestcase> MatrixDiagBwdSmokeTestConfigs()
{
    return MatrixDiagConfigs({{{2, 3, 3}, -1, 1, 3, 3}});
}

inline std::vector<MatrixDiagTestcase> MatrixDiagBwdPerfTestConfigs()
{
    return MatrixDiagConfigs({});
}

inline std::vector<MatrixDiagTestcase> MatrixDiagBwdFullTestConfigs()
{
    std::vector<MatrixDiagTestcase> tcs;
    auto smoke_test = MatrixDiagBwdSmokeTestConfigs();
    auto perf_test  = MatrixDiagBwdPerfTestConfigs();
    tcs.insert(tcs.end(), smoke_test.begin(), smoke_test.end());
    tcs.insert(tcs.end(), perf_test.begin(), perf_test.end());
    return tcs;
}

template <typename TIO = float>
struct MatrixDiagTestForward : public ::testing::TestWithParam<MatrixDiagTestcase>
{
protected:
    void SetUp() override
    {
        auto&& handle      = get_handle();
        matrix_diag_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(1e-2, 100); };

        k0            = matrix_diag_config.diagOffset0;
        k1            = matrix_diag_config.diagOffset1;
        align         = matrix_diag_config.align;
        auto diagSize = matrix_diag_config.diagSize;
        auto num_rows = matrix_diag_config.num_rows;
        auto num_cols = matrix_diag_config.num_cols;

        diag   = tensor<TIO>{diagSize}.generate(gen_value);
        pad    = tensor<TIO>{{1}};
        pad[0] = -1;

        auto outputSize = diagSize;
        if(k0 == k1)
            outputSize.push_back(0);
        outputSize[outputSize.size() - 2] = num_rows;
        outputSize[outputSize.size() - 1] = num_cols;

        output = tensor<TIO>{outputSize};
        std::fill(output.begin(), output.end(), std::numeric_limits<TIO>::quiet_NaN());

        ref_output = tensor<TIO>{outputSize};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<TIO>::quiet_NaN());

        diag_dev   = handle.Write(diag.data);
        pad_dev    = handle.Write(pad.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_matrix_set_diag_forward(pad, diag, ref_output, k0, k1, true, align);
        miopenStatus_t status = miopen::MatrixDiagForward(handle,
                                                          diag.desc,
                                                          diag_dev.get(),
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
    MatrixDiagTestcase matrix_diag_config;

    tensor<TIO> diag;
    tensor<TIO> pad;
    tensor<TIO> output;

    tensor<TIO> ref_output;

    miopen::Allocator::ManageDataPtr diag_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr pad_dev;

    int64_t k0, k1;

    miopenMatrixDiagAlignMode_t align;
};

template <typename TIO = float>
struct MatrixDiagTestBackward : public ::testing::TestWithParam<MatrixDiagTestcase>
{
protected:
    void SetUp() override
    {
        auto&& handle      = get_handle();
        matrix_diag_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(1e-2, 100); };

        k0            = matrix_diag_config.diagOffset0;
        k1            = matrix_diag_config.diagOffset1;
        align         = matrix_diag_config.align;
        auto diagSize = matrix_diag_config.diagSize;
        auto num_rows = matrix_diag_config.num_rows;
        auto num_cols = matrix_diag_config.num_cols;

        pad    = tensor<TIO>{{1}};
        pad[0] = 0;

        auto outputSize = diagSize;
        if(k0 == k1)
            outputSize.push_back(0);
        outputSize[outputSize.size() - 2] = num_rows;
        outputSize[outputSize.size() - 1] = num_cols;
        output_grad                       = tensor<TIO>{outputSize}.generate(gen_value);

        diag_grad = tensor<TIO>{diagSize};
        std::fill(diag_grad.begin(), diag_grad.end(), std::numeric_limits<TIO>::quiet_NaN());

        ref_diag_grad = tensor<TIO>{diagSize};
        std::fill(
            ref_diag_grad.begin(), ref_diag_grad.end(), std::numeric_limits<TIO>::quiet_NaN());

        output_grad_dev = handle.Write(output_grad.data);
        diag_grad_dev   = handle.Write(diag_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_matrix_diag_part_forward(output_grad, pad, ref_diag_grad, k0, k1, align);
        miopenStatus_t status = miopen::MatrixDiagBackward(handle,
                                                           output_grad.desc,
                                                           output_grad_dev.get(),
                                                           diag_grad.desc,
                                                           diag_grad_dev.get(),
                                                           k0,
                                                           k1,
                                                           align);
        ASSERT_EQ(status, miopenStatusSuccess);

        diag_grad.data = handle.Read<TIO>(diag_grad_dev, diag_grad.data.size());
    }

    void Verify()
    {
        auto error = miopen::rms_range(ref_diag_grad, diag_grad);

        ASSERT_EQ(miopen::range_distance(ref_diag_grad), miopen::range_distance(diag_grad));
        EXPECT_EQ(error, 0) << "Error! Incorrect diagonal gradient!";
    }
    MatrixDiagTestcase matrix_diag_config;

    tensor<TIO> pad;
    tensor<TIO> output_grad;
    tensor<TIO> diag_grad;

    tensor<TIO> ref_diag_grad;

    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr diag_grad_dev;

    int64_t k0, k1;

    miopenMatrixDiagAlignMode_t align;
};
