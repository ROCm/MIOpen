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

inline std::vector<MatrixDiagPartTestcase> MatrixDiagPartSmokeTestConfigs()
{
    return MatrixDiagPartConfigs({{{2, 3, 4}, 0, 0},
                                  {{2, 3, 4}, 1, 1},
                                  {{2, 3, 4}, -1, 2},
                                  {{2, 3, 4}, -2, -1},
                                  {{2, 3, 4}, 1, 3}});
}

inline std::vector<MatrixDiagPartTestcase> MatrixDiagPartPerfTestConfigs()
{
    return MatrixDiagPartConfigs({});
}

inline std::vector<MatrixDiagPartTestcase> MatrixDiagPartFullTestConfigs()
{
    std::vector<MatrixDiagPartTestcase> tcs;

    auto smoke_test = MatrixDiagPartSmokeTestConfigs();
    auto perf_test  = MatrixDiagPartPerfTestConfigs();

    tcs.reserve(smoke_test.size() + perf_test.size());
    for(const auto& test : smoke_test)
        tcs.push_back(test);
    for(const auto& test : perf_test)
        tcs.push_back(test);

    return tcs;
}

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
        auto outSize      = inputSize;
        if(k0 == k1)
            outSize.pop_back();
        else
            outSize[outSize.size() - 2] = k1 - k0 + 1;
        outSize.back() = max_diag_len;

        doutput = tensor<TIO>{outSize}.generate(gen_value);

        pad    = tensor<TIO>{{1}};
        pad[0] = -1;

        dinput = tensor<TIO>{inputSize};
        std::fill(dinput.begin(), dinput.end(), std::numeric_limits<TIO>::quiet_NaN());

        ref_dinput = tensor<TIO>{inputSize};
        std::fill(ref_dinput.begin(), ref_dinput.end(), std::numeric_limits<TIO>::quiet_NaN());

        pad_dev     = handle.Write(pad.data);
        doutput_dev = handle.Write(doutput.data);
        dinput_dev  = handle.Write(dinput.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_matrix_set_diag(pad, doutput, ref_dinput, k0, k1, true, align);
        miopenStatus_t status = miopen::MatrixDiagPartBackward(handle,
                                                               pad.desc,
                                                               pad_dev.get(),
                                                               doutput.desc,
                                                               doutput_dev.get(),
                                                               dinput.desc,
                                                               dinput_dev.get(),
                                                               k0,
                                                               k1,
                                                               align);
        ASSERT_EQ(status, miopenStatusSuccess);

        dinput.data = handle.Read<TIO>(dinput_dev, dinput.data.size());
    }

    void Verify()
    {
        auto error = miopen::rms_range(ref_dinput, dinput);

        ASSERT_EQ(miopen::range_distance(ref_dinput), miopen::range_distance(dinput));
        EXPECT_EQ(error, 0) << "Error! Incorrect input gradient!";
    }
    MatrixDiagPartTestcase matrix_set_diag_config;

    tensor<TIO> pad;
    tensor<TIO> doutput;
    tensor<TIO> dinput;

    tensor<TIO> ref_dinput;

    miopen::Allocator::ManageDataPtr pad_dev;
    miopen::Allocator::ManageDataPtr doutput_dev;
    miopen::Allocator::ManageDataPtr dinput_dev;

    int64_t k0, k1;

    miopenMatrixDiagAlignMode_t align;
};
