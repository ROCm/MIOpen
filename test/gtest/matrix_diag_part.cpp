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

#include "matrix_diag_part.hpp"
#include <miopen/bfloat16.hpp>

namespace matrix_diag {

using GPU_MatrixDiagPartForward_FP32   = MatrixDiagPartTestForward<float>;
using GPU_MatrixDiagPartForward_FP16   = MatrixDiagPartTestForward<half>;
using GPU_MatrixDiagPartForward_BFP16  = MatrixDiagPartTestForward<bfloat16>;
using GPU_MatrixDiagPartBackward_FP32  = MatrixDiagPartTestBackward<float>;
using GPU_MatrixDiagPartBackward_FP16  = MatrixDiagPartTestBackward<half>;
using GPU_MatrixDiagPartBackward_BFP16 = MatrixDiagPartTestBackward<bfloat16>;

} // namespace matrix_diag
using namespace matrix_diag;

TEST_P(GPU_MatrixDiagPartForward_FP32, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_MatrixDiagPartForward_FP16, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_MatrixDiagPartForward_BFP16, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_MatrixDiagPartBackward_FP32, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_MatrixDiagPartBackward_FP16, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_MatrixDiagPartBackward_BFP16, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_MatrixDiagPartForward_FP32,
                         testing::ValuesIn(MatrixDiagPartSmokeTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_MatrixDiagPartForward_FP16,
                         testing::ValuesIn(MatrixDiagPartSmokeTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_MatrixDiagPartForward_BFP16,
                         testing::ValuesIn(MatrixDiagPartSmokeTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_MatrixDiagPartForward_FP32,
                         testing::ValuesIn(MatrixDiagPartPerfTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_MatrixDiagPartForward_FP16,
                         testing::ValuesIn(MatrixDiagPartPerfTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_MatrixDiagPartForward_BFP16,
                         testing::ValuesIn(MatrixDiagPartPerfTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MatrixDiagPartForward_FP32,
                         testing::ValuesIn(MatrixDiagPartFullTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MatrixDiagPartForward_FP16,
                         testing::ValuesIn(MatrixDiagPartFullTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MatrixDiagPartForward_BFP16,
                         testing::ValuesIn(MatrixDiagPartFullTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_MatrixDiagPartBackward_FP32,
                         testing::ValuesIn(MatrixDiagPartSmokeTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_MatrixDiagPartBackward_FP16,
                         testing::ValuesIn(MatrixDiagPartSmokeTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_MatrixDiagPartBackward_BFP16,
                         testing::ValuesIn(MatrixDiagPartSmokeTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_MatrixDiagPartBackward_FP32,
                         testing::ValuesIn(MatrixDiagPartPerfTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_MatrixDiagPartBackward_FP16,
                         testing::ValuesIn(MatrixDiagPartPerfTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_MatrixDiagPartBackward_BFP16,
                         testing::ValuesIn(MatrixDiagPartPerfTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MatrixDiagPartBackward_FP32,
                         testing::ValuesIn(MatrixDiagPartFullTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MatrixDiagPartBackward_FP16,
                         testing::ValuesIn(MatrixDiagPartFullTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MatrixDiagPartBackward_BFP16,
                         testing::ValuesIn(MatrixDiagPartFullTestConfigs()));
