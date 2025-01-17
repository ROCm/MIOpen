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

#include "matrix_diag.hpp"
#include <miopen/bfloat16.hpp>

namespace matrix_diag {

using GPU_MatrixDiagForward_FP32   = MatrixDiagTestForward<float>;
using GPU_MatrixDiagForward_FP16   = MatrixDiagTestForward<half>;
using GPU_MatrixDiagForward_BFP16  = MatrixDiagTestForward<bfloat16>;
using GPU_MatrixDiagBackward_FP32  = MatrixDiagTestBackward<float>;
using GPU_MatrixDiagBackward_FP16  = MatrixDiagTestBackward<half>;
using GPU_MatrixDiagBackward_BFP16 = MatrixDiagTestBackward<bfloat16>;

} // namespace matrix_diag
using namespace matrix_diag;

TEST_P(GPU_MatrixDiagForward_FP32, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_MatrixDiagForward_FP16, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_MatrixDiagForward_BFP16, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_MatrixDiagBackward_FP32, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_MatrixDiagBackward_FP16, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_MatrixDiagBackward_BFP16, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_MatrixDiagForward_FP32,
                         testing::ValuesIn(MatrixDiagSmokeTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_MatrixDiagForward_FP16,
                         testing::ValuesIn(MatrixDiagSmokeTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_MatrixDiagForward_BFP16,
                         testing::ValuesIn(MatrixDiagSmokeTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_MatrixDiagForward_FP32,
                         testing::ValuesIn(MatrixDiagPerfTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_MatrixDiagForward_FP16,
                         testing::ValuesIn(MatrixDiagPerfTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_MatrixDiagForward_BFP16,
                         testing::ValuesIn(MatrixDiagPerfTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MatrixDiagForward_FP32,
                         testing::ValuesIn(MatrixDiagFullTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MatrixDiagForward_FP16,
                         testing::ValuesIn(MatrixDiagFullTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MatrixDiagForward_BFP16,
                         testing::ValuesIn(MatrixDiagFullTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_MatrixDiagBackward_FP32,
                         testing::ValuesIn(MatrixDiagSmokeTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_MatrixDiagBackward_FP16,
                         testing::ValuesIn(MatrixDiagSmokeTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_MatrixDiagBackward_BFP16,
                         testing::ValuesIn(MatrixDiagSmokeTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_MatrixDiagBackward_FP32,
                         testing::ValuesIn(MatrixDiagPerfTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_MatrixDiagBackward_FP16,
                         testing::ValuesIn(MatrixDiagPerfTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_MatrixDiagBackward_BFP16,
                         testing::ValuesIn(MatrixDiagPerfTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MatrixDiagBackward_FP32,
                         testing::ValuesIn(MatrixDiagFullTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MatrixDiagBackward_FP16,
                         testing::ValuesIn(MatrixDiagFullTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MatrixDiagBackward_BFP16,
                         testing::ValuesIn(MatrixDiagFullTestConfigs()));
