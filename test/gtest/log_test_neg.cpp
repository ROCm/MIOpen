/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include <miopen/config.h>

#if MIOPEN_BACKEND_HIP
#include "log.hpp"

TEST(CPU_LOG_TEST_NONE, AssertLogCmdOutput_Neg)
{
    TestLogFun(miopen::debug::LogCmdConvolution, logConv, false);
}

TEST(CPU_LOG_TEST_NONE, AssertLogFindCmdOutput_Neg)
{
    TestLogFun(miopen::debug::LogCmdFindConvolution, logFindConv, false);
}

TEST(CPU_LOG_TEST_NONE, AssertTestLogCmdCBAFusionOutput_Neg)
{
    TestLogCmdCBAFusion(miopen::debug::LogCmdFusion, logFusionConvBiasActiv, false);
}

TEST(CPU_LOG_TEST_NONE, AssertTestLogCmdBNormFusionOutput_Neg)
{
    TestLogCmdBNormFusion(miopen::debug::LogCmdFusion, logBnormActiv, false);
}
#endif
