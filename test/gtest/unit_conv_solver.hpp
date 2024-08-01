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
#pragma once

#include <miopen/solver.hpp>

#include "gtest_common.hpp"
#include "solver_fwd.hpp"
#include "solver_bwd.hpp"
#include "solver_wrw.hpp"

class UnitTestConvSolver
{
public:
    virtual ~UnitTestConvSolver()            = default;
    virtual void RunTest(const miopen::solver::conv::ConvSolverBase& solver,
                         Gpu supported_gpus) = 0;

protected:
    static bool CheckTestSupportedForDevice(Gpu supported_gpus);
};

template <typename T>
class UnitTestConvSolverFwd : public ConvFwdSolverTest<T, T>, UnitTestConvSolver
{
public:
    void RunTest(const miopen::solver::conv::ConvSolverBase& solver, Gpu supported_gpus) final
    {
        if(!this->CheckTestSupportedForDevice(supported_gpus))
        {
            GTEST_SKIP();
        }
        this->SolverFwd(solver);
    }
};

template <typename T>
class UnitTestConvSolverBwd : public ConvBwdSolverTest<T, T>, UnitTestConvSolver
{
public:
    void RunTest(const miopen::solver::conv::ConvSolverBase& solver, Gpu supported_gpus) final
    {
        if(!this->CheckTestSupportedForDevice(supported_gpus))
        {
            GTEST_SKIP();
        }
        this->SolverBwd(solver);
    }
};

template <typename T>
class UnitTestConvSolverWrw : public ConvWrwSolverTest<T, T>, UnitTestConvSolver
{
public:
    void RunTest(const miopen::solver::conv::ConvSolverBase& solver, Gpu supported_gpus) final
    {
        if(!this->CheckTestSupportedForDevice(supported_gpus))
        {
            GTEST_SKIP();
        }
        this->SolverWrw(solver);
    }
};

class GPU_UnitTestConvSolver_fwd_fp16 : public UnitTestConvSolverFwd<half_float::half>
{
};

class GPU_UnitTestConvSolver_bwd_fp16 : public UnitTestConvSolverBwd<half_float::half>
{
};

class GPU_UnitTestConvSolver_wrw_fp16 : public UnitTestConvSolverWrw<half_float::half>
{
};
