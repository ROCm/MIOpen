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

#include "solver_fwd.hpp"
#include "solver_bwd.hpp"
#include "solver_wrw.hpp"

struct ConvTestCase
{
    ConvTestCase(const std::initializer_list<size_t>& x,
                 const std::initializer_list<size_t>& w,
                 const std::initializer_list<int>& pad,
                 const std::initializer_list<int>& stride,
                 const std::initializer_list<int>& dilation,
                 miopenDataType_t type);

    ConvTestCase(const std::initializer_list<size_t>& x,
                 const std::initializer_list<size_t>& w,
                 const std::initializer_list<int>& pad,
                 const std::initializer_list<int>& stride,
                 const std::initializer_list<int>& dilation,
                 miopenDataType_t type_x,
                 miopenDataType_t type_w,
                 miopenDataType_t type_y);

    const std::vector<size_t>& GetXDims() const;
    const std::vector<size_t>& GetWDims() const;

    miopenDataType_t GetXDataType() const;
    miopenDataType_t GetWDataType() const;
    miopenDataType_t GetYDataType() const;

    miopen::ConvolutionDescriptor GetConv() const;

    friend std::ostream& operator<<(std::ostream& os, const ConvTestCase& tc);

private:
    const std::vector<size_t> x;
    const std::vector<size_t> w;
    const std::vector<int> pad;
    const std::vector<int> stride;
    const std::vector<int> dilation;
    const miopenDataType_t type_x;
    const miopenDataType_t type_w;
    const miopenDataType_t type_y;
};

// Unit test for convolution solver

class UnitTestConvSolver
{
public:
    virtual ~UnitTestConvSolver()                                            = default;
    virtual void RunTest(const miopen::solver::conv::ConvSolverBase& solver) = 0;
};

template <typename T>
class UnitTestConvSolverFwd : public ConvFwdSolverTest<T, T>, UnitTestConvSolver
{
public:
    void RunTest(const miopen::solver::conv::ConvSolverBase& solver) final
    {
        this->SolverFwd(solver);
    }
};

template <typename T>
class UnitTestConvSolverBwd : public ConvBwdSolverTest<T, T>, UnitTestConvSolver
{
public:
    void RunTest(const miopen::solver::conv::ConvSolverBase& solver) final
    {
        this->SolverBwd(solver);
    }
};

template <typename T>
class UnitTestConvSolverWrw : public ConvWrwSolverTest<T, T>, UnitTestConvSolver
{
public:
    void RunTest(const miopen::solver::conv::ConvSolverBase& solver) final
    {
        this->SolverWrw(solver);
    }
};

class GPU_UnitTestConvSolver_fwd_FP16 : public UnitTestConvSolverFwd<half_float::half>
{
};

class GPU_UnitTestConvSolver_bwd_FP16 : public UnitTestConvSolverBwd<half_float::half>
{
};

class GPU_UnitTestConvSolver_wrw_FP16 : public UnitTestConvSolverWrw<half_float::half>
{
};

// This test is designed to detect the expansion of the solver's device applicability

class UnitTestConvSolverDevApplicabilityBase
{
public:
    void RunTestImpl(const miopen::solver::conv::ConvSolverBase& solver,
                     Gpu supported_devs,
                     miopenDataType_t datatype,
                     miopen::conv::Direction direction,
                     const ConvTestCaseBase& conv_config);
};

template <miopenDataType_t datatype, miopen::conv::Direction direction>
class UnitTestConvSolverDevApplicability
    : public UnitTestConvSolverDevApplicabilityBase,
      public ::testing::TestWithParam<std::tuple<Gpu, ConvTestCaseBase>>
{
public:
    void RunTest(const miopen::solver::conv::ConvSolverBase& solver)
    {
        Gpu supported_devs;
        ConvTestCaseBase conv_config;
        std::tie(supported_devs, conv_config) = GetParam();
        this->RunTestImpl(solver, supported_devs, datatype, direction, conv_config);
    }
};

template <miopenDataType_t datatype>
class UnitTestConvSolverDevApplicabilityFwd
    : public UnitTestConvSolverDevApplicability<datatype, miopen::conv::Direction::Forward>
{
};

class CPU_UnitTestConvSolverDevApplicability_fwd_FP16
    : public UnitTestConvSolverDevApplicabilityFwd<miopenHalf>
{
};
