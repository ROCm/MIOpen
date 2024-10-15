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

#include <miopen/conv/solvers.hpp>

#include "gtest_common.hpp"
#include "unit_conv_ConvolutionDescriptor.hpp"
#include "unit_TensorDescriptor.hpp"

namespace miopen {
namespace unit_tests {

//************************************************************************************
// ConvTestCase
//************************************************************************************

struct ConvTestCase
{
    ConvTestCase();

    ConvTestCase(std::vector<size_t>&& x,
                 std::vector<size_t>&& w,
                 std::vector<int>&& pad,
                 std::vector<int>&& stride,
                 std::vector<int>&& dilation,
                 miopenDataType_t type);

    ConvTestCase(std::vector<size_t>&& x,
                 std::vector<size_t>&& w,
                 std::vector<int>&& pad,
                 std::vector<int>&& stride,
                 std::vector<int>&& dilation,
                 miopenDataType_t type_x,
                 miopenDataType_t type_w,
                 miopenDataType_t type_y);

    ConvTestCase(TensorDescriptorParams&& x,
                 TensorDescriptorParams&& w,
                 miopenDataType_t type_y,
                 ConvolutionDescriptorParams&& conv);

    miopen::TensorDescriptor GetXTensorDescriptor() const;
    miopen::TensorDescriptor GetWTensorDescriptor() const;

    miopenDataType_t GetXDataType() const;
    miopenDataType_t GetWDataType() const;
    miopenDataType_t GetYDataType() const;

    miopen::ConvolutionDescriptor GetConv() const;

    friend std::ostream& operator<<(std::ostream& os, const ConvTestCase& tc);

private:
    TensorDescriptorParams x;
    TensorDescriptorParams w;
    miopenDataType_t type_y;
    ConvolutionDescriptorParams conv;
};

//************************************************************************************
// Unit test for convolution solver
//************************************************************************************

class UnitTestConvSolverBase
{
public:
    void RunTestImpl(const miopen::solver::conv::ConvSolverBase& solver,
                     miopen::conv::Direction direction,
                     const ConvTestCase& conv_config,
                     miopenConvAlgorithm_t algo,
                     bool use_cpu_ref);

protected:
    void SetUpImpl(Gpu supported_devs);
};

template <miopen::conv::Direction direction>
class UnitTestConvSolver
    : public UnitTestConvSolverBase,
      public ::testing::TestWithParam<std::tuple<Gpu, miopenConvAlgorithm_t, ConvTestCase>>
{
public:
    void RunTest(const miopen::solver::conv::ConvSolverBase& solver, bool use_cpu_ref = false)
    {
        miopenConvAlgorithm_t algo;
        ConvTestCase conv_config;
        std::tie(std::ignore, algo, conv_config) = GetParam();
        this->RunTestImpl(solver, direction, conv_config, algo, use_cpu_ref);
    }

protected:
    void SetUp() override
    {
        Gpu supported_devs;
        std::tie(supported_devs, std::ignore, std::ignore) = GetParam();
        this->SetUpImpl(supported_devs);
    }
};

using UnitTestConvSolverFwd = UnitTestConvSolver<miopen::conv::Direction::Forward>;
using UnitTestConvSolverBwd = UnitTestConvSolver<miopen::conv::Direction::BackwardData>;
using UnitTestConvSolverWrw = UnitTestConvSolver<miopen::conv::Direction::BackwardWeights>;

//************************************************************************************
// This test is designed to detect the expansion of the solver's device applicability
//************************************************************************************

class UnitTestConvSolverDevApplicabilityBase
{
public:
    void RunTestImpl(const miopen::solver::conv::ConvSolverBase& solver,
                     Gpu supported_devs,
                     miopen::conv::Direction direction,
                     const ConvTestCase& conv_config);
};

template <miopen::conv::Direction direction>
class UnitTestConvSolverDevApplicability
    : public UnitTestConvSolverDevApplicabilityBase,
      public ::testing::TestWithParam<std::tuple<Gpu, ConvTestCase>>
{
public:
    void RunTest(const miopen::solver::conv::ConvSolverBase& solver)
    {
        Gpu supported_devs;
        ConvTestCase conv_config;
        std::tie(supported_devs, conv_config) = GetParam();
        this->RunTestImpl(solver, supported_devs, direction, conv_config);
    }
};

using UnitTestConvSolverDevApplicabilityFwd =
    UnitTestConvSolverDevApplicability<miopen::conv::Direction::Forward>;
using UnitTestConvSolverDevApplicabilityBwd =
    UnitTestConvSolverDevApplicability<miopen::conv::Direction::BackwardData>;
using UnitTestConvSolverDevApplicabilityWrw =
    UnitTestConvSolverDevApplicability<miopen::conv::Direction::BackwardWeights>;

} // namespace unit_tests
} // namespace miopen

//************************************************************************************
// Unit test for convolution solver
//************************************************************************************
using GPU_UnitTestConvSolverFwd_FP16 = miopen::unit_tests::UnitTestConvSolverFwd;
using GPU_UnitTestConvSolverBwd_FP16 = miopen::unit_tests::UnitTestConvSolverBwd;
using GPU_UnitTestConvSolverWrw_FP16 = miopen::unit_tests::UnitTestConvSolverWrw;

using GPU_UnitTestConvSolverFwd_BFP16 = miopen::unit_tests::UnitTestConvSolverFwd;
using GPU_UnitTestConvSolverBwd_BFP16 = miopen::unit_tests::UnitTestConvSolverBwd;
using GPU_UnitTestConvSolverWrw_BFP16 = miopen::unit_tests::UnitTestConvSolverWrw;

using GPU_UnitTestConvSolverFwd_FP32 = miopen::unit_tests::UnitTestConvSolverFwd;
using GPU_UnitTestConvSolverBwd_FP32 = miopen::unit_tests::UnitTestConvSolverBwd;
using GPU_UnitTestConvSolverWrw_FP32 = miopen::unit_tests::UnitTestConvSolverWrw;

using GPU_UnitTestConvSolverFwd_I8 = miopen::unit_tests::UnitTestConvSolverFwd;
using GPU_UnitTestConvSolverBwd_I8 = miopen::unit_tests::UnitTestConvSolverBwd;
using GPU_UnitTestConvSolverWrw_I8 = miopen::unit_tests::UnitTestConvSolverWrw;

//************************************************************************************
// This test is designed to detect the expansion of the solver's device applicability
//************************************************************************************

using CPU_UnitTestConvSolverDevApplicabilityFwd_NONE =
    miopen::unit_tests::UnitTestConvSolverDevApplicabilityFwd;
using CPU_UnitTestConvSolverDevApplicabilityBwd_NONE =
    miopen::unit_tests::UnitTestConvSolverDevApplicabilityBwd;
using CPU_UnitTestConvSolverDevApplicabilityWrw_NONE =
    miopen::unit_tests::UnitTestConvSolverDevApplicabilityWrw;
