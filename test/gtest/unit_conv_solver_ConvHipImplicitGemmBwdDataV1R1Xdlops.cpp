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

#include "unit_conv_solver.hpp"

#ifndef HIP_PACKAGE_VERSION_FLAT
#error "HIP_PACKAGE_VERSION_FLAT undefined"
#endif

// WORKAROUND_SWDEV_251757 disables this solver due to precision issues.
// However we still want to check that solver is not broken and therefore use
// MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS=1 to enable it.
#define WORKAROUND_SWDEV_251757 1

#define SOLVER_NAME ConvHipImplicitGemmBwdDataV1R1Xdlops

#if WORKAROUND_SWDEV_251757
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS)
#endif

namespace {

#if WORKAROUND_SWDEV_251757
class SolverEnabler
{
public:
    SolverEnabler()
    {
        if(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS)
            prev = lib_env::value<bool>(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS);
        if(prev != true)
            lib_env::update(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS, true);
    }

    ~SolverEnabler()
    {
        if(prev)
        {
            if(prev != true)
                lib_env::update(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS, false);
        }
        else
        {
            lib_env::clear(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS);
        }
    }

private:
    std::optional<bool> prev;
};
#endif

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{32, 128, 32, 32}, {12, 128, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

template <miopenDataType_t datatype>
const auto& GetTestParams()
{
    static const auto params = [] {
        Gpu supported_gpus = Gpu::gfx908 | Gpu::gfx90A;
        if constexpr(datatype != miopenBFloat16)
        {
            supported_gpus = supported_gpus | Gpu::gfx94X | Gpu::gfx950;
        }
        auto p = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
        p.EnableDeprecatedSolvers();
        p.Tunable(5);
        p.SetConvAttrFp16Alt(0);
        return p;
    }();
    return params;
}

const auto& GetTestParamsFP16() { return GetTestParams<miopenHalf>(); }
const auto& GetTestParamsBFP16() { return GetTestParams<miopenBFloat16>(); }
const auto& GetTestParamsFP32() { return GetTestParams<miopenFloat>(); }

} // namespace

using GPU_UnitTestConvSolverHipImplicitGemmBwdDataV1R1XdlopsBwd_FP16 =
    GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverHipImplicitGemmBwdDataV1R1XdlopsBwd_BFP16 =
    GPU_UnitTestConvSolverBwd_BFP16;
using GPU_UnitTestConvSolverHipImplicitGemmBwdDataV1R1XdlopsBwd_FP32 =
    GPU_UnitTestConvSolverBwd_FP32;
using CPU_UnitTestConvSolverHipImplicitGemmBwdDataV1R1XdlopsDevApplicabilityBwd_BFP16 =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;
using CPU_UnitTestConvSolverHipImplicitGemmBwdDataV1R1XdlopsDevApplicabilityBwd_FP32 =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmBwdDataV1R1XdlopsBwd_FP16, SOLVER_NAME)
{
#if WORKAROUND_SWDEV_251757
    SolverEnabler solver_enabler;
#endif
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmBwdDataV1R1Xdlops{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmBwdDataV1R1XdlopsBwd_BFP16, SOLVER_NAME)
{
#if WORKAROUND_SWDEV_251757
    SolverEnabler solver_enabler;
#endif
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmBwdDataV1R1Xdlops{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmBwdDataV1R1XdlopsBwd_FP32, SOLVER_NAME)
{
#if WORKAROUND_SWDEV_251757
    SolverEnabler solver_enabler;
#endif
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmBwdDataV1R1Xdlops{});
};

TEST_P(CPU_UnitTestConvSolverHipImplicitGemmBwdDataV1R1XdlopsDevApplicabilityBwd_BFP16, SOLVER_NAME)
{
#if WORKAROUND_SWDEV_251757
    SolverEnabler solver_enabler;
#endif
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmBwdDataV1R1Xdlops{});
};

TEST_P(CPU_UnitTestConvSolverHipImplicitGemmBwdDataV1R1XdlopsDevApplicabilityBwd_FP32, SOLVER_NAME)
{
#if WORKAROUND_SWDEV_251757
    SolverEnabler solver_enabler;
#endif
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmBwdDataV1R1Xdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmBwdDataV1R1XdlopsBwd_FP16,
                         testing::Combine(testing::Values(GetTestParamsFP16()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmBwdDataV1R1XdlopsBwd_BFP16,
                         testing::Combine(testing::Values(GetTestParamsBFP16()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmBwdDataV1R1XdlopsBwd_FP32,
                         testing::Combine(testing::Values(GetTestParamsFP32()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverHipImplicitGemmBwdDataV1R1XdlopsDevApplicabilityBwd_BFP16,
    testing::Combine(testing::Values(GetTestParamsBFP16()),
                     testing::Values(GetConvTestCases(miopenBFloat16)[0])));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverHipImplicitGemmBwdDataV1R1XdlopsDevApplicabilityBwd_FP32,
    testing::Combine(testing::Values(GetTestParamsFP32()),
                     testing::Values(GetConvTestCases(miopenFloat)[0])));
