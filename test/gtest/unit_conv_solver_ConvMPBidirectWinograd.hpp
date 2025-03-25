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

#ifndef WINO_DATA_H
#error "WINO_DATA_H undefined"
#endif

#ifndef WINO_FILTER_H
#error "WINO_FILTER_H undefined"
#endif

// WORKAROUND_LWPMIOPEN_1388 Disabling tests due to tolerance issues
#define WORKAROUND_LWPMIOPEN_1388 1

// WORKAROUND_SWDEV_257202 disables these solvers due to SSD convergence issues.
// However we still want to check that solver is not broken and therefore use
// MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_FnX3=1.

#define CONCAT2_HELPER(str1, str2) str1##str2
#define CONCAT2(str1, str2) CONCAT2_HELPER(str1, str2)

#define CONCAT3_HELPER(str1, str2, str3) str1##str2##str3
#define CONCAT3(str1, str2, str3) CONCAT3_HELPER(str1, str2, str3)

#define CONCAT5_HELPER(str1, str2, str3, str4, str5) str1##str2##str3##str4##str5
#define CONCAT5(str1, str2, str3, str4, str5) CONCAT5_HELPER(str1, str2, str3, str4, str5)

#define MAKE_SUFFIX(wino_data_h, wino_filter_h, sep) CONCAT3(wino_data_h, sep, wino_filter_h)
#define MAKE_SUFFIX_UPPER(wino_data_h, wino_filter_h) MAKE_SUFFIX(wino_data_h, wino_filter_h, X)
#define MAKE_SUFFIX_LOWER(wino_data_h, wino_filter_h) MAKE_SUFFIX(wino_data_h, wino_filter_h, x)

#define SHORT_SOLVER_NAME \
    CONCAT2(MPBidirectWinogradF, MAKE_SUFFIX_LOWER(WINO_DATA_H, WINO_FILTER_H))

#if WORKAROUND_LWPMIOPEN_1388
#define SOLVER_NAME \
    CONCAT2(DISABLED_ConvMPBidirectWinogradF, MAKE_SUFFIX_LOWER(WINO_DATA_H, WINO_FILTER_H))
#else
#define SOLVER_NAME CONCAT2(ConvMPBidirectWinogradF, MAKE_SUFFIX_LOWER(WINO_DATA_H, WINO_FILTER_H))
#endif

#define TESTSUITE_NAME_GENERIC(hw_type, name, datatype) CONCAT5(hw_type, _, name, _, datatype)

#define TESTSUITE_NAME_GENERIC_DIR(hw_type, name, direction, datatype) \
    TESTSUITE_NAME_GENERIC(hw_type, CONCAT2(name, direction), datatype)

#define TESTSUITE_NAME(hw_type, direction, datatype) \
    TESTSUITE_NAME_GENERIC_DIR(                      \
        hw_type, CONCAT2(UnitTestConvSolver, SHORT_SOLVER_NAME), direction, datatype)

#define TESTSUITE_NAME_DEV_APP(hw_type, direction, datatype)                                     \
    TESTSUITE_NAME_GENERIC_DIR(hw_type,                                                          \
                               CONCAT3(UnitTestConvSolver, SHORT_SOLVER_NAME, DevApplicability), \
                               direction,                                                        \
                               datatype)

#define TESTSUITE_NAME_FWD_FP16 TESTSUITE_NAME(GPU, Fwd, FP16)
#define TESTSUITE_NAME_BWD_FP16 TESTSUITE_NAME(GPU, Bwd, FP16)
#define TESTSUITE_NAME_FWD_FP32 TESTSUITE_NAME(GPU, Fwd, FP32)
#define TESTSUITE_NAME_BWD_FP32 TESTSUITE_NAME(GPU, Bwd, FP32)

#define TESTSUITE_NAME_DEVAPP TESTSUITE_NAME_DEV_APP(CPU, Fwd, NONE)

#define MP_BD_WINOGRAD_ENV_VAR \
    CONCAT2(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F, MAKE_SUFFIX_UPPER(WINO_DATA_H, WINO_FILTER_H))

MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F2X3)
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F3X3)
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F4X3)
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F5X3)
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F6X3)

namespace {

class SolverEnabler
{
public:
    SolverEnabler()
    {
        if(MP_BD_WINOGRAD_ENV_VAR)
            prev = lib_env::value<bool>(MP_BD_WINOGRAD_ENV_VAR);
        if(prev != true)
            lib_env::update(MP_BD_WINOGRAD_ENV_VAR, true);
    }

    ~SolverEnabler()
    {
        if(prev)
        {
            if(prev != true)
                lib_env::update(MP_BD_WINOGRAD_ENV_VAR, false);
        }
        else
        {
            lib_env::clear(MP_BD_WINOGRAD_ENV_VAR);
        }
    }

private:
    std::optional<bool> prev;
};

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{8, 8, 8, 8}, {8, 8, 3, 3}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        Gpu supported_gpus = Gpu::gfx900 | Gpu::gfx906 | Gpu::gfx908;
        auto p             = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
        p.CheckXnackDisabled();
        return p;
    }();
    return params;
}

} // namespace

using TESTSUITE_NAME_FWD_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using TESTSUITE_NAME_BWD_FP16 = GPU_UnitTestConvSolverBwd_FP16;
using TESTSUITE_NAME_FWD_FP32 = GPU_UnitTestConvSolverFwd_FP32;
using TESTSUITE_NAME_BWD_FP32 = GPU_UnitTestConvSolverBwd_FP32;

using TESTSUITE_NAME_DEVAPP = CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(TESTSUITE_NAME_FWD_FP16, SOLVER_NAME)
{
    SolverEnabler solver_enabler;
    this->RunTest(miopen::solver::conv::ConvMPBidirectWinograd<WINO_DATA_H, WINO_FILTER_H>{});
};

TEST_P(TESTSUITE_NAME_BWD_FP16, SOLVER_NAME)
{
    SolverEnabler solver_enabler;
    this->RunTest(miopen::solver::conv::ConvMPBidirectWinograd<WINO_DATA_H, WINO_FILTER_H>{});
};

TEST_P(TESTSUITE_NAME_FWD_FP32, SOLVER_NAME)
{
    SolverEnabler solver_enabler;
    this->RunTest(miopen::solver::conv::ConvMPBidirectWinograd<WINO_DATA_H, WINO_FILTER_H>{});
};

TEST_P(TESTSUITE_NAME_BWD_FP32, SOLVER_NAME)
{
    SolverEnabler solver_enabler;
    this->RunTest(miopen::solver::conv::ConvMPBidirectWinograd<WINO_DATA_H, WINO_FILTER_H>{});
};

TEST_P(TESTSUITE_NAME_DEVAPP, SOLVER_NAME)
{
    SolverEnabler solver_enabler;
    this->RunTest(miopen::solver::conv::ConvMPBidirectWinograd<WINO_DATA_H, WINO_FILTER_H>{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TESTSUITE_NAME_FWD_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         TESTSUITE_NAME_BWD_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         TESTSUITE_NAME_FWD_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         TESTSUITE_NAME_BWD_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TESTSUITE_NAME_DEVAPP,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));
