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
#include <gtest/gtest.h>
#include <miopen/batchnorm/solvers.hpp>
#include <miopen/batchnorm/problem_description.hpp>
#include <miopen/tensor.hpp>
#include <miopen/activ.hpp>
#include <miopen/handle.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/datatype.hpp>

using namespace miopen;
using namespace miopen::batchnorm;
using namespace miopen::solver::batchnorm;

namespace {

struct ApplicabilityParams
{
    miopenDataType_t x_type;
    miopenDataType_t y_type;
    miopenDataType_t scale_type;
    miopenDataType_t bias_type;
    miopenDataType_t mean_type;
    miopenDataType_t var_type;
    bool expect_applicable;
    std::string test_name;

    friend std::ostream& operator<<(std::ostream& os, const ApplicabilityParams& p)
    {
        os << "ApplicabilityParams(";
        os << "x_type=" << GetDataType(p.x_type) << ", ";
        os << "y_type=" << GetDataType(p.y_type) << ", ";
        os << "scale_type=" << GetDataType(p.scale_type) << ", ";
        os << "bias_type=" << GetDataType(p.bias_type) << ", ";
        os << "mean_type=" << GetDataType(p.mean_type) << ", ";
        os << "var_type=" << GetDataType(p.var_type) << ", ";
        os << "expect_applicable=" << (p.expect_applicable ? "true" : "false") << ", ";
        os << "test_name=" << p.test_name;
        os << ")";
        return os;
    }
};

TensorDescriptor MakeTensorDesc(miopenDataType_t type)
{
    return TensorDescriptor(type, {1, 2, 3, 4});
}

ActivationDescriptor MakeActivationDesc(miopenActivationMode_t mode = miopenActivationPASTHRU)
{
    ActivationDescriptor desc(mode, 1.0, 1.0, 1.0);
    return desc;
}

ExecutionContext MakeContext()
{
    ExecutionContext ctx;
    return ctx;
}

std::vector<ApplicabilityParams> GetParams()
{
    std::vector<ApplicabilityParams> params;
    // Valid cases
    for(auto dt : {miopenFloat, miopenHalf, miopenBFloat16})
    {
        params.push_back({dt,
                          dt,
                          miopenFloat,
                          miopenFloat,
                          miopenFloat,
                          miopenFloat,
                          true,
                          "X_Y_" + GetDataType(dt) + "_Stats_" + GetDataType(miopenFloat)});
    }
    // Invalid: X & Y are invalid types
    params.push_back({miopenInt8,
                      miopenInt8,
                      miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      false,
                      "X_INT8_Y_INT8"});
    params.push_back({miopenInt32,
                      miopenInt32,
                      miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      false,
                      "X_INT32_Y_INT32"});
    // Invalid: X & Y are mixed
    params.push_back({miopenFloat,
                      miopenHalf,
                      miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      false,
                      "X_FP32_Y_FP16"});
    params.push_back({miopenHalf,
                      miopenBFloat16,
                      miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      false,
                      "X_FP16_Y_BFP16"});
    params.push_back({miopenBFloat16,
                      miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      false,
                      "X_BFP16_Y_FP32"});
    // Invalid: stats not FP32
    params.push_back({miopenFloat,
                      miopenFloat,
                      miopenHalf,
                      miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      false,
                      "Scale_FP16"});
    params.push_back({miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      miopenHalf,
                      miopenFloat,
                      miopenFloat,
                      false,
                      "Bias_FP16"});
    params.push_back({miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      miopenHalf,
                      miopenFloat,
                      false,
                      "Mean_FP16"});
    params.push_back({miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      miopenFloat,
                      miopenHalf,
                      false,
                      "Var_FP16"});
    // Invalid: all stats not FP32
    params.push_back({miopenHalf,
                      miopenHalf,
                      miopenHalf,
                      miopenHalf,
                      miopenHalf,
                      miopenHalf,
                      false,
                      "AllStats_FP16"});
    return params;
}

} // namespace

class CPU_BatchNormFwdTrainingSpatialApplicabilityTest_NONE
    : public ::testing::TestWithParam<ApplicabilityParams>
{
};

TEST_P(CPU_BatchNormFwdTrainingSpatialApplicabilityTest_NONE, IsApplicable)
{
    const auto& p  = GetParam();
    auto ctx       = MakeContext();
    auto xDesc     = MakeTensorDesc(p.x_type);
    auto yDesc     = MakeTensorDesc(p.y_type);
    auto scaleDesc = MakeTensorDesc(p.scale_type);
    auto biasDesc  = MakeTensorDesc(p.bias_type);
    auto meanDesc  = MakeTensorDesc(p.mean_type);
    auto varDesc   = MakeTensorDesc(p.var_type);
    auto actDesc   = MakeActivationDesc();

    ProblemDescription problem(miopenBNSpatial,
                               xDesc,
                               yDesc,
                               scaleDesc,
                               biasDesc,
                               meanDesc,
                               varDesc,
                               1.0,
                               1e-5,
                               true,
                               true,
                               1,
                               actDesc);

    BnFwdTrainingSpatial solver;
    EXPECT_EQ(solver.IsApplicable(ctx, problem), p.expect_applicable) << p.test_name;
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_BatchNormFwdTrainingSpatialApplicabilityTest_NONE,
                         ::testing::ValuesIn(GetParams()));

class CPU_BatchNormFwdTrainingPerActivationApplicabilityTest_NONE
    : public ::testing::TestWithParam<ApplicabilityParams>
{
};

TEST_P(CPU_BatchNormFwdTrainingPerActivationApplicabilityTest_NONE, IsApplicable)
{
    const auto& p  = GetParam();
    auto ctx       = MakeContext();
    auto xDesc     = MakeTensorDesc(p.x_type);
    auto yDesc     = MakeTensorDesc(p.y_type);
    auto scaleDesc = MakeTensorDesc(p.scale_type);
    auto biasDesc  = MakeTensorDesc(p.bias_type);
    auto meanDesc  = MakeTensorDesc(p.mean_type);
    auto varDesc   = MakeTensorDesc(p.var_type);
    auto actDesc   = MakeActivationDesc();

    ProblemDescription problem(miopenBNPerActivation,
                               xDesc,
                               yDesc,
                               scaleDesc,
                               biasDesc,
                               meanDesc,
                               varDesc,
                               1.0,
                               1e-5,
                               true,
                               true,
                               1,
                               actDesc);

    BnFwdTrainingPerActivation solver;
    EXPECT_EQ(solver.IsApplicable(ctx, problem), p.expect_applicable) << p.test_name;
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_BatchNormFwdTrainingPerActivationApplicabilityTest_NONE,
                         ::testing::ValuesIn(GetParams()));

class CPU_BatchNormFwdInferenceApplicabilityTest_NONE
    : public ::testing::TestWithParam<ApplicabilityParams>
{
};

TEST_P(CPU_BatchNormFwdInferenceApplicabilityTest_NONE, IsApplicable)
{
    const auto& p  = GetParam();
    auto ctx       = MakeContext();
    auto xDesc     = MakeTensorDesc(p.x_type);
    auto yDesc     = MakeTensorDesc(p.y_type);
    auto scaleDesc = MakeTensorDesc(p.scale_type);
    auto biasDesc  = MakeTensorDesc(p.bias_type);
    auto meanDesc  = MakeTensorDesc(p.mean_type);
    auto varDesc   = MakeTensorDesc(p.var_type);
    auto actDesc   = MakeActivationDesc();

    ProblemDescription problem(
        miopenBNSpatial, xDesc, yDesc, scaleDesc, biasDesc, meanDesc, varDesc, 1e-5, actDesc);

    BnFwdInference solver;
    EXPECT_EQ(solver.IsApplicable(ctx, problem), p.expect_applicable) << p.test_name;
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_BatchNormFwdInferenceApplicabilityTest_NONE,
                         ::testing::ValuesIn(GetParams()));

class CPU_BatchNormBwdTrainingSpatialApplicabilityTest_NONE
    : public ::testing::TestWithParam<ApplicabilityParams>
{
};

TEST_P(CPU_BatchNormBwdTrainingSpatialApplicabilityTest_NONE, IsApplicable)
{
    const auto& p  = GetParam();
    auto ctx       = MakeContext();
    auto xDesc     = MakeTensorDesc(p.x_type);
    auto dyDesc    = MakeTensorDesc(p.y_type);
    auto dxDesc    = MakeTensorDesc(p.y_type); // dx usually same type as y
    auto scaleDesc = MakeTensorDesc(p.scale_type);
    auto biasDesc  = MakeTensorDesc(p.bias_type);
    auto meanDesc  = MakeTensorDesc(p.mean_type);
    auto varDesc   = MakeTensorDesc(p.var_type);
    auto actDesc   = MakeActivationDesc();

    ProblemDescription problem(miopenBNSpatial,
                               xDesc,
                               dyDesc,
                               dxDesc,
                               scaleDesc,
                               biasDesc,
                               meanDesc,
                               varDesc,
                               1e-5,
                               false,
                               1,
                               actDesc);

    BnBwdTrainingSpatial solver;
    EXPECT_EQ(solver.IsApplicable(ctx, problem), p.expect_applicable) << p.test_name;
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_BatchNormBwdTrainingSpatialApplicabilityTest_NONE,
                         ::testing::ValuesIn(GetParams()));

class CPU_BatchNormBwdTrainingPerActivationApplicabilityTest_NONE
    : public ::testing::TestWithParam<ApplicabilityParams>
{
};

TEST_P(CPU_BatchNormBwdTrainingPerActivationApplicabilityTest_NONE, IsApplicable)
{
    const auto& p  = GetParam();
    auto ctx       = MakeContext();
    auto xDesc     = MakeTensorDesc(p.x_type);
    auto dyDesc    = MakeTensorDesc(p.y_type);
    auto dxDesc    = MakeTensorDesc(p.y_type); // dx usually same type as y
    auto scaleDesc = MakeTensorDesc(p.scale_type);
    auto biasDesc  = MakeTensorDesc(p.bias_type);
    auto meanDesc  = MakeTensorDesc(p.mean_type);
    auto varDesc   = MakeTensorDesc(p.var_type);
    auto actDesc   = MakeActivationDesc();

    ProblemDescription problem(miopenBNPerActivation,
                               xDesc,
                               dyDesc,
                               dxDesc,
                               scaleDesc,
                               biasDesc,
                               meanDesc,
                               varDesc,
                               1e-5,
                               false,
                               1,
                               actDesc);

    BnBwdTrainingPerActivation solver;
    EXPECT_EQ(solver.IsApplicable(ctx, problem), p.expect_applicable) << p.test_name;
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_BatchNormBwdTrainingPerActivationApplicabilityTest_NONE,
                         ::testing::ValuesIn(GetParams()));
