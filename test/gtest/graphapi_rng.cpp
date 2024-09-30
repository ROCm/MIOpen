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

#include <miopen/graphapi/rng.hpp>

#include <tuple>

#include <gtest/gtest.h>

#include "graphapi_gtest_common.hpp"

namespace {

using miopen::graphapi::ValidatedValue;
using miopen::graphapi::ValidatedVector;
using DescriptorTuple = std::tuple<bool,
                                   miopenRngDistribution_t,
                                   double,
                                   ValidatedValue<double>,
                                   double,
                                   double,
                                   ValidatedValue<double>>;

using miopen::graphapi::Rng;
using miopen::graphapi::RngBuilder;

} // namespace

class CPU_GraphApiRngBuilder_NONE : public testing::TestWithParam<DescriptorTuple>
{
protected:
    bool mAttrsValid;
    miopenRngDistribution_t distribution;
    double mNormalMean;
    ValidatedValue<double> mNormalStdev;
    double mUniformMin;
    double mUniformMax;
    ValidatedValue<double> mBernoulliProb;

    void SetUp() override
    {
        std::tie(mAttrsValid,
                 distribution,
                 mNormalMean,
                 mNormalStdev,
                 mUniformMin,
                 mUniformMax,
                 mBernoulliProb) = GetParam();
    }
};

TEST_P(CPU_GraphApiRngBuilder_NONE, ValidateAttributes)
{
    if(mAttrsValid && mNormalStdev.valid && mBernoulliProb.valid)
    {
        EXPECT_NO_THROW({
            RngBuilder()
                .setDistribution(distribution)
                .setNormalMean(mNormalMean)
                .setNormalStdev(mNormalStdev.value)
                .setUniformMin(mUniformMin)
                .setUniformMax(mUniformMax)
                .setBernoulliProb(mBernoulliProb.value)
                .build();
        }) << "Builder failed on valid attributes";
    }
    else
    {
        EXPECT_ANY_THROW({
            RngBuilder()
                .setDistribution(distribution)
                .setNormalMean(mNormalMean)
                .setNormalStdev(mNormalStdev.value)
                .setUniformMin(mUniformMin)
                .setUniformMax(mUniformMax)
                .setBernoulliProb(mBernoulliProb.value)
                .build();
        }) << "Buider failed to detect invalid attributes";
    }
    if(mNormalStdev.valid)
    {
        EXPECT_NO_THROW({ RngBuilder().setNormalStdev(mNormalStdev.value); })
            << "RngBuilder::setNormalStdev(double) failed on valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ RngBuilder().setNormalStdev(mNormalStdev.value); })
            << "RngBuilder::setNormalStdev(double) failed on invalid attribute";
    }
    if(mBernoulliProb.valid)
    {
        EXPECT_NO_THROW({ RngBuilder().setBernoulliProb(mBernoulliProb.value); })
            << "RngBuilder::setBernoulliProb(double) failed on valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ RngBuilder().setBernoulliProb(mBernoulliProb.value); })
            << "RngBuilder::setBernoulliProb(double) failed on invalid attribute";
    }
}

namespace {

using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::GTestGraphApiExecute;

} // namespace

class CPU_GraphApiRng_NONE : public testing::TestWithParam<DescriptorTuple>
{
private:
    // Pointers to these are used in mExecute object below
    GTestDescriptorSingleValueAttribute<miopenRngDistribution_t, char> mDistribution;
    GTestDescriptorSingleValueAttribute<double, char> mNormalMean;
    GTestDescriptorSingleValueAttribute<double, char> mNormalStdev;
    GTestDescriptorSingleValueAttribute<double, char> mUniformMin;
    GTestDescriptorSingleValueAttribute<double, char> mUniformMax;
    GTestDescriptorSingleValueAttribute<double, char> mBernoulliProb;

protected:
    GTestGraphApiExecute<GTestDescriptorAttribute*> mExecute;

    void SetUp() override
    {
        auto [valid, distribution, normalMean, normalStdev, uniformMin, uniformMax, bernoulliProb] =
            GetParam();

        mDistribution = {true,
                         "MIOPEN_ATTR_RNG_DISTRIBUTION",
                         MIOPEN_ATTR_RNG_DISTRIBUTION,
                         MIOPEN_TYPE_RNG_DISTRIBUTION,
                         MIOPEN_TYPE_CHAR,
                         2,
                         distribution};

        mNormalMean = {true,
                       "MIOPEN_ATTR_RNG_NORMAL_DIST_MEAN",
                       MIOPEN_ATTR_RNG_NORMAL_DIST_MEAN,
                       MIOPEN_TYPE_DOUBLE,
                       MIOPEN_TYPE_CHAR,
                       2,
                       normalMean};

        mNormalStdev = {normalStdev.valid,
                        "MIOPEN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION",
                        MIOPEN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION,
                        MIOPEN_TYPE_DOUBLE,
                        MIOPEN_TYPE_CHAR,
                        2,
                        normalStdev.value};

        mUniformMin = {true,
                       "MIOPEN_ATTR_RNG_UNIFORM_DIST_MINIMUM",
                       MIOPEN_ATTR_RNG_UNIFORM_DIST_MINIMUM,
                       MIOPEN_TYPE_DOUBLE,
                       MIOPEN_TYPE_CHAR,
                       2,
                       uniformMin};

        mUniformMax = {true,
                       "MIOPEN_ATTR_RNG_UNIFORM_DIST_MAXIMUM",
                       MIOPEN_ATTR_RNG_UNIFORM_DIST_MAXIMUM,
                       MIOPEN_TYPE_DOUBLE,
                       MIOPEN_TYPE_CHAR,
                       2,
                       uniformMax};

        mBernoulliProb = {bernoulliProb.valid,
                          "MIOPEN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY",
                          MIOPEN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY,
                          MIOPEN_TYPE_DOUBLE,
                          MIOPEN_TYPE_CHAR,
                          2,
                          bernoulliProb.value};

        mExecute.descriptor.textName   = "MIOPEN_BACKEND_RNG_DESCRIPTOR";
        mExecute.descriptor.type       = MIOPEN_BACKEND_RNG_DESCRIPTOR;
        mExecute.descriptor.attrsValid = valid;

        mExecute.descriptor.attributes = {&mDistribution,
                                          &mNormalMean,
                                          &mNormalStdev,
                                          &mUniformMin,
                                          &mUniformMax,
                                          &mBernoulliProb};
    }
};

TEST_P(CPU_GraphApiRng_NONE, CFunctions) { mExecute(); }

static auto validAttributesNormal =
    testing::Combine(testing::Values(true),
                     testing::Values(MIOPEN_RNG_DISTRIBUTION_NORMAL),
                     testing::Values(0.0),
                     testing::Values(ValidatedValue<double>{true, 0.5}),
                     testing::Values(0.0, 1.0),
                     testing::Values(0.0, 1.0),
                     testing::Values(ValidatedValue<double>{true, 0.5},
                                     ValidatedValue<double>{false, -0.5},
                                     ValidatedValue<double>{false, 1.5}));

static auto validAttributesUniform = testing::Combine(
    testing::Values(true),
    testing::Values(MIOPEN_RNG_DISTRIBUTION_UNIFORM),
    testing::Values(0.0),
    testing::Values(ValidatedValue<double>{true, 0.5}, ValidatedValue<double>{false, -0.5}),
    testing::Values(0.0),
    testing::Values(1.0),
    testing::Values(ValidatedValue<double>{true, 0.5},
                    ValidatedValue<double>{false, -0.5},
                    ValidatedValue<double>{false, 1.5}));

static auto validAttributesBernoulli = testing::Combine(
    testing::Values(true),
    testing::Values(MIOPEN_RNG_DISTRIBUTION_BERNOULLI),
    testing::Values(0.0),
    testing::Values(ValidatedValue<double>{true, 0.5}, ValidatedValue<double>{false, -0.5}),
    testing::Values(0.0, 1.0),
    testing::Values(0.0, 1.0),
    testing::Values(ValidatedValue<double>{true, 0.5}));

static auto invalidAttributesNormal =
    testing::Combine(testing::Values(false),
                     testing::Values(MIOPEN_RNG_DISTRIBUTION_NORMAL),
                     testing::Values(0.0),
                     testing::Values(ValidatedValue<double>{false, -0.5}),
                     testing::Values(0.0, 1.0),
                     testing::Values(0.0, 1.0),
                     testing::Values(ValidatedValue<double>{true, 0.5},
                                     ValidatedValue<double>{false, -0.5},
                                     ValidatedValue<double>{false, 1.5}));

static auto invalidAttributesUniform = testing::Combine(
    testing::Values(false),
    testing::Values(MIOPEN_RNG_DISTRIBUTION_UNIFORM),
    testing::Values(0.0),
    testing::Values(ValidatedValue<double>{true, 0.5}, ValidatedValue<double>{false, -0.5}),
    testing::Values(1.0),
    testing::Values(0.0),
    testing::Values(ValidatedValue<double>{true, 0.5},
                    ValidatedValue<double>{false, -0.5},
                    ValidatedValue<double>{false, 1.5}));

static auto invalidAttributesBernoulli = testing::Combine(
    testing::Values(false),
    testing::Values(MIOPEN_RNG_DISTRIBUTION_BERNOULLI),
    testing::Values(0.0),
    testing::Values(ValidatedValue<double>{true, 0.5}, ValidatedValue<double>{false, -0.5}),
    testing::Values(0.0, 1.0),
    testing::Values(0.0, 1.0),
    testing::Values(ValidatedValue<double>{false, -0.5}, ValidatedValue<double>{false, 1.5}));

INSTANTIATE_TEST_SUITE_P(UnitVAN, CPU_GraphApiRngBuilder_NONE, validAttributesNormal);
INSTANTIATE_TEST_SUITE_P(UnitVAU, CPU_GraphApiRngBuilder_NONE, validAttributesUniform);
INSTANTIATE_TEST_SUITE_P(UnitVAB, CPU_GraphApiRngBuilder_NONE, validAttributesBernoulli);

INSTANTIATE_TEST_SUITE_P(UnitIAN, CPU_GraphApiRngBuilder_NONE, invalidAttributesNormal);
INSTANTIATE_TEST_SUITE_P(UnitIAU, CPU_GraphApiRngBuilder_NONE, invalidAttributesUniform);
INSTANTIATE_TEST_SUITE_P(UnitIAB, CPU_GraphApiRngBuilder_NONE, invalidAttributesBernoulli);

INSTANTIATE_TEST_SUITE_P(UnitVAN, CPU_GraphApiRng_NONE, validAttributesNormal);
INSTANTIATE_TEST_SUITE_P(UnitVAU, CPU_GraphApiRng_NONE, validAttributesUniform);
INSTANTIATE_TEST_SUITE_P(UnitVAB, CPU_GraphApiRng_NONE, validAttributesBernoulli);

INSTANTIATE_TEST_SUITE_P(UnitIAN, CPU_GraphApiRng_NONE, invalidAttributesNormal);
INSTANTIATE_TEST_SUITE_P(UnitIAU, CPU_GraphApiRng_NONE, invalidAttributesUniform);
INSTANTIATE_TEST_SUITE_P(UnitIAB, CPU_GraphApiRng_NONE, invalidAttributesBernoulli);
