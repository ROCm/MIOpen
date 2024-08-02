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
#include <miopen/graphapi/tensor.hpp>

#include <array>
#include <tuple>
#include <variant>

#include <gtest/gtest.h>

#include "graphapi_gtest_common.hpp"

namespace miopen {

namespace graphapi {

std::ostream& operator<<(std::ostream& os, std::variant<int64_t, Tensor*> seed)
{
    if(seed.index() == 0)
        return os << std::get<0>(seed);
    else
        return os << std::get<1>(seed);
};

} // namespace graphapi

} // namespace miopen

namespace {

using miopen::graphapi::OperationRng;
using miopen::graphapi::OperationRngBuilder;
using miopen::graphapi::Rng;
using miopen::graphapi::Tensor;

using miopen::graphapi::ValidatedValue;

using DescriptorTuple = std::tuple<bool,
                                   ValidatedValue<Rng*>,
                                   ValidatedValue<Tensor*>,
                                   ValidatedValue<std::variant<int64_t, Tensor*>>,
                                   ValidatedValue<Tensor*>>;

} // namespace

class CPU_GraphApiOperationRngBuilder_NONE : public testing::TestWithParam<DescriptorTuple>
{
protected:
    bool mAttrsValid;
    ValidatedValue<Rng*> mRng;
    ValidatedValue<Tensor*> mOutput;
    ValidatedValue<std::variant<int64_t, Tensor*>> mSeed;
    ValidatedValue<Tensor*> mOffset;

    void SetUp() override { std::tie(mAttrsValid, mRng, mOutput, mSeed, mOffset) = GetParam(); }

    OperationRng buildWithDefaultSeed()
    {
        return OperationRngBuilder()
            .setRng(mRng.value)
            .setOutput(mOutput.value)
            .setOffset(mOffset.value)
            .build();
    }

    OperationRng build()
    {
        if(mSeed.value.index() == 0)
        {
            return OperationRngBuilder()
                .setRng(mRng.value)
                .setOutput(mOutput.value)
                .setSeed(std::get<0>(mSeed.value))
                .setOffset(mOffset.value)
                .build();
        }
        else
        {
            return OperationRngBuilder()
                .setRng(mRng.value)
                .setOutput(mOutput.value)
                .setSeed(std::get<1>(mSeed.value))
                .setOffset(mOffset.value)
                .build();
        }
    }
};

TEST_P(CPU_GraphApiOperationRngBuilder_NONE, ValidateAttributes)
{
    if(mAttrsValid)
    {
        EXPECT_NO_THROW({ build(); }) << "Builder failed on valid attributes";
        EXPECT_NO_THROW({ buildWithDefaultSeed(); })
            << "Builder failed on valid attributes and default seed";
    }
    else
    {
        EXPECT_ANY_THROW({ build(); }) << "Builder failed to detect invalid attributes";
        if(!mRng.valid || !mOutput.valid || !mOffset.valid)
        {
            EXPECT_ANY_THROW({ buildWithDefaultSeed(); })
                << "Builder failed to detect invalid attributes with default seed";
        }
    }

    if(mRng.valid)
    {
        EXPECT_NO_THROW({ OperationRngBuilder().setRng(mRng.value); })
            << "OperationRngBuilder::setRng failed with a valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ OperationRngBuilder().setRng(mRng.value); })
            << "OperationRngBuilder::setRng failed with an invalid attribute";
    }

    if(mOutput.valid)
    {
        EXPECT_NO_THROW({ OperationRngBuilder().setOutput(mOutput.value); })
            << "OperationRngBuilder::setOutput failed with a valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ OperationRngBuilder().setOutput(mOutput.value); })
            << "OperationRngBuilder::setOutput failed with an invalid attribute";
    }

    if(mSeed.valid)
    {
        if(mSeed.value.index() == 0)
        {
            EXPECT_NO_THROW({ OperationRngBuilder().setSeed(std::get<0>(mSeed.value)); })
                << "OperationRngBuilder::setSeed(int64_t) failed with a valid attribute";
        }
        else
        {
            EXPECT_NO_THROW({ OperationRngBuilder().setSeed(std::get<1>(mSeed.value)); })
                << "OperationRngBuilder::setSeed(Tensor*) failed with a valid attribute";
        }
    }
    else
    {
        if(mSeed.value.index() == 0)
        {
            EXPECT_ANY_THROW({ OperationRngBuilder().setSeed(std::get<0>(mSeed.value)); })
                << "OperationRngBuilder::setSeed(int64_t) failed with an invalid attribute";
        }
        else
        {
            EXPECT_ANY_THROW({ OperationRngBuilder().setSeed(std::get<1>(mSeed.value)); })
                << "OperationRngBuilder::setSeed(Tensor*) failed with an invalid attribute";
        }
    }

    if(mOffset.valid)
    {
        EXPECT_NO_THROW({ OperationRngBuilder().setOffset(mOffset.value); })
            << "OperationRngBuilder::setOffset failed with a valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ OperationRngBuilder().setOffset(mOffset.value); })
            << "OperationRngBuilder::setOffset failed with an invalid attribute";
    }
}

TEST_P(CPU_GraphApiOperationRngBuilder_NONE, MissingSetter)
{
    EXPECT_ANY_THROW({
        OperationRngBuilder().setOutput(mOutput.value).setOffset(mOffset.value).build();
    }) << "Builder with default seed failed to detect missing setRng() call";
    EXPECT_ANY_THROW({ OperationRngBuilder().setRng(mRng.value).setOffset(mOffset.value).build(); })
        << "Builder with default seed failed to detect missing setOutput() call";
    EXPECT_ANY_THROW({ OperationRngBuilder().setRng(mRng.value).setOutput(mOutput.value).build(); })
        << "Builder with default seed failed to detect missing setOffset() call";
    if(mSeed.value.index() == 0)
    {
        EXPECT_ANY_THROW({
            OperationRngBuilder()
                .setSeed(std::get<0>(mSeed.value))
                .setOutput(mOutput.value)
                .setOffset(mOffset.value)
                .build();
        }) << "Builder failed to detect missing setRng() call";
        EXPECT_ANY_THROW({
            OperationRngBuilder()
                .setRng(mRng.value)
                .setSeed(std::get<0>(mSeed.value))
                .setOffset(mOffset.value)
                .build();
        }) << "Builder failed to detect missing setOutput() call";
        EXPECT_ANY_THROW({
            OperationRngBuilder()
                .setRng(mRng.value)
                .setSeed(std::get<0>(mSeed.value))
                .setOutput(mOutput.value)
                .build();
        }) << "Builder failed to detect missing setOffset() call";
    }
    else
    {
        EXPECT_ANY_THROW({
            OperationRngBuilder()
                .setSeed(std::get<1>(mSeed.value))
                .setOutput(mOutput.value)
                .setOffset(mOffset.value)
                .build();
        }) << "Builder failed to detect missing setRng() call";
        EXPECT_ANY_THROW({
            OperationRngBuilder()
                .setRng(mRng.value)
                .setSeed(std::get<1>(mSeed.value))
                .setOffset(mOffset.value)
                .build();
        }) << "Builder failed to detect missing setOutput() call";
        EXPECT_ANY_THROW({
            OperationRngBuilder()
                .setRng(mRng.value)
                .setSeed(std::get<1>(mSeed.value))
                .setOutput(mOutput.value)
                .build();
        }) << "Builder failed to detect missing setOffset() call";
    }
}

namespace {

using miopen::graphapi::BackendRngDescriptor;
using miopen::graphapi::GMockBackendTensorDescriptor;
using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::GTestGraphApiExecute;

class Seed
{
private:
    std::variant<GTestDescriptorSingleValueAttribute<int64_t, char>,
                 GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char>>
        mAttribute;
    GMockBackendTensorDescriptor mTensor;

public:
    Seed& operator=(ValidatedValue<std::variant<int64_t, Tensor*>>& testCaseSeed)
    {
        constexpr const char* textName              = "MIOPEN_ATTR_OPERATION_RNG_SEED";
        constexpr miopenBackendAttributeName_t name = MIOPEN_ATTR_OPERATION_RNG_SEED;

        if(testCaseSeed.value.index() == 0)
        {
            mAttribute =
                GTestDescriptorSingleValueAttribute<int64_t, char>(testCaseSeed.valid,
                                                                   textName,
                                                                   name,
                                                                   MIOPEN_TYPE_INT64,
                                                                   MIOPEN_TYPE_CHAR,
                                                                   2,
                                                                   std::get<0>(testCaseSeed.value));
        }
        else
        {
            if(testCaseSeed.valid)
            {
                mTensor = *std::get<1>(testCaseSeed.value);
            }
            mAttribute = GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char>(
                testCaseSeed.valid,
                textName,
                name,
                MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                MIOPEN_TYPE_CHAR,
                2,
                &mTensor);
        }

        return *this;
    }
    GTestDescriptorAttribute* get()
    {
        if(mAttribute.index() == 0)
        {
            return &std::get<0>(mAttribute);
        }
        else
        {
            return &std::get<1>(mAttribute);
        }
    }
};

class GMockBackendRngDescriptor : public BackendRngDescriptor
{
public:
    GMockBackendRngDescriptor& operator=(const ValidatedValue<Rng*>& testCaseRng)
    {
        if(!testCaseRng.valid)
        {
            return *this;
        }

        auto& theRng = *testCaseRng.value;

        auto distr = theRng.getDistribution();
        setAttribute(MIOPEN_ATTR_RNG_DISTRIBUTION, MIOPEN_TYPE_RNG_DISTRIBUTION, 1, &distr);

        auto normalMean  = theRng.getNormalMean();
        auto normalStdev = theRng.getNormalStdev();

        auto uniformMin = theRng.getUniformMin();
        auto uniformMax = theRng.getUniformMax();

        auto bernoulliProb = theRng.getBernoulliProb();

        switch(distr)
        {
        case MIOPEN_RNG_DISTRIBUTION_NORMAL:
            setAttribute(MIOPEN_ATTR_RNG_NORMAL_DIST_MEAN, MIOPEN_TYPE_DOUBLE, 1, &normalMean);
            setAttribute(MIOPEN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION,
                         MIOPEN_TYPE_DOUBLE,
                         1,
                         &normalStdev);
            break;

        case MIOPEN_RNG_DISTRIBUTION_UNIFORM:
            setAttribute(MIOPEN_ATTR_RNG_UNIFORM_DIST_MINIMUM, MIOPEN_TYPE_DOUBLE, 1, &uniformMin);
            setAttribute(MIOPEN_ATTR_RNG_UNIFORM_DIST_MAXIMUM, MIOPEN_TYPE_DOUBLE, 1, &uniformMax);
            break;

        case MIOPEN_RNG_DISTRIBUTION_BERNOULLI:
            setAttribute(
                MIOPEN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY, MIOPEN_TYPE_DOUBLE, 1, &bernoulliProb);
            break;
        }

        finalize();

        return *this;
    }
};

} // namespace

class CPU_GraphApiOperationRng_NONE : public ::testing::TestWithParam<DescriptorTuple>
{
private:
    // Pointers to these are stored in the objects below
    GMockBackendRngDescriptor mRngDescriptor;
    GMockBackendTensorDescriptor mOutputDescriptor;
    GMockBackendTensorDescriptor mOffsetDesctiptor;

    // Pointers to these are stored in mExecute object below
    GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char> mRng;
    GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char> mOutput;
    Seed mSeed;
    GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char> mOffset;

protected:
    GTestGraphApiExecute<GTestDescriptorAttribute*> mExecute;

    void SetUp() override
    {
        auto [valid, rng, output, seed, offset] = GetParam();

        try
        {
            mRngDescriptor    = rng;
            mOutputDescriptor = output;
            mSeed             = seed;
            mOffsetDesctiptor = offset;
        }
        catch(const std::exception& e)
        {
            FAIL() << e.what();
        }

        mRng = {rng.valid,
                "MIOPEN_ATTR_OPERATION_RNG_DESC",
                MIOPEN_ATTR_OPERATION_RNG_DESC,
                MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                MIOPEN_TYPE_CHAR,
                2,
                &mRngDescriptor};

        mOutput = {output.valid,
                   "MIOPEN_ATTR_OPERATION_RNG_YDESC",
                   MIOPEN_ATTR_OPERATION_RNG_YDESC,
                   MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                   MIOPEN_TYPE_CHAR,
                   2,
                   &mOutputDescriptor};

        mOffset = {offset.valid,
                   "MIOPEN_ATTR_OPERATION_RNG_OFFSET_DESC",
                   MIOPEN_ATTR_OPERATION_RNG_OFFSET_DESC,
                   MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                   MIOPEN_TYPE_CHAR,
                   2,
                   &mOffsetDesctiptor};

        mExecute.descriptor.attrsValid = valid;
        mExecute.descriptor.textName   = "MIOPEN_BACKEND_OPERATION_RNG_DESCRIPTOR";
        mExecute.descriptor.type       = MIOPEN_BACKEND_OPERATION_RNG_DESCRIPTOR;
        mExecute.descriptor.attributes = {&mRng, &mOutput, mSeed.get(), &mOffset};
    }
};

TEST_P(CPU_GraphApiOperationRng_NONE, CFunctions) { mExecute(); }

static Rng anRng(MIOPEN_RNG_DISTRIBUTION_BERNOULLI, 0, 0, 0, 0, 0.5);

static std::array<ValidatedValue<Rng*>, 2> anyRngs{ValidatedValue<Rng*>{true, &anRng},
                                                   ValidatedValue<Rng*>{false, nullptr}};

static std::array<ValidatedValue<Rng*>, 1> validRngs{ValidatedValue<Rng*>{true, &anRng}};

static std::array<ValidatedValue<Rng*>, 1> invalidRngs{ValidatedValue<Rng*>{false, nullptr}};

static Tensor anOutput(miopenFloat, {10, 100, 100}, {100 * 100, 100, 1}, 1, false);

static std::array<ValidatedValue<Tensor*>, 2> anyOutputs{ValidatedValue<Tensor*>{true, &anOutput},
                                                         ValidatedValue<Tensor*>{false, nullptr}};

static std::array<ValidatedValue<Tensor*>, 1> validOutputs{
    ValidatedValue<Tensor*>{true, &anOutput}};

static std::array<ValidatedValue<Tensor*>, 1> invalidOutputs{
    ValidatedValue<Tensor*>{false, nullptr}};

static Tensor aValidSeedOrOffset(miopenFloat, {1, 1, 1}, {1, 1, 1}, 2, false);
static Tensor anInvalidSeedOrOffset(miopenFloat, {10, 100, 100}, {100 * 100, 100, 1}, 3, false);

static std::array<ValidatedValue<std::variant<int64_t, Tensor*>>, 4> anySeeds{
    ValidatedValue<std::variant<int64_t, Tensor*>>{true, 1},
    ValidatedValue<std::variant<int64_t, Tensor*>>{true, &aValidSeedOrOffset},
    ValidatedValue<std::variant<int64_t, Tensor*>>{false, nullptr},
    ValidatedValue<std::variant<int64_t, Tensor*>>{false, &anInvalidSeedOrOffset}};

static std::array<ValidatedValue<std::variant<int64_t, Tensor*>>, 2> validSeeds{
    ValidatedValue<std::variant<int64_t, Tensor*>>{true, 1},
    ValidatedValue<std::variant<int64_t, Tensor*>>{true, &aValidSeedOrOffset}};

static std::array<ValidatedValue<std::variant<int64_t, Tensor*>>, 2> invalidSeeds{
    ValidatedValue<std::variant<int64_t, Tensor*>>{false, nullptr},
    ValidatedValue<std::variant<int64_t, Tensor*>>{false, &anInvalidSeedOrOffset}};

static std::array<ValidatedValue<Tensor*>, 3> anyOffsets{
    ValidatedValue<Tensor*>{true, &aValidSeedOrOffset},
    ValidatedValue<Tensor*>{false, nullptr},
    ValidatedValue<Tensor*>{false, &anInvalidSeedOrOffset}};

static std::array<ValidatedValue<Tensor*>, 1> validOffsets{
    ValidatedValue<Tensor*>{true, &aValidSeedOrOffset}};

static std::array<ValidatedValue<Tensor*>, 2> invalidOffsets{
    ValidatedValue<Tensor*>{false, nullptr},
    ValidatedValue<Tensor*>{false, &anInvalidSeedOrOffset}};

static auto validAttributes = testing::Combine(testing::Values(true),
                                               testing::ValuesIn(validRngs),
                                               testing::ValuesIn(validOutputs),
                                               testing::ValuesIn(validSeeds),
                                               testing::ValuesIn(validOffsets));

static auto invalidAtLeastRngs = testing::Combine(testing::Values(false),
                                                  testing::ValuesIn(invalidRngs),
                                                  testing::ValuesIn(anyOutputs),
                                                  testing::ValuesIn(anySeeds),
                                                  testing::ValuesIn(anyOffsets));

static auto invalidAtLeastOutputs = testing::Combine(testing::Values(false),
                                                     testing::ValuesIn(anyRngs),
                                                     testing::ValuesIn(invalidOutputs),
                                                     testing::ValuesIn(anySeeds),
                                                     testing::ValuesIn(anyOffsets));

static auto invalidAtLeastSeeds = testing::Combine(testing::Values(false),
                                                   testing::ValuesIn(anyRngs),
                                                   testing::ValuesIn(anyOutputs),
                                                   testing::ValuesIn(invalidSeeds),
                                                   testing::ValuesIn(anyOffsets));

static auto invalidAtLeastOffsets = testing::Combine(testing::Values(false),
                                                     testing::ValuesIn(anyRngs),
                                                     testing::ValuesIn(anyOutputs),
                                                     testing::ValuesIn(anySeeds),
                                                     testing::ValuesIn(invalidOffsets));

INSTANTIATE_TEST_SUITE_P(UnitVA, CPU_GraphApiOperationRngBuilder_NONE, validAttributes);
INSTANTIATE_TEST_SUITE_P(UnitIR, CPU_GraphApiOperationRngBuilder_NONE, invalidAtLeastRngs);
INSTANTIATE_TEST_SUITE_P(UnitIO, CPU_GraphApiOperationRngBuilder_NONE, invalidAtLeastOutputs);
INSTANTIATE_TEST_SUITE_P(UnitIS, CPU_GraphApiOperationRngBuilder_NONE, invalidAtLeastSeeds);
INSTANTIATE_TEST_SUITE_P(UnitIOff, CPU_GraphApiOperationRngBuilder_NONE, invalidAtLeastOffsets);

INSTANTIATE_TEST_SUITE_P(UnitVA, CPU_GraphApiOperationRng_NONE, validAttributes);
INSTANTIATE_TEST_SUITE_P(UnitIR, CPU_GraphApiOperationRng_NONE, invalidAtLeastRngs);
INSTANTIATE_TEST_SUITE_P(UnitIO, CPU_GraphApiOperationRng_NONE, invalidAtLeastOutputs);
INSTANTIATE_TEST_SUITE_P(UnitIOff, CPU_GraphApiOperationRng_NONE, invalidAtLeastOffsets);

/* This one won't work as intended because seed is an optional attribute with a default value
 * and Graph API allows to finalize() if other attributes are valid.

INSTANTIATE_TEST_SUITE_P(InvalidAtLeastSeeds, GraphApiOperationRng, invalidAtLeastSeeds);
*/
