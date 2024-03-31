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

class GraphApiOperationRngBuilder : public testing::TestWithParam<DescriptorTuple>
{
protected:
    bool attrsValid;
    ValidatedValue<Rng*> rng;
    ValidatedValue<Tensor*> output;
    ValidatedValue<std::variant<int64_t, Tensor*>> seed;
    ValidatedValue<Tensor*> offset;

    void SetUp() override { std::tie(attrsValid, rng, output, seed, offset) = GetParam(); }

    OperationRng buildWithDefaultSeed()
    {
        return OperationRngBuilder()
            .setRng(rng.value)
            .setOutput(output.value)
            .setOffset(offset.value)
            .build();
    }

    OperationRng build()
    {
        if(seed.value.index() == 0)
        {
            return OperationRngBuilder()
                .setRng(rng.value)
                .setOutput(output.value)
                .setSeed(std::get<0>(seed.value))
                .setOffset(offset.value)
                .build();
        }
        else
        {
            return OperationRngBuilder()
                .setRng(rng.value)
                .setOutput(output.value)
                .setSeed(std::get<1>(seed.value))
                .setOffset(offset.value)
                .build();
        }
    }
};

TEST_P(GraphApiOperationRngBuilder, ValidateAttributes)
{
    if(attrsValid)
    {
        EXPECT_NO_THROW({ build(); }) << "Builder failed on valid attributes";
        EXPECT_NO_THROW({ buildWithDefaultSeed(); })
            << "Builder failed on valid attributes and default seed";
    }
    else
    {
        EXPECT_ANY_THROW({ build(); }) << "Builder failed to detect invalid attributes";
        if(!rng.valid || !output.valid || !offset.valid)
        {
            EXPECT_ANY_THROW({ buildWithDefaultSeed(); })
                << "Builder failed to detect invalid attributes with default seed";
        }
    }

    if(rng.valid)
    {
        EXPECT_NO_THROW({ OperationRngBuilder().setRng(rng.value); })
            << "OperationRngBuilder::setRng failed with a valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ OperationRngBuilder().setRng(rng.value); })
            << "OperationRngBuilder::setRng failed with an invalid attribute";
    }

    if(output.valid)
    {
        EXPECT_NO_THROW({ OperationRngBuilder().setOutput(output.value); })
            << "OperationRngBuilder::setOutput failed with a valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ OperationRngBuilder().setOutput(output.value); })
            << "OperationRngBuilder::setOutput failed with an invalid attribute";
    }

    if(seed.valid)
    {
        if(seed.value.index() == 0)
        {
            EXPECT_NO_THROW({ OperationRngBuilder().setSeed(std::get<0>(seed.value)); })
                << "OperationRngBuilder::setSeed(int64_t) failed with a valid attribute";
        }
        else
        {
            EXPECT_NO_THROW({ OperationRngBuilder().setSeed(std::get<1>(seed.value)); })
                << "OperationRngBuilder::setSeed(Tensor*) failed with a valid attribute";
        }
    }
    else
    {
        if(seed.value.index() == 0)
        {
            EXPECT_ANY_THROW({ OperationRngBuilder().setSeed(std::get<0>(seed.value)); })
                << "OperationRngBuilder::setSeed(int64_t) failed with an invalid attribute";
        }
        else
        {
            EXPECT_ANY_THROW({ OperationRngBuilder().setSeed(std::get<1>(seed.value)); })
                << "OperationRngBuilder::setSeed(Tensor*) failed with an invalid attribute";
        }
    }

    if(offset.valid)
    {
        EXPECT_NO_THROW({ OperationRngBuilder().setOffset(offset.value); })
            << "OperationRngBuilder::setOffset failed with a valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ OperationRngBuilder().setOffset(offset.value); })
            << "OperationRngBuilder::setOffset failed with an invalid attribute";
    }
}

TEST_P(GraphApiOperationRngBuilder, MissingSetter)
{
    EXPECT_ANY_THROW({
        OperationRngBuilder().setOutput(output.value).setOffset(offset.value).build();
    }) << "Builder with default seed failed to detect missing setRng() call";
    EXPECT_ANY_THROW({ OperationRngBuilder().setRng(rng.value).setOffset(offset.value).build(); })
        << "Builder with default seed failed to detect missing setOutput() call";
    EXPECT_ANY_THROW({ OperationRngBuilder().setRng(rng.value).setOutput(output.value).build(); })
        << "Builder with default seed failed to detect missing setOffset() call";
    if(seed.value.index() == 0)
    {
        EXPECT_ANY_THROW({
            OperationRngBuilder()
                .setSeed(std::get<0>(seed.value))
                .setOutput(output.value)
                .setOffset(offset.value)
                .build();
        }) << "Builder failed to detect missing setRng() call";
        EXPECT_ANY_THROW({
            OperationRngBuilder()
                .setRng(rng.value)
                .setSeed(std::get<0>(seed.value))
                .setOffset(offset.value)
                .build();
        }) << "Builder failed to detect missing setOutput() call";
        EXPECT_ANY_THROW({
            OperationRngBuilder()
                .setRng(rng.value)
                .setSeed(std::get<0>(seed.value))
                .setOutput(output.value)
                .build();
        }) << "Builder failed to detect missing setOffset() call";
    }
    else
    {
        EXPECT_ANY_THROW({
            OperationRngBuilder()
                .setSeed(std::get<1>(seed.value))
                .setOutput(output.value)
                .setOffset(offset.value)
                .build();
        }) << "Builder failed to detect missing setRng() call";
        EXPECT_ANY_THROW({
            OperationRngBuilder()
                .setRng(rng.value)
                .setSeed(std::get<1>(seed.value))
                .setOffset(offset.value)
                .build();
        }) << "Builder failed to detect missing setOutput() call";
        EXPECT_ANY_THROW({
            OperationRngBuilder()
                .setRng(rng.value)
                .setSeed(std::get<1>(seed.value))
                .setOutput(output.value)
                .build();
        }) << "Builder failed to detect missing setOffset() call";
    }
}

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

INSTANTIATE_TEST_SUITE_P(ValidAttributes, GraphApiOperationRngBuilder, validAttributes);
INSTANTIATE_TEST_SUITE_P(InvalidAtLeastRngs, GraphApiOperationRngBuilder, invalidAtLeastRngs);
INSTANTIATE_TEST_SUITE_P(InvalidAtLeastOutputs, GraphApiOperationRngBuilder, invalidAtLeastOutputs);
INSTANTIATE_TEST_SUITE_P(InvalidAtLeastSeeds, GraphApiOperationRngBuilder, invalidAtLeastSeeds);
INSTANTIATE_TEST_SUITE_P(InvalidAtLeastOffsets, GraphApiOperationRngBuilder, invalidAtLeastOffsets);
