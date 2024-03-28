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

#include <miopen/graphapi/variant_pack.hpp>
#include <miopen/miopen.h>

#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

namespace {

template <typename T>
struct ValidatedVector
{
    bool valid;
    std::vector<T> values;

    friend void PrintTo(const ValidatedVector& v, std::ostream* os)
    {
        *os << '{';
        auto begin = v.values.cbegin();
        auto end   = v.values.cend();
        if(begin != end)
            *os << *begin++;
        while(begin != end)
            *os << ' ' << *begin++;
        *os << '}';
    }
};

template <typename T>
struct ValidatedValue
{
    bool valid;
    T value;

    friend void PrintTo(const ValidatedValue& v, std::ostream* os) { *os << v.value; }
};

} // namespace

using GraphApiVariantPackTuple =
    std::tuple<bool, ValidatedVector<int64_t>, ValidatedVector<void*>, ValidatedValue<void*>>;

class GraphApiVariantPackBuilder : public testing::TestWithParam<GraphApiVariantPackTuple>
{
protected:
    bool attrsValid;
    ValidatedVector<int64_t> tensorIds;
    ValidatedVector<void*> dataPointers;
    ValidatedValue<void*> workspace;

    void SetUp() override { std::tie(attrsValid, tensorIds, dataPointers, workspace) = GetParam(); }
    miopen::graphapi::VariantPack buildByLValue()
    {
        miopen::graphapi::VariantPackBuilder builder;
        return builder.setTensorIds(tensorIds.values)
            .setDataPointers(dataPointers.values)
            .setWorkspace(workspace.value)
            .build();
    }
    miopen::graphapi::VariantPack buildByRValue()
    {
        return miopen::graphapi::VariantPackBuilder()
            .setTensorIds(tensorIds.values)
            .setDataPointers(dataPointers.values)
            .setWorkspace(workspace.value)
            .build();
    }
    void setIdsByRValue(bool passAttrByRValue)
    {
        if(passAttrByRValue)
        {
            auto attr = tensorIds.values;
            miopen::graphapi::VariantPackBuilder().setTensorIds(std::move(attr));
        }
        else
        {
            miopen::graphapi::VariantPackBuilder().setTensorIds(tensorIds.values);
        }
    }
    void setIdsByLValue(bool passAttrByRValue)
    {
        miopen::graphapi::VariantPackBuilder builder;
        if(passAttrByRValue)
        {
            auto attr = tensorIds.values;
            builder.setTensorIds(std::move(attr));
        }
        else
        {
            builder.setTensorIds(tensorIds.values);
        }
    }
    void setPointersByRValue(bool passAttrByRValue)
    {
        if(passAttrByRValue)
        {
            auto attr = dataPointers.values;
            miopen::graphapi::VariantPackBuilder().setDataPointers(std::move(attr));
        }
        else
        {
            miopen::graphapi::VariantPackBuilder().setDataPointers(dataPointers.values);
        }
    }
    void setPointersByLValue(bool passAttrByRValue)
    {
        miopen::graphapi::VariantPackBuilder builder;
        if(passAttrByRValue)
        {
            auto attr = dataPointers.values;
            builder.setDataPointers(std::move(attr));
        }
        else
        {
            builder.setDataPointers(dataPointers.values);
        }
    }
    void setWorkspace(bool byRValue)
    {
        if(byRValue)
        {
            miopen::graphapi::VariantPackBuilder().setWorkspace(workspace.value);
        }
        else
        {
            miopen::graphapi::VariantPackBuilder builder;
            builder.setWorkspace(workspace.value);
        }
    }
};

TEST_P(GraphApiVariantPackBuilder, ValidateAttributes)
{
    if(attrsValid)
    {
        EXPECT_NO_THROW({ buildByRValue(); }) << "R-value builder failed on valid attributes";
        EXPECT_NO_THROW({ buildByLValue(); }) << "L-value builder failed on valid attribures";
    }
    else
    {
        EXPECT_ANY_THROW({ buildByRValue(); })
            << "R-value builder failed to detect invalid attributes";
        EXPECT_ANY_THROW({ buildByLValue(); })
            << "L-value builder failed to detect invalid attributes";
    }
    if(tensorIds.valid)
    {
        EXPECT_NO_THROW({ setIdsByRValue(false); })
            << "VariantPackBuilder::setTensorIds(const std::vector<int64_t>&)&& failed on valid "
               "attribute";
        EXPECT_NO_THROW({ setIdsByRValue(true); })
            << "VariantPackBuilder::setTensorIds(std::vector<int64_t>&&)&& failed on valid "
               "attribute";
        EXPECT_NO_THROW({ setIdsByLValue(false); })
            << "VariantPackBuilder::setTensorIds(const std::vector<int64_t>&)& failed on valid "
               "attribute";
        EXPECT_NO_THROW({
            setIdsByLValue(true);
        }) << "VariantPackBuilder::setTensorIds(std::vector<int64_t>&&)& failed on valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ setIdsByRValue(false); })
            << "VariantPackBuilder::setTensorIds(const std::vector<int64_t>&)&& failed on invalid "
               "attribute";
        EXPECT_ANY_THROW({ setIdsByRValue(true); })
            << "VariantPackBuilder::setTensorIds(std::vector<int64_t>&&)&& failed on invalid "
               "attribute";
        EXPECT_ANY_THROW({ setIdsByLValue(false); })
            << "VariantPackBuilder::setTensorIds(const std::vector<int64_t>&)& failed on invalid "
               "attribute";
        EXPECT_ANY_THROW({ setIdsByLValue(true); })
            << "VariantPackBuilder::setTensorIds(std::vector<int64_t>&&)& failed on invalid "
               "attribute";
    }
    if(dataPointers.valid)
    {
        EXPECT_NO_THROW({ setPointersByRValue(false); })
            << "VariantPackBuilder::setDataPointers(const std::vector<void*>&)&& failed on valid "
               "attribute";
        EXPECT_NO_THROW({ setPointersByRValue(true); })
            << "VariantPackBuilder::setDataPointers(std::vector<void*>&&)&& failed on valid "
               "attribute";
        EXPECT_NO_THROW({ setPointersByLValue(false); })
            << "VariantPackBuilder::setDataPointers(const std::vector<void*>&)& failed on valid "
               "attribute";
        EXPECT_NO_THROW({ setPointersByLValue(true); })
            << "VariantPackBuilder::setDataPointers(std::vector<void*>&&)& failed on valid "
               "attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ setPointersByRValue(false); })
            << "VariantPackBuilder::setDataPointers(const std::vector<void*>&)&& failed on invalid "
               "attribute";
        EXPECT_ANY_THROW({ setPointersByRValue(true); })
            << "VariantPackBuilder::setDataPointers(std::vector<void*>&&)&& failed on invalid "
               "attribute";
        EXPECT_ANY_THROW({ setPointersByLValue(false); })
            << "VariantPackBuilder::setDataPointers(const std::vector<void*>&)& failed on invalid "
               "attribute";
        EXPECT_ANY_THROW({ setPointersByLValue(true); })
            << "VariantPackBuilder::setDataPointers(std::vector<void*>&&)& failed on invalid "
               "attribute";
    }
    if(workspace.valid)
    {
        EXPECT_NO_THROW({ setWorkspace(true); })
            << "VariantPackBuilder::setWorkspace(void*)&& failed on valid attribute";
        EXPECT_NO_THROW({ setWorkspace(false); })
            << "VariantPackBuilder::setWorkspace(void*)& failed on valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ setWorkspace(true); })
            << "VariantPackBuilder::setWorkspace(void*)&& failed on invalid attribute";
        EXPECT_ANY_THROW({ setWorkspace(false); })
            << "VariantPackBuilder::setWorkspace(void*)& failed on invalid attribute";
    }
}

TEST_P(GraphApiVariantPackBuilder, MissingSetter)
{
    EXPECT_ANY_THROW({
        miopen::graphapi::VariantPackBuilder()
            .setDataPointers(dataPointers.values)
            .setWorkspace(workspace.value)
            .build();
    }) << "R-value builder validated attributes despite missing setTensorIds() call";
    EXPECT_ANY_THROW({
        miopen::graphapi::VariantPackBuilder()
            .setTensorIds(tensorIds.values)
            .setWorkspace(workspace.value)
            .build();
    }) << "R-value builder validated attributes despite missing setDataPointers() call";
    EXPECT_ANY_THROW({
        miopen::graphapi::VariantPackBuilder()
            .setTensorIds(tensorIds.values)
            .setDataPointers(dataPointers.values)
            .build();
    }) << "R-value builder validated attributes despite missing setWorkspace() call";
    EXPECT_ANY_THROW({
        miopen::graphapi::VariantPackBuilder builder;
        builder.setDataPointers(dataPointers.values).setWorkspace(workspace.value).build();
    }) << "L-value builder validated attributes despite missing setTensorIds() call";
    EXPECT_ANY_THROW({
        miopen::graphapi::VariantPackBuilder builder;
        builder.setTensorIds(tensorIds.values).setWorkspace(workspace.value).build();
    }) << "L-value builder validated attributes despite missing setDataPointers() call";
    EXPECT_ANY_THROW({
        miopen::graphapi::VariantPackBuilder builder;
        builder.setTensorIds(tensorIds.values).setDataPointers(dataPointers.values).build();
    }) << "L-value builder validated attributes despite missing setWorkspace() call";
}

TEST_P(GraphApiVariantPackBuilder, RetrieveAttributes)
{
    miopen::graphapi::VariantPack vPack;
    if(attrsValid)
    {
        ASSERT_NO_THROW({ vPack = buildByRValue(); }) << "Builder failed on valid attributes";
    }
    else
    {
        ASSERT_ANY_THROW({ vPack = buildByRValue(); })
            << "Builder failed to detect invalid attributes";
        return;
    }

    auto idsAndPointersCorrect =
        std::inner_product(tensorIds.values.cbegin(),
                           tensorIds.values.cend(),
                           dataPointers.values.cbegin(),
                           true,
                           std::logical_and<>(),
                           [&vPack](auto id, auto ptr) { return vPack.getDataPointer(id) == ptr; });
    EXPECT_TRUE(idsAndPointersCorrect)
        << "Tensor ids or data pointers are set or retrieved incorrectly";
    EXPECT_EQ(vPack.getWorkspace(), workspace.value) << "Workspace is set or retrieved incorrectly";
}

static char mem[10][256];

INSTANTIATE_TEST_SUITE_P(
    ValidAttributes,
    GraphApiVariantPackBuilder,
    testing::Values(GraphApiVariantPackTuple{true, {true, {}}, {true, {}}, {true, mem[9]}},
                    GraphApiVariantPackTuple{
                        true, {true, {1, 2, 3}}, {true, {mem[0], mem[1], mem[2]}}, {true, mem[9]}},
                    GraphApiVariantPackTuple{true,
                                             {true, {1, 2, 3, 4, 5}},
                                             {true, {mem[0], mem[1], mem[2], mem[3], mem[4]}},
                                             {true, mem[9]}}));
INSTANTIATE_TEST_SUITE_P(
    InvalidIds,
    GraphApiVariantPackBuilder,
    testing::Combine(
        testing::Values(false),
        testing::Values(ValidatedVector<int64_t>{false, {1, 2, 1, 4, 5}},
                        ValidatedVector<int64_t>{false, {1, 2, 3, 2, 5}},
                        ValidatedVector<int64_t>{false, {1, 5, 3, 4, 5}}),
        testing::Values(ValidatedVector<void*>{true, {mem[0], mem[1], mem[2], mem[3], mem[4]}},
                        ValidatedVector<void*>{true, {mem[0], mem[1], mem[2]}},
                        ValidatedVector<void*>{true, {}}),
        testing::Values(ValidatedValue<void*>{true, mem[9]})));
INSTANTIATE_TEST_SUITE_P(
    InvalidPointers,
    GraphApiVariantPackBuilder,
    testing::Combine(
        testing::Values(false),
        testing::Values(ValidatedVector<int64_t>{true, {1, 2, 3, 4, 5}},
                        ValidatedVector<int64_t>{true, {1, 2, 3}},
                        ValidatedVector<int64_t>{true, {}}),
        testing::Values(ValidatedVector<void*>{false, {nullptr, mem[1], mem[2], mem[3], mem[4]}},
                        ValidatedVector<void*>{false, {mem[0], nullptr, mem[2], mem[3], mem[4]}},
                        ValidatedVector<void*>{false, {mem[0], mem[1], nullptr, mem[3], mem[4]}},
                        ValidatedVector<void*>{false, {mem[0], mem[1], mem[2], nullptr, mem[4]}},
                        ValidatedVector<void*>{false, {mem[0], mem[1], mem[2], mem[3], nullptr}},
                        ValidatedVector<void*>{false, {mem[0], mem[1], mem[2], mem[0], mem[4]}},
                        ValidatedVector<void*>{false, {mem[0], mem[4], mem[2], mem[3], mem[4]}},
                        ValidatedVector<void*>{false, {mem[0], mem[1], mem[2], mem[2], mem[4]}}),
        testing::Values(ValidatedValue<void*>{false, nullptr},
                        ValidatedValue<void*>{true, mem[2]},
                        ValidatedValue<void*>{true, mem[9]})));
INSTANTIATE_TEST_SUITE_P(
    NullWorkspace,
    GraphApiVariantPackBuilder,
    testing::Combine(testing::Values(false),
                     testing::Values(ValidatedVector<int64_t>{true, {}},
                                     ValidatedVector<int64_t>{true, {1, 2, 3, 4, 5}}),
                     testing::Values(ValidatedVector<void*>{true, {}},
                                     ValidatedVector<void*>{
                                         true, {mem[0], mem[1], mem[2], mem[3], mem[4]}}),
                     testing::Values(ValidatedValue<void*>{false, nullptr})));
INSTANTIATE_TEST_SUITE_P(InvalidWorkspace,
                         GraphApiVariantPackBuilder,
                         testing::Combine(testing::Values(false),
                                          testing::Values(ValidatedVector<int64_t>{true, {}},
                                                          ValidatedVector<int64_t>{
                                                              true, {1, 2, 3, 4, 5}}),
                                          testing::Values(ValidatedVector<void*>{
                                              true, {mem[0], mem[1], mem[2], mem[3], mem[4]}}),
                                          testing::Values(ValidatedValue<void*>{true, mem[0]},
                                                          ValidatedValue<void*>{true, mem[1]},
                                                          ValidatedValue<void*>{true, mem[2]},
                                                          ValidatedValue<void*>{true, mem[3]},
                                                          ValidatedValue<void*>{true, mem[4]})));
INSTANTIATE_TEST_SUITE_P(
    SizeMismatch,
    GraphApiVariantPackBuilder,
    testing::Combine(testing::Values(false),
                     testing::Values(ValidatedVector<int64_t>{true, {}},
                                     ValidatedVector<int64_t>{true, {1, 2, 3, 4, 5}},
                                     ValidatedVector<int64_t>{false, {5, 2, 3, 4, 5}}),
                     testing::Values(ValidatedVector<void*>{true, {mem[0]}},
                                     ValidatedVector<void*>{false, {nullptr}},
                                     ValidatedVector<void*>{true, {mem[0], mem[1], mem[2], mem[3]}},
                                     ValidatedVector<void*>{false,
                                                            {mem[0], mem[3], mem[2], mem[3]}}),
                     testing::Values(ValidatedValue<void*>{true, mem[0]},
                                     ValidatedValue<void*>{true, mem[1]},
                                     ValidatedValue<void*>{true, mem[2]},
                                     ValidatedValue<void*>{true, mem[3]},
                                     ValidatedValue<void*>{true, mem[9]})));
