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

#include "graphapi_gtest_common.hpp"

namespace {

using miopen::graphapi::ValidatedValue;
using miopen::graphapi::ValidatedVector;
using GraphApiVariantPackTuple =
    std::tuple<bool, ValidatedVector<int64_t>, ValidatedVector<void*>, ValidatedValue<void*>>;

} // namespace

class CPU_GraphApiVariantPackBuilder_NONE : public testing::TestWithParam<GraphApiVariantPackTuple>
{
protected:
    bool mAttrsValid;
    ValidatedVector<int64_t> mTensorIds;
    ValidatedVector<void*> mDataPointers;
    ValidatedValue<void*> mWorkspace;

    void SetUp() override
    {
        std::tie(mAttrsValid, mTensorIds, mDataPointers, mWorkspace) = GetParam();
    }
    miopen::graphapi::VariantPack buildByLValue()
    {
        miopen::graphapi::VariantPackBuilder builder;
        return builder.setTensorIds(mTensorIds.values)
            .setDataPointers(mDataPointers.values)
            .setWorkspace(mWorkspace.value)
            .build();
    }
    miopen::graphapi::VariantPack buildByRValue()
    {
        return miopen::graphapi::VariantPackBuilder()
            .setTensorIds(mTensorIds.values)
            .setDataPointers(mDataPointers.values)
            .setWorkspace(mWorkspace.value)
            .build();
    }
    void setIdsByRValue(bool passAttrByRValue)
    {
        if(passAttrByRValue)
        {
            auto attr = mTensorIds.values;
            miopen::graphapi::VariantPackBuilder().setTensorIds(std::move(attr));
        }
        else
        {
            miopen::graphapi::VariantPackBuilder().setTensorIds(mTensorIds.values);
        }
    }
    void setIdsByLValue(bool passAttrByRValue)
    {
        miopen::graphapi::VariantPackBuilder builder;
        if(passAttrByRValue)
        {
            auto attr = mTensorIds.values;
            builder.setTensorIds(std::move(attr));
        }
        else
        {
            builder.setTensorIds(mTensorIds.values);
        }
    }
    void setPointersByRValue(bool passAttrByRValue)
    {
        if(passAttrByRValue)
        {
            auto attr = mDataPointers.values;
            miopen::graphapi::VariantPackBuilder().setDataPointers(std::move(attr));
        }
        else
        {
            miopen::graphapi::VariantPackBuilder().setDataPointers(mDataPointers.values);
        }
    }
    void setPointersByLValue(bool passAttrByRValue)
    {
        miopen::graphapi::VariantPackBuilder builder;
        if(passAttrByRValue)
        {
            auto attr = mDataPointers.values;
            builder.setDataPointers(std::move(attr));
        }
        else
        {
            builder.setDataPointers(mDataPointers.values);
        }
    }
    void setWorkspace(bool byRValue)
    {
        if(byRValue)
        {
            miopen::graphapi::VariantPackBuilder().setWorkspace(mWorkspace.value);
        }
        else
        {
            miopen::graphapi::VariantPackBuilder builder;
            builder.setWorkspace(mWorkspace.value);
        }
    }
};

TEST_P(CPU_GraphApiVariantPackBuilder_NONE, ValidateAttributes)
{
    if(mAttrsValid)
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
    if(mTensorIds.valid)
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
    if(mDataPointers.valid)
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
    if(mWorkspace.valid)
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

TEST_P(CPU_GraphApiVariantPackBuilder_NONE, MissingSetter)
{
    EXPECT_ANY_THROW({
        miopen::graphapi::VariantPackBuilder()
            .setDataPointers(mDataPointers.values)
            .setWorkspace(mWorkspace.value)
            .build();
    }) << "R-value builder validated attributes despite missing setTensorIds() call";
    EXPECT_ANY_THROW({
        miopen::graphapi::VariantPackBuilder()
            .setTensorIds(mTensorIds.values)
            .setWorkspace(mWorkspace.value)
            .build();
    }) << "R-value builder validated attributes despite missing setDataPointers() call";
    EXPECT_ANY_THROW({
        miopen::graphapi::VariantPackBuilder()
            .setTensorIds(mTensorIds.values)
            .setDataPointers(mDataPointers.values)
            .build();
    }) << "R-value builder validated attributes despite missing setWorkspace() call";
    EXPECT_ANY_THROW({
        miopen::graphapi::VariantPackBuilder builder;
        builder.setDataPointers(mDataPointers.values).setWorkspace(mWorkspace.value).build();
    }) << "L-value builder validated attributes despite missing setTensorIds() call";
    EXPECT_ANY_THROW({
        miopen::graphapi::VariantPackBuilder builder;
        builder.setTensorIds(mTensorIds.values).setWorkspace(mWorkspace.value).build();
    }) << "L-value builder validated attributes despite missing setDataPointers() call";
    EXPECT_ANY_THROW({
        miopen::graphapi::VariantPackBuilder builder;
        builder.setTensorIds(mTensorIds.values).setDataPointers(mDataPointers.values).build();
    }) << "L-value builder validated attributes despite missing setWorkspace() call";
}

TEST_P(CPU_GraphApiVariantPackBuilder_NONE, RetrieveAttributes)
{
    miopen::graphapi::VariantPack vPack;
    if(mAttrsValid)
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
        std::inner_product(mTensorIds.values.cbegin(),
                           mTensorIds.values.cend(),
                           mDataPointers.values.cbegin(),
                           true,
                           std::logical_and<>(),
                           [&vPack](auto id, auto ptr) { return vPack.getDataPointer(id) == ptr; });
    EXPECT_TRUE(idsAndPointersCorrect)
        << "Tensor ids or data pointers are set or retrieved incorrectly";
    EXPECT_EQ(vPack.getWorkspace(), mWorkspace.value)
        << "Workspace is set or retrieved incorrectly";
}

namespace {

using miopen::graphapi::GTestDescriptor;
using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::GTestDescriptorVectorAttribute;
using miopen::graphapi::GTestGraphApiExecute;

class TensorIds : public GTestDescriptorVectorAttribute<int64_t, char>
{
public:
    TensorIds() = default;
    TensorIds(const ValidatedVector<int64_t>& vv)
        : GTestDescriptorVectorAttribute<int64_t, char>(vv.valid,
                                                        "MIOPEN_ATTR_VARIANT_PACK_UNIQUE_IDS",
                                                        MIOPEN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                                        MIOPEN_TYPE_INT64,
                                                        MIOPEN_TYPE_CHAR,
                                                        -1,
                                                        vv.values)
    {
    }
};

class DataPointers : public GTestDescriptorVectorAttribute<void*, char>
{
public:
    DataPointers() = default;
    DataPointers(const ValidatedVector<void*>& vv)
        : GTestDescriptorVectorAttribute<void*, char>(vv.valid,
                                                      "MIOPEN_ATTR_VARIANT_PACK_DATA_POINTERS",
                                                      MIOPEN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                      MIOPEN_TYPE_VOID_PTR,
                                                      MIOPEN_TYPE_CHAR,
                                                      -1,
                                                      vv.values)
    {
    }
};

class Workspace : public GTestDescriptorSingleValueAttribute<void*, char>
{
public:
    Workspace() = default;
    Workspace(const ValidatedValue<void*>& vv)
        : GTestDescriptorSingleValueAttribute<void*, char>(vv.valid,
                                                           "MIOPEN_ATTR_VARIANT_PACK_WORKSPACE",
                                                           MIOPEN_ATTR_VARIANT_PACK_WORKSPACE,
                                                           MIOPEN_TYPE_VOID_PTR,
                                                           MIOPEN_TYPE_CHAR,
                                                           2,
                                                           vv.value)
    {
    }
};

} // namespace

class CPU_GraphApiVariantPack_NONE : public testing::TestWithParam<GraphApiVariantPackTuple>
{
private:
    // Pointers to these are used in mExecute object below
    TensorIds mTensorIds;
    DataPointers mDataPointers;
    Workspace mWorkspace;

protected:
    GTestGraphApiExecute<GTestDescriptorAttribute*> mExecute;

    void SetUp() override
    {
        bool valid                                             = false;
        std::tie(valid, mTensorIds, mDataPointers, mWorkspace) = GetParam();
        mExecute.descriptor = {"MIOPEN_BACKEND_VARIANT_PACK_DESCRIPTOR",
                               MIOPEN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                               valid,
                               {&mTensorIds, &mDataPointers, &mWorkspace}};
    }
};

TEST_P(CPU_GraphApiVariantPack_NONE, CFunctions) { mExecute(); }

static char mem[10][256];

static auto ValidAttributesTestCases =
    testing::Values(GraphApiVariantPackTuple{true, {true, {}}, {true, {}}, {true, mem[9]}},
                    GraphApiVariantPackTuple{true, {true, {}}, {true, {}}, {true, nullptr}},
                    GraphApiVariantPackTuple{
                        true, {true, {1, 2, 3}}, {true, {mem[0], mem[1], mem[2]}}, {true, mem[9]}},
                    GraphApiVariantPackTuple{true,
                                             {true, {1, 2, 3, 4, 5}},
                                             {true, {mem[0], mem[1], mem[2], mem[3], mem[4]}},
                                             {true, mem[9]}});

static auto InvalidIdsTestCases = testing::Combine(
    testing::Values(false),
    testing::Values(ValidatedVector<int64_t>{false, {1, 2, 1, 4, 5}},
                    ValidatedVector<int64_t>{false, {1, 2, 3, 2, 5}},
                    ValidatedVector<int64_t>{false, {1, 5, 3, 4, 5}}),
    testing::Values(ValidatedVector<void*>{true, {mem[0], mem[1], mem[2], mem[3], mem[4]}},
                    ValidatedVector<void*>{true, {mem[0], mem[1], mem[2]}},
                    ValidatedVector<void*>{true, {}}),
    testing::Values(ValidatedValue<void*>{true, mem[9]}));

static auto InvalidPointersTestCases = testing::Combine(
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
    testing::Values(ValidatedValue<void*>{true, nullptr},
                    ValidatedValue<void*>{true, mem[2]},
                    ValidatedValue<void*>{true, mem[9]}));

static auto InvalidWorkspace = testing::Combine(
    testing::Values(false),
    testing::Values(ValidatedVector<int64_t>{true, {}},
                    ValidatedVector<int64_t>{true, {1, 2, 3, 4, 5}}),
    testing::Values(ValidatedVector<void*>{true, {mem[0], mem[1], mem[2], mem[3], mem[4]}}),
    testing::Values(ValidatedValue<void*>{true, mem[0]},
                    ValidatedValue<void*>{true, mem[1]},
                    ValidatedValue<void*>{true, mem[2]},
                    ValidatedValue<void*>{true, mem[3]},
                    ValidatedValue<void*>{true, mem[4]}));

static auto SizeMismatch = testing::Combine(
    testing::Values(false),
    testing::Values(ValidatedVector<int64_t>{true, {}},
                    ValidatedVector<int64_t>{true, {1, 2, 3, 4, 5}},
                    ValidatedVector<int64_t>{false, {5, 2, 3, 4, 5}}),
    testing::Values(ValidatedVector<void*>{true, {mem[0]}},
                    ValidatedVector<void*>{false, {nullptr}},
                    ValidatedVector<void*>{true, {mem[0], mem[1], mem[2], mem[3]}},
                    ValidatedVector<void*>{false, {mem[0], mem[3], mem[2], mem[3]}}),
    testing::Values(ValidatedValue<void*>{true, mem[0]},
                    ValidatedValue<void*>{true, mem[1]},
                    ValidatedValue<void*>{true, mem[2]},
                    ValidatedValue<void*>{true, mem[3]},
                    ValidatedValue<void*>{true, mem[9]}));

INSTANTIATE_TEST_SUITE_P(UnitVA, CPU_GraphApiVariantPackBuilder_NONE, ValidAttributesTestCases);
INSTANTIATE_TEST_SUITE_P(UnitII, CPU_GraphApiVariantPackBuilder_NONE, InvalidIdsTestCases);
INSTANTIATE_TEST_SUITE_P(UnitIP, CPU_GraphApiVariantPackBuilder_NONE, InvalidPointersTestCases);
INSTANTIATE_TEST_SUITE_P(UnitIW, CPU_GraphApiVariantPackBuilder_NONE, InvalidWorkspace);
INSTANTIATE_TEST_SUITE_P(UnitSM, CPU_GraphApiVariantPackBuilder_NONE, SizeMismatch);

INSTANTIATE_TEST_SUITE_P(UnitVA, CPU_GraphApiVariantPack_NONE, ValidAttributesTestCases);
INSTANTIATE_TEST_SUITE_P(UnitII, CPU_GraphApiVariantPack_NONE, InvalidIdsTestCases);
INSTANTIATE_TEST_SUITE_P(UnitIP, CPU_GraphApiVariantPack_NONE, InvalidPointersTestCases);
INSTANTIATE_TEST_SUITE_P(UnitIW, CPU_GraphApiVariantPack_NONE, InvalidWorkspace);
INSTANTIATE_TEST_SUITE_P(UnitSM, CPU_GraphApiVariantPack_NONE, SizeMismatch);
