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

using miopen::graphapi::ValidatedVector;
using miopen::graphapi::ValidatedValue;
using GraphApiVariantPackTuple =
    std::tuple<bool, ValidatedVector<int64_t>, ValidatedVector<void*>, ValidatedValue<void*>>;

} // namespace

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

namespace {

using miopen::graphapi::GTestDescriptor;
using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::GTestDescriptorVectorAttribute;

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

class GraphApiVariantPack : public testing::TestWithParam<GraphApiVariantPackTuple>
{
protected:
    // to be used in the test
    GTestDescriptor<GTestDescriptorAttribute*> descriptor;

    // descriptor above contains pointer to these:
    TensorIds mTensorIds;
    DataPointers mDataPointers;
    Workspace mWorkspace;

    void SetUp() override
    {
        bool valid                                             = false;
        std::tie(valid, mTensorIds, mDataPointers, mWorkspace) = GetParam();
        descriptor = {"MIOPEN_BACKEND_VARIANT_PACK_DESCRIPTOR",
                      MIOPEN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                      valid,
                      {&mTensorIds, &mDataPointers, &mWorkspace}};
    }
};

TEST_P(GraphApiVariantPack, CFuncions)
{
    auto [descrTextName, descrType, attrsValid, attributes] = descriptor;

    // Create Desctiptor
    miopenBackendDescriptor_t descr;
    // clang-format off
    miopenStatus_t status = miopenBackendCreateDescriptor(descrType, &descr);
    ASSERT_EQ(status, miopenStatusSuccess) << descrTextName << " wasn't created";
    ASSERT_NE(descr, nullptr) << "A null " << descrTextName << " was created";
    // clang-format on

    // Finalize before setting attributes
    status = miopenBackendFinalize(descr);
    if(status == miopenStatusSuccess)
    {
        miopenBackendDestroyDescriptor(descr);
        FAIL() << descrTextName << " was finalized without setting attributes";
    }

    // Set attributes (should succeed)
    bool anyAttributeFailed = false;
    for(auto& attrPtr : attributes)
    {
        auto [isCorrect,
              textName,
              name,
              type,
              count,
              data,
              invalidType,
              invalidTypeData,
              invalidCount,
              invalidCountData,
              readBuffer] = attrPtr->getTestCase();

        // clang-format off
        status = miopenBackendSetAttribute(descr, name, invalidType, count, invalidTypeData);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was set with invalid type";

        status = miopenBackendSetAttribute(descr, name, type, invalidCount, invalidCountData);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was set with invalid element count";

        status = miopenBackendSetAttribute(descr, name, type, count, nullptr);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was set with null array of elements";

        status = miopenBackendSetAttribute(descr, name, type, count, data);
        if(attrsValid) // implementation may postpone validating values to finalize()
            EXPECT_EQ(status, miopenStatusSuccess) << textName << " wasn't set";
        // clang-format on

        anyAttributeFailed = anyAttributeFailed || (status != miopenStatusSuccess);
    }

    // Get attibute before finalizing (not a one should succeed)
    bool anyAttributeGot = false;
    for(auto& attrPtr : attributes)
    {
        auto [isCorrect,
              textName,
              name,
              type,
              count,
              data,
              invalidType,
              invalidTypeData,
              invalidCount,
              invalidCountData,
              readBuffer] = attrPtr->getTestCase();

        int64_t elementCount = 0;

        status = miopenBackendGetAttribute(descr, name, type, count, &elementCount, readBuffer);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was retrieved before finalize()";

        anyAttributeGot = anyAttributeGot || (status == miopenStatusSuccess);
    }

    // Stop further execution if needed
    if(anyAttributeGot)
    {
        miopenBackendDestroyDescriptor(descr);
        FAIL() << "Some attributes of " << descrTextName << " were retrieved before finalize()";
    }
    if(anyAttributeFailed && attrsValid)
    {
        miopenBackendDestroyDescriptor(descr);
        FAIL() << "Not all attributes of " << descrTextName << " were set";
    }

    // Finalize
    status = miopenBackendFinalize(descr);

    // Stop further execution if finalize() acted incorrectly
    if(attrsValid && status != miopenStatusSuccess)
    {
        miopenBackendDestroyDescriptor(descr);
        FAIL() << descrTextName << " wasn't finalized";
    }
    else if(!attrsValid)
    {
        miopenBackendDestroyDescriptor(descr);
        ASSERT_NE(status, miopenStatusSuccess)
            << descrTextName << " was finalized on invalid attributes";

        // No need to proceed with invalid attributes
        return;
    }

    // Set attributes after finalizing (not a one should succeed)
    bool anyAttributeSet = false;
    for(auto& attrPtr : attributes)
    {
        auto [isCorrect,
              textName,
              name,
              type,
              count,
              data,
              invalidType,
              invalidTypeData,
              invalidCount,
              invalidCountData,
              readBuffer] = attrPtr->getTestCase();

        status = miopenBackendSetAttribute(descr, name, type, count, data);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was set after finalize()";

        anyAttributeSet = anyAttributeSet || (status == miopenStatusSuccess);
    }

    // Stop if an attribute was set
    if(anyAttributeSet)
    {
        miopenBackendDestroyDescriptor(descr);
        ASSERT_NE(status, miopenStatusSuccess)
            << "An attribute of " << descrTextName << " was set after finalize()";
    }

    // Get attributes
    for(auto& attrPtr : attributes)
    {
        auto [isCorrect,
              textName,
              name,
              type,
              count,
              data,
              invalidType,
              invalidTypeData,
              invalidCount,
              invalidCountData,
              readBuffer] = attrPtr->getTestCase();

        int64_t elementCount = 0;
        // clang-format off
        status = miopenBackendGetAttribute(descr, name, invalidType, count, &elementCount, invalidTypeData);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was retrieved with invalid type";
        status = miopenBackendGetAttribute(descr, name, type, invalidCount, &elementCount, invalidCountData);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was retrieved with invalid element count";
        status = miopenBackendGetAttribute(descr, name, type, count, nullptr, readBuffer);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was retrieved with null element count";
        status = miopenBackendGetAttribute(descr, name, type, count, &elementCount, nullptr);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was retrieved with null array of elements";
        status = miopenBackendGetAttribute(descr, name, type, count, &elementCount, readBuffer);
        EXPECT_EQ(status, miopenStatusSuccess) << textName << " wasn't retrieved";
        EXPECT_EQ(count, elementCount) << textName << " set and retrieved number of elements differ";
        if(status == miopenStatusSuccess && count == elementCount)
            EXPECT_TRUE(attrPtr->isSetAndGotEqual()) << textName << " set and retrieved values differ";
        // clang-format on
    }
}

static char mem[10][256];

static auto ValidAttributesTestCases =
    testing::Values(GraphApiVariantPackTuple{true, {true, {}}, {true, {}}, {true, mem[9]}},
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
    testing::Values(ValidatedValue<void*>{false, nullptr},
                    ValidatedValue<void*>{true, mem[2]},
                    ValidatedValue<void*>{true, mem[9]}));

static auto NullWorkspace = testing::Combine(
    testing::Values(false),
    testing::Values(ValidatedVector<int64_t>{true, {}},
                    ValidatedVector<int64_t>{true, {1, 2, 3, 4, 5}}),
    testing::Values(ValidatedVector<void*>{true, {}},
                    ValidatedVector<void*>{true, {mem[0], mem[1], mem[2], mem[3], mem[4]}}),
    testing::Values(ValidatedValue<void*>{false, nullptr}));

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

INSTANTIATE_TEST_SUITE_P(ValidAttributes, GraphApiVariantPackBuilder, ValidAttributesTestCases);
INSTANTIATE_TEST_SUITE_P(InvalidIds, GraphApiVariantPackBuilder, InvalidIdsTestCases);
INSTANTIATE_TEST_SUITE_P(InvalidPointers, GraphApiVariantPackBuilder, InvalidPointersTestCases);
INSTANTIATE_TEST_SUITE_P(NullWorkspace, GraphApiVariantPackBuilder, NullWorkspace);
INSTANTIATE_TEST_SUITE_P(InvalidWorkspace, GraphApiVariantPackBuilder, InvalidWorkspace);
INSTANTIATE_TEST_SUITE_P(SizeMismatch, GraphApiVariantPackBuilder, SizeMismatch);

INSTANTIATE_TEST_SUITE_P(ValidAttributes, GraphApiVariantPack, ValidAttributesTestCases);
INSTANTIATE_TEST_SUITE_P(InvalidIds, GraphApiVariantPack, InvalidIdsTestCases);
INSTANTIATE_TEST_SUITE_P(InvalidPointers, GraphApiVariantPack, InvalidPointersTestCases);
INSTANTIATE_TEST_SUITE_P(NullWorkspace, GraphApiVariantPack, NullWorkspace);
INSTANTIATE_TEST_SUITE_P(InvalidWorkspace, GraphApiVariantPack, InvalidWorkspace);
INSTANTIATE_TEST_SUITE_P(SizeMismatch, GraphApiVariantPack, SizeMismatch);
