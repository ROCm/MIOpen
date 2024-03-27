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

#include <miopen/graphapi/pointwise.hpp>

#include <memory>
#include <tuple>

#include <gtest/gtest.h>

#include "graphapi_gtest_common.hpp"

TEST(GraphApiPointwiseBuilder, Attributes)
{
    EXPECT_ANY_THROW({
        miopen::graphapi::PointwiseBuilder().setMode(MIOPEN_POINTWISE_ADD).build();
    }) << "Builder produced Pointwise despite missing setMathPrecision() call";
    EXPECT_ANY_THROW({
        miopen::graphapi::PointwiseBuilder().setMathPrecision(miopenFloat).build();
    }) << "Builder produced Pointwise despite missing setMode() call";
    EXPECT_NO_THROW({
        miopen::graphapi::PointwiseBuilder()
            .setMode(MIOPEN_POINTWISE_ADD)
            .setMathPrecision(miopenFloat)
            .build();
    }) << "Builder failed to produce Pointwise with valid attributes";
}

namespace {

using miopen::graphapi::GTestDescriptor;
using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;

class Mode : public GTestDescriptorSingleValueAttribute<miopenPointwiseMode_t, char>
{
public:
    Mode(miopenPointwiseMode_t mode)
        : GTestDescriptorSingleValueAttribute<miopenPointwiseMode_t, char>(
              true,
              "MIOPEN_ATTR_POINTWISE_MODE",
              MIOPEN_ATTR_POINTWISE_MODE,
              MIOPEN_TYPE_POINTWISE_MODE,
              MIOPEN_TYPE_CHAR,
              2,
              mode)
    {
    }
};

class Precision : public GTestDescriptorSingleValueAttribute<miopenDataType_t, char>
{
public:
    Precision(miopenDataType_t precision)
        : GTestDescriptorSingleValueAttribute<miopenDataType_t, char>(
              true,
              "MIOPEN_ATTR_POINTWISE_MATH_PREC",
              MIOPEN_ATTR_POINTWISE_MATH_PREC,
              MIOPEN_TYPE_DATA_TYPE,
              MIOPEN_TYPE_CHAR,
              2,
              precision)
    {
    }
};

class NanPropagation : public GTestDescriptorSingleValueAttribute<miopenNanPropagation_t, char>
{
public:
    NanPropagation(miopenNanPropagation_t value)
        : GTestDescriptorSingleValueAttribute<miopenNanPropagation_t, char>(
              true,
              "MIOPEN_ATTR_POINTWISE_NAN_PROPAGATION",
              MIOPEN_ATTR_POINTWISE_NAN_PROPAGATION,
              MIOPEN_TYPE_NAN_PROPOGATION,
              MIOPEN_TYPE_CHAR,
              2,
              value)
    {
    }
};

class DoubleAttribute : public GTestDescriptorSingleValueAttribute<double, char>
{
public:
    DoubleAttribute(const char* textName, miopenBackendAttributeName_t name)
        : GTestDescriptorSingleValueAttribute<double, char>(
              true, textName, name, MIOPEN_TYPE_DOUBLE, MIOPEN_TYPE_CHAR, 2, 0.7)
    {
    }
};

class FloatAttribute : public GTestDescriptorSingleValueAttribute<float, char>
{
public:
    FloatAttribute(const char* textName, miopenBackendAttributeName_t name)
        : GTestDescriptorSingleValueAttribute<float, char>(
              true, textName, name, MIOPEN_TYPE_DOUBLE, MIOPEN_TYPE_CHAR, 2, 0.7f)
    {
    }
};

class Axis : public GTestDescriptorSingleValueAttribute<int64_t, char>
{
public:
    Axis(int64_t value)
        : GTestDescriptorSingleValueAttribute<int64_t, char>(true,
                                                             "MIOPEN_ATTR_POINTWISE_AXIS",
                                                             MIOPEN_ATTR_POINTWISE_AXIS,
                                                             MIOPEN_TYPE_INT64,
                                                             MIOPEN_TYPE_CHAR,
                                                             2,
                                                             value)
    {
    }
};

class Descriptor : public GTestDescriptor
{
public:
    Descriptor();
    Descriptor(std::initializer_list<std::shared_ptr<GTestDescriptorAttribute>> attributes_)
        : GTestDescriptor{"MIOPEN_BACKEND_POINTWISE_DESCRIPTOR",
                          MIOPEN_BACKEND_POINTWISE_DESCRIPTOR,
                          true,
                          attributes_}
    {
    }
};

} // namespace

using TestCaseType = std::tuple<miopenPointwiseMode_t,
                                miopenDataType_t,
                                miopenNanPropagation_t,
                                bool,
                                bool,
                                bool,
                                bool,
                                bool,
                                int64_t>;

void PrintTo(const TestCaseType& v, std::ostream* os)
{
    *os << "mode: " << std::get<0>(v) << ", prec: " << std::get<1>(v)
        << ", nan: " << (std::get<2>(v) == MIOPEN_PROPAGATE_NAN ? "propagage" : "not propagate")
        << ", relu_lower_clip: " << (std::get<3>(v) ? "double" : "float")
        << ", relu_upper_clip: " << (std::get<4>(v) ? "double" : "float")
        << ", relu_lower_clip_slope: " << (std::get<5>(v) ? "double" : "float")
        << ", elu_alpha: " << (std::get<6>(v) ? "double" : "float")
        << ", softplus_beta: " << (std::get<7>(v) ? "double" : "float")
        << ", axis: " << std::get<8>(v);
}

class GraphApiPointwise : public testing::TestWithParam<TestCaseType>
{
protected:
    Descriptor descriptor;

    std::shared_ptr<GTestDescriptorAttribute>
    createDoubleOrFloat(bool isDouble, const char* textName, miopenBackendAttributeName_t name)
    {
        if(isDouble)
            return std::make_shared<DoubleAttribute>(textName, name);
        else
            return std::make_shared<FloatAttribute>(textName, name);
    }

    void SetUp() override
    {
        auto [mode,
              precision,
              nanPropagation,
              reluLowerClip,
              reluUpperClip,
              reluLowerClipSlope,
              eluAlpha,
              softPlusBeta,
              axis] = GetParam();
        descriptor  = {std::make_shared<Mode>(mode),
                      std::make_shared<Precision>(precision),
                      std::make_shared<NanPropagation>(nanPropagation),
                      createDoubleOrFloat(reluLowerClip,
                                          "MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP",
                                          MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP),
                      createDoubleOrFloat(reluUpperClip,
                                          "MIOPEN_ATTR_POINTWISE_RELU_UPPER_CLIP",
                                          MIOPEN_ATTR_POINTWISE_RELU_UPPER_CLIP),
                      createDoubleOrFloat(reluLowerClipSlope,
                                          "MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE",
                                          MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE),
                      createDoubleOrFloat(eluAlpha,
                                          "MIOPEN_ATTR_POINTWISE_ELU_ALPHA",
                                          MIOPEN_ATTR_POINTWISE_ELU_ALPHA),
                      createDoubleOrFloat(softPlusBeta,
                                          "MIOPEN_ATTR_POINTWISE_SOFTPLUS_BETA",
                                          MIOPEN_ATTR_POINTWISE_SOFTPLUS_BETA),
                      std::make_shared<Axis>(axis)};
    }
};

TEST_P(GraphApiPointwise, CFuncions)
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
        if(status == miopenStatusSuccess)
        EXPECT_TRUE(attrPtr->isSetAndGotEqual()) << textName << " set and retrieved values differ";
        // clang-format on
    }
}

INSTANTIATE_TEST_SUITE_P(
    CFuncionsTest,
    GraphApiPointwise,
    testing::Combine(testing::Values(MIOPEN_POINTWISE_ADD, MIOPEN_POINTWISE_MUL),
                     testing::Values(miopenFloat, miopenHalf),
                     testing::Values(MIOPEN_NOT_PROPAGATE_NAN, MIOPEN_PROPAGATE_NAN),
                     testing::Values(true, false),
                     testing::Values(true, false),
                     testing::Values(true, false),
                     testing::Values(true, false),
                     testing::Values(true, false),
                     testing::Values(-1, 0)));
