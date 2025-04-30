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

#include <tuple>
#include <variant>

#include <gtest/gtest.h>

#include "graphapi_gtest_common.hpp"

TEST(CPU_GraphApiPointwiseBuilder_NONE, Attributes)
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
using miopen::graphapi::GTestGraphApiExecute;

class Mode : public GTestDescriptorSingleValueAttribute<miopenPointwiseMode_t, char>
{
public:
    Mode() = default;
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
    Precision() = default;
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
    NanPropagation() = default;
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
    DoubleAttribute() = default;
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
    Axis() = default;
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

class DoubleOrFloatAttribute
{
private:
    std::variant<DoubleAttribute, FloatAttribute> mAttribute;

public:
    DoubleOrFloatAttribute() = default;

    void set(bool isDouble, const char* textName, miopenBackendAttributeName_t name)
    {
        if(isDouble)
        {
            mAttribute = DoubleAttribute(textName, name);
        }
        else
        {
            mAttribute = FloatAttribute(textName, name);
        }
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

} // namespace

using TestCaseType = std::tuple<miopenPointwiseMode_t,
                                miopenDataType_t,
                                miopenNanPropagation_t,
                                bool,
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
        << ", swish_beta: " << (std::get<8>(v) ? "double" : "float")
        << ", axis: " << std::get<9>(v);
}

class CPU_GraphApiPointwise_NONE : public testing::TestWithParam<TestCaseType>
{
private:
    Mode mMode;
    Precision mPrecision;
    NanPropagation mNanPropagation;
    DoubleOrFloatAttribute mReluLowerClip;
    DoubleOrFloatAttribute mReluUpperClip;
    DoubleOrFloatAttribute mReluLowerClipSlope;
    DoubleOrFloatAttribute mEluAlpha;
    DoubleOrFloatAttribute mSoftPlusBeta;
    DoubleOrFloatAttribute mSwishBeta;
    Axis mAxis;

protected:
    GTestGraphApiExecute<GTestDescriptorAttribute*> mExecute;

    void SetUp() override
    {
        auto [mode,
              precision,
              nanPropagation,
              isReluLowerClipDouble,
              isReluUpperClipDouble,
              isReluLowerClipSlopeDouble,
              isEluAlphaDouble,
              isSoftPlusBetaDouble,
              isSwishBetaDouble,
              axis] = GetParam();

        mMode           = {mode};
        mPrecision      = {precision};
        mNanPropagation = {nanPropagation};
        mAxis           = axis;

        mReluLowerClip.set(isReluLowerClipDouble,
                           "MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP",
                           MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP);

        mReluUpperClip.set(isReluUpperClipDouble,
                           "MIOPEN_ATTR_POINTWISE_RELU_UPPER_CLIP",
                           MIOPEN_ATTR_POINTWISE_RELU_UPPER_CLIP);

        mReluLowerClipSlope.set(isReluLowerClipSlopeDouble,
                                "MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE",
                                MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE);

        mEluAlpha.set(
            isEluAlphaDouble, "MIOPEN_ATTR_POINTWISE_ELU_ALPHA", MIOPEN_ATTR_POINTWISE_ELU_ALPHA);

        mSoftPlusBeta.set(isSoftPlusBetaDouble,
                          "MIOPEN_ATTR_POINTWISE_SOFTPLUS_BETA",
                          MIOPEN_ATTR_POINTWISE_SOFTPLUS_BETA);

        mSwishBeta.set(isSwishBetaDouble,
                       "MIOPEN_ATTR_POINTWISE_SWISH_BETA",
                       MIOPEN_ATTR_POINTWISE_SWISH_BETA);

        mExecute.descriptor.attributes = {&mMode,
                                          &mPrecision,
                                          &mNanPropagation,
                                          mReluLowerClip.get(),
                                          mReluUpperClip.get(),
                                          mReluLowerClipSlope.get(),
                                          mEluAlpha.get(),
                                          mSoftPlusBeta.get(),
                                          mSwishBeta.get(),
                                          &mAxis};

        mExecute.descriptor.attrsValid = true;
        mExecute.descriptor.textName   = "MIOPEN_BACKEND_POINTWISE_DESCRIPTOR";
        mExecute.descriptor.type       = MIOPEN_BACKEND_POINTWISE_DESCRIPTOR;
    }
};

TEST_P(CPU_GraphApiPointwise_NONE, CFunctions) { mExecute(); }

INSTANTIATE_TEST_SUITE_P(
    Unit,
    CPU_GraphApiPointwise_NONE,
    testing::Combine(testing::Values(MIOPEN_POINTWISE_ADD, MIOPEN_POINTWISE_MUL),
                     testing::Values(miopenFloat, miopenHalf),
                     testing::Values(MIOPEN_NOT_PROPAGATE_NAN, MIOPEN_PROPAGATE_NAN),
                     testing::Values(true, false),
                     testing::Values(true, false),
                     testing::Values(true, false),
                     testing::Values(true, false),
                     testing::Values(true, false),
                     testing::Values(true, false),
                     testing::Values(-1, 0)));
