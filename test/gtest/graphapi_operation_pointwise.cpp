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

namespace {

using miopen::graphapi::OperationPointwise;
using miopen::graphapi::OperationPointwiseBuilder;
using miopen::graphapi::Pointwise;
using miopen::graphapi::Tensor;

using OneInputTuple   = std::tuple<bool, Pointwise*, Tensor*, Tensor*>;
using TwoInputTuple   = std::tuple<bool, Pointwise*, Tensor*, Tensor*, Tensor*>;
using ThreeInputTuple = std::tuple<bool, Pointwise*, Tensor*, Tensor*, Tensor*, Tensor*>;

} // namespace

TEST(GraphApiOperationPointwiseBuilderSingleSetter, AnyAttribute)
{
    EXPECT_ANY_THROW({ OperationPointwiseBuilder().setPointwise(nullptr); })
        << "OperationPointwiseBuilder::setPointwise failed on an invalid attribute";
    EXPECT_ANY_THROW({ OperationPointwiseBuilder().setX(nullptr); })
        << "OperationPointwiseBuilder::setX failed on an invalid attribute";
    EXPECT_ANY_THROW({ OperationPointwiseBuilder().setB(nullptr); })
        << "OperationPointwiseBuilder::setB failed on an invalid attribute";
    EXPECT_ANY_THROW({ OperationPointwiseBuilder().setY(nullptr); })
        << "OperationPointwiseBuilder::setY failed on an invalid attribute";
    EXPECT_ANY_THROW({ OperationPointwiseBuilder().setT(nullptr); })
        << "OperationPointwiseBuilder::setT failed on an invalid attribute";
    EXPECT_ANY_THROW({ OperationPointwiseBuilder().setDx(nullptr); })
        << "OperationPointwiseBuilder::setDx failed on an invalid attribute";
    EXPECT_ANY_THROW({ OperationPointwiseBuilder().setDy(nullptr); })
        << "OperationPointwiseBuilder::setDy failed on an invalid attribute";

    Pointwise pointwise{MIOPEN_POINTWISE_ADD, miopenFloat};
    EXPECT_NO_THROW({ OperationPointwiseBuilder().setPointwise(&pointwise); })
        << "OperationPointwiseBuilder::setPointwise failed on a valid attribute";

    Tensor tensor{miopenFloat, {8, 64, 64}, {64 * 64, 64, 1}, 1, false};
    EXPECT_NO_THROW({ OperationPointwiseBuilder().setX(&tensor); })
        << "OperationPointwiseBuilder::setX failed on a valid attribute";
    EXPECT_NO_THROW({ OperationPointwiseBuilder().setB(&tensor); })
        << "OperationPointwiseBuilder::setB failed on a ivalid attribute";
    EXPECT_NO_THROW({ OperationPointwiseBuilder().setY(&tensor); })
        << "OperationPointwiseBuilder::setY failed on a ivalid attribute";
    EXPECT_NO_THROW({ OperationPointwiseBuilder().setT(&tensor); })
        << "OperationPointwiseBuilder::setT failed on a valid attribute";
    EXPECT_NO_THROW({ OperationPointwiseBuilder().setDx(&tensor); })
        << "OperationPointwiseBuilder::setDx failed on a valid attribute";
    EXPECT_NO_THROW({ OperationPointwiseBuilder().setDy(&tensor); })
        << "OperationPointwiseBuilder::setDy failed on a valid attribute";
}

class GraphApiOperationPointwiseBuilderOneInput : public testing::TestWithParam<OneInputTuple>
{
protected:
    bool mValid;
    Pointwise* mPointwise;
    Tensor* mX;
    Tensor* mY;

    void SetUp() override { std::tie(mValid, mPointwise, mX, mY) = GetParam(); }
};

TEST_P(GraphApiOperationPointwiseBuilderOneInput, Test)
{
    if(mValid)
    {
        EXPECT_NO_THROW({
            OperationPointwiseBuilder().setPointwise(mPointwise).setX(mX).setY(mY).build();
        }) << "Builder failed on valid attributes";
        EXPECT_NO_THROW({
            OperationPointwiseBuilder()
                .setPointwise(mPointwise)
                .setX(mX)
                .setY(mY)
                .setAlpha1(0.4f)
                .build();
        }) << "Builder failed on valid attributes with alpha1";
    }
    else
    {
        EXPECT_ANY_THROW({
            OperationPointwiseBuilder().setPointwise(mPointwise).setX(mX).setY(mY).build();
        }) << "Builder failed on invalid attributes";
    }
    EXPECT_ANY_THROW({ OperationPointwiseBuilder().setX(mX).setY(mY).build(); })
        << "Builder failed to detect missing setPointwise call";
    EXPECT_ANY_THROW({ OperationPointwiseBuilder().setPointwise(mPointwise).setY(mY).build(); })
        << "Builder failed to detect missing setX call";
    EXPECT_ANY_THROW({ OperationPointwiseBuilder().setPointwise(mPointwise).setX(mX).build(); })
        << "Builder failed to detect missing setY call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder().setPointwise(mPointwise).setX(mX).setY(mY).setB(mX).build();
    }) << "Builder failed to detect unwanted setB call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder().setPointwise(mPointwise).setX(mX).setY(mY).setT(mX).build();
    }) << "Builder failed to detect unwanted setT call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder().setPointwise(mPointwise).setX(mX).setY(mY).setDx(mX).build();
    }) << "Builder failed to detect unwanted setDx call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder().setPointwise(mPointwise).setX(mX).setY(mY).setDy(mX).build();
    }) << "Builder failed to detect unwanted setDy call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder()
            .setPointwise(mPointwise)
            .setX(mX)
            .setY(mY)
            .setAlpha2(0.5f)
            .build();
    }) << "Builder failed to detect unwanted setAlpha2 call";
}

class GraphApiOperationPointwiseBuilderTwoInput : public testing::TestWithParam<TwoInputTuple>
{
protected:
    bool mValid;
    Pointwise* mPointwise;
    Tensor* mX;
    Tensor* mB;
    Tensor* mY;

    void SetUp() override { std::tie(mValid, mPointwise, mX, mB, mY) = GetParam(); }
};

TEST_P(GraphApiOperationPointwiseBuilderTwoInput, Test)
{
    if(mValid)
    {
        EXPECT_NO_THROW({
            OperationPointwiseBuilder().setPointwise(mPointwise).setX(mX).setB(mB).setY(mY).build();
        }) << "Builder failed on valid attributes";
        EXPECT_NO_THROW({
            OperationPointwiseBuilder()
                .setPointwise(mPointwise)
                .setX(mX)
                .setB(mB)
                .setY(mY)
                .setAlpha1(0.4f)
                .build();
        }) << "Builder failed on valid attributes with alpha1";
        EXPECT_NO_THROW({
            OperationPointwiseBuilder()
                .setPointwise(mPointwise)
                .setX(mX)
                .setB(mB)
                .setY(mY)
                .setAlpha2(0.7f)
                .build();
        }) << "Builder failed on valid attributes with alpha2";
        EXPECT_NO_THROW({
            OperationPointwiseBuilder()
                .setPointwise(mPointwise)
                .setX(mX)
                .setB(mB)
                .setY(mY)
                .setAlpha1(0.3f)
                .setAlpha2(0.8f)
                .build();
        }) << "Builder failed on valid attributes with alpha1 and alpha2";
    }
    else
    {
        EXPECT_ANY_THROW({
            OperationPointwiseBuilder().setPointwise(mPointwise).setX(mX).setB(mB).setY(mY).build();
        }) << "Builder failed on invalid attributes";
    }
    EXPECT_ANY_THROW({ OperationPointwiseBuilder().setX(mX).setB(mB).setY(mY).build(); })
        << "Builder failed to detect missing setPointwise call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder().setPointwise(mPointwise).setB(mB).setY(mY).build();
    }) << "Builder failed to detect missing setX call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder().setPointwise(mPointwise).setX(mX).setY(mY).build();
    }) << "Builder failed to detect missing setB call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder().setPointwise(mPointwise).setX(mX).setB(mB).build();
    }) << "Builder failed to detect missing setY call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder()
            .setPointwise(mPointwise)
            .setX(mX)
            .setB(mB)
            .setY(mY)
            .setT(mX)
            .build();
    }) << "Builder failed to detect unwanted setT call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder()
            .setPointwise(mPointwise)
            .setX(mX)
            .setB(mB)
            .setY(mY)
            .setDx(mX)
            .build();
    }) << "Builder failed to detect unwanted setDx call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder()
            .setPointwise(mPointwise)
            .setX(mX)
            .setB(mB)
            .setY(mY)
            .setDy(mX)
            .build();
    }) << "Builder failed to detect unwanted setDy call";
}

class GraphApiOperationPointwiseBuilderBwd : public testing::TestWithParam<TwoInputTuple>
{
protected:
    bool mValid;
    Pointwise* mPointwise;
    Tensor* mY;
    Tensor* mDy;
    Tensor* mDx;

    void SetUp() override { std::tie(mValid, mPointwise, mY, mDy, mDx) = GetParam(); }
};

TEST_P(GraphApiOperationPointwiseBuilderBwd, Test)
{
    if(mValid)
    {
        EXPECT_NO_THROW({
            OperationPointwiseBuilder()
                .setPointwise(mPointwise)
                .setY(mY)
                .setDy(mDy)
                .setDx(mDx)
                .build();
        }) << "Builder failed on valid attributes";
        EXPECT_NO_THROW({
            OperationPointwiseBuilder()
                .setPointwise(mPointwise)
                .setY(mY)
                .setDy(mDy)
                .setDx(mDx)
                .setAlpha1(0.4f)
                .build();
        }) << "Builder failed on valid attributes with alpha1";
        EXPECT_NO_THROW({
            OperationPointwiseBuilder()
                .setPointwise(mPointwise)
                .setY(mY)
                .setDy(mDy)
                .setDx(mDx)
                .setAlpha2(0.7f)
                .build();
        }) << "Builder failed on valid attributes with alpha2";
        EXPECT_NO_THROW({
            OperationPointwiseBuilder()
                .setPointwise(mPointwise)
                .setY(mY)
                .setDy(mDy)
                .setDx(mDx)
                .setAlpha1(0.3f)
                .setAlpha2(0.8f)
                .build();
        }) << "Builder failed on valid attributes with alpha1 and alpha2";
    }
    else
    {
        EXPECT_ANY_THROW({
            OperationPointwiseBuilder()
                .setPointwise(mPointwise)
                .setY(mY)
                .setDy(mDy)
                .setDx(mDx)
                .build();
        }) << "Builder failed on invalid attributes";
    }
    EXPECT_ANY_THROW({ OperationPointwiseBuilder().setY(mY).setDy(mDy).setDx(mDx).build(); })
        << "Builder failed to detect missing setPointwise call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder().setPointwise(mPointwise).setDy(mDy).setDx(mDx).build();
    }) << "Builder failed to detect missing setY call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder().setPointwise(mPointwise).setY(mY).setDx(mDx).build();
    }) << "Builder failed to detect missing setDy call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder().setPointwise(mPointwise).setY(mY).setDy(mDy).build();
    }) << "Builder failed to detect missing setDx call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder()
            .setPointwise(mPointwise)
            .setY(mY)
            .setDy(mDy)
            .setDx(mDx)
            .setX(mY)
            .build();
    }) << "Builder failed to detect unwanted setX call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder()
            .setPointwise(mPointwise)
            .setY(mY)
            .setDy(mDy)
            .setDx(mDx)
            .setB(mY)
            .build();
    }) << "Builder failed to detect unwanted setB call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder()
            .setPointwise(mPointwise)
            .setY(mY)
            .setDy(mDy)
            .setDx(mDx)
            .setT(mY)
            .build();
    }) << "Builder failed to detect unwanted setT call";
}

class GraphApiOperationPointwiseBuilderThreeInput : public testing::TestWithParam<ThreeInputTuple>
{
protected:
    bool mValid;
    Pointwise* mPointwise;
    Tensor* mX;
    Tensor* mB;
    Tensor* mY;
    Tensor* mT;

    void SetUp() override { std::tie(mValid, mPointwise, mX, mB, mY, mT) = GetParam(); }
};

TEST_P(GraphApiOperationPointwiseBuilderThreeInput, Test)
{
    if(mValid)
    {
        EXPECT_NO_THROW({
            OperationPointwiseBuilder()
                .setPointwise(mPointwise)
                .setX(mX)
                .setB(mB)
                .setY(mY)
                .setT(mT)
                .build();
        }) << "Builder failed on valid attributes";
        EXPECT_NO_THROW({
            OperationPointwiseBuilder()
                .setPointwise(mPointwise)
                .setX(mX)
                .setB(mB)
                .setY(mY)
                .setT(mT)
                .setAlpha1(0.4f)
                .build();
        }) << "Builder failed on valid attributes with alpha1";
        EXPECT_NO_THROW({
            OperationPointwiseBuilder()
                .setPointwise(mPointwise)
                .setX(mX)
                .setB(mB)
                .setY(mY)
                .setT(mT)
                .setAlpha2(0.6f)
                .build();
        }) << "Builder failed on valid attributes with alpha2";
        EXPECT_NO_THROW({
            OperationPointwiseBuilder()
                .setPointwise(mPointwise)
                .setX(mX)
                .setB(mB)
                .setY(mY)
                .setT(mT)
                .setAlpha1(0.3f)
                .setAlpha1(0.7f)
                .build();
        }) << "Builder failed on valid attributes with alpha1 and alpha2";
    }
    else
    {
        EXPECT_ANY_THROW({
            OperationPointwiseBuilder()
                .setPointwise(mPointwise)
                .setX(mX)
                .setB(mB)
                .setY(mY)
                .setT(mT)
                .build();
        }) << "Builder failed on invalid attributes";
    }
    EXPECT_ANY_THROW({ OperationPointwiseBuilder().setX(mX).setB(mB).setY(mY).setT(mT).build(); })
        << "Builder failed to detect missing setPointwise call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder().setPointwise(mPointwise).setB(mB).setY(mY).setT(mT).build();
    }) << "Builder failed to detect missing setX call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder().setPointwise(mPointwise).setX(mX).setY(mY).setT(mT).build();
    }) << "Builder failed to detect missing setB call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder().setPointwise(mPointwise).setX(mX).setB(mB).setT(mT).build();
    }) << "Builder failed to detect missing setY call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder().setPointwise(mPointwise).setX(mX).setB(mB).setY(mY).build();
    }) << "Builder failed to detect missing setT call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder()
            .setPointwise(mPointwise)
            .setX(mX)
            .setB(mB)
            .setY(mY)
            .setT(mT)
            .setDx(mX)
            .build();
    }) << "Builder failed to detect unwanted setDx call";
    EXPECT_ANY_THROW({
        OperationPointwiseBuilder()
            .setPointwise(mPointwise)
            .setX(mX)
            .setB(mB)
            .setY(mY)
            .setT(mT)
            .setDy(mX)
            .build();
    }) << "Builder failed to detect unwanted setDy call";
}

static Tensor x{miopenFloat, {8, 64, 64}, {64 * 64, 64, 1}, 1, false};
static Tensor x2{miopenFloat, {8, 1, 64}, {64 * 1, 64, 1}, 2, false};
static Tensor x3{miopenFloat, {8, 64, 1}, {64 * 1, 1, 1}, 3, false};
static Tensor b{miopenFloat, {8, 64, 64}, {64 * 64, 64, 1}, 4, false};
static Tensor b2{miopenFloat, {8, 1, 64}, {64 * 1, 64, 1}, 5, false};
static Tensor b3{miopenFloat, {8, 64, 1}, {64 * 1, 1, 1}, 6, false};
static Tensor y{miopenFloat, {8, 64, 64}, {64 * 64, 64, 1}, 7, false};
static Tensor y2{miopenFloat, {8, 1, 64}, {64 * 1, 64, 1}, 8, false};
static Tensor t{miopenFloat, {8, 64, 64}, {64 * 64, 64, 1}, 9, false};
static Pointwise pointwiseAdd{MIOPEN_POINTWISE_ADD, miopenFloat};
static Pointwise pointwiseAbs{MIOPEN_POINTWISE_ABS, miopenFloat};
static Pointwise pointwiseReluBwd{MIOPEN_POINTWISE_RELU_BWD, miopenFloat};
static Pointwise pointwiseBinSel{MIOPEN_POINTWISE_BINARY_SELECT, miopenFloat};

static auto oneInputValid   = testing::Combine(testing::Values(true),
                                             testing::Values(&pointwiseAbs),
                                             testing::Values(&x),
                                             testing::Values(&y));
static auto oneInputInvalid = testing::Combine(testing::Values(false),
                                               testing::Values(&pointwiseAbs),
                                               testing::Values(&x2, &x3),
                                               testing::Values(&y));

static auto twoInputValid1  = testing::Combine(testing::Values(true),
                                              testing::Values(&pointwiseAdd),
                                              testing::Values(&x, &x2, &x3),
                                              testing::Values(&b),
                                              testing::Values(&y));
static auto twoInputValid2  = testing::Combine(testing::Values(true),
                                              testing::Values(&pointwiseAdd),
                                              testing::Values(&x),
                                              testing::Values(&b, &b2, &b3),
                                              testing::Values(&y));
static auto twoInputInvalid = testing::Combine(testing::Values(false),
                                               testing::Values(&pointwiseAdd),
                                               testing::Values(&x, &x2, &x3),
                                               testing::Values(&b),
                                               testing::Values(&y2));

static auto twoInputValidBwd1  = testing::Combine(testing::Values(true),
                                                 testing::Values(&pointwiseReluBwd),
                                                 testing::Values(&x, &x2, &x3),
                                                 testing::Values(&b),
                                                 testing::Values(&y));
static auto twoInputValidBwd2  = testing::Combine(testing::Values(true),
                                                 testing::Values(&pointwiseReluBwd),
                                                 testing::Values(&x),
                                                 testing::Values(&b, &b2, &b3),
                                                 testing::Values(&y));
static auto twoInputInvalidBwd = testing::Combine(testing::Values(false),
                                                  testing::Values(&pointwiseReluBwd),
                                                  testing::Values(&x, &x2, &x3),
                                                  testing::Values(&b),
                                                  testing::Values(&y2));

static auto threeInputValid   = testing::Combine(testing::Values(true),
                                               testing::Values(&pointwiseBinSel),
                                               testing::Values(&x),
                                               testing::Values(&b),
                                               testing::Values(&y),
                                               testing::Values(&t));
static auto threeInputInvalid = testing::Combine(testing::Values(false),
                                                 testing::Values(&pointwiseBinSel),
                                                 testing::Values(&x2, &x3),
                                                 testing::Values(&b, &b2, &b3),
                                                 testing::Values(&y, &y2),
                                                 testing::Values(&t));

INSTANTIATE_TEST_SUITE_P(OneInputValid, GraphApiOperationPointwiseBuilderOneInput, oneInputValid);
INSTANTIATE_TEST_SUITE_P(OneInputInvalid,
                         GraphApiOperationPointwiseBuilderOneInput,
                         oneInputInvalid);

INSTANTIATE_TEST_SUITE_P(TwoInputValid1, GraphApiOperationPointwiseBuilderTwoInput, twoInputValid1);
INSTANTIATE_TEST_SUITE_P(TwoInputValid2, GraphApiOperationPointwiseBuilderTwoInput, twoInputValid2);
INSTANTIATE_TEST_SUITE_P(TwoInputInvalid,
                         GraphApiOperationPointwiseBuilderTwoInput,
                         twoInputInvalid);

INSTANTIATE_TEST_SUITE_P(TwoInputValidBwd1,
                         GraphApiOperationPointwiseBuilderBwd,
                         twoInputValidBwd1);
INSTANTIATE_TEST_SUITE_P(TwoInputValidBwd2,
                         GraphApiOperationPointwiseBuilderBwd,
                         twoInputValidBwd2);
INSTANTIATE_TEST_SUITE_P(TwoInputInvalidBwd,
                         GraphApiOperationPointwiseBuilderBwd,
                         twoInputInvalidBwd);

INSTANTIATE_TEST_SUITE_P(ThreeInputValid,
                         GraphApiOperationPointwiseBuilderThreeInput,
                         threeInputValid);
INSTANTIATE_TEST_SUITE_P(ThreeInputInvalid,
                         GraphApiOperationPointwiseBuilderThreeInput,
                         threeInputInvalid);
