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

TEST(CPU_GraphApiOperationPointwiseBuilderSingleSetter_NONE, AnyAttribute)
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

class CPU_GraphApiOperationPointwiseBuilderOneInput_NONE
    : public testing::TestWithParam<OneInputTuple>
{
protected:
    bool mValid;
    Pointwise* mPointwise;
    Tensor* mX;
    Tensor* mY;

    void SetUp() override { std::tie(mValid, mPointwise, mX, mY) = GetParam(); }
};

TEST_P(CPU_GraphApiOperationPointwiseBuilderOneInput_NONE, Test)
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

class CPU_GraphApiOperationPointwiseBuilderTwoInput_NONE
    : public testing::TestWithParam<TwoInputTuple>
{
protected:
    bool mValid;
    Pointwise* mPointwise;
    Tensor* mX;
    Tensor* mB;
    Tensor* mY;

    void SetUp() override { std::tie(mValid, mPointwise, mX, mB, mY) = GetParam(); }
};

TEST_P(CPU_GraphApiOperationPointwiseBuilderTwoInput_NONE, Test)
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

class CPU_GraphApiOperationPointwiseBuilderBwd_NONE : public testing::TestWithParam<TwoInputTuple>
{
protected:
    bool mValid;
    Pointwise* mPointwise;
    Tensor* mY;
    Tensor* mDy;
    Tensor* mDx;

    void SetUp() override { std::tie(mValid, mPointwise, mY, mDy, mDx) = GetParam(); }
};

TEST_P(CPU_GraphApiOperationPointwiseBuilderBwd_NONE, Test)
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

class CPU_GraphApiOperationPointwiseBuilderThreeInput_NONE
    : public testing::TestWithParam<ThreeInputTuple>
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

TEST_P(CPU_GraphApiOperationPointwiseBuilderThreeInput_NONE, Test)
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

namespace {

using miopen::graphapi::GMockBackendTensorDescriptor;
using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::GTestGraphApiExecute;
using miopen::graphapi::ValidatedValue;

class GMockBackendPointwiseDescriptor : public miopen::graphapi::BackendPointwiseDescriptor
{
public:
    GMockBackendPointwiseDescriptor&
    operator=(Pointwise* pointwise) // we don't bother with ValidatedValue here
    {
        if(pointwise == nullptr)
        {
            return *this;
        }

        auto mode = pointwise->getMode();
        setAttribute(MIOPEN_ATTR_POINTWISE_MODE, MIOPEN_TYPE_POINTWISE_MODE, 1, &mode);

        auto prec = pointwise->getMathPrecision();
        setAttribute(MIOPEN_ATTR_POINTWISE_MATH_PREC, MIOPEN_TYPE_DATA_TYPE, 1, &prec);

        finalize();

        return *this;
    }
};

class GraphApiOperationPointwiseBase
{
private:
    // Pointers to these are stored in the objects below
    GMockBackendPointwiseDescriptor mPointwiseDescriptor;
    std::vector<GMockBackendTensorDescriptor> mTensorDescriptors;

    // Pointers to these are stored in mExecute object below
    GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char> mPointwiseAttribute;
    std::vector<GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char>>
        mTensorAttributes;

protected:
    GTestGraphApiExecute<GTestDescriptorAttribute*> mExecute;

    void prepareExecute(
        bool valid,
        Pointwise* pointwise,
        std::initializer_list<std::tuple<Tensor*, const char*, miopenBackendAttributeName_t>>
            tensors)
    {
        try
        {
            mTensorDescriptors.reserve(tensors.size()); // to prevent ptr invalidation
            mTensorAttributes.reserve(tensors.size());  // to prevent ptr invalidation
            mExecute.descriptor.attributes.reserve(tensors.size() + 1);

            mExecute.descriptor.textName   = "MIOPEN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR";
            mExecute.descriptor.type       = MIOPEN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR;
            mExecute.descriptor.attrsValid = valid;

            mPointwiseDescriptor = pointwise;
            mPointwiseAttribute  = {pointwise != nullptr,
                                   "MIOPEN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR",
                                   MIOPEN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                                   MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                                   MIOPEN_TYPE_CHAR,
                                   2,
                                   &mPointwiseDescriptor};
            mExecute.descriptor.attributes.push_back(&mPointwiseAttribute);

            std::for_each(tensors.begin(), tensors.end(), [this](const auto& tpl) {
                auto ptr = std::get<Tensor*>(tpl);

                auto& descriptor = mTensorDescriptors.emplace_back();
                descriptor       = ValidatedValue<Tensor*>{ptr != nullptr, ptr};

                auto& attribute =
                    mTensorAttributes.emplace_back(ptr != nullptr,
                                                   std::get<const char*>(tpl),
                                                   std::get<miopenBackendAttributeName_t>(tpl),
                                                   MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                                                   MIOPEN_TYPE_CHAR,
                                                   2,
                                                   &descriptor);
                mExecute.descriptor.attributes.push_back(&attribute);
            });
        }
        catch(const std::exception& e)
        {
            FAIL() << e.what();
        }
    }
};

} // namespace

class CPU_GraphApiOperationPointwiseOneInput_NONE : public testing::TestWithParam<OneInputTuple>,
                                                    protected GraphApiOperationPointwiseBase
{
protected:
    void SetUp() override
    {
        auto [valid, pointwise, x, y] = GetParam();

        // clang-format off
        prepareExecute(
            valid,
            pointwise,
            {{x, "MIOPEN_ATTR_OPERATION_POINTWISE_XDESC", MIOPEN_ATTR_OPERATION_POINTWISE_XDESC},
             {y, "MIOPEN_ATTR_OPERATION_POINTWISE_YDESC", MIOPEN_ATTR_OPERATION_POINTWISE_YDESC}});
        // clang-format on
    }
};

class CPU_GraphApiOperationPointwiseTwoInput_NONE : public testing::TestWithParam<TwoInputTuple>,
                                                    protected GraphApiOperationPointwiseBase
{
protected:
    void SetUp() override
    {
        auto [valid, pointwise, x, b, y] = GetParam();

        // clang-format off
        prepareExecute(
            valid,
            pointwise,
            {{x, "MIOPEN_ATTR_OPERATION_POINTWISE_XDESC", MIOPEN_ATTR_OPERATION_POINTWISE_XDESC},
             {b, "MIOPEN_ATTR_OPERATION_POINTWISE_BDESC", MIOPEN_ATTR_OPERATION_POINTWISE_BDESC},
             {y, "MIOPEN_ATTR_OPERATION_POINTWISE_YDESC", MIOPEN_ATTR_OPERATION_POINTWISE_YDESC}});
        // clang-format on
    }
};

class CPU_GraphApiOperationPointwiseBwd_NONE : public testing::TestWithParam<TwoInputTuple>,
                                               protected GraphApiOperationPointwiseBase
{
protected:
    void SetUp() override
    {
        auto [valid, pointwise, y, dy, dx] = GetParam();

        // clang-format off
        prepareExecute(
            valid,
            pointwise,
            {{y, "MIOPEN_ATTR_OPERATION_POINTWISE_YDESC", MIOPEN_ATTR_OPERATION_POINTWISE_YDESC},
             {dy, "MIOPEN_ATTR_OPERATION_POINTWISE_DYDESC", MIOPEN_ATTR_OPERATION_POINTWISE_DYDESC},
             {dx, "MIOPEN_ATTR_OPERATION_POINTWISE_DXDESC", MIOPEN_ATTR_OPERATION_POINTWISE_DXDESC}});
        // clang-format on
    }
};

class CPU_GraphApiOperationPointwiseThreeInput_NONE
    : public testing::TestWithParam<ThreeInputTuple>,
      protected GraphApiOperationPointwiseBase
{
protected:
    void SetUp() override
    {
        auto [valid, pointwise, x, b, y, t] = GetParam();

        // clang-format off
        prepareExecute(
            valid,
            pointwise,
            {{x, "MIOPEN_ATTR_OPERATION_POINTWISE_XDESC", MIOPEN_ATTR_OPERATION_POINTWISE_XDESC},
             {b, "MIOPEN_ATTR_OPERATION_POINTWISE_BDESC", MIOPEN_ATTR_OPERATION_POINTWISE_BDESC},
             {y, "MIOPEN_ATTR_OPERATION_POINTWISE_YDESC", MIOPEN_ATTR_OPERATION_POINTWISE_YDESC},
             {t, "MIOPEN_ATTR_OPERATION_POINTWISE_TDESC", MIOPEN_ATTR_OPERATION_POINTWISE_TDESC}});
        // clang-format on
    }
};

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
                                               testing::Values(&pointwiseAbs, nullptr),
                                               testing::Values(&x2, &x3, nullptr),
                                               testing::Values(&y, nullptr));

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
                                               testing::Values(&pointwiseAdd, nullptr),
                                               testing::Values(&x, &x2, &x3, nullptr),
                                               testing::Values(&b, nullptr),
                                               testing::Values(&y2, nullptr));

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
                                                  testing::Values(&pointwiseReluBwd, nullptr),
                                                  testing::Values(&x, &x2, &x3, nullptr),
                                                  testing::Values(&b, nullptr),
                                                  testing::Values(&y2, nullptr));

static auto threeInputValid   = testing::Combine(testing::Values(true),
                                               testing::Values(&pointwiseBinSel),
                                               testing::Values(&x),
                                               testing::Values(&b),
                                               testing::Values(&y),
                                               testing::Values(&t));
static auto threeInputInvalid = testing::Combine(testing::Values(false),
                                                 testing::Values(&pointwiseBinSel, nullptr),
                                                 testing::Values(&x2, &x3, nullptr),
                                                 testing::Values(&b, &b2, &b3, nullptr),
                                                 testing::Values(&y, &y2, nullptr),
                                                 testing::Values(&t, nullptr));

INSTANTIATE_TEST_SUITE_P(UnitIV, CPU_GraphApiOperationPointwiseBuilderOneInput_NONE, oneInputValid);
INSTANTIATE_TEST_SUITE_P(UnitII,
                         CPU_GraphApiOperationPointwiseBuilderOneInput_NONE,
                         oneInputInvalid);

INSTANTIATE_TEST_SUITE_P(Unit2IV1,
                         CPU_GraphApiOperationPointwiseBuilderTwoInput_NONE,
                         twoInputValid1);
INSTANTIATE_TEST_SUITE_P(Unit2IV2,
                         CPU_GraphApiOperationPointwiseBuilderTwoInput_NONE,
                         twoInputValid2);
INSTANTIATE_TEST_SUITE_P(Unit2II,
                         CPU_GraphApiOperationPointwiseBuilderTwoInput_NONE,
                         twoInputInvalid);

INSTANTIATE_TEST_SUITE_P(Unit2IV, CPU_GraphApiOperationPointwiseBuilderBwd_NONE, twoInputValidBwd1);
INSTANTIATE_TEST_SUITE_P(Unit2IV2,
                         CPU_GraphApiOperationPointwiseBuilderBwd_NONE,
                         twoInputValidBwd2);
INSTANTIATE_TEST_SUITE_P(Unit2II,
                         CPU_GraphApiOperationPointwiseBuilderBwd_NONE,
                         twoInputInvalidBwd);

INSTANTIATE_TEST_SUITE_P(Unit3IV,
                         CPU_GraphApiOperationPointwiseBuilderThreeInput_NONE,
                         threeInputValid);
INSTANTIATE_TEST_SUITE_P(Unit3II,
                         CPU_GraphApiOperationPointwiseBuilderThreeInput_NONE,
                         threeInputInvalid);

TEST_P(CPU_GraphApiOperationPointwiseOneInput_NONE, CFunctions) { mExecute(); }
TEST_P(CPU_GraphApiOperationPointwiseTwoInput_NONE, CFunctions) { mExecute(); }
TEST_P(CPU_GraphApiOperationPointwiseBwd_NONE, CFunctions) { mExecute(); }
TEST_P(CPU_GraphApiOperationPointwiseThreeInput_NONE, CFunctions) { mExecute(); }

INSTANTIATE_TEST_SUITE_P(UnitIV, CPU_GraphApiOperationPointwiseOneInput_NONE, oneInputValid);
INSTANTIATE_TEST_SUITE_P(UnitII, CPU_GraphApiOperationPointwiseOneInput_NONE, oneInputInvalid);

INSTANTIATE_TEST_SUITE_P(Unit2IV1, CPU_GraphApiOperationPointwiseTwoInput_NONE, twoInputValid1);
INSTANTIATE_TEST_SUITE_P(Unit2IV2, CPU_GraphApiOperationPointwiseTwoInput_NONE, twoInputValid2);
INSTANTIATE_TEST_SUITE_P(Unit2II, CPU_GraphApiOperationPointwiseTwoInput_NONE, twoInputInvalid);

INSTANTIATE_TEST_SUITE_P(Unit2IV1, CPU_GraphApiOperationPointwiseBwd_NONE, twoInputValidBwd1);
INSTANTIATE_TEST_SUITE_P(Unit2IV2, CPU_GraphApiOperationPointwiseBwd_NONE, twoInputValidBwd2);
INSTANTIATE_TEST_SUITE_P(Unit2II, CPU_GraphApiOperationPointwiseBwd_NONE, twoInputInvalidBwd);

INSTANTIATE_TEST_SUITE_P(Unit3IV, CPU_GraphApiOperationPointwiseThreeInput_NONE, threeInputValid);
INSTANTIATE_TEST_SUITE_P(Unit3II, CPU_GraphApiOperationPointwiseThreeInput_NONE, threeInputInvalid);
