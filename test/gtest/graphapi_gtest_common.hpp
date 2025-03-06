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
#pragma once

#include <miopen/graphapi/tensor.hpp>
#include <miopen/miopen.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>

namespace miopen {

namespace graphapi {

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

class GTestDescriptorAttribute
{
public:
    struct TestCase
    {
        bool isCorrect;

        const char* textName;
        miopenBackendAttributeName_t name;
        miopenBackendAttributeType_t type;
        int64_t count;
        void* data;

        miopenBackendAttributeType_t invalidType;
        void* invalidTypeData;

        int64_t invalidCount;
        void* invalidCountData;

        void* readBuffer;
    };

protected:
    TestCase mTestCase;

public:
    GTestDescriptorAttribute() = default;
    GTestDescriptorAttribute(const TestCase& testCase) : mTestCase(testCase) {}

    TestCase getTestCase() const { return mTestCase; }

    virtual testing::AssertionResult isSetAndGotEqual() = 0;

    virtual ~GTestDescriptorAttribute() = default;
};

template <typename ValueType, typename InvalidValueType>
class GTestDescriptorVectorAttribute : public GTestDescriptorAttribute
{
protected:
    // For several values we use a vector.
    // For a single value attribute here would have been
    // a single member of type ValueType.
    std::vector<ValueType> mValues;
    // But the rest of the fields should be nevertheless
    // vectors, so we'll treat a single value attribute
    // as special case of vector attribute
    std::vector<InvalidValueType> mInvalidTypeValues;
    std::vector<ValueType> mInvalidCountValues;
    std::vector<ValueType> mReadValues;

public:
    GTestDescriptorVectorAttribute(bool isCorrect,
                                   const char* textName,
                                   miopenBackendAttributeName_t name,
                                   miopenBackendAttributeType_t type,
                                   miopenBackendAttributeType_t invalidType,
                                   int64_t invalidCount,
                                   std::initializer_list<ValueType> values)
        : GTestDescriptorVectorAttribute(isCorrect,
                                         textName,
                                         name,
                                         type,
                                         invalidType,
                                         invalidCount,
                                         std::vector<ValueType>(values))
    {
    }

    GTestDescriptorVectorAttribute() = default;
    GTestDescriptorVectorAttribute(bool isCorrect,
                                   const char* textName,
                                   miopenBackendAttributeName_t name,
                                   miopenBackendAttributeType_t type,
                                   miopenBackendAttributeType_t invalidType,
                                   int64_t invalidCount,
                                   const std::vector<ValueType>& values)
        : mValues(values),
          mInvalidTypeValues(std::max(static_cast<decltype(values.size())>(1), values.size())),
          mInvalidCountValues(std::max(static_cast<decltype(invalidCount)>(1), invalidCount),
                              values.empty() ? ValueType() : *values.begin()),
          mReadValues(std::max(static_cast<decltype(values.size())>(1), values.size()))
    {
        mTestCase.isCorrect = isCorrect;

        mTestCase.textName = textName;
        mTestCase.name     = name;
        mTestCase.type     = type;
        mTestCase.count    = mValues.size();
        mTestCase.data     = mValues.empty() ? mReadValues.data() : mValues.data();

        mTestCase.invalidType     = invalidType;
        mTestCase.invalidTypeData = mInvalidTypeValues.data();

        mTestCase.invalidCount     = invalidCount;
        mTestCase.invalidCountData = mInvalidCountValues.data();

        mTestCase.readBuffer = mReadValues.data();
    }

    virtual testing::AssertionResult isSetAndGotEqual() override
    {
        if(std::equal(mValues.begin(), mValues.end(), mReadValues.begin()))
        {
            return testing::AssertionSuccess();
        }
        else
        {
            return testing::AssertionFailure();
        }
    }
};

template <typename ValueType, typename InvalidValueType>
class GTestDescriptorSingleValueAttribute
    : public GTestDescriptorVectorAttribute<ValueType, InvalidValueType>
{
public:
    GTestDescriptorSingleValueAttribute() = default;
    GTestDescriptorSingleValueAttribute(bool isCorrect,
                                        const char* textName,
                                        miopenBackendAttributeName_t name,
                                        miopenBackendAttributeType_t type,
                                        miopenBackendAttributeType_t invalidType,
                                        int64_t invalidCount,
                                        ValueType value)
        : GTestDescriptorVectorAttribute<ValueType, InvalidValueType>(
              isCorrect, textName, name, type, invalidType, invalidCount, {value})
    {
    }

    virtual testing::AssertionResult isSetAndGotEqual() override
    {
        assert(this->mValues.size() == this->mReadValues.size());
        if(this->mValues[0] == this->mReadValues[0])
        {
            return testing::AssertionSuccess();
        }
        else
        {
            return testing::AssertionFailure()
                   << "is " << this->mReadValues[0] << " but should be " << this->mValues[0];
        }
    }
};

template <typename AttributePointer = std::shared_ptr<GTestDescriptorAttribute>>
struct GTestDescriptor
{
    const char* textName               = "";
    miopenBackendDescriptorType_t type = miopenBackendDescriptorType_t(0);
    bool attrsValid                    = false;
    std::vector<AttributePointer> attributes;
};

template <typename AttributePointer = std::shared_ptr<GTestDescriptorAttribute>>
struct GTestGraphApiExecute
{
    GTestDescriptor<AttributePointer> descriptor;

    void operator()()
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
            if(isCorrect) EXPECT_EQ(status, miopenStatusSuccess) << textName << " wasn't set";
            else EXPECT_NE(status, miopenStatusSuccess) << textName << " was set to invalid value";
            // clang-format on
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
            EXPECT_NE(status, miopenStatusSuccess)
                << textName << " was retrieved before finalize()";

            anyAttributeGot = anyAttributeGot || (status == miopenStatusSuccess);
        }

        // Stop further execution if needed
        if(anyAttributeGot)
        {
            miopenBackendDestroyDescriptor(descr);
            FAIL() << "Some attributes of " << descrTextName << " were retrieved before finalize()";
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
            FAIL() << "An attribute of " << descrTextName << " was set after finalize()";
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

            if(isCorrect)
            {
                status = miopenBackendGetAttribute(descr, name, type, count, &elementCount, readBuffer);

                EXPECT_EQ(status, miopenStatusSuccess) << textName << " wasn't retrieved";
                EXPECT_EQ(count, elementCount) << textName << " set and retrieved number of elements differ";

                if(status == miopenStatusSuccess && count == elementCount)
                {
                    EXPECT_TRUE(attrPtr->isSetAndGotEqual()) << textName << " set and retrieved values differ";
                }
            }
            // clang-format on
        }
    }
};

class GMockBackendTensorDescriptor : public BackendTensorDescriptor
{
public:
    GMockBackendTensorDescriptor& operator=(const Tensor& testCaseTensor)
    {
        auto dataType = testCaseTensor.GetType();
        setAttribute(MIOPEN_ATTR_TENSOR_DATA_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &dataType);

        auto& d = testCaseTensor.GetLengths();
        std::vector<int64_t> dims{d.cbegin(), d.cend()};
        setAttribute(MIOPEN_ATTR_TENSOR_DIMENSIONS, MIOPEN_TYPE_INT64, dims.size(), dims.data());

        auto& s = testCaseTensor.GetStrides();
        std::vector<int64_t> strides{s.cbegin(), s.cend()};
        setAttribute(MIOPEN_ATTR_TENSOR_STRIDES, MIOPEN_TYPE_INT64, strides.size(), strides.data());

        auto id = testCaseTensor.getId();
        setAttribute(MIOPEN_ATTR_TENSOR_UNIQUE_ID, MIOPEN_TYPE_INT64, 1, &id);

        auto isVirtual = testCaseTensor.isVirtual();
        setAttribute(MIOPEN_ATTR_TENSOR_IS_VIRTUAL, MIOPEN_TYPE_BOOLEAN, 1, &isVirtual);

        finalize();

        return *this;
    }

    GMockBackendTensorDescriptor& operator=(const ValidatedValue<Tensor*>& validatedTestCaseTensor)
    {
        if(validatedTestCaseTensor.valid)
        {
            return *this = *validatedTestCaseTensor.value;
        }
        else
        {
            return *this;
        }
    }
};

} // namespace graphapi

} // namespace miopen
