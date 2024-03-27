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

#include <miopen/miopen.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>

namespace miopen {

namespace graphapi {

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
        : mValues(values),
          mInvalidTypeValues(std::max(1ul, values.size())),
          mInvalidCountValues(std::max(1l, invalidCount),
                              values.size() > 0 ? *values.begin() : ValueType{}),
          mReadValues(values.size())
    {
        mTestCase.isCorrect = isCorrect;

        mTestCase.textName = textName;
        mTestCase.name     = name;
        mTestCase.type     = type;
        mTestCase.count    = mValues.size();
        mTestCase.data     = mValues.data();

        mTestCase.invalidType     = invalidType;
        mTestCase.invalidTypeData = mInvalidTypeValues.data();

        mTestCase.invalidCount     = invalidCount;
        mTestCase.invalidCountData = mInvalidCountValues.data();

        mTestCase.readBuffer = mReadValues.data();
    }

    virtual testing::AssertionResult isSetAndGotEqual() override
    {
        if(std::equal(mValues.begin(), mValues.end(), mReadValues.begin(), mReadValues.end()))
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

struct GTestDescriptor
{
    const char* textName               = "";
    miopenBackendDescriptorType_t type = miopenBackendDescriptorType_t(0);
    bool attrsValid                    = false;
    std::vector<std::shared_ptr<GTestDescriptorAttribute>> attributes;
};

} // namespace graphapi

} // namespace miopen
