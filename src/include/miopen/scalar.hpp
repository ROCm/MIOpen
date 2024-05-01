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

#include <miopen/common.hpp>
#include <miopen/errors.hpp>
#include <miopen/miopen.h>

#include <cmath>
#include <limits>
#include <cassert>

namespace miopen {
struct Scalar
{
    explicit Scalar(double default_val = 1.0)
        : mVal(static_cast<double>(default_val)), mType(miopenDouble)
    {
    }

    // Any type of data in ptr are converted to double
    Scalar(ConstData_t ptr, miopenDataType_t type = miopenDouble, double default_val = 1.0)
    {
        double temp = 1.0;
        if(ptr != nullptr)
        {
            memcpy(&temp, ptr, sizeof(double));
        }
        else
        {
            temp = default_val;
        }
        mVal = temp;
    }

    int32_t GetAsInt32() const { return static_cast<int32_t>(mVal); }
    float GetAsFloat() const { return static_cast<float>(mVal); }
    double GetAsDouble() const { return static_cast<double>(mVal); }

    miopenDataType_t GetType() const { return mType; }

private:
    double mVal;
    miopenDataType_t mType;
};

} // namespace miopen
