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
#include <miopen/miopen.h>

#include <cmath>
#include <limits>
#include <cassert>

template <typename T>
inline bool isCloseToZero(T value)
{
    return std::abs(value) <= std::numeric_limits<T>::epsilon();
}

template <typename T>
inline bool isCloseToOne(float value)
{
    return std::abs(value - T(1)) <= std::numeric_limits<T>::epsilon();
}

namespace miopen {
template <int default_val = 1>
struct ScalarMul
{
    ScalarMul()
    {
        mVal  = static_cast<float>(default_val);
        mType = miopenFloat;
    }

    // only supports double or float
    ScalarMul(ConstData_t ptr, miopenDataType_t type) : mType(type)
    {

        if(ptr == nullptr)
        {
            mVal = static_cast<float>(default_val);
        }
        else
        {
            if(type == miopenDouble)
            {
                mVal = *static_cast<const double*>(ptr);
            }
            else
            {
                mVal = *static_cast<const float*>(ptr);
                type = miopenFloat;
            }
        }
    }

    template <typename T>
    T GetVal() const
    {
        if constexpr(std::is_same_v<float, T>)
        {
            return GetAsFloat();
        }
        else if constexpr(std::is_same_v<double, T>)
        {
            return GetAsDouble();
        }
        else
        {
            throw std::runtime_error("ERROR: only expected float or double\n");
        }
    }

    float GetAsFloat() const { return (float)mVal; }
    double GetAsDouble() const { return mVal; }

    miopenDataType_t GetType() const { return mType; }

private:
    double mVal;
    miopenDataType_t mType;
};

using Alpha = ScalarMul<>;
using Beta  = ScalarMul<0>;

enum class EncodeAlphaBeta
{
    /* IDENTITY      alpha = 1.0 and beta = 0.0 */
    /* SCALE         alpha = 4.2 and beta = 0.0 */
    /* BILINEAR      alpha = 3.2 and beta = 1.1 */
    /* ERROR_STATE   alpha = 0.0 and beta = 3.1 */

    IDENTITY    = 0, /* alpha = 1.0 and beta = 0.0.*/
    SCALE       = 1, /* alpha with some value and beta 0.0*/
    BILINEAR    = 2, /* both alpha and beta with some value*/
    ERROR_STATE = 3, /* alpha 0.0 and beta with some value, this should not occur.
                        But used to check for errors.*/
};

EncodeAlphaBeta GetEncodedAlphaBeta(const Alpha& alpha, const Beta& beta);
} // namespace miopen
