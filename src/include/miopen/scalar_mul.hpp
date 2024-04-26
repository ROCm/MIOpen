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
    Scalar(double default_val = 1.0)
    {
        mVal  = static_cast<double>(default_val);
        mType = miopenDouble;
    }

    // only supports double or float
    Scalar(ConstData_t ptr, miopenDataType_t type = miopenDouble, double default_val = 1.0)
        : mType(type)
    {

        if(ptr == nullptr)
        {
            mVal = static_cast<double>(default_val);
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
            MIOPEN_THROW("Only expected float or double.");
        }
    }

    float GetAsFloat() const { return (float)mVal; }
    double GetAsDouble() const { return mVal; }

    miopenDataType_t GetType() const { return mType; }

private:
    double mVal;
    miopenDataType_t mType;
};

namespace conv {
enum class AlphaBetaCase
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

AlphaBetaCase GetAlphaBetaCase(const Scalar& alpha, const Scalar& beta);
} // namespace conv
} // namespace miopen
