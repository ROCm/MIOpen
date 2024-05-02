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
#include <miopen/scalar.hpp>
#include <miopen/conv/problem_description.hpp>

namespace miopen {

Scalar::Scalar(ConstData_t ptr, miopenDataType_t type)
{
    if(type == miopenFloat)
    {
        mVal = *static_cast<const float*>(ptr);
    }
    else if(type == miopenDouble)
    {
        mVal = *static_cast<const double*>(ptr);
    }
    else
    {
        MIOPEN_THROW("ERROR: Only accepts float or double type for now.");
    }
}

float Scalar::GetAsFloat() const
{
    if(mType == miopenFloat)
    {
        return std::get<float>(mVal);
    }

    return static_cast<float>(std::get<double>(mVal));
}

double Scalar::GetAsDouble() const
{
    if(mType == miopenDouble)
    {
        return std::get<double>(mVal);
    }

    return static_cast<double>(std::get<float>(mVal));
}

} // namespace miopen
