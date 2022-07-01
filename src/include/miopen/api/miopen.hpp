/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_HPP_
#define GUARD_MIOPEN_HPP_

#include <miopen/miopen.h>
#include <iosfwd>

namespace miopen {

enum class DataType
{
    Half = miopenHalf,
    Float = miopenFloat,
    Int32 = miopenInt32,
    Int8 = miopenInt8,
    Int8x4 = miopenInt8x4,
    BFloat16 = miopenBFloat16,
    Double = miopenDouble
};

inline DataType ToInternal(miopenDataType_t type) {
    return DataType(type);
}

inline miopenDataType_t ToApi(DataType wrapper_type) {
    return miopenDataType_t(wrapper_type);
}

inline std::ostream& operator<<(std::ostream& os, DataType wrapper_type) {
    return os << ToApi(wrapper_type);
}

inline std::string to_string(DataType type) {
    return std::to_string(ToApi(type));
}

using api_miopenDataType_t = miopenDataType_t;

} // miopen namespace


#define miopenDataType_t miopen::DataType

#define miopenHalf      miopen::DataType::Half
#define miopenFloat     miopen::DataType::Float
#define miopenInt32     miopen::DataType::Int32
#define miopenInt8      miopen::DataType::Int8
#define miopenInt8x4    miopen::DataType::Int8x4
#define miopenBFloat16  miopen::DataType::BFloat16
#define miopenDouble    miopen::DataType::Double

#endif // GUARD_MIOPEN_HPP_
