/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_API_WRAPPERS_HPP_
#define GUARD_MIOPEN_API_WRAPPERS_HPP_

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

inline DataType miopenLegacyToWrapper(miopenDataType_t type) {
    return DataType(type);
}

inline miopenDataType_t miopenWrapperToLegacy(DataType wrapper_type) {
    return miopenDataType_t(wrapper_type);
}

inline std::ostream& operator<<(std::ostream& os, DataType wrapper_type) {
    return os << miopenWrapperToLegacy(wrapper_type);
}

} // miopen namespace

#define miopenHalf ERROR_USE_DATATYPE_ENUM_CLASS_INSTEAD
#define miopenFloat ERROR_USE_DATATYPE_ENUM_CLASS_INSTEAD
#define miopenInt32 ERROR_USE_DATATYPE_ENUM_CLASS_INSTEAD
#define miopenInt8 ERROR_USE_DATATYPE_ENUM_CLASS_INSTEAD
#define miopenInt8x4 ERROR_USE_DATATYPE_ENUM_CLASS_INSTEAD
#define miopenBFloat16 ERROR_USE_DATATYPE_ENUM_CLASS_INSTEAD
#define miopenDouble ERROR_USE_DATATYPE_ENUM_CLASS_INSTEAD

#endif // GUARD_MIOPEN_API_WRAPPERS_HPP_
