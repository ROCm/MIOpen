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
#ifndef GUARD_MIOPEN_DATATYPE_HPP
#define GUARD_MIOPEN_DATATYPE_HPP

#include <sstream>
#include <string>
#include <limits>

namespace miopen {

inline std::string GetDataType(miopenDataType_t type)
{
    std::string type_str;
    switch(type)
    {
    case miopenFloat: {
        type_str = "float";
    }
    break;
    case miopenHalf: {
        type_str = "half";
    }
    break;
    case miopenBFloat16: {
        type_str = "bfloat16";
    }
    break;
    case miopenInt8x4:
    case miopenInt8: {
        type_str = "int8_t";
    }
    break;
    case miopenInt32: {
        type_str = "int";
    }
    break;
    case miopenDouble: {
        type_str = "double";
    }
    break;
    }
    return type_str;
}

inline std::size_t get_data_size(miopenDataType_t) { MIOPEN_THROW("not implemented"); }

inline std::size_t get_data_size(miopenIndexType_t index_type)
{
    switch(index_type)
    {
    case miopenIndexUint8: {
        return sizeof(uint8_t);
    }
    case miopenIndexUint16: {
        return sizeof(uint16_t);
    }
    case miopenIndexUint32: {
        return sizeof(uint32_t);
    }
    case miopenIndexUint64: {
        return sizeof(uint64_t);
    }
    }

    MIOPEN_THROW("not belong to any case");
}

inline std::size_t get_index_max(miopenIndexType_t index_type)
{
    // Basically, constants defined in cl.h, like CL_UCHAR_MAX, shall be used here.
    //    However, these are not available for HIP backend.
    switch(index_type)
    {
    case miopenIndexUint8: {
        return std::numeric_limits<uint8_t>::max();
    }
    case miopenIndexUint16: {
        return std::numeric_limits<uint16_t>::max();
    }
    case miopenIndexUint32: {
        return std::numeric_limits<uint32_t>::max();
    }
    case miopenIndexUint64: {
        return std::numeric_limits<uint64_t>::max();
    }
    }

    MIOPEN_THROW("not belong to any case");
}

inline std::string GetDataTypeKernelParams(miopenDataType_t type)
{
    // values for MIOPEN_USE_ macros
    int use_fp16               = 0;
    int use_fp32               = 0;
    int use_int8               = 0;
    int use_int8x4             = 0;
    int use_int32              = 0;
    int use_bfp16              = 0;
    int use_fp64               = 0;
    const int use_rne_bfloat16 = MIOPEN_USE_RNE_BFLOAT16;

    switch(type)
    {
    case miopenHalf: use_fp16 = 1; break;
    case miopenFloat: use_fp32 = 1; break;
    case miopenInt8: use_int8 = 1; break;
    case miopenInt8x4: use_int8x4 = 1; break;
    case miopenBFloat16: use_bfp16 = 1; break;
    case miopenInt32: use_int32 = 1; break;
    case miopenDouble: use_fp64 = 1; break;
    default:
        MIOPEN_THROW("Only float, half, bfloat16, int8, int8x4 data type is supported.");
        break;
    }

    std::ostringstream ss;
    ss << " -DMIOPEN_USE_FP16=" << use_fp16;
    ss << " -DMIOPEN_USE_FP32=" << use_fp32;
    ss << " -DMIOPEN_USE_INT8=" << use_int8;
    ss << " -DMIOPEN_USE_INT8x4=" << use_int8x4;
    ss << " -DMIOPEN_USE_BFP16=" << use_bfp16;
    ss << " -DMIOPEN_USE_INT32=" << use_int32;
    ss << " -DMIOPEN_USE_RNE_BFLOAT16=" << use_rne_bfloat16;
    if(use_fp64 != 0)
        ss << " -DMIOPEN_USE_FP64=" << use_fp64;
    return ss.str();
}

} // namespace miopen

#endif // GUARD_MIOPEN_DATATYPE_HPP
