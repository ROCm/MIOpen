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

#include <string>
#include <limits>

namespace miopen {

inline std::string GetDataType(miopenDataType_t type)
{
    std::string type_str;
    switch(type)
    {
    case miopenFloat: { type_str = "float";
    }
    break;
    case miopenHalf: { type_str = "half";
    }
    break;
    case miopenBFloat16: { type_str = "bfloat16";
    }
    break;
    case miopenInt8x4:
    case miopenInt8: { type_str = "int8_t";
    }
    break;
    case miopenInt32: { type_str = "int";
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
    case miopenIndexUint8: { return sizeof(uint8_t);
    }
    case miopenIndexUint16: { return sizeof(uint16_t);
    }
    case miopenIndexUint32: { return sizeof(uint32_t);
    }
    case miopenIndexUint64: { return sizeof(uint64_t);
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
    case miopenIndexUint8: { return std::numeric_limits<uint8_t>::max();
    }
    case miopenIndexUint16: { return std::numeric_limits<uint16_t>::max();
    }
    case miopenIndexUint32: { return std::numeric_limits<uint32_t>::max();
    }
    case miopenIndexUint64: { return std::numeric_limits<uint64_t>::max();
    }
    }

    MIOPEN_THROW("not belong to any case");
}

inline std::string GetDataTypeKernelParams(miopenDataType_t type)
{
    std::string s{};

    switch(type)
    {
    case miopenHalf:
    {
        s = " -DMIOPEN_USE_FP16=1 -DMIOPEN_USE_FP32=0 -DMIOPEN_USE_BFP16=0 -DMIOPEN_USE_INT8=0 "
            "-DMIOPEN_USE_INT8x4=0";
        break;
    }
    case miopenFloat:
    {
        s = " -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_BFP16=0  -DMIOPEN_USE_INT8=0 "
            "-DMIOPEN_USE_INT8x4=0";
        break;
    }
    case miopenInt8:
    {
        s = " -DMIOPEN_USE_INT8=1 -DMIOPEN_USE_INT8x4=0 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=0 "
            "-DMIOPEN_USE_BFP16=0";
        break;
    }
    case miopenInt8x4:
    {
        s = " -DMIOPEN_USE_INT8x4=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=0 -DMIOPEN_USE_BFP16=0 "
            "-DMIOPEN_USE_INT8=0";
        break;
    }
    case miopenBFloat16:
    {
        s = " -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=0 -DMIOPEN_USE_BFP16=1 -DMIOPEN_USE_INT8=0 "
            "-DMIOPEN_USE_INT8x4=0";
#if MIOPEN_USE_RNE_BFLOAT16 == 1
        s += " -DMIOPEN_USE_RNE_BFLOAT16=1";
#endif
        break;
    }
    // In special case of int8 and int8x4, the output datatype is specified to be in32
    // by frameworks. However, inside the kernel the outputs are marked fp32. That setting
    // is done outside of this function.
    case miopenInt32: break;
    default:
        MIOPEN_THROW("Only float, half, bfloat16, int8, int8x4 data type is supported.");
        break;
    }

    return s;
}

} // namespace miopen

#endif // GUARD_MIOPEN_DATATYPE_HPP
