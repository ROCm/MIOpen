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

#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>

namespace miopen {
namespace unit_tests {

struct TensorDescriptorParams
{
    TensorDescriptorParams(miopenDataType_t datatype_in, std::vector<std::size_t>&& lens_in)
        : datatype(datatype_in), lens(std::move(lens_in))
    {
    }

    TensorDescriptorParams(miopenDataType_t datatype_in,
                           miopenTensorLayout_t layout_in,
                           std::vector<std::size_t>&& lens_in)
        : datatype(datatype_in), layout(layout_in), lens(std::move(lens_in))
    {
    }

    TensorDescriptorParams(miopenDataType_t datatype_in,
                           std::vector<std::size_t>&& lens_in,
                           std::vector<std::size_t>&& strides_in)
        : datatype(datatype_in), lens(std::move(lens_in)), strides(std::move(strides_in))
    {
    }

    TensorDescriptorParams(miopenDataType_t datatype_in,
                           miopenTensorLayout_t layout_in,
                           std::vector<std::size_t>&& lens_in,
                           std::vector<std::size_t>&& strides_in)
        : datatype(datatype_in),
          layout(layout_in),
          lens(std::move(lens_in)),
          strides(std::move(strides_in))
    {
    }

    miopen::TensorDescriptor GetTensorDescriptor() const
    {
        if(layout)
        {
            if(!strides.empty())
                return {datatype, layout.value(), lens, strides};
            else
                return {datatype, layout.value(), lens};
        }
        else
        {
            if(!strides.empty())
                return {datatype, lens, strides};
            else
                return {datatype, lens};
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const TensorDescriptorParams& tp)
    {
        os << tp.datatype << ", ";
        if(tp.layout)
            os << tp.layout.value() << ", ";
        else
            os << "{}, ";
        miopen::LogRange(os << "{", tp.lens, ",") << "}, ";
        miopen::LogRange(os << "{", tp.strides, ",") << "}";
        return os;
    }

private:
    miopenDataType_t datatype;
    std::optional<miopenTensorLayout_t> layout;
    std::vector<std::size_t> lens;
    std::vector<std::size_t> strides;
};

} // namespace unit_tests
} // namespace miopen
