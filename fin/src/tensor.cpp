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
 * The above copyright notice and this permission notice shall be included in
 *all
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
#include "tensor.hpp"

namespace fin {

tensor::randInit(double dataScale, double min, double max)
{
    // different for fwd/wrw and bwd
    for(auto& it : cpuData)
    {
        // it =
    }
}

size_t tensor::size()
{
    // TODO: check that all internal storages have the same size
    return desc.GetElementSize();
}

size_t tensor::elem_size(miopenDataType_t data_type)
{
    switch(data_type)
    {
    case miopenFloat: return sizeof(float);
    case miopenHalf: return sizeof(float16);
    case miopenBFloat16: return sizeof(bfloat16);
    case miopenInt8: return sizeof(int8_t);
    }
}

} // namespace fin
