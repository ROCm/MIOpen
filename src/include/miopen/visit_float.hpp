/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MLOPEN_VISIT_FLOAT_HPP
#define GUARD_MLOPEN_VISIT_FLOAT_HPP

#include <miopen/miopen.h>
#include <half.hpp>
#include <miopen/bfloat16.hpp>

namespace miopen {

template <class T>
struct as_float
{
    using type = T;
    template <class X>
    type operator()(X x) const
    {
        return static_cast<T>(x);
    }

    template <class X>
    type* operator()(X* x) const
    {
        return static_cast<T*>(x);
    }

    template <class X>
    const type* operator()(const X* x) const
    {
        return static_cast<const T*>(x);
    }
};

template <class F>
void visit_float(miopenDataType_t t, F f)
{
    switch(t)
    {
    case miopenFloat:
    {
        f(as_float<float>{});
        break;
    }
    case miopenHalf:
    {
        f(as_float<half_float::half>{});
        break;
    }
    case miopenBFloat16:
    {
        f(as_float<bfloat16>{});
        break;
    }
    case miopenInt8x4:
    case miopenInt8:
    {
        f(as_float<int8_t>{});
        break;
    }
    case miopenInt32:
    {
        f(as_float<int>{});
        break;
    }
    case miopenDouble:
    {
        f(as_float<double>{});
        break;
    }
    }
}

} // namespace miopen

#endif
