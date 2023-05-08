/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#ifndef GUARD_CALC_ERR_
#define GUARD_CALC_ERR_

#include <cmath>
#include <cstdint>

// This works well when difference is small. When one value
// is 0 and another is not, result is incorrect (very big)
// due to zero exponent in one number and non-zero exponent
// in another, because exponent resides in MSBs of a floating
// point number.
template <typename T_>
float ApproxUlps(T_ c_val, T_ g_val)
{
    double err = -1.0;
    if(sizeof(T_) == 2)
    {
        int16_t* c_uval = reinterpret_cast<int16_t*>(&c_val);
        int16_t* g_uval = reinterpret_cast<int16_t*>(&g_val);
        err             = static_cast<double>(std::abs(*c_uval - *g_uval));
    }
    else if(sizeof(T_) == 4)
    {
        int32_t* c_uval = reinterpret_cast<int32_t*>(&c_val);
        int32_t* g_uval = reinterpret_cast<int32_t*>(&g_val);
        err             = static_cast<double>(std::abs(*c_uval - *g_uval));
    }
    else if(sizeof(T_) == 8)
    {
        int64_t* c_uval = reinterpret_cast<int64_t*>(&c_val);
        int64_t* g_uval = reinterpret_cast<int64_t*>(&g_val);
        err             = static_cast<double>(std::abs(*c_uval - *g_uval));
    }

    // double delta = abs(c_val - g_val);
    // double nextafter_delta = nextafterf(min(abs(c_val), abs(g_val)), (T_)INFINITY) -
    // min(abs(c_val), abs(g_val));
    // err = delta / nextafter_delta;
    return err;
}

#endif // GUARD_GUARD_CALC_ERR_
