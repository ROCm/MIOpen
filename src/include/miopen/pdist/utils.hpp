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

#include <math.h>

namespace miopen {

namespace solver {

namespace pdist {

inline bool is_approx_equal(double a, double b, double tolerance = 1e-6)
{
    return (fabs(a - b) < tolerance);
}

inline double sign_(double val) { return (0 < val) - (val < 0); }

inline double backward(const double diff, const double grad, const double dist, const double p)
{
    if(p == 1.f)
    { // one
        return grad * sign_(diff);
    }
    else if(p < 2.f)
    { // lt_two
        return (dist == 0.0 || (diff == 0.0 && p < 1))
                   ? 0
                   : (sign_(diff) * pow(fabs(diff), p - 1) * grad / pow(dist, p - 1));
    }
    else if(p == 2.f)
    { // two
        return dist == 0.0 ? 0 : grad * diff / dist;
    }
    else if(isinf(p))
    { // inf
        return grad * sign_(diff) * is_approx_equal(fabs(diff), dist);
    }
    else
    { // p
        return dist == 0.0 ? 0 : diff * pow(fabs(diff), p - 2) * grad / pow(dist, p - 1);
    }
}

} // namespace pdist

} // namespace solver

} // namespace miopen
