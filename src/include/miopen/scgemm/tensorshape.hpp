/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_SCGEMM_TENSORSHAPE_HPP_
#define GUARD_MIOPEN_SCGEMM_TENSORSHAPE_HPP_

#include <cstdint>
#include <vector>

namespace miopen {
namespace scgemm {

#define STR(s) #s
#define PSIZE(n, m) (((n) + (m)-1) & (~((m)-1)))

struct tensorshape_t
{
    uint32_t nx   = 0;
    uint32_t ny   = 0;
    uint32_t nz   = 0;
    uint32_t bt   = 0; // if is filter then bt means output channel
    uint32_t nc   = 0; // if is filter then nc means input channel
    uint32_t type = 0; // 0 means input/output, 1 means filter
    uint32_t dim  = 0; // 2 or 3

    tensorshape_t(){};
    tensorshape_t(
        uint32_t x, uint32_t y, uint32_t z, uint32_t bs, uint32_t c, uint32_t t, uint32_t d)
        : nx(x), ny(y), nz(z), bt(bs), nc(c), type(t), dim(d){};
};

struct group_prop_t
{
    uint32_t pnx = 0;
    uint32_t pny = 0;
    uint32_t pnz = 0;
    uint32_t qnx = 0;
    uint32_t qny = 0;
    uint32_t qnz = 0;
    uint32_t fnx = 0;
    uint32_t fny = 0;
    uint32_t fnz = 0;
    uint32_t pnc = 0;
    uint32_t qnc = 0;
    uint32_t bat = 0;
};

tensorshape_t create_tensorshape_4d(uint32_t, uint32_t, uint32_t, uint32_t);
tensorshape_t create_tensorshape_5d(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
tensorshape_t create_tensorshape_filter_4d(uint32_t, uint32_t, uint32_t, uint32_t);
tensorshape_t create_tensorshape_filter_5d(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
void get_group_prop(group_prop_t*,
                    const tensorshape_t&,
                    const tensorshape_t&,
                    const std::vector<uint32_t>&,
                    const std::vector<uint32_t>&);
} // namespace scgemm
} // namespace miopen
#endif
