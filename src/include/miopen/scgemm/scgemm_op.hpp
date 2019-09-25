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
#ifndef GUARD_MIOPEN_SCGEMM_SCGEMMOP_HPP_
#define GUARD_MIOPEN_SCGEMM_SCGEMMOP_HPP_
#include <cstdint>
#include <memory>
namespace miopen {
namespace scgemm {

struct kernel_prop_t
{
    std::string name;
    uint32_t block_size;
    uint32_t tile_x;
    uint32_t tile_y;
};

struct scgemm_fconv_params
{
    uint32_t ntidx;
    uint32_t nb_amap;
    uint32_t aozero;
    uint32_t bozero;
    uint32_t onc;
    uint32_t ocs;
    uint32_t ls;
    uint32_t sgs;
    uint32_t onpx;

    uint32_t pnx;
    uint32_t pny;
    uint32_t pnz;
    uint32_t qnx;
    uint32_t qny;
    uint32_t qnz;
    uint32_t fnx;
    uint32_t fny;
    uint32_t fnz;
    uint32_t pnc;
    uint32_t qnc;
    uint32_t bat;
};

struct scgemm_fgemm_params
{
    uint32_t ntidx;
    uint32_t dimx;
    uint32_t bs;
    uint32_t m;
    uint32_t n;
    uint32_t k;
};
} // namespace scgemm
} // namespace miopen
#endif
