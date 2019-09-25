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
#include <miopen/scgemm/tensorshape.hpp>
#include <miopen/logger.hpp>

namespace miopen {
namespace scgemm {

tensorshape_t create_tensorshape_4d(uint32_t nx, uint32_t ny, uint32_t bt, uint32_t nc)
{
    return tensorshape_t{nx, ny, 1, bt, nc, 0, 2};
}

tensorshape_t create_tensorshape_5d(uint32_t nx, uint32_t ny, uint32_t nz, uint32_t bt, uint32_t nc)
{
    return tensorshape_t{nx, ny, nz, bt, nc, 0, 3};
}

tensorshape_t create_tensorshape_filter_4d(uint32_t nx, uint32_t ny, uint32_t pnc, uint32_t qnc)
{
    return tensorshape_t{nx, ny, 1, qnc, pnc, 1, 2};
}

tensorshape_t
create_tensorshape_filter_5d(uint32_t nx, uint32_t ny, uint32_t nz, uint32_t pnc, uint32_t qnc)
{
    return tensorshape_t{nx, ny, nz, qnc, pnc, 1, 3};
}

void get_group_prop(group_prop_t* gprop,
                    const tensorshape_t& pS,
                    const tensorshape_t& fS,
                    const std::vector<uint32_t>& strides,
                    const std::vector<uint32_t>& dilations)
{
    uint32_t su, sv, sd, du, dv, dd;
    gprop->pnx = pS.nx;
    gprop->pny = pS.ny;
    gprop->pnz = pS.nz;
    gprop->bat = pS.bt;
    gprop->fnx = fS.nx;
    gprop->fny = fS.ny;
    gprop->fnz = fS.nz;
    gprop->pnc = fS.nc;
    gprop->qnc = fS.bt;

    if(pS.dim == 2)
    {
        gprop->qnz = 1;
    }

    su = sv = sd = du = dv = dd = 1;
    if(!strides.empty())
    {
        su = strides[0];
        sv = strides[1];
        if(pS.dim == 3)
        {
            sd = strides[2];
        }
    }
    if(!dilations.empty())
    {
        du += dilations[0];
        dv += dilations[1];
        if(pS.dim == 3)
        {
            dd += dilations[2];
        }
    }
    gprop->qnx = (gprop->pnx - 1 - du * (gprop->fnx - 1)) / su + 1;
    gprop->qny = (gprop->pny - 1 - dv * (gprop->fny - 1)) / sv + 1;
    if(pS.dim == 3)
    {
        gprop->qnz = (gprop->pnz - 1 - dd * (gprop->fnz - 1)) / sd + 1;
    }
}
} // namespace scgemm
} // namespace miopen
