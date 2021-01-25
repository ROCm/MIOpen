/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#include <miopen/idiv.hpp>
#include <cassert>

// Find the position of the least significant bit set to 1 in a 32 bit integer
static inline uint32_t bfls(uint32_t n)
{
    uint8_t shift;
    for(shift = 0; shift < 32; shift++)
        if((1U << shift) >= n)
            break;
    return shift;
}

namespace miopen {
magic_t idiv_magic(uint32_t nmax, uint32_t d)
{
    assert(d != 0);
    if(d == 1)
        return {1, 0};
    uint64_t nc      = ((nmax + 1) / d) * d - 1;
    const auto nbits = bfls(nmax);
    const auto r     = (nbits << 1) + 1;
    magic_t magic    = {static_cast<uint32_t>(-1), static_cast<uint32_t>(-1)};
    for(uint32_t s = 0; s < r; s++)
    {
        uint64_t exp = static_cast<uint64_t>(1) << s;
        uint64_t mod = d - 1 - (exp - 1) % d;
        if(exp > (nc * mod))
        {
            magic.m = static_cast<uint32_t>((exp + mod) / d);
            magic.s = s;
            break;
        }
    }
    return magic;
}
} // namespace miopen
