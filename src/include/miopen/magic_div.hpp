/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MLOPEN_MAGIC_DIV_HPP_
#define GUARD_MLOPEN_MAGIC_DIV_HPP_

#include <cstdint>
#include <cassert>

namespace miopen {

struct magic_div_u32
{
    uint32_t magic;
    uint8_t shift;
};

using magic_div_u32_t = magic_div_u32;

/*
 *
 * numer / denom = quotient, reminder
 *
 * use magic number to do integer division of uint32 (acctually INT32_MAX, the 31 bit divisoin)
 * most algorithm to compute uint32 need branching if cover all 32 bit of uint32.
 * since we compute the magic number on host side, implement the division in gpu side, it is better
 * not use branching
 * hence add more restriction to numer and denom, to be 1 bit less. hence need less-or-equal than
 * INT32_MAX
 *
 * magic_div_u32_gen() compute from input arg d, to get a magic and a shift.
 * to use the value, below is a example host-side code to do this
 *
 * // host side version
 * static inline uint32_t magic_div_mulhi_u32(uint32_t x, uint32_t y) {
 *     uint64_t xl = x, yl = y;
 *     uint64_t rl = xl * yl;
 *     return (uint32_t)(rl >> 32);
 * }
 * uint32_t magic_div_u32_do(uint32_t numer, const struct magic_div_u32_t *denom) {
 *     uint32_t tmp = magic_div_mulhi_u32(denom->magic, numer);
 *     return (tmp + numer) >> denom->shift;
 * }
 *
 */
static inline magic_div_u32_t magic_div_u32_gen(uint32_t d)
{
    assert(d >= 1 && d <= INT32_MAX);
    uint8_t shift;
    for(shift = 0; shift < 32; shift++)
        if((1U << shift) >= d)
            break;

    constexpr uint64_t one = 1;
    uint64_t magic         = ((one << 32) * ((one << shift) - d)) / d + 1;
    assert(magic <= 0xffffffffUL);

    return {static_cast<uint32_t>(magic), shift};
}

static inline uint32_t magic_div_u32_pack_shift(uint8_t s0, uint8_t s1, uint8_t s2, uint8_t s3)
{
    uint32_t shift_0 = s0;
    uint32_t shift_1 = s1;
    uint32_t shift_2 = s2;
    uint32_t shift_3 = s3;
    return (shift_3 << 24) | (shift_2 << 16) | (shift_1 << 8) | shift_0;
}

} // namespace miopen

#endif
