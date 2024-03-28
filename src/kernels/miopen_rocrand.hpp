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

namespace miopen::prng {
// borrowed from rocrand, since hiprtc/comgr has some problems to compile kernels with rocrand
struct xorwow_state
{
    // Weyl sequence value
    unsigned int d;
    // Xorshift values (160 bits)
    unsigned int x[5];
};

inline constexpr unsigned int xorwow_next(xorwow_state* state)
{
    const unsigned int t = state->x[0] ^ (state->x[0] >> 2);
    state->x[0]          = state->x[1];
    state->x[1]          = state->x[2];
    state->x[2]          = state->x[3];
    state->x[3]          = state->x[4];
    state->x[4]          = (state->x[4] ^ (state->x[4] << 4)) ^ (t ^ (t << 1));

    state->d += 362437;

    return state->d + state->x[4];
}

inline constexpr float uniform_distribution(unsigned int v)
{
    constexpr float ROCRAND_2POW32_INV = 2.3283064e-10f;
    return ROCRAND_2POW32_INV + (static_cast<float>(v) * ROCRAND_2POW32_INV);
}

inline constexpr float xorwow_uniform(xorwow_state* state)
{
    return uniform_distribution(xorwow_next(state));
}

} // namespace miopen::prng
