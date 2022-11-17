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

#include "test.hpp"

#define CHECK(...)                                    \
    do                                                \
    {                                                 \
        if(!(__VA_ARGS__))                            \
            failed(#__VA_ARGS__, __FILE__, __LINE__); \
                                                      \
    } while(false)

#define EXPECT(...)                                         \
    do                                                      \
    {                                                       \
        if(!(__VA_ARGS__))                                  \
            failed_abort(#__VA_ARGS__, __FILE__, __LINE__); \
                                                            \
    } while(false)

#define EXPECT_OP(LEFT, OP, RIGHT)                                 \
    expect_op((LEFT),                                              \
              [](const auto& l, const auto& r) { return l OP r; }, \
              (RIGHT),                                             \
              #LEFT,                                               \
              #OP,                                                 \
              #RIGHT,                                              \
              __FILE__,                                            \
              __LINE__)
#define EXPECT_EQUAL(LEFT, RIGHT) EXPECT_OP(LEFT, ==, RIGHT)
#define STATUS(...) EXPECT((__VA_ARGS__) == 0)

#define FAIL(...) failed(__VA_ARGS__, __FILE__, __LINE__)
