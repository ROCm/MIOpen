/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <cassert>
#include <cstdio>
#include <cstdlib>

#ifndef GUARD_TEST_TEST_HPP_
#define GUARD_TEST_TEST_HPP_

[[gnu::noreturn]] void failed_abort(const char* msg, const char* file, int line)
{
    printf("FAILED: %s: %s:%i\n", msg, file, line);
    std::abort();
}

void failed(const char* msg, const char* file, int line)
{
    printf("FAILED: %s: %s:%i\n", msg, file, line);
}

#define CHECK(...)     \
    if(!(__VA_ARGS__)) \
    failed(#__VA_ARGS__, __FILE__, __LINE__)
#define EXPECT(...)    \
    if(!(__VA_ARGS__)) \
    failed_abort(#__VA_ARGS__, __FILE__, __LINE__)
#define STATUS(...) EXPECT((__VA_ARGS__) == 0)

template <class T>
void run_test()
{
    T t = {};
    t.run();
}

#endif
