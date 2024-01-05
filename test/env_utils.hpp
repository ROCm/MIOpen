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

#ifndef GUARD_MIOPEN_ENV_UTILS_HPP
#define GUARD_MIOPEN_ENV_UTILS_HPP

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

#include "test.hpp"
#include <string_view>

inline void setEnvironmentVariable(std::string_view name, std::string_view value)
{
#ifdef _WIN32
    BOOL ret = SetEnvironmentVariable(name.data(), value.data());
    EXPECT_EQUAL(ret, TRUE);
#else
    int ret = setenv(name.data(), value.data(), 1);
    EXPECT_EQUAL(ret, 0);
#endif
}

inline void unsetEnvironmentVariable(std::string_view name)
{
#ifdef _WIN32
    BOOL ret = SetEnvironmentVariable(name.data(), nullptr);
    EXPECT_EQUAL(ret, TRUE);
#else
    int ret = unsetenv(name.data());
    EXPECT_EQUAL(ret, 0);
#endif
}

#endif // GUARD_MIOPEN_ENV_UTILS_HPP
