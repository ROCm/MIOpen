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

#include <miopen/env.hpp>

#ifndef _WIN32
#include <cstdlib>
#endif

#include <optional>
#include <string>
#include <string_view>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

namespace miopen::env {

std::optional<std::string> getEnvironmentVariable(std::string_view name)
{
#ifdef _WIN32
    auto required_size = GetEnvironmentVariable(name.data(), nullptr, 0);
    if(required_size == 0) // usually ERROR_ENVVAR_NOT_FOUND
    {
        return std::nullopt;
    }
    // requires size to hold the string and its terminating null character.
    std::string value(required_size - 1, 0);
    GetEnvironmentVariable(name.data(), value.data(), required_size);
    return {value};
#else
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    auto value = std::getenv(name.data());
    return value == nullptr ? std::nullopt : std::make_optional<std::string>(value);
#endif
}

} // namespace miopen::env
