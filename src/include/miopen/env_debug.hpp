/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#ifdef MIOPEN_BUILD_TESTING

#include <optional>
#include <string>
#include <string_view>

#include <miopen/config.hpp>

namespace miopen {
namespace debug {
namespace env {

// MT-Unsafe
MIOPEN_INTERNALS_EXPORT std::optional<std::string> GetEnvVariable(std::string_view name);

// MT-Unsafe
MIOPEN_INTERNALS_EXPORT void UpdateEnvVariable(std::string_view name, std::string_view value);

// MT-Unsafe
MIOPEN_INTERNALS_EXPORT void ClearEnvVariable(std::string_view name);

} // namespace env
} // namespace debug
} // namespace miopen

#endif
