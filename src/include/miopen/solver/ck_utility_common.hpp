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

#pragma once

#include <miopen/handle.hpp>
#include <miopen/stringutils.hpp>

namespace miopen {
namespace solver {
namespace ck_utility {

// MI100 : gfx908
// MI200 : gfx90a
// MI300 : gfx942
/// \todo This function should probably always return true, since the list of supported devices
/// depends on which devices CK was compiled for, and the CK itself includes a check whether is an
/// instance for the device.
static inline bool is_ck_whitelist(const std::string& device_name)
{
    return (StartsWith(device_name, "gfx908") || StartsWith(device_name, "gfx90a") ||
            StartsWith(device_name, "gfx942") || StartsWith(device_name, "gfx950"));
}

static inline bool is_ck_whitelist(const Handle& handle)
{
    return is_ck_whitelist(handle.GetDeviceName());
}

} // namespace ck_utility
} // namespace solver
} // namespace miopen
