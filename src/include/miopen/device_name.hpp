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
#ifndef GUARD_MIOPEN_DEVICE_NAME_HPP
#define GUARD_MIOPEN_DEVICE_NAME_HPP

#include <map>
#include <string>
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_ENFORCE_DEVICE)

namespace miopen {

std::string inline GetDeviceNameFromMap(const std::string& name)
{

    static std::map<std::string, std::string> device_name_map = {
        {"Ellesmere", "gfx803"},
        {"Baffin", "gfx803"},
        {"RacerX", "gfx803"},
        {"Polaris10", "gfx803"},
        {"Polaris11", "gfx803"},
        {"Tonga", "gfx803"},
        {"Fiji", "gfx803"},
        {"gfx800", "gfx803"},
        {"gfx802", "gfx803"},
        {"gfx803", "gfx803"},
        {"gfx804", "gfx803"},
        {"Vega10", "gfx900"},
        {"gfx900", "gfx900"},
        {"gfx901", "gfx900"},
        {"gfx906", "gfx906"},
        {"gfx908", "gfx908"},
        {"10.3.0 Sienna_Cichlid 18", "gfx1030"},
    };

    std::string n(name);
    const char* const p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_ENFORCE_DEVICE{});
    if(p_asciz != nullptr && strlen(p_asciz) > 0)
    {
        n = p_asciz;
    }

    auto device_name_iterator = device_name_map.find(n);
    if(device_name_iterator != device_name_map.end())
    {
        return device_name_iterator->second;
    }
    else
    {
        return n;
    }
}

} // namespace miopen

#endif // GUARD_MIOPEN_DEVICE_NAME_HPP
