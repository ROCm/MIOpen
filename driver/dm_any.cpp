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
#include "registry_driver_maker.hpp"
#include "any_driver.hpp"

static Driver* makeDriver(const std::string& base_arg)
{
    // NOTE: there's no bool dtype in miopen_type so I commented out the support for bool.
    // if(base_arg == "any")
    //     return new AnyDriver<float, bool>();
    // if(base_arg == "anyfp16")
    //     return new AnyDriver<float16, bool>();
    // if(base_arg == "anybfp16")
    //     return new AnyDriver<bfloat16, bool>();
    // if(base_arg == "anyint32")
    //     return new AnyDriver<int32_t, bool>();
    // if(base_arg == "anyint16")
    //     return new AnyDriver<int16_t, bool>();
    if(base_arg == "anyuint8")
        return new AnyDriver<uint8_t, uint8_t>();
    // NOTE: It seems like MIOpen doesn't support for miopen dtype bool yet (?) so I commented out the support for bool.
    // if(base_arg == "anybool")
    //     return new AnyDriver<bool, bool>();
    return nullptr;
}

// REGISTER_DRIVER_MAKER(makeDriver);
REGISTER_DRIVER_MAKER(makeDriver);
