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
#include "miopen/bfloat16.hpp"
#include "registry_driver_maker.hpp"
#include "any_driver.hpp"

static Driver* makeDriver(const std::string& base_arg)
{
    // TODO: Add support for uint8, bool, fp16, int16, fp32, int32
    // Tref cannot be "bool". Being bool dtype make it unable to use outhost.data()
    if(base_arg == "anychar") // signed char
                              // template <typename Tgpu, typename Tref>
        return new AnyDriver<signed char, uint8_t>();
    // if(base_arg == "anyuchar")
    // return new AnyDriver<unsigned char, uint8_t>(); // uint8_t is actually the same with int8_t
    //                                                 // as MIOpen automatically convert to int8_t
    //                                                 // if dtype input is uint8_t
    // if(base_arg == "any")                               // float
    //     return new AnyDriver<float, uint8_t>();
    // TODO: Add conversion for those half dtype in kernel function
    if(base_arg == "anyfp16")
        return new AnyDriver<float16, uint8_t>();
    // if(base_arg == "anybfp16")
    //     return new AnyDriver<bfloat16, float>;

    return nullptr;
}

REGISTER_DRIVER_MAKER(makeDriver);
