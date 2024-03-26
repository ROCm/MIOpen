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
#include "conv.hpp"
#include "conv_driver.hpp"

Driver* makeDriverConv(const std::string& base_arg)
{
    if(base_arg == "conv")
        return new ConvDriver<float, float>();
    if(base_arg == "convfp16")
        return new ConvDriver<float16, float>();
    if(base_arg == "convbfp16")
        return new ConvDriver<bfloat16, float>();
    if(base_arg == "convint8")
        return new ConvDriver<int8_t, int32_t>();
    if(base_arg == "convfp8")
        return new ConvDriver<float8, float>();
    if(base_arg == "convbfp8")
        return new ConvDriver<bfloat8, float>();
    return nullptr;
}
