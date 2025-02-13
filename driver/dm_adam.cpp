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
#include "adam_driver.hpp"
#include "registry_driver_maker.hpp"

static Driver* makeDriver(const std::string& base_arg)
{
    if(base_arg == "adam")
        return new AdamDriver<float, float>();
    else if(base_arg == "adamfp16")
        return new AdamDriver<float16, float>();
    else if(base_arg == "ampadam")
        return new AdamDriver<float, float, float16>(false, true);
    else if(base_arg == "adamw")
        return new AdamDriver<float, float>(true);
    else if(base_arg == "adamwfp16")
        return new AdamDriver<float16, float>(true);
    else if(base_arg == "ampadamw")
        return new AdamDriver<float, float, float16>(true, true);
    return nullptr;
}

REGISTER_DRIVER_MAKER(makeDriver);
