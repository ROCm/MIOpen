/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <miopen/readonlyramdb.hpp>
#include <miopen/logger.hpp>

#include <fstream>
#include <mutex>
#include <sstream>
#include <map>

namespace miopen {

const std::unordered_map<std::string, std::string>& PerfRamDb::perf_db_init(std::string arch_cu)
{
#include "perf_db_init.h"
#if 0
    if(arch_cu == "gfx906-60")
    {
        static const std::unordered_map<std::string, std::string> data
        {{"3-64-32-1x1-32-64-32-100-0x0-1x1-1x1-0-NCHW-FP32-F",
              "ConvBinWinogradRxSf2x3:60;ConvOclDirectFwd1x1:1,64,1,1,0,2,4,4,0;ConvAsm1x1U:1,8,"
              "1,64,3,1,1,1"},
            {"3-32-32-1x1-32-32-32-100-0x0-1x1-1x1-0-NCHW-FP32-F",
              "ConvBinWinogradRxSf2x3:58;ConvOclDirectFwd1x1:1,64,1,1,0,2,4,4,0;ConvAsm1x1U:1,8,"
              "1,64,3,1,1,1"}};
        return data;
    }
    else
    {
        static const std::unordered_map<std::string, std::string> data{};
        return data;
    }
#endif
}

} // namespace miopen
