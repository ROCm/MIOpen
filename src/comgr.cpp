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

#include <miopen/logger.hpp>
#include <amd_comgr.h>

namespace miopen {
namespace comgr {

namespace {

bool PrintVersion()
{
    std::size_t major = 0;
    std::size_t minor = 0;
    amd_comgr_get_version(&major, &minor);
    MIOPEN_LOG_I("comgr v." << major << '.' << minor);
    return true;
}

bool once = PrintVersion(); /// FIXME remove this

} // namespace

void BuildOcl(const std::string& text, const std::string& options, std::string& binary)
{
    (void)text;
    (void)options;
    (void)binary;
#if 0
    if(return_code != 0)
    {
        MIOPEN_LOG_W(error_message);
        MIOPEN_THROW("Assembly error(" + std::to_string(return_code) + ")");
    }
#endif
}

} // namespace comgr
} // namespace miopen
