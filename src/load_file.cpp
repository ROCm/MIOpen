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

#include <miopen/errors.hpp>
#include <miopen/load_file.hpp>

#include <fstream>
#include <ios>
#include <iterator>
#include <vector>
#include <sstream>

namespace miopen {

std::vector<char> LoadFile(const fs::path& path)
{
#if MIOPEN_WORKAROUND_USE_BOOST_FILESYSTEM
    boost::system::error_code error_code;
#else
    std::error_code error_code;
#endif
    const auto size = fs::file_size(path, error_code);
    if(error_code)
        MIOPEN_THROW(path.string() + ": " + error_code.message());
    std::ifstream in(path, std::ios::binary);
    if(!in.is_open())
        MIOPEN_THROW(path.string() + ": file opening error");
    std::vector<char> v(size);
    if(in.read(v.data(), v.size()).fail())
        MIOPEN_THROW(path.string() + ": file reading error");
    return v;
}

} // namespace miopen
