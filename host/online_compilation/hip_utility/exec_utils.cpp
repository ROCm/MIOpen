/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#include <exec_utils.hpp>
#include <manage_ptr.hpp>
#include <istream>
#include <ostream>
#include <string>
#include <cstdio>
#include <array>
#include <cassert>

#ifdef __linux__
#include <unistd.h>
#include <cstdio>
#include <sys/wait.h>
#endif // __linux__

namespace olCompile {
namespace exec {

int Run(const std::string& p, std::istream* in, std::ostream* out)
{
#ifdef __linux__
    const auto redirect_stdin  = (in != nullptr);
    const auto redirect_stdout = (out != nullptr);

    assert(!(redirect_stdin && redirect_stdout));

    const auto file_mode = redirect_stdout ? "r" : "w";
    OLC_MANAGE_PTR(FILE*, pclose) pipe{popen(p.c_str(), file_mode)};

    if(!pipe)
        throw std::runtime_error("olCompile::exec::Run(): popen(" + p + ", " + file_mode +
                                 ") failed");

    if(redirect_stdin || redirect_stdout)
    {
        std::array<char, 1024> buffer{};

        if(redirect_stdout)
        {
            while(feof(pipe.get()) == 0)
                if(fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
                    *out << buffer.data();
        }
        else
        {
            while(!in->eof())
            {
                in->read(buffer.data(), buffer.size() - 1);
                buffer[in->gcount()] = 0;

                if(fputs(buffer.data(), pipe.get()) == EOF)
                    throw std::runtime_error("olCompile::exec::Run(): fputs() failed");
            }
        }
    }

    auto status = pclose(pipe.release());
    return WEXITSTATUS(status);
#else
    (void)p;
    (void)in;
    (void)out;
    return -1;
#endif // __linux__
}

} // namespace exec
} // namespace olCompile
