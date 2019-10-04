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

#include <miopen/tmp_dir.hpp>
#include <miopen/env.hpp>
#include <boost/filesystem.hpp>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_SAVE_TEMP_DIR)

namespace miopen {

void SystemCmd(std::string cmd)
{
#ifndef NDEBUG
    MIOPEN_LOG_I(cmd);
#endif
// We shouldn't call system commands
#ifdef MIOPEN_USE_CLANG_TIDY
    (void)cmd;
#else
    if(std::system(cmd.c_str()) != 0)
        MIOPEN_THROW("Can't execute " + cmd);
#endif
}

TmpDir::TmpDir(std::string prefix)
    : path(boost::filesystem::temp_directory_path() /
           boost::filesystem::unique_path("miopen-" + prefix + "-%%%%-%%%%-%%%%-%%%%"))
{
    boost::filesystem::create_directories(this->path);
}

void TmpDir::Execute(std::string exe, std::string args)
{
    std::string cd  = "cd " + this->path.string() + "; ";
    std::string cmd = cd + exe + " " + args; // + " > /dev/null";
    SystemCmd(cmd);
}

TmpDir::~TmpDir()
{
    if(!miopen::IsEnabled(MIOPEN_DEBUG_SAVE_TEMP_DIR{}))
    {
        boost::filesystem::remove_all(this->path);
    }
}

} // namespace miopen
