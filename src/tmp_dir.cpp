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
#include <miopen/filesystem.hpp>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <miopen/process.hpp>
#include <boost/filesystem/operations.hpp>

#include <thread>
#include <string_view>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_SAVE_TEMP_DIR)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_EXIT_STATUS_TEMP_DIR)

namespace miopen {

TmpDir::TmpDir(std::string_view prefix) : path{fs::temp_directory_path()}
{
    std::string p{prefix.empty() ? "" : (prefix[0] == '-' ? "" : "-")};

    path /= boost::filesystem::unique_path("miopen" + p.append(prefix) + "-%%%%-%%%%-%%%%-%%%%")
                .string();

    fs::create_directories(path);
}

int TmpDir::Execute(std::string_view cmd, std::string_view args) const
{
    if(env::enabled(MIOPEN_DEBUG_SAVE_TEMP_DIR))
    {
        MIOPEN_LOG_I2(path);
    }
    auto status = Process{cmd}.Arguments(args).WorkingDirectory(path).Execute().Wait();
    if(env::enabled(MIOPEN_DEBUG_EXIT_STATUS_TEMP_DIR))
    {
        MIOPEN_LOG_I2(status);
    }
    return status;
}

TmpDir::~TmpDir()
{
    if(!env::enabled(MIOPEN_DEBUG_SAVE_TEMP_DIR))
    {
#ifdef _WIN32
        constexpr int remove_max_retries = 5;
        int count                        = 0;
        while(count < remove_max_retries)
        {
            try
            {
                fs::remove_all(path);
                break;
            }
            catch(const fs::filesystem_error& err)
            {
                MIOPEN_LOG_W(err.what());
                std::this_thread::sleep_for(std::chrono::milliseconds{125});
            }
            ++count;
        }
#else
        if(!this->path.empty())
            fs::remove_all(this->path);
#endif
    }
}

} // namespace miopen
