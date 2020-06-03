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

#include <miopen/readonlyramdb.hpp>
#include <miopen/logger.hpp>

#include <fstream>
#include <mutex>
#include <sstream>
#include <map>

namespace miopen {

std::unordered_map<std::string, std::string> FindRamDb::find_db_init(std::string /*path*/)
{
    // use the path to switch between arches
    return {{"3-32-32-3x3-32-30-30-100-0x0-1x1-1x1-0-NCHW-FP32-F",
              "miopenConvolutionFwdAlgoDirect:ConvOclDirectFwd,0.07792,0,"
              "miopenConvolutionFwdAlgoDirect,<unused>;miopenConvolutionFwdAlgoGEMM:gemm,2.736,"
              "97200,rocBlas,<unused>;miopenConvolutionFwdAlgoWinograd:ConvBinWinogradRxSf2x3,0."
              "07104,0,miopenConvolutionFwdAlgoWinograd,<unused>"},
            {"32-30-30-3x3-3-32-32-100-0x0-1x1-1x1-0-NCHW-FP32-B",
              "miopenConvolutionBwdDataAlgoGEMM:gemm,3.904,97200,rocBlas,<unused>;"
              "miopenConvolutionBwdDataAlgoWinograd:ConvBinWinogradRxS,0.13632,0,"
              "miopenConvolutionBwdDataAlgoWinograd,<unused>;miopenConvolutionBwdDataAlgoDirect:"
              "ConvOclDirectFwd,0.12496,0,miopenConvolutionBwdDataAlgoDirect,<unused>"},
            {"32-30-30-3x3-3-32-32-100-0x0-1x1-1x1-0-NCHW-FP32-W",
              "miopenConvolutionBwdWeightsAlgoWinograd:ConvWinograd3x3MultipassWrW<3-6>,0.31824,"
              "22424576,miopenConvolutionBwdWeightsAlgoWinograd,"
              "32x30x30x3x3x3x32x32x100xNCHWxFP32x0x0x1x1x1x1x1x0;"
              "miopenConvolutionBwdWeightsAlgoGEMM:gemm,13.712,97200,rocBlas,<unused>;"
              "miopenConvolutionBwdWeightsAlgoDirect:ConvOclBwdWrW53,0.0856,345600,"
              "miopenConvolutionBwdWeightsAlgoDirect,<unused>"},
            {"key0", "value0"},
            {"key1", "value1"}};
}

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

ReadonlyRamDb& ReadonlyRamDb::GetCached(const std::string& path,
                                        bool warn_if_unreadable,
                                        const std::string& /*arch*/,
                                        const std::size_t /*num_cu*/)
{
    MIOPEN_LOG_I("");
    static std::mutex mutex;
    static const std::lock_guard<std::mutex> lock{mutex};

    static auto instances = std::map<std::string, ReadonlyRamDb*>{};
    const auto it         = instances.find(path);

    if(it != instances.end())
        return *it->second;

    // The ReadonlyRamDb objects allocated here by "new" shall be alive during
    // the calling app lifetime. Size of each is very small, and there couldn't
    // be many of them (max number is number of _different_ GPU board installed
    // in the user's system, which is _one_ for now). Therefore the total
    // footprint in heap is very small. That is why we can omit deletion of
    // these objects thus avoiding bothering with MP/MT syncronization.
    // These will be destroyed altogether with heap.
    auto instance = new ReadonlyRamDb{path};
    instances.emplace(path, instance);
    instance->Prefetch(path, warn_if_unreadable);
    return *instance;
}

template <class TFunc>
static auto Measure(const std::string& funcName, TFunc&& func)
{
    if(!miopen::IsLogging(LoggingLevel::Info))
        return func();

    const auto start = std::chrono::high_resolution_clock::now();
    func();
    const auto end = std::chrono::high_resolution_clock::now();
    MIOPEN_LOG_I("Db::" << funcName << " time: " << (end - start).count() * .000001f << " ms");
}

void ReadonlyRamDb::Prefetch(const std::string& path, bool warn_if_unreadable)
{
    Measure("Prefetch", [this, &path, warn_if_unreadable]() {
        auto file = std::ifstream{path};

        if(!file)
        {
            const auto log_level = warn_if_unreadable ? LoggingLevel::Warning : LoggingLevel::Info;
            MIOPEN_LOG(log_level, "File is unreadable: " << path);
            return;
        }

        auto line   = std::string{};
        auto n_line = 0;

        while(std::getline(file, line))
        {
            ++n_line;

            if(line.empty())
                continue;

            const auto key_size = line.find('=');
            const bool is_key   = (key_size != std::string::npos && key_size != 0);

            if(!is_key)
            {
                MIOPEN_LOG_E("Ill-formed record: key not found: " << path << "#" << n_line);
                continue;
            }

            const auto key      = line.substr(0, key_size);
            const auto contents = line.substr(key_size + 1);

            cache.emplace(key, contents);
        }
    });
}
} // namespace miopen
