/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
/* ************************************************************************
 * Copyright 2015 Vratis, Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

#include <miopen/errors.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/logger.hpp>

#include <iostream>
#include <iterator>

namespace miopen {

static std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
    return LogRange(os, v, ",");
}

static void AddKernelDumpKernelParams(const std::string& program_name,
                                      const std::string& kernel_name,
                                      const std::vector<size_t>& vld,
                                      const std::vector<size_t>& vgd,
                                      const std::string& params)
{
    const char* keys[] = {"MLO_FILTER_SIZE0",
                          "MLO_FILTER_SIZE1",
                          "MLO_N_INPUTS",
                          "MLO_N_OUTPUTS",
                          "MLO_BATCH_SZ",
                          "MLO_IN_HEIGHT",
                          "MLO_IN_WIDTH",
                          "MLO_OUT_HEIGHT",
                          "MLO_OUT_WIDTH",
                          "MLO_FLTR_SZ0",
                          "MLO_FLTR_SZ1",
                          "MLO_N_IN_CHNLS",
                          "MLO_N_OUT_CHNLS"};
    int value[sizeof(keys) / sizeof(keys[0])] = {0};
    for(const char* p = params.c_str(); p != nullptr && (p = strstr(p, "-D")) != nullptr;)
    {
        p += (p[2] == ' ') ? 3 : 2;
        const char* q = strstr(p, "=");
        if(q == nullptr)
            break;
        q++;
        for(unsigned long i = 0; i < sizeof(keys) / sizeof(keys[0]); i++)
        {
            if(strncmp(p, keys[i], strlen(keys[i])) == 0)
            {
                value[i - ((i >= 9) ? 9 : 0)] = static_cast<int>(strtol(q, nullptr, 10));
                break;
            }
        }
    }
    // for(int i = 0; i < sizeof(keys)/sizeof(keys[0]); i++) printf("%s = %d\n", keys[i], value[i]);
    int msize = value[0] * value[1] * value[2] * value[3];
    int isize = value[4] * value[2] * value[5] * value[6];
    int osize = value[4] * value[3] * value[7] * value[8];
    MIOPEN_LOG_I2("runcl " << program_name << " -k " << kernel_name << " -dumpilisa -r 10"
                           << " if#"
                           << isize * 4
                           << ": if#"
                           << msize * 4
                           << ": if#"
                           << osize * 4
                           << ": iv#0 "
                           << vgd
                           << "/"
                           << vld
                           << ' '
                           << params);
}

const std::vector<Kernel>& KernelCache::GetKernels(const std::string& algorithm,
                                                   const std::string& network_config)
{

    std::pair<std::string, std::string> key = std::make_pair(algorithm, network_config);

    const auto it = kernel_map.find(key);
    if(it != kernel_map.end())
    {
        MIOPEN_LOG_I2(it->second.size() << " kernels for key: " << key.first << " \"" << key.second
                                        << '\"');
        return it->second;
    }

    static const std::vector<Kernel> empty{};
    MIOPEN_LOG_I2("0 kernels for key: " << key.first << " \"" << key.second << '\"');
    return empty;
}

bool KernelCache::HasKernels(const std::string& algorithm, const std::string& network_config) const
{
    const auto key = std::make_pair(algorithm, network_config);
#ifndef NDEBUG
    MIOPEN_LOG_I("Key: " << key.first << " \"" << key.second << '\"');
#endif
    const auto it = kernel_map.find(key);
    if(it == kernel_map.end())
        return false;

    if(it->second.empty())
    {
        MIOPEN_THROW("There should be at least one kernel in kernel cache if an entry exists");
    }

    return true;
}

Kernel KernelCache::AddKernel(Handle& h,
                              const std::string& algorithm,
                              const std::string& network_config,
                              const std::string& program_name,
                              const std::string& kernel_name,
                              const std::vector<size_t>& vld,
                              const std::vector<size_t>& vgd,
                              std::string params,
                              std::size_t cache_index,
                              bool is_kernel_miopengemm_str,
                              const std::string& kernel_src)
{
    if(params.length() > 0)
    {
        // Ensure only one space after the -cl-std.
        // >1 space can cause an Apple compiler bug. See clSPARSE issue #141.
        if(params.at(0) != ' ')
        {
            params = " " + params;
        }
    }

    const std::pair<std::string, std::string> key = std::make_pair(algorithm, network_config);
    if(!network_config.empty() || !algorithm.empty()) // Don't log only _empty_ keys.
        MIOPEN_LOG_I2("Key: " << key.first << " \"" << key.second << '\"');

    Program program;

    auto program_it = program_map.find(std::make_pair(program_name, params));
    if(program_it != program_map.end())
    {
        program = program_it->second;
    }
    else
    {
        if(!is_kernel_miopengemm_str) // default value
            is_kernel_miopengemm_str = algorithm.find("ImplicitGEMM") == std::string::npos &&
                                       algorithm.find("GEMM") != std::string::npos &&
                                       algorithm.find("StaticCompiledGEMM") == std::string::npos;

        if(miopen::IsLogging(miopen::LoggingLevel::Info2))
        {
            AddKernelDumpKernelParams(is_kernel_miopengemm_str
                                          ? std::string("(source provided by miopengemm)")
                                          : program_name,
                                      kernel_name,
                                      vld,
                                      vgd,
                                      params);
        }
        program = h.LoadProgram(program_name, params, is_kernel_miopengemm_str, kernel_src);
        program_map[std::make_pair(program_name, params)] = program;
    }
    Kernel kernel{program, kernel_name, vld, vgd};
    if(!network_config.empty() && !algorithm.empty())
    {
        this->AddKernel(key, kernel, cache_index);
    }
    return kernel;
}

void KernelCache::AddKernel(Key key, Kernel k, std::size_t cache_index)
{
    auto&& v = kernel_map[key];
    if(cache_index >= v.size())
    {
        v.resize(cache_index + 1);
    }
    v[cache_index] = k;
}

void KernelCache::ClearKernels(const std::string& algorithm, const std::string& network_config)
{
    if(network_config.empty() || algorithm.empty())
    {
        MIOPEN_THROW("Network config or algorithm empty.");
    }
    const std::pair<std::string, std::string> key = std::make_pair(algorithm, network_config);
    auto&& v = this->kernel_map[key];
    if(!v.empty())
    {
        MIOPEN_LOG_I2(v.size() << " kernels for key: " << key.first << " \"" << key.second << '\"');
    }
    v.clear();
}

KernelCache::KernelCache() {}

} // namespace miopen
