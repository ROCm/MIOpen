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

#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/logger.hpp>
#include <miopen/stringutils.hpp>

#include <iostream>
#include <iterator>
#include <mutex>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEVICE_ARCH)

namespace miopen {

using WriteLock = std::unique_lock<std::shared_mutex>;
using ReadLock  = std::shared_lock<std::shared_mutex>;

std::vector<Kernel> KernelCache::GetKernels(const std::string& algorithm,
                                            const std::string& network_config)
{
    ReadLock readLock(lock);

    std::pair<std::string, std::string> key = std::make_pair(algorithm, network_config);

    const auto it = kernel_map.find(key);
    if(it != kernel_map.end())
    {
        MIOPEN_LOG_I2(it->second.size()
                      << " kernels for key: " << key.first << " \"" << key.second << '\"');
        return it->second;
    }

    static const std::vector<Kernel> empty{};
    MIOPEN_LOG_I2("0 kernels for key: " << key.first << " \"" << key.second << '\"');
    return empty;
}

bool KernelCache::HasProgram(const fs::path& name, const std::string& params) const
{
    ReadLock readLock(lock);

    const auto key = std::make_pair(name, params);
    return program_map.count(key) > 0;
}

void KernelCache::ClearProgram(const fs::path& name, const std::string& params)
{
    WriteLock writeLock(lock);

    const auto key  = std::make_pair(name, params);
    auto program_it = program_map.find(key);
    if(program_it != program_map.end())
    {
        program_map.erase(program_it);
    }
}

void KernelCache::AddProgram(Program prog, const fs::path& program_name, std::string params)
{
    WriteLock writeLock(lock);

    program_map[std::make_pair(program_name, params)] = prog;
}

Kernel KernelCache::AddKernel(const Handle& h,
                              const std::string& algorithm,
                              const std::string& network_config,
                              const fs::path& program_name,
                              const std::string& kernel_name,
                              const std::vector<size_t>& vld,
                              const std::vector<size_t>& vgd,
                              std::string params,
                              std::size_t cache_index,
                              const std::string& kernel_src,
                              Program* program_out)
{
    WriteLock writeLock(lock);

    const std::pair<std::string, std::string> key = std::make_pair(algorithm, network_config);
    if(!network_config.empty() || !algorithm.empty()) // Don't log only _empty_ keys.
        MIOPEN_LOG_I2("Key: " << key.first << " \"" << key.second << '\"');

    const auto program = [&] {
        auto program_it = program_map.find(std::make_pair(program_name, params));
        if(program_it != program_map.end())
        {
            auto& program = program_it->second;

            if(program_out != nullptr && !program.IsCodeObjectInMemory() &&
               !program.IsCodeObjectInFile())
            {
                // We need the binaries attached to the program.
                // This may happen if someone calls immediate mode and then find 2.0 with request
                // for binaries.
                program = h.LoadProgram(program_name, params, kernel_src, true);
            }

            return program;
        }
        else
        {
            auto program = h.LoadProgram(program_name, params, kernel_src, program_out != nullptr);

            program_map[std::make_pair(program_name, params)] = program;
            return program;
        }
    }();

    if(program_out != nullptr)
        *program_out = program;

    Kernel kernel{};
    const auto& arch = env::value(MIOPEN_DEVICE_ARCH);
    if(!arch.empty())
    {
        kernel = Kernel{program, kernel_name};
    }
    else
    {
        kernel = Kernel{program, kernel_name, vld, vgd};
    }

    if(!network_config.empty() && !algorithm.empty())
    {
        this->AddKernelUnsafe(key, kernel, cache_index);
    }
    return kernel;
}

void KernelCache::AddKernel(Key key, Kernel k, std::size_t cache_index)
{
    WriteLock writeLock(lock);
    AddKernelUnsafe(key, k, cache_index);
}

void KernelCache::AddKernelUnsafe(Key key, Kernel k, std::size_t cache_index)
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
    WriteLock writeLock(lock);

    if(network_config.empty() || algorithm.empty())
    {
        MIOPEN_THROW("Network config or algorithm empty.");
    }
    const std::pair<std::string, std::string> key = std::make_pair(algorithm, network_config);
    auto&& v                                      = this->kernel_map[key];
    if(!v.empty())
    {
        MIOPEN_LOG_I2(v.size() << " kernels for key: " << key.first << " \"" << key.second << '\"');
    }
    v.clear();
}

KernelCache::KernelCache() {}

} // namespace miopen
