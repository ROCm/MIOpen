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

#pragma once
#ifndef GUARD_MIOPEN_KERNEL_CACHE_HPP_
#define GUARD_MIOPEN_KERNEL_CACHE_HPP_

#include <miopen/handle.hpp>
#include <miopen/kernel.hpp>
#include <miopen/miopen.h>
#include <string>
#include <unordered_map>

namespace miopen {

struct SimpleHash
{
    size_t operator()(const std::pair<std::string, std::string>& p) const
    {
        using std::hash;
        return (hash<std::string>()(p.first) ^ hash<std::string>()(p.second));
    }
};

/**
 * @brief The KernelCache class Build and cache kernels
 *
 */
class KernelCache
{

    public:
    using Key        = std::pair<std::string, std::string>;
    using KernelMap  = std::unordered_map<Key, Kernel, SimpleHash>;
    using ProgramMap = std::unordered_map<Key, Program, SimpleHash>;

    Kernel GetKernel(Handle& h,
                     const std::string& algorithm,
                     const std::string& network_config,
                     const std::string& program_name,
                     const std::string& kernel_name,
                     const std::vector<size_t>& vld,
                     const std::vector<size_t>& vgd,
                     std::string params = "");

    Kernel GetKernel(const std::string& algorithm, const std::string& network_config);

    KernelCache();

    private:
    KernelMap kernel_map;
    ProgramMap program_map;
};

} // namespace miopen

#endif // GUARD_MIOPEN_KERNEL_CACHE_HPP_
