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

#ifndef GUARD_MLOPEN_BINARY_CACHE_HPP
#define GUARD_MLOPEN_BINARY_CACHE_HPP

#include <miopen/config.h>
#include <miopen/target_properties.hpp>
#include <boost/filesystem/path.hpp>
#include <string>

namespace miopen {

bool IsCacheDisabled();

boost::filesystem::path GetCacheFile(const std::string& device,
                                     const std::string& name,
                                     const std::string& args,
                                     bool is_kernel_str);

boost::filesystem::path GetCachePath(bool is_system);

#if !MIOPEN_ENABLE_SQLITE_KERN_CACHE
boost::filesystem::path LoadBinary(const TargetProperties& target,
                                   std::size_t num_cu,
                                   const std::string& name,
                                   const std::string& args,
                                   bool is_kernel_str = false);
void SaveBinary(const boost::filesystem::path& binary_path,
                const TargetProperties& target,
                const std::string& name,
                const std::string& args,
                bool is_kernel_str = false);
#else
std::string LoadBinary(const TargetProperties& target,
                       std::size_t num_cu,
                       const std::string& name,
                       const std::string& args,
                       bool is_kernel_str = false);

void SaveBinary(const std::string& hsaco,
                const TargetProperties& target,
                std::size_t num_cu,
                const std::string& name,
                const std::string& args,
                bool is_kernel_str = false);
#endif

} // namespace miopen

#endif
