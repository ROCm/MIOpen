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
#ifndef MIOPEN_GUARD_MLIR_BUILD_HPP
#define MIOPEN_GUARD_MLIR_BUILD_HPP

#include <miopen/config.h>
#if MIOPEN_USE_MLIR

#include <miopen/target_properties.hpp>
#include <miopen/tmp_dir.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/optional.hpp>
#include <string>

namespace miopen {

void MiirGenLaunchParams(const std::string& params, size_t& local_size, size_t& global_size);

bool MiirIsConfigApplicable(const std::string& params);

int MiirGetKernelCount(const std::string& params);

void MiirGenBin(const std::string& params, std::vector<char>& buffer);

int MiirGetWorkspaceSize(const std::string& params);
} // namespace miopen

#endif // MIOPEN_USE_MLIR
#endif // MIOPEN_GUARD_MLIR_BUILD_HPP
