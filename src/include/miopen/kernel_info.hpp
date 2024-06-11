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

#ifndef GUARD_MLOPEN_KERNEL_INFO_HPP
#define GUARD_MLOPEN_KERNEL_INFO_HPP

#include <ostream>
#include <string>
#include <vector>
#include <miopen/kernel.hpp>

namespace miopen {

struct Handle;

namespace solver {

/// Describes a kernel source and whatever information required in order
/// to build and run it (the former is unused for binary kernels).
struct KernelInfo
{
    std::string comp_options;
    std::vector<size_t> l_wk;
    std::vector<size_t> g_wk;
    fs::path kernel_file;
    std::string kernel_name;
    friend std::ostream& operator<<(std::ostream& os, const KernelInfo& k);
};

std::vector<Program> PrecompileKernels(const Handle& h,
                                       const std::vector<KernelInfo>& kernels,
                                       bool force_attach_binary = false);

} // namespace solver
} // namespace miopen

#endif
