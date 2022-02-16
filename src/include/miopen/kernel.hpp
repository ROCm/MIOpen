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
#ifndef GUARD_MIOPEN_KERNEL_HPP
#define GUARD_MIOPEN_KERNEL_HPP

#include <string>
#include <vector>

#include <miopen/config.h>

namespace miopen {
std::string GetKernelSrc(std::string name);
std::string GetKernelInc(std::string key);
const std::string* GetKernelIncPtr(std::string key);
std::vector<std::string> GetKernelIncList();
std::vector<std::string> GetHipKernelIncList();
} // namespace miopen

#if MIOPEN_BACKEND_OPENCL
#include <miopen/clhelper.hpp>
#include <miopen/oclkernel.hpp>

namespace miopen {
using Kernel       = OCLKernel;
using KernelInvoke = OCLKernelInvoke;
using Program      = SharedProgramPtr;

} // namespace miopen

#elif MIOPEN_BACKEND_HIP
#include <miopen/hipoc_kernel.hpp>

namespace miopen {
using Kernel       = HIPOCKernel;
using KernelInvoke = HIPOCKernelInvoke;
using Program      = HIPOCProgram;

} // namespace miopen
#endif

#endif
