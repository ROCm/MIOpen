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
#ifndef MIOPEN_GUARD_OCL_HELPER_HPP_
#define MIOPEN_GUARD_OCL_HELPER_HPP_

#include <iostream>
#include <miopen/manage_ptr.hpp>
#include <miopen/miopen.h>
#include <string>

namespace miopen {

using ClProgramPtr = MIOPEN_MANAGE_PTR(cl_program, clReleaseProgram);
using ClKernelPtr  = MIOPEN_MANAGE_PTR(cl_kernel, clReleaseKernel);
using ClAqPtr      = MIOPEN_MANAGE_PTR(miopenAcceleratorQueue_t, clReleaseCommandQueue);

ClProgramPtr LoadBinaryProgram(cl_context ctx, cl_device_id device, const std::string& source);

ClProgramPtr LoadProgram(cl_context ctx,
                         cl_device_id device,
                         const std::string& program_name,
                         std::string params,
                         bool is_kernel_str,
                         const std::string& kernel_src);
void SaveProgramBinary(const ClProgramPtr& program, const std::string& name);
ClKernelPtr CreateKernel(cl_program program, const std::string& kernel_name);
inline ClKernelPtr CreateKernel(const ClProgramPtr& program, const std::string& kernel_name)
{
    return CreateKernel(program.get(), kernel_name);
}
#if 0 /// \todo Dead code?
ClAqPtr CreateQueueWithProfiling(cl_context ctx, cl_device_id dev);
#endif
cl_device_id GetDevice(cl_command_queue q);
cl_context GetContext(cl_command_queue q);
} // namespace miopen

#endif // MIOPEN_GUARD_OCL_HELPER_HPP_
