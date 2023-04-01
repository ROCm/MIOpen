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
#include <miopen/env.hpp>
#include <miopen/handle_lock.hpp>
#include <miopen/logger.hpp>
#include <miopen/oclkernel.hpp>

namespace miopen {

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEVICE_ARCH)

#ifndef NDEBUG
static std::string DimToFormattedString(const size_t* dims, size_t count)
{
    std::stringstream ss;
    ss << '{';
    for(size_t i = 0; i < count; ++i)
    {
        if(i > 0)
            ss << ", ";
        else
            ss << ' ';
        ss << dims[i];
    }
    ss << " }";
    return ss.str();
}
#endif // !NDEBUG

void OCLKernelInvoke::run() const
{
#ifndef NDEBUG
    MIOPEN_LOG_I2("kernel_name = "
                  << GetName() << ", work_dim = " << work_dim << ", global_work_offset = "
                  << DimToFormattedString(global_work_offset.data(), work_dim)
                  << ", global_work_dim = " << DimToFormattedString(gdims.data(), work_dim)
                  << ", local_work_dim = " << DimToFormattedString(ldims.data(), work_dim));
#endif // !NDEBUG

    MIOPEN_HANDLE_LOCK

    const char* const arch = miopen::GetStringEnv(MIOPEN_DEVICE_ARCH{});
    if(arch != nullptr && strlen(arch) > 0)
    {
        MIOPEN_THROW("MIOPEN_DEVICE_ARCH used, escaping launching kernel");
    }

    cl_event ev;
    /* way to run OCL group larger than 256
     * hack to ensure local_size == 0, just checking that the 1st dim is 0
     * may want to use a better solution*/
    cl_int status = clEnqueueNDRangeKernel(queue,
                                           kernel.get(),
                                           work_dim,
                                           ((work_dim == 0) ? nullptr : global_work_offset.data()),
                                           gdims.data(),
                                           ((ldims[0] == 0) ? nullptr : ldims.data()),
                                           0,
                                           nullptr,
                                           callback ? &ev : nullptr);

    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "Running kernel failed: ");
    }
    else if(callback)
    {
        clWaitForEvents(1, &ev);
        callback(ev);
    }
}

std::string OCLKernelInvoke::GetName() const
{
    std::array<char, 200> buffer{};

    cl_int status =
        clGetKernelInfo(kernel.get(), CL_KERNEL_FUNCTION_NAME, 200, buffer.data(), nullptr);

    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "Error getting kernel name");
    }
    return buffer.data();
}

OCLKernelInvoke OCLKernel::Invoke(cl_command_queue q, std::function<void(cl_event&)> callback) const
{
#ifndef NDEBUG
    MIOPEN_LOG_I(GetName());
#endif
    OCLKernelInvoke result{q, kernel, gdims.size(), {}, {}, {}, callback};
    std::copy(gdims.begin(), gdims.end(), result.gdims.begin());
    std::copy(ldims.begin(), ldims.end(), result.ldims.begin());
    return result;
}

std::string OCLKernel::GetName() const
{
    std::array<char, 200> buffer{};

    cl_int status =
        clGetKernelInfo(kernel.get(), CL_KERNEL_FUNCTION_NAME, 200, buffer.data(), nullptr);

    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "Error getting kernel name");
    }
    return buffer.data();
}

} // namespace miopen
