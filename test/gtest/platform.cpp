/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <stdexcept>
#include <gtest/gtest.h>
#include "platform.hpp"

#define THROW_PLATFORM_EXCEPTION(what) \
    throw std::runtime_error(what + std::string(" at ") + __FILE__ + ":" + std::to_string(__LINE__))

// ==================== Device ====================

#if MIOPEN_BACKEND_HIP

Device::Device(miopenHandle_t) {}

Device::~Device() {}

#endif // MIOPEN_BACKEND_HIP

#if MIOPEN_BACKEND_OPENCL

Device::Device(miopenHandle_t handle)
{
    miopenStatus_t miopen_status;
    cl_int status;

    miopen_status = miopenGetStream(handle, &cmd_queue);
    if(miopen_status != miopenStatusSuccess)
    {
        THROW_PLATFORM_EXCEPTION("miopenGetStream error");
    }

    status = clRetainCommandQueue(cmd_queue);
    if(status != CL_SUCCESS)
    {
        THROW_PLATFORM_EXCEPTION("clRetainCommandQueue error");
    }

    status =
        clGetCommandQueueInfo(cmd_queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);
    if(status != CL_SUCCESS)
    {
        THROW_PLATFORM_EXCEPTION("clGetCommandQueueInfo error");
    }

    status = clRetainContext(context);
    if(status != CL_SUCCESS)
    {
        THROW_PLATFORM_EXCEPTION("clRetainContext error");
    }
}

Device::~Device()
{
    EXPECT_EQ(clReleaseCommandQueue(cmd_queue), CL_SUCCESS);
    EXPECT_EQ(clReleaseContext(context), CL_SUCCESS);
}

#endif // MIOPEN_BACKEND_OPENCL

DevMem Device::Malloc(size_t size) const { return {*this, size}; }

#if MIOPEN_BACKEND_HIP

bool Device::Synchronize() const
{
    auto status = hipDeviceSynchronize();
    if(status != hipSuccess)
        return false;
    return true;
}

#endif // MIOPEN_BACKEND_HIP

#if MIOPEN_BACKEND_OPENCL

bool Device::Synchronize() const
{
    auto status = clFinish(cmd_queue);
    if(status != CL_SUCCESS)
        return false;
    return true;
}

#endif // MIOPEN_BACKEND_OPENCL

// ==================== DevMem ====================

#if MIOPEN_BACKEND_HIP

DevMem::DevMem(const Device&, size_t size)
{
    auto status = hipMalloc(&ptr, size);
    if(status != hipSuccess)
    {
        THROW_PLATFORM_EXCEPTION("hipMalloc error");
    }
}

DevMem::~DevMem() { EXPECT_EQ(hipFree(ptr), hipSuccess); }

#endif // MIOPEN_BACKEND_HIP

#if MIOPEN_BACKEND_OPENCL

DevMem::DevMem(const Device& device, size_t size)
{
    cl_int status;

    if(size == 0)
    {
        cmd_queue = nullptr;
        ptr       = nullptr;
        return;
    }

    cmd_queue = device.cmd_queue;
    status    = clRetainCommandQueue(cmd_queue);
    if(status != CL_SUCCESS)
    {
        THROW_PLATFORM_EXCEPTION("clRetainCommandQueue error");
    }

    ptr = clCreateBuffer(device.context, CL_MEM_READ_WRITE, size, nullptr, &status);
    if(status != CL_SUCCESS)
    {
        THROW_PLATFORM_EXCEPTION("clCreateBuffer error");
    }
}

DevMem::~DevMem()
{
    if(ptr == nullptr)
        return;

    EXPECT_EQ(clReleaseMemObject(ptr), CL_SUCCESS);
    EXPECT_EQ(clReleaseCommandQueue(cmd_queue), CL_SUCCESS);
}

#endif // MIOPEN_BACKEND_OPENCL

void* DevMem::Data() const { return ptr; }

#if MIOPEN_BACKEND_HIP

bool DevMem::CopyToDevice(const void* src, size_t size) const
{
    auto status = hipMemcpy(ptr, src, size, hipMemcpyHostToDevice);
    if(status != hipSuccess)
        return false;
    return true;
}

bool DevMem::CopyFromDevice(void* dst, size_t size) const
{
    auto status = hipMemcpy(dst, ptr, size, hipMemcpyDeviceToHost);
    if(status != hipSuccess)
        return false;
    return true;
}

#endif // MIOPEN_BACKEND_HIP

#if MIOPEN_BACKEND_OPENCL

bool DevMem::CopyToDevice(const void* src, size_t size) const
{
    auto status = clEnqueueWriteBuffer(cmd_queue, ptr, CL_TRUE, 0, size, src, 0, nullptr, nullptr);
    if(status != CL_SUCCESS)
        return false;
    return true;
}

bool DevMem::CopyFromDevice(void* dst, size_t size) const
{
    auto status = clEnqueueReadBuffer(cmd_queue, ptr, CL_TRUE, 0, size, dst, 0, nullptr, nullptr);
    if(status != CL_SUCCESS)
        return false;
    return true;
}

#endif // MIOPEN_BACKEND_OPENCL
