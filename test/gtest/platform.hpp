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

#pragma once

#include <miopen/miopen.h>

class DevMem;

class Device
{
public:
    Device(miopenHandle_t handle);
    Device(const Device& other) = delete;
    ~Device();

    Device& operator=(const Device& other) = delete;

    DevMem Malloc(size_t size) const;
    bool Synchronize() const;

private:
#if MIOPEN_BACKEND_OPENCL
    cl_command_queue cmd_queue;
    cl_context context;
#endif

    friend class DevMem;
};

class DevMem
{
public:
    DevMem(const DevMem& other) = delete;
    ~DevMem();

    DevMem& operator=(const DevMem& other) = delete;

    void* Data() const;

    bool CopyToDevice(const void* src, size_t size) const;
    bool CopyFromDevice(void* dst, size_t size) const;

private:
    DevMem(const Device& device, size_t size);

#if MIOPEN_BACKEND_HIP
    void* ptr;
#elif MIOPEN_BACKEND_OPENCL
    cl_command_queue cmd_queue;
    cl_mem ptr;
#endif

    friend class Device;
};
