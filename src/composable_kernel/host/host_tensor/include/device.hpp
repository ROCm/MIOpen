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
#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <memory>
#include <functional>
#include <thread>
#include <chrono>
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"

struct DeviceMem
{
    DeviceMem() = delete;
    DeviceMem(std::size_t mem_size);
    void* GetDeviceBuffer();
    void ToDevice(const void* p);
    void FromDevice(void* p);
    ~DeviceMem();

    void* mpDeviceBuf;
    std::size_t mMemSize;
};

struct KernelTimerImpl;

struct KernelTimer
{
    KernelTimer();
    ~KernelTimer();
    void Start();
    void End();
    float GetElapsedTime() const;

    std::unique_ptr<KernelTimerImpl> impl;
};

using device_stream_t = hipStream_t;

template <typename... Args, typename F>
void launch_kernel(F kernel, dim3 grid_dim, dim3 block_dim, std::size_t lds_byte, Args... args)
{
    hipStream_t stream_id = nullptr;

    hipLaunchKernelGGL(kernel, grid_dim, block_dim, lds_byte, stream_id, args...);
}

template <typename... Args, typename F>
float launch_and_time_kernel(
    F kernel, int nrepeat, dim3 grid_dim, dim3 block_dim, std::size_t lds_byte, Args... args)
{
    KernelTimer timer;

    printf("%s: grid_dim {%d, %d, %d}, block_dim {%d, %d, %d} \n",
           __func__,
           grid_dim.x,
           grid_dim.y,
           grid_dim.z,
           block_dim.x,
           block_dim.y,
           block_dim.z);

    printf("Warm up\n");

    hipStream_t stream_id = nullptr;

    // warm up
    hipLaunchKernelGGL(kernel, grid_dim, block_dim, lds_byte, stream_id, args...);

    printf("Start running %d times...\n", nrepeat);

    timer.Start();

    for(int i = 0; i < nrepeat; ++i)
    {
        hipLaunchKernelGGL(kernel, grid_dim, block_dim, lds_byte, stream_id, args...);
    }

    timer.End();

    // std::this_thread::sleep_for (std::chrono::microseconds(10));

    return timer.GetElapsedTime() / nrepeat;
}
#endif
