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

#include <chrono>
#include <miopen/errors.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <thread>
#include <hip/hip_hcc.h>

namespace miopen {

void HIPOCKernelInvoke::run(void* args, std::size_t size) const
{
    HipEventPtr start = nullptr;
    HipEventPtr stop  = nullptr;
    void* config[]    = {
// HIP_LAUNCH_PARAM_* are macros that do horrible things
#ifdef MIOPEN_USE_CLANG_TIDY
        nullptr, args, nullptr, &size, nullptr
#else
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &size,
        HIP_LAUNCH_PARAM_END
#endif
    };
    if(callback)
    {
        start = make_hip_event();
        stop  = make_hip_event();
    }

    // std::cerr << "Launch kernel: " << name << std::endl;

    auto status = hipHccModuleLaunchKernel(fun,
                                           gdims[0],
                                           gdims[1],
                                           gdims[2],
                                           ldims[0],
                                           ldims[1],
                                           ldims[2],
                                           0,
                                           stream,
                                           nullptr,
                                           reinterpret_cast<void**>(&config),
                                           start.get(),
                                           stop.get());
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed to launch kernel");

    if(callback)
    {
#if 0
        auto start_time = std::chrono::system_clock::now();
        while(hipEventQuery(stop.get()) == hipErrorNotReady)
        {
            std::this_thread::yield();
            if((std::chrono::system_clock::now() - start_time) > std::chrono::seconds(60))
            {
                std::cerr << "Timeout: HIPOCKernelInvoke::run" << std::endl;
                std::abort();
            }
        }
#else
        hipEventSynchronize(stop.get());
#endif
        callback(start.get(), stop.get());
    }
}

HIPOCKernelInvoke HIPOCKernel::Invoke(hipStream_t stream,
                                      std::function<void(hipEvent_t, hipEvent_t)> callback)
{
    return HIPOCKernelInvoke{stream, fun, ldims, gdims, name, callback};
}
} // namespace miopen
