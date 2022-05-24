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
#include <miopen/errors.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/handle_lock.hpp>
#include <miopen/logger.hpp>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <thread>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEVICE_ARCH)

namespace miopen {

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

void HIPOCKernelInvoke::run(void* args, std::size_t size) const
{
#ifndef NDEBUG
    MIOPEN_LOG_I2("kernel_name = "
                  << GetName() << ", global_work_dim = " << DimToFormattedString(gdims.data(), 3)
                  << ", local_work_dim = " << DimToFormattedString(ldims.data(), 3));
#endif // !NDEBUG

    HipEventPtr start = nullptr;
    HipEventPtr stop  = nullptr;
    void* config[]    = {// HIP_LAUNCH_PARAM_* are macros that do horrible things
                      // NOLINTNEXTLINE cppcoreguidelines-pro-type-cstyle-cast
                      HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      args,
                      // NOLINTNEXTLINE cppcoreguidelines-pro-type-cstyle-cast
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &size,
                      // NOLINTNEXTLINE cppcoreguidelines-pro-type-cstyle-cast
                      HIP_LAUNCH_PARAM_END};
    if(callback)
    {
        start = make_hip_event();
        stop  = make_hip_event();
    }

    const char* const arch = miopen::GetStringEnv(MIOPEN_DEVICE_ARCH{});
    if(arch != nullptr && strlen(arch) > 0)
    {
        MIOPEN_THROW("MIOPEN_DEVICE_ARCH used, escaping launching kernel");
    }

    MIOPEN_HANDLE_LOCK

    auto status = hipExtModuleLaunchKernel(fun,
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
                                      std::function<void(hipEvent_t, hipEvent_t)> callback) const
{
    return HIPOCKernelInvoke{stream, fun, ldims, gdims, name, callback};
}
} // namespace miopen
