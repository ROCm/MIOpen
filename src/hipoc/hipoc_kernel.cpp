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
#include <miopen/handle.hpp>
#include <miopen/handle_lock.hpp>
#include <miopen/logger.hpp>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <thread>

#define WORKAROUND_SWDEV_448157 1

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEVICE_ARCH)

namespace miopen {

HipEventProfiler::HipEventProfiler(const Handle& handle_)
    : handle(handle_), start(nullptr), stop(nullptr)
{
    if(handle.IsProfilingEnabled())
    {
        start = make_hip_event();
        stop  = make_hip_event();
        hipEventRecord(start.get(), handle.GetStream());
    }
}

HipEventProfiler::~HipEventProfiler()
{
    if(start)
    {
        hipEventRecord(stop.get(), handle.GetStream());
        hipEventSynchronize(stop.get());
        float event_time = 0.0f;
        hipEventElapsedTime(&event_time, start.get(), stop.get());
        handle.ResetKernelTime();
        handle.AccumKernelTime(event_time);
    }
}

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

void HIPOCKernelInvoke::run(void* args, std::size_t size) const
{
    MIOPEN_LOG_I2("kernel_name = "
                  << GetName() << ", global_work_dim = " << DimToFormattedString(gdims.data(), 3)
                  << ", local_work_dim = " << DimToFormattedString(ldims.data(), 3));

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

    const auto& arch = env::value(MIOPEN_DEVICE_ARCH);
    if(!arch.empty())
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

void HIPOCKernelInvoke::run_cooperative(void** kern_args) const
{
    hipError_t status;

    MIOPEN_LOG_I2("kernel_name = "
                  << GetName() << ", global_work_dim = " << DimToFormattedString(gdims.data(), 3)
                  << ", local_work_dim = " << DimToFormattedString(ldims.data(), 3));

    const auto& arch = env::value(MIOPEN_DEVICE_ARCH);
    if(!arch.empty())
    {
        MIOPEN_THROW("MIOPEN_DEVICE_ARCH used, escaping launching kernel");
    }

    HipEventPtr start = nullptr;
    HipEventPtr stop  = nullptr;

    if(callback)
    {
        start = make_hip_event();
        stop  = make_hip_event();
    }

#if WORKAROUND_SWDEV_448157
    if(gdims[0] >= (1ULL << 32) || gdims[1] >= (1ULL << 32) || gdims[2] >= (1ULL << 32))
        MIOPEN_THROW("gridDim x blockDim >= 2^32");

    if(gdims[0] % ldims[0] != 0 || gdims[1] % ldims[1] != 0 || gdims[2] % ldims[2] != 0)
        MIOPEN_THROW(miopenStatusInternalError);

    unsigned grid_dim_x = gdims[0] / ldims[0];
    unsigned grid_dim_y = gdims[1] / ldims[1];
    unsigned grid_dim_z = gdims[2] / ldims[2];

    MIOPEN_HANDLE_LOCK

    if(callback)
    {
        status = hipEventRecord(start.get(), stream);
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status, "hipEventRecord() failed");
    }

    status = hipModuleLaunchCooperativeKernel(fun,
                                              grid_dim_x,
                                              grid_dim_y,
                                              grid_dim_z,
                                              ldims[0],
                                              ldims[1],
                                              ldims[2],
                                              0,
                                              stream,
                                              kern_args);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed to launch kernel");

    if(callback)
    {
        status = hipEventRecord(stop.get(), stream);
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status, "hipEventRecord() failed");
    }
#else
#error "Doesn't work without workaround"
#endif // WORKAROUND_SWDEV_448157

    if(callback)
    {
        hipEventSynchronize(stop.get());
        callback(start.get(), stop.get());
    }
}

HIPOCKernelInvoke HIPOCKernel::Invoke(hipStream_t stream,
                                      std::function<void(hipEvent_t, hipEvent_t)> callback,
                                      bool coop_launch) const
{
    return HIPOCKernelInvoke{stream, fun, ldims, gdims, name, callback, coop_launch};
}
} // namespace miopen
