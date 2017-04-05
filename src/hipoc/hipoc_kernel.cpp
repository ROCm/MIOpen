
#include <miopen/hipoc_kernel.hpp>
#include <miopen/errors.hpp>
#include <hip/hip_hcc.h>

namespace miopen {

void HIPOCKernelInvoke::run(void* args, std::size_t size) const
{
    HipEventPtr start = nullptr;
    HipEventPtr stop = nullptr;
    void *config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
        HIP_LAUNCH_PARAM_END
    };
    if (callback)
    {
        start = make_hip_event();
        stop = make_hip_event();

        hipEventRecord(start.get(), stream);
    }

    // std::cerr << "Launch kernel: " << name << std::endl;
    auto status = hipHccModuleLaunchKernel(fun, gdims[0], gdims[1], gdims[2], ldims[0], ldims[1], ldims[2], 0, stream, nullptr, (void**)&config);
    if(status != hipSuccess) MIOPEN_THROW_HIP_STATUS(status, "Failed to launch kernel");

    if (callback)
    {
        hipEventRecord(stop.get(), stream);
        hipEventSynchronize(stop.get());
        callback(start.get(), stop.get());
    }
}

HIPOCKernelInvoke HIPOCKernel::Invoke(hipStream_t stream, std::function<void(hipEvent_t, hipEvent_t)> callback)
{
    return HIPOCKernelInvoke{stream, fun, ldims, gdims, name, callback};
}
}
