
#include <mlopen/hipoc_kernel.hpp>
#include <mlopen/errors.hpp>

namespace mlopen {

void HIPOCKernelInvoke::run(void* args, std::size_t size) const
{
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    void *config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
        HIP_LAUNCH_PARAM_END
    };
    if (callback)
    {
        hipEventCreate(&start);
        hipEventCreate(&stop);

        hipEventRecord(start, nullptr);
    }

    // std::cerr << "Launch kernel: " << name << std::endl;
    auto status = hipModuleLaunchKernel(fun, gdims[0], gdims[1], gdims[2], ldims[0], ldims[1], ldims[2], 0, stream, nullptr, (void**)&config);
    if(status != hipSuccess) MLOPEN_THROW_HIP_STATUS(status, "Failed to launch kernel");

    if (callback)
    {
        hipEventRecord(stop, nullptr);
        hipEventSynchronize(stop);
        callback(start, stop);
    }
}

HIPOCKernelInvoke HIPOCKernel::Invoke(hipStream_t stream, std::function<void(hipEvent_t, hipEvent_t)> callback)
{
    return HIPOCKernelInvoke{stream, fun, ldims, gdims, name, callback};
}
}
