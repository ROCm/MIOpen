
#include <mlopen/hipoc_kernel.hpp>
#include <mlopen/errors.hpp>

namespace mlopen {

void HIPOCKernel::run(void* args, std::size_t size) const
{
    void *config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
        HIP_LAUNCH_PARAM_END
    };

    // hipModuleLaunchKernel(Function, g_id[0], g_id[1], g_id[2], l_id[0], l_id[1], l_id[2], 0, 0, NULL, config);
    auto status = hipModuleLaunchKernel(fun, gdims[0], gdims[1], gdims[2], ldims[0], ldims[1], ldims[2], 0, 0, nullptr, (void**)&config);
    if(status != hipSuccess) MLOPEN_THROW_HIP_STATUS(status, "Failed to launch kernel");
}
}
