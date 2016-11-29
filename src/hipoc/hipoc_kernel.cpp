
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

    std::string kernel_module = "&__OpenCL_" + name + "_kernel";

    hipFunction_t fun;
    if (hipSuccess != hipModuleGetFunction(&fun, program.module.get(), kernel_module.c_str()))
        MLOPEN_THROW("Failed to get function");
    hipModuleLaunchKernel(fun, gdims[0], gdims[1], gdims[2], ldims[0], ldims[1], ldims[2], 0, 0, nullptr, (void**)&config);
}
}
