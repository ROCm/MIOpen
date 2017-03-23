#ifndef GUARD_MLOPEN_HIPOC_KERNEL_HPP
#define GUARD_MLOPEN_HIPOC_KERNEL_HPP

#include <mlopen/hipoc_program.hpp>
#include <mlopen/errors.hpp>
#include <array>
#include <vector>
#include <cassert>

namespace mlopen {

template<class... Ts>
struct KernelArgsPack;

template<class T, class... Ts>
struct KernelArgsPack<T, Ts...>
{
    KernelArgsPack(T x, Ts... xs)
    : head(x), tail(xs...)
    {}
    T head;
    KernelArgsPack<Ts...> tail;
};

template<>
struct KernelArgsPack<>
{};

template<class... Ts>
struct KernelArgs
{
    KernelArgs(Ts... xs)
    : pack(xs...)
    {
        for(int i=0;i<6;i++) hidden[i] = 0;
    }
    uint64_t hidden[6];
    KernelArgsPack<Ts...> pack;
};

struct HIPOCKernelInvoke
{
    hipStream_t stream;
    hipFunction_t fun;
    std::array<size_t, 3> ldims;
    std::array<size_t, 3> gdims;
    std::string name;
    std::function<void(hipEvent_t, hipEvent_t)> callback;

    template<class... Ts>
    void operator()(Ts... xs) const
    {
        KernelArgs<Ts...> args{xs...};
        run(&args, sizeof(args));
    }

    void run(void* args, std::size_t size) const;

    const std::string& GetName() const
    {
        return name;
    }
};

struct HIPOCKernel
{
    HIPOCProgram program;
    std::string name;
    std::array<size_t, 3> ldims;
    std::array<size_t, 3> gdims;
    std::string kernel_module;
    hipFunction_t fun;

    HIPOCKernel()
    {}
    HIPOCKernel(HIPOCProgram p, const std::string kernel_name, std::vector<size_t> local_dims, std::vector<size_t> global_dims)
    : program(p), name(kernel_name)
    {
        assert(!local_dims.empty() && local_dims.size() <= 3);
        assert(!global_dims.empty() && global_dims.size() <= 3);
        ldims.fill(1);
        gdims.fill(1);
        std::copy(local_dims.begin(), local_dims.end(), ldims.begin());
        for(int i=0;i<global_dims.size();i++)
        {
            gdims[i] = (global_dims[i] - 1)/ldims[i] + 1;
            if(global_dims[i] != (gdims[i] * ldims[i]))
            {
                std::cerr 
                    << "Warning: Extra read guard is needed for kernel " 
                    << kernel_name << " as global_dims are not equal(" 
                    << global_dims[i] << " != " << (gdims[i] * ldims[i]) 
                    << ")" << std::endl;
            }
        }

        kernel_module = "&__OpenCL_" + name + "_kernel";
        auto status = hipModuleGetFunction(&fun, program.module.get(), kernel_module.c_str());
        if (hipSuccess != status)
            MLOPEN_THROW_HIP_STATUS(status, "Failed to get function: " + kernel_module);
    }

    HIPOCKernelInvoke Invoke(hipStream_t stream, std::function<void(hipEvent_t, hipEvent_t)> callback=nullptr);
};

}

#endif
