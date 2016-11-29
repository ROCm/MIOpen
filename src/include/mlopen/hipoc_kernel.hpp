#ifndef GUARD_MLOPEN_HIPOC_KERNEL_HPP
#define GUARD_MLOPEN_HIPOC_KERNEL_HPP

#include <mlopen/hipoc_program.hpp>

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
    {}
    uint64_t hidden[6];
    KernelArgsPack<Ts...> pack;
};

struct HIPOCKernel
{
    HIPOCProgram program;
    std::string name;
    std::vector<size_t> ldims;
    std::vector<size_t> gdims;

    HIPOCKernel()
    {}
    HIPOCKernel(HIPOCProgram p, const std::string kernel_name, std::vector<size_t> local_dims, std::vector<size_t> global_dims)
    : program(p), name(kernel_name), ldims(local_dims), gdims(3, 1)
    {
        for(int i=0;i<global_dims.size();i++)
        {
            gdims[i] = (global_dims[i] - 1)/local_dims[i] + 1;
        }
    }

    template<class... Ts>
    void operator()(Ts... xs) const
    {
        KernelArgs<Ts...> args{xs...};
        run(&args, sizeof(args));
    }

    void run(void* args, std::size_t size) const;
};

typedef HIPOCKernel HIPOCKernelInvoke;

}

#endif
