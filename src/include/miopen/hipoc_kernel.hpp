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
#ifndef GUARD_MIOPEN_HIPOC_KERNEL_HPP
#define GUARD_MIOPEN_HIPOC_KERNEL_HPP

#include <miopen/config.hpp>
#include <miopen/errors.hpp>
#include <miopen/hipoc_program.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/op_kernel_args.hpp>

#include <array>
#include <cassert>
#include <cstring>
#include <vector>

namespace miopen {

using HipEventPtr = MIOPEN_MANAGE_PTR(hipEvent_t, hipEventDestroy);
inline HipEventPtr make_hip_event()
{
    hipEvent_t result = nullptr;
    hipEventCreate(&result);
    return HipEventPtr{result};
}

struct HipEventProfiler
{
    const Handle& handle;
    HipEventPtr start;
    HipEventPtr stop;

    HipEventProfiler(const Handle& handle_);
    ~HipEventProfiler();
};

#if 1 // Keep around other storage techinques -- @pfultz2 27.03.2017

template <class T, class U>
struct KernelArgsPair
{
    constexpr static auto alignU       = alignof(U);
    constexpr static auto padding      = (alignU - (sizeof(T) % alignU)) % alignU;
    constexpr static auto second_index = sizeof(T) + padding;
    KernelArgsPair(T x, U y)
    {
        new(buffer) T(x); // NOLINT (clang-analyzer-cplusplus.PlacementNew)
        new(buffer + second_index) U(y);
    }

    alignas(U) char buffer[second_index + sizeof(U)] = {};
};

template <class... Ts>
struct KernelArgsPack;

template <class T, class U, class... Ts>
struct KernelArgsPack<T, U, Ts...>
{
    using data_t = KernelArgsPack<KernelArgsPair<T, U>, Ts...>;
    KernelArgsPack(T x, U y, Ts... xs) : data(KernelArgsPair<T, U>(x, y), xs...) {}
    data_t data;
};

template <class T>
struct KernelArgsPack<T>
{
    KernelArgsPack(T x) : head(x) {}
    T head;
};

#else

template <class... Ts>
struct KernelArgsPack;

template <class T, class... Ts>
struct KernelArgsPack<T, Ts...>
{
    KernelArgsPack(T x, Ts... xs) : head(x), tail(xs...) {}
    T head;
    KernelArgsPack<Ts...> tail;
};

template <>
struct KernelArgsPack<>
{
};

#endif
template <class... Ts>
struct KernelArgs
{
    KernelArgs(Ts... xs) : pack(xs...) { std::fill(std::begin(hidden), std::end(hidden), 0); }
    KernelArgsPack<Ts...> pack;
    uint64_t hidden[6] = {};
};

struct MIOPEN_INTERNALS_EXPORT HIPOCKernelInvoke
{
    HIPOCKernelInvoke() {}
    HIPOCKernelInvoke(hipStream_t pstream,
                      hipFunction_t pfun,
                      std::array<size_t, 3> pldims,
                      std::array<size_t, 3> pgdims,
                      std::string pname,
                      std::function<void(hipEvent_t, hipEvent_t)> pcallback,
                      bool pcoop_launch)
        : stream(pstream),
          fun(pfun),
          ldims(pldims),
          gdims(pgdims),
          name(pname),
          callback(pcallback),
          coop_launch(pcoop_launch)
    {
    }

    void operator()(std::vector<OpKernelArg>& any_args) const
    {
        if(coop_launch)
            MIOPEN_THROW(miopenStatusNotImplemented);

        char hip_args[256] = {0};
        auto sz_left       = any_args[0].size();

        memcpy(hip_args, &(any_args[0].buffer[0]), any_args[0].size());
        //        copy_arg(any_args[0], hip_args, 0);

        for(std::size_t idx = 1; idx < any_args.size(); idx++)
        {
            auto& any_arg            = any_args[idx];
            std::size_t alignment    = any_arg.size();
            std::size_t padding      = (alignment - (sz_left % alignment)) % alignment;
            std::size_t second_index = sz_left + padding;
            memcpy(hip_args + second_index, &(any_arg.buffer[0]), any_arg.size());
            // copy_arg(any_arg, hip_args, second_index);
            sz_left = second_index + alignment;
        }
        run(hip_args, sz_left);
    }

    template <class... Ts>
    void operator()(Ts... xs) const
    {
        if(coop_launch)
        {
            auto args = std::array<void*, sizeof...(xs)>{(&xs)...};
            run_cooperative(args.data());
        }
        else
        {
            KernelArgs<Ts...> args{xs...};
            run(&args, sizeof(args));
        }
    }

    void SetLocalDims(size_t dim_x, size_t dim_y, size_t dim_z) { ldims = {dim_x, dim_y, dim_z}; }

    void SetGlobalDims(size_t dim_x, size_t dim_y, size_t dim_z) { gdims = {dim_x, dim_y, dim_z}; }

    const std::string& GetName() const { return name; }

private:
    void run(void* args, std::size_t size) const;
    void run_cooperative(void** kern_args) const;

    hipStream_t stream          = nullptr;
    hipFunction_t fun           = nullptr;
    std::array<size_t, 3> ldims = {};
    std::array<size_t, 3> gdims = {};
    std::string name;
    std::function<void(hipEvent_t, hipEvent_t)> callback;
    bool coop_launch;
};

struct MIOPEN_INTERNALS_EXPORT HIPOCKernel
{
    HIPOCProgram program;
    std::string name;
    std::array<size_t, 3> ldims = {};
    std::array<size_t, 3> gdims = {};
    std::string kernel_module;
    hipFunction_t fun = nullptr;

    HIPOCKernel() {}
    HIPOCKernel(HIPOCProgram p, const std::string kernel_name) : program(p), name(kernel_name) {}
    HIPOCKernel(HIPOCProgram p,
                const std::string kernel_name,
                std::vector<size_t> local_dims,
                std::vector<size_t> global_dims)
        : program(p), name(kernel_name)
    {
        assert(!local_dims.empty() && local_dims.size() <= 3);
        assert(!global_dims.empty() && global_dims.size() <= 3);
        ldims.fill(1);
        gdims.fill(1);
        std::copy(local_dims.begin(), local_dims.end(), ldims.begin());
        std::copy(global_dims.begin(), global_dims.end(), gdims.begin());

        kernel_module = name;
        auto status   = hipModuleGetFunction(&fun, program.GetModule(), kernel_module.c_str());
        if(hipSuccess != status)
        {
            MIOPEN_THROW_HIP_STATUS(status,
                                    "Failed to get function: " + kernel_module + " from " +
                                        program.GetCodeObjectPathname());
        }
    }

    HIPOCKernelInvoke Invoke(hipStream_t stream,
                             std::function<void(hipEvent_t, hipEvent_t)> callback = nullptr,
                             bool coop_launch                                     = false) const;
};

} // namespace miopen

#endif
