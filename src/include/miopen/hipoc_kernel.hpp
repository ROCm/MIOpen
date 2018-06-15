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

#include <array>
#include <cassert>
#include <miopen/errors.hpp>
#include <miopen/hipoc_program.hpp>
#include <miopen/stringutils.hpp>
#include <vector>
#include <boost/spirit/home/support/detail/hold_any.hpp>

namespace miopen {

using HipEventPtr = MIOPEN_MANAGE_PTR(hipEvent_t, hipEventDestroy);
inline HipEventPtr make_hip_event()
{
    hipEvent_t result = nullptr;
    hipEventCreate(&result);
    return HipEventPtr{result};
}

#if 1

#if 1
template <class T, class U>
struct KernelArgsPair
{
    static const int alignment    = sizeof(U);
    static const int padding      = (alignment - (sizeof(T) % alignment)) % alignment;
    static const int second_index = sizeof(T) + padding;
    KernelArgsPair(T x, U y)
    {
        new(buffer) T(x);
        new(buffer + second_index) U(y);
    }
    char buffer[second_index + sizeof(U)] = {};
};
#else
template <class T, class U>
struct KernelArgsPair
{
    KernelArgsPair(T x, U y) : first(x), second(y) {}
    T first;
    U second;
};
#endif

template <class... Ts>
struct KernelArgsPack;

template <class T, class U, class... Ts>
struct KernelArgsPack<T, U, Ts...>
{
    using data_t = KernelArgsPack<KernelArgsPair<T, U>, Ts...>;
    KernelArgsPack(T x, U y, Ts... xs) : data(KernelArgsPair<T, U>(x, y), xs...)
    {
        std::cout << "T: " << typeid(T).name() << " U: " << typeid(U).name() << "\n";
    }
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
    KernelArgs(Ts... xs) : pack(xs...)
    {
        for(auto& x : hidden)
            x = 0;
    }
    KernelArgsPack<Ts...> pack;
    uint64_t hidden[6] = {};
};

struct HIPOCKernelInvoke
{
    hipStream_t stream = nullptr;
    hipFunction_t fun  = nullptr;
    std::array<size_t, 3> ldims = {};
    std::array<size_t, 3> gdims = {};
    std::string name;
    std::function<void(hipEvent_t, hipEvent_t)> callback;

    // Workaround for aggregate types in c++11
    HIPOCKernelInvoke() {}
    HIPOCKernelInvoke(hipStream_t pstream,
                      hipFunction_t pfun,
                      std::array<size_t, 3> pldims,
                      std::array<size_t, 3> pgdims,
                      std::string pname,
                      std::function<void(hipEvent_t, hipEvent_t)> pcallback)
        : stream(pstream), fun(pfun), ldims(pldims), gdims(pgdims), name(pname), callback(pcallback)
    {
    }

    auto sizeof_any(boost::spirit::hold_any& a) const
    {
        if(a.type() == typeid(float))
            return sizeof(float);
        else if(a.type() == typeid(int))
            return sizeof(int);
        else if(a.type() == typeid(char))
            return sizeof(char);
        else if(a.type() == typeid(long long))
            return sizeof(long long);
        else if(a.type() == typeid(double))
            return sizeof(double);
        else if(a.type() == typeid(void*))
            return sizeof(void*);
        else
            MIOPEN_THROW("Unknown type in any container");
    }

    void copy_arg(boost::spirit::hold_any& arg, char* buf, size_t offset) const
    {
        if(arg.type() == typeid(float))
        {
            auto orig_arg = boost::spirit::any_cast<float>(arg);
            memcpy(buf + offset, &orig_arg, sizeof(float));
        }
        else if(arg.type() == typeid(int))
        {
            auto orig_arg = boost::spirit::any_cast<int>(arg);
            memcpy(buf + offset, &orig_arg, sizeof(int));
        }
        else if(arg.type() == typeid(char))
        {
            auto orig_arg = boost::spirit::any_cast<char>(arg);
            memcpy(buf + offset, &orig_arg, sizeof(char));
        }
        else if(arg.type() == typeid(long long))
        {
            auto orig_arg = boost::spirit::any_cast<long long>(arg);
            memcpy(buf + offset, &orig_arg, sizeof(long long));
        }
        else if(arg.type() == typeid(double))
        {
            auto orig_arg = boost::spirit::any_cast<double>(arg);
            memcpy(buf + offset, &orig_arg, sizeof(double));
        }
        else if(arg.type() == typeid(void*))
        {
            auto orig_arg = boost::spirit::any_cast<void*>(arg);
            memcpy(buf + offset, &orig_arg, sizeof(void*));
        }
        else
            MIOPEN_THROW("");
    }

    void operator()(std::vector<boost::spirit::hold_any>& any_args) const
    {
        char hip_args[256] = {0};
        auto sz_left       = sizeof_any(any_args[0]);
        int alignment, padding, second_index;
        copy_arg(any_args[0], hip_args, 0);

        for(auto idx = 1; idx < any_args.size(); idx++)
        {
            auto any_arg = any_args[idx];
            alignment    = sizeof_any(any_arg);
            padding      = (alignment - (sz_left % alignment)) % alignment;
            second_index = sz_left + padding;
            copy_arg(any_arg, hip_args, second_index);
            sz_left = second_index + alignment;
        }
        run(hip_args, sz_left);
    }

    template <class... Ts>
    void operator()(Ts... xs) const
    {
        KernelArgs<Ts...> args{xs...};
        run(&args, sizeof(args));
    }

    void run(void* args, std::size_t size) const;

    const std::string& GetName() const { return name; }
};

struct HIPOCKernel
{
    HIPOCProgram program;
    std::string name;
    std::array<size_t, 3> ldims = {};
    std::array<size_t, 3> gdims = {};
    std::string kernel_module;
    hipFunction_t fun = nullptr;

    HIPOCKernel() {}
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
            MIOPEN_THROW_HIP_STATUS(status,
                                    "Failed to get function: " + kernel_module + " from " +
                                        program.GetBinary().string());
    }

    HIPOCKernelInvoke Invoke(hipStream_t stream,
                             std::function<void(hipEvent_t, hipEvent_t)> callback = nullptr);
};

} // namespace miopen

#endif
