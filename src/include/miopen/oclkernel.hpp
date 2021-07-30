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
#ifndef GUARD_MIOPEN_OCL_KERNEL_HPP_
#define GUARD_MIOPEN_OCL_KERNEL_HPP_

#include <algorithm>
#include <array>
#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <miopen/miopen.h>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

#include <miopen/clhelper.hpp>
#include <miopen/each_args.hpp>
#include <miopen/errors.hpp>
#include <miopen/op_kernel_args.hpp>

namespace miopen {

using SharedKernelPtr  = std::shared_ptr<typename std::remove_pointer<cl_kernel>::type>;
using SharedProgramPtr = std::shared_ptr<typename std::remove_pointer<cl_program>::type>;

struct LocalMemArg
{
    LocalMemArg(size_t _size) : size(_size) {}
    size_t GetSize() const { return size; }

    private:
    size_t size;
};

struct OCLSetKernelArg
{
    template <class I, class T>
    void operator()(cl_kernel kernel, I i, const T& x) const
    {
        cl_int status =
            clSetKernelArg(kernel, i, sizeof(T), reinterpret_cast<const void*>(&x)); // NOLINT
        if(status != CL_SUCCESS)
        {
            MIOPEN_THROW("Error setting argument #" + std::to_string(i) + " to kernel (size = " +
                         std::to_string(sizeof(T)) + "): " + OpenCLErrorMessage(status)); // NOLINT
        }
    }

    template <class I>
    void operator()(cl_kernel kernel, I i, const LocalMemArg& lmem) const
    {
        cl_int status = clSetKernelArg(kernel, i, lmem.GetSize(), NULL);
        if(status != CL_SUCCESS)
        {
            MIOPEN_THROW("Error setting argument #" + std::to_string(i) +
                         " to kernel: " + OpenCLErrorMessage(status));
        }
    }
};

struct OCLKernelInvoke
{
    cl_command_queue queue                   = nullptr;
    SharedKernelPtr kernel                   = nullptr;
    size_t work_dim                          = 0;
    std::array<size_t, 3> global_work_offset = {};
    // std::array<size_t, 3> global_work_dim    = {};
    // std::array<size_t, 3> local_work_dim     = {};
    std::array<size_t, 3> gdims = {};
    std::array<size_t, 3> ldims = {};
    std::function<void(cl_event&)> callback;

    void operator()(std::vector<OpKernelArg> args) const
    {
        for(size_t idx = 0; idx < args.size(); idx++)
        {
            auto arg      = args[idx];
            cl_int status = clSetKernelArg(
                kernel.get(), idx, arg.size(), reinterpret_cast<const void*>(&arg.buffer[0]));
            if(status != CL_SUCCESS)
            {
                MIOPEN_THROW("Error setting argument #" + std::to_string(idx) +
                             " to kernel (size = " + std::to_string(arg.size()) +
                             "): " + OpenCLErrorMessage(status));
            }
        }
        run();
    }

    template <class... Ts>
    void operator()(const Ts&... xs) const
    {
        each_args_i(
            std::bind(
                OCLSetKernelArg{}, kernel.get(), std::placeholders::_1, std::placeholders::_2),
            xs...);
        run();
    }

    void run() const;
    std::string GetName() const;
};

class OCLKernel
{

    public:
    OCLKernel() {}
    OCLKernel(ClKernelPtr k) : kernel(std::move(k)) {}
    OCLKernel(ClKernelPtr k, std::vector<size_t> local_dims, std::vector<size_t> global_dims)
        : kernel(std::move(k)), ldims(std::move(local_dims)), gdims(std::move(global_dims))
    {
        assert(ldims.size() == gdims.size());
        assert(!ldims.empty() && ldims.size() <= 3);
    }

    OCLKernel(SharedProgramPtr p,
              const std::string& kernel_name,
              std::vector<size_t> local_dims,
              std::vector<size_t> global_dims)
        : program(p),
          kernel(CreateKernel(p.get(), kernel_name)),
          ldims(std::move(local_dims)),
          gdims(std::move(global_dims))
    {
        assert(!gdims.empty() && gdims.size() <= 3);
        assert(!ldims.empty() && ldims.size() <= 3);
        if(std::accumulate(ldims.begin(), ldims.end(), 1, std::multiplies<size_t>{}) >
           256) // FIXME: get ldims limit from runtime
        {
            std::fill(ldims.begin(), ldims.end(), 0);
        }
    }

    OCLKernel(SharedProgramPtr p, const std::string& kernel_name)
        : program(p), kernel(CreateKernel(p.get(), kernel_name))
    {
    }

    OCLKernelInvoke Invoke(cl_command_queue q,
                           std::function<void(cl_event&)> callback = nullptr) const;

    cl_kernel GetKernel() { return kernel.get(); }

    std::string GetName() const;

    inline const std::vector<size_t>& GetLocalDims() const { return ldims; }
    inline const std::vector<size_t>& GetGlobalDims() const { return gdims; }

    private:
    SharedProgramPtr program;
    SharedKernelPtr kernel;
    std::vector<size_t> ldims;
    std::vector<size_t> gdims;
};

} // namespace miopen

#endif // GUARD_MIOPEN_OCL_KERNEL_HPP_
