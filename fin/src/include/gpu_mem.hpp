/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all
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
#ifndef GUARD_FIN_GPUMEM_HPP
#define GUARD_FIN_GPUMEM_HPP

#include "half.hpp"
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <numeric>
#include <vector>

#include <hip/hip_runtime_api.h>

namespace fin {

struct relMem
{
#if FIN_BACKEND_OPENCL
    void operator()(cl_mem ptr) { clReleaseMemObject(ptr); }
#elif FIN_BACKEND_HIP
    void operator()(void* ptr) { hipFree(ptr); }
#endif
};
#if FIN_BACKEND_OPENCL
using gpu_mem_ptr = std::unique_ptr<cl_mem, relMem>;
#elif FIN_BACKEND_HIP
using gpu_mem_ptr = std::unique_ptr<void, relMem>;
#endif

struct GPUMem
{

#if FIN_BACKEND_OPENCL
    GPUMem(){};
    GPUMem(cl_context& ctx, size_t psz, size_t pdata_sz) : sz(psz), data_sz(pdata_sz)
    {
        buf = gpu_mem_ptr{clCreateBuffer(ctx, CL_MEM_READ_WRITE, data_sz * sz, nullptr, nullptr)};
    }

    int ToGPU(cl_command_queue& q, void* p)
    {
        return clEnqueueWriteBuffer(q, buf.get(), CL_TRUE, 0, data_sz * sz, p, 0, nullptr, nullptr);
    }
    int FromGPU(cl_command_queue& q, void* p)
    {
        return clEnqueueReadBuffer(q, buf.get(), CL_TRUE, 0, data_sz * sz, p, 0, nullptr, nullptr);
    }

    cl_mem GetMem() { return buf.get(); }
    size_t GetSize() { return sz * data_sz; }

    // ~GPUMem() { clReleaseMemObject(buf); }

    gpu_mem_ptr buf;
#elif FIN_BACKEND_HIP

    GPUMem(){};
    GPUMem(uint32_t ctx, size_t psz, size_t pdata_sz) : _ctx(ctx), sz(psz), data_sz(pdata_sz)
    {
        void* tmp = nullptr;
        hipMalloc(static_cast<void**>(&(tmp)), data_sz * sz);
        if(tmp == nullptr && sz > 0)
            throw std::runtime_error("Unable to allocate GPU memory");
        buf = gpu_mem_ptr{tmp};
    }

    int ToGPU(hipStream_t q, void* p)
    {
        _q = q;
        return static_cast<int>(hipMemcpy(buf.get(), p, data_sz * sz, hipMemcpyHostToDevice));
    }
    int FromGPU(hipStream_t q, void* p)
    {
        hipDeviceSynchronize();
        _q = q;
        return static_cast<int>(hipMemcpy(p, buf.get(), data_sz * sz, hipMemcpyDeviceToHost));
    }

    void* GetMem() { return buf.get(); }
    size_t GetSize() { return sz * data_sz; }

    // ~GPUMem() { hipFree(buf); }
    hipStream_t _q; // Place holder for opencl context
    uint32_t _ctx;
    gpu_mem_ptr buf;
#endif
    size_t sz;
    size_t data_sz;
};

} // namespace fin
#endif // #define GUARD_FIN_GPUMEM_HPP
