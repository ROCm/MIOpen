/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#pragma once

#if MIOPEN_BACKEND_OPENCL
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#elif MIOPEN_BACKEND_HIP
#include <hip/hip_runtime_api.h>
#endif
#include <miopen/tensor_holder.hpp>
#include <miopen/rocrand_wrapper.hpp>
#include <miopen/logger.hpp>

#if MIOPEN_BACKEND_OPENCL
#define STATUS_SUCCESS CL_SUCCESS
typedef cl_int status_t;
typedef cl_context context_t;
#define DEFINE_CONTEXT(name) context_t name
typedef cl_command_queue stream;
#elif MIOPEN_BACKEND_HIP
#define STATUS_SUCCESS 0
typedef int status_t;
typedef uint32_t context_t;
#define DEFINE_CONTEXT(name) context_t name = 0
typedef hipStream_t stream;
#else // Unknown backend.
// No definitions -> build errors if used.
#endif

struct GPUMem
{

#if MIOPEN_BACKEND_OPENCL
    GPUMem(){};
    GPUMem(cl_context& ctx, size_t psz, size_t pdata_sz) : sz(psz), data_sz(pdata_sz)
    {
        buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, data_sz * sz, nullptr, nullptr);
    }

    int ToGPU(cl_command_queue& q, void* p) const
    {
        return clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, data_sz * sz, p, 0, nullptr, nullptr);
    }
    int FromGPU(cl_command_queue& q, void* p) const
    {
        return clEnqueueReadBuffer(q, buf, CL_TRUE, 0, data_sz * sz, p, 0, nullptr, nullptr);
    }

    cl_mem GetMem() const { return buf; }
    size_t GetSize() const { return sz * data_sz; }

    ~GPUMem() { clReleaseMemObject(buf); }

    cl_mem buf;
    size_t sz;
    size_t data_sz;

#elif MIOPEN_BACKEND_HIP

    GPUMem(){};
    GPUMem(uint32_t ctx, size_t psz, size_t pdata_sz) : _ctx(ctx), sz(psz), data_sz(pdata_sz)
    {
        auto status = hipMalloc(static_cast<void**>(&buf), GetSize());
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status,
                                    "[MIOpenDriver] hipMalloc " + std::to_string(GetSize()));
        MIOPEN_LOG_CUSTOM(miopen::LoggingLevel::Info2,
                          "MIOpenDriver",
                          "hipMalloc " << GetSize() << " at " << buf << " Ok");
    }

    int ToGPU(hipStream_t q, void* p)
    {
        _q = q;
        return static_cast<int>(hipMemcpy(buf, p, GetSize(), hipMemcpyHostToDevice));
    }
    int FromGPU(hipStream_t q, void* p)
    {
        hipDeviceSynchronize();
        _q = q;
        return static_cast<int>(hipMemcpy(p, buf, GetSize(), hipMemcpyDeviceToHost));
    }

    void* GetMem() { return buf; }
    size_t GetSize() { return sz * data_sz; }

    ~GPUMem()
    {
        size_t size = 0;
        auto status = hipMemPtrGetInfo(buf, &size);
        if(status != hipSuccess)
            MIOPEN_LOG_CUSTOM(miopen::LoggingLevel::Warning,
                              "MIOpenDriver",
                              "hipMemPtrGetInfo at " << buf << ' '
                                                     << miopen::HIPErrorMessage(status, ""));
        status = hipFree(buf);
        if(status != hipSuccess)
            MIOPEN_LOG_CUSTOM(miopen::LoggingLevel::Error,
                              "MIOpenDriver",
                              "hipFree " << size << " at " << buf << ' '
                                         << miopen::HIPErrorMessage(status, ""));
        else
            MIOPEN_LOG_CUSTOM(miopen::LoggingLevel::Info2,
                              "MIOpenDriver",
                              "hipFree " << size << " at " << buf << " Ok");
    }

    hipStream_t _q; // Place holder for opencl context
    uint32_t _ctx;
    void* buf;
    size_t sz;
    size_t data_sz;
#endif
};

template <typename Tgpu>
class GpumemTensor
{
    std::unique_ptr<GPUMem> dev;
    tensor<Tgpu> host;
    bool is_gpualloc = false;

public:
    void SetGpuallocMode(bool v) { is_gpualloc = v; }
    tensor<Tgpu>& GetTensor() { return host; }

    void AllocOnHost(miopenTensorDescriptor_t t)
    {
        host = tensor<Tgpu>(miopen::deref(t));
        if(is_gpualloc) // We do not need host data.
        {
            host.data.clear();
            host.data.shrink_to_fit(); // To free host memory.
        }
    }
    template <typename T>
    void AllocOnHost(tensor<T> t)
    {
        AllocOnHost(&t.desc);
    }

    std::vector<Tgpu>& GetVector()
    {
        if(is_gpualloc)
            MIOPEN_THROW("[MIOpenDriver] GpumemTensor::GetVector should not be called in "
                         "'--gpualloc 1' mode");
        return host.data;
    }

    Tgpu* GetVectorData() { return is_gpualloc ? nullptr : host.data.data(); }
    std::size_t GetVectorSize() const { return is_gpualloc ? 0 : host.data.size(); }

    void
    InitHostData(const size_t sz,     //
                 const bool do_write, // If set to false, then only generate random data. This is
                                      // necessary to reproduce values in input buffers even if some
                                      // directions are skipped. For example, inputs for Backward
                                      // will be the same for both "-F 0" and "-F 2".
                 std::function<Tgpu()> generator)
    {
        if(is_gpualloc)
        {
            /// In gpualloc mode, we do not care about reproducibility of results, because
            /// validation is not used. Therefore, we do not have to always generate random value
            /// (\ref move_rand)
            return;
        }

        for(size_t i = 0; i < sz; ++i)
        {
            /// \anchor move_rand
            /// Generate random value, even if buffer is unused. This provides the same
            /// initialization of input buffers regardless of which kinds of
            /// convolutions are currently selectedfor testing (see the "-F" option).
            /// Verification cache would be broken otherwise.
            auto val = generator();
            if(do_write)
                GetVector()[i] = val;
        }
    }

    status_t AllocOnDevice(stream, context_t ctx, const size_t sz)
    {
        dev = std::make_unique<GPUMem>(ctx, sz, sizeof(Tgpu));
        return STATUS_SUCCESS;
    }

    status_t AllocOnDeviceAndInit(stream q, context_t ctx, const size_t sz)
    {
        AllocOnDevice(q, ctx, sz);
        if(is_gpualloc)
        {
            /// \anchor gpualloc_random_init
            /// In gpualloc mode, we do not want to leave input buffers uninitialized, because
            /// there could be NaNs and Infs, which may affect the performance (which we are
            /// interested to evaluate in this mode). Initialization with all 0's is not the
            /// best choice as well, because GPU HW may optimize out computations with 0's and
            /// that could affect performance of kernels too. That is why we are using
            /// rocrand to initialize input buffers.
            ///
            /// However we do not care about precision in gpualloc mode, because validation
            /// is not used. Therefore, range (0,1] is fine.
            return gpumemrand::gen_0_1(static_cast<Tgpu*>(GetDevicePtr()), sz);
        }
        return dev->ToGPU(q, GetVectorData());
    }

    template <typename T>
    status_t AllocOnDevice(stream, context_t ctx, const size_t sz, std::vector<T>&)
    {
        static_assert(std::is_same<T, float>::value           //
                          || std::is_same<T, int32_t>::value, //
                      "Before enabling more types, check thoroughly.");
        dev = std::make_unique<GPUMem>(ctx, sz, sizeof(T));
        return STATUS_SUCCESS;
    }

    template <typename T>
    status_t AllocOnDeviceAndInit(stream q, context_t ctx, const size_t sz, std::vector<T>& init)
    {
        AllocOnDevice(q, ctx, sz, init);
        if(is_gpualloc)
        {
            /// \ref gpualloc_random_init
            return gpumemrand::gen_0_1(static_cast<Tgpu*>(GetDevicePtr()), sz);
        }
        return dev->ToGPU(q, init.data());
    }

    status_t CopyFromDeviceToHost(stream q)
    {
        return is_gpualloc ? STATUS_SUCCESS : dev->FromGPU(q, GetVectorData());
    }

    template <typename T>
    status_t CopyFromDeviceToHost(stream q, tensor<T>& t)
    {
        return is_gpualloc ? STATUS_SUCCESS : dev->FromGPU(q, t.data.data());
    }

    template <typename T>
    status_t CopyFromDeviceToHost(stream q, std::vector<T>& v)
    {
        return is_gpualloc ? STATUS_SUCCESS : dev->FromGPU(q, v.data());
    }

    auto GetDevicePtr() -> auto { return dev->GetMem(); }
};
