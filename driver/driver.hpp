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
#ifndef GUARD_MIOPEN_DRIVER_HPP
#define GUARD_MIOPEN_DRIVER_HPP

#include <half/half.hpp>
#include "random.hpp"

#include "InputFlags.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <memory>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/bfloat16.hpp>
#include <../test/tensor_holder.hpp>
#include "util_driver.hpp"
#include "rocrand_wrapper.hpp"
using half         = half_float::half;
using hip_bfloat16 = bfloat16;
#include <hip_float8.hpp>
using float16      = half_float::half;
using float8_fnuz  = miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>;
using bfloat8_fnuz = miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>;
#include <numeric>
#include <vector>

#if MIOPEN_BACKEND_OPENCL
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#elif MIOPEN_BACKEND_HIP
#include <hip/hip_runtime_api.h>
#endif

#define UNPACK_VEC4(v) (v[0]), (v[1]), (v[2]), (v[3])

// Use values which are distinctively greater then miopenStatus_t,
// so that these can be ORed with any miopen status code
// without loss of information.
typedef enum
{
    // These four codes could be returned together, ORed:
    EC_VerifyFwd     = 0x100,
    EC_VerifyBwd     = 0x200,
    EC_VerifyWrw     = 0x400,
    EC_VerifyBwdBias = 0x800,
} errorCode_t;

struct GPUMem
{
    enum class Check
    {
        None,
        Front,
        Back,
    };

#if MIOPEN_BACKEND_OPENCL
    GPUMem(){};
    GPUMem(cl_context& ctx, size_t psz, size_t pdata_sz, Check ch = Check::None)
        : sz(psz), data_sz(pdata_sz)
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
    GPUMem(uint32_t ctx, size_t psz, size_t pdata_sz, Check ch = Check::None)
        : _ctx(ctx), sz(psz), data_sz(pdata_sz), check(ch)
    {
        auto status = hipMalloc(static_cast<void**>(&buf), GetTotalSize(GetSize()));
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status,
                                    "[MIOpenDriver] hipMalloc " + std::to_string(GetSize()));
        buf = static_cast<char*>(buf) + GetOffsetToUserBuffer();
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

    template <typename Tgpu>
    status_t FillBufferWithNans(miopenHandle_t handle, const miopenTensorDescriptor_t tensorDesc)
    {
        // In the past we have had some issues with incorrect results due to Nans in the output
        // buffers.  In order to test the clearing of the output buffers, you can
        // init the buffers with NaNs.

        if(std::is_same<Tgpu, int8_t>::value)
        {
            // ints dont have Nan so use max value.
            Tgpu max = std::numeric_limits<Tgpu>::max();
            miopenSetTensor(handle, tensorDesc, GetMem(), &max);
        }
        else
        {
            Tgpu nan = std::numeric_limits<Tgpu>::quiet_NaN();
            miopenSetTensor(handle, tensorDesc, GetMem(), &nan);
        }

        return STATUS_SUCCESS;
    }

    void* GetMem() { return buf; }
    size_t GetSize() { return sz * data_sz; }

    size_t GetTotalSize(size_t userSize)
    {
        if(check == Check::None)
            return userSize;

        constexpr size_t maxPadding = 2ULL * 1024 * 1024 - 1;

        auto roundUpToPageAlignment = [&](size_t bytes) {
            return (bytes + maxPadding) & ~maxPadding;
        };

        return roundUpToPageAlignment(userSize);
    }

    size_t GetOffsetToUserBuffer()
    {
        if(check == Check::Back)
        {
            auto userSize = GetSize();
            return GetTotalSize(userSize) - userSize;
        }
        return 0;
    }

    ~GPUMem()
    {
        buf = static_cast<char*>(buf) - GetOffsetToUserBuffer();

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
    Check check;
#endif
};

template <typename Tgpu>
class GpumemTensor
{
    std::unique_ptr<GPUMem> dev;
    tensor<Tgpu> host;
    bool is_gpualloc         = false;
    bool init_gpu_output_nan = false;

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

    status_t FillGpuBufferWithNans(miopenHandle_t handle, const miopenTensorDescriptor_t tensorDesc)
    {
        return dev->FillBufferWithNans<Tgpu>(handle, tensorDesc);
    }

    status_t
    AllocOnDevice(stream, context_t ctx, const size_t sz, GPUMem::Check check = GPUMem::Check::None)
    {
        dev = std::make_unique<GPUMem>(ctx, sz, sizeof(Tgpu), check);
        return STATUS_SUCCESS;
    }

    status_t AllocOnDeviceAndInit(stream q,
                                  context_t ctx,
                                  const size_t sz,
                                  GPUMem::Check check = GPUMem::Check::None)
    {
        AllocOnDevice(q, ctx, sz, check);
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
    status_t AllocOnDevice(stream,
                           context_t ctx,
                           const size_t sz,
                           std::vector<T>&,
                           GPUMem::Check check = GPUMem::Check::None)
    {
        static_assert(std::is_same<T, float>::value           //
                          || std::is_same<T, int32_t>::value, //
                      "Before enabling more types, check thoroughly.");
        dev = std::make_unique<GPUMem>(ctx, sz, sizeof(T), check);
        return STATUS_SUCCESS;
    }

    template <typename T>
    status_t AllocOnDeviceAndInit(stream q,
                                  context_t ctx,
                                  const size_t sz,
                                  std::vector<T>& init,
                                  GPUMem::Check check = GPUMem::Check::None)
    {
        AllocOnDevice(q, ctx, sz, init, check);
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

inline void PadBufferSize(size_t& sz, int datatype_sz)
{
    size_t page_sz = (2 * 1024 * 1024) / datatype_sz;
    if(sz % page_sz != 0)
    {
        sz = ((sz + page_sz) / page_sz) * page_sz;
    }
}

[[noreturn]] inline void Usage(int e)
{
    printf("Usage: ./driver *base_arg* *other_args*\n");
    printf("Supported Base Arguments: conv[fp16|int8|bfp16], CBAInfer[fp16|bfp16], "
           "CAInfer[fp16|bfp16], pool[fp16], lrn[fp16], "
           "activ[fp16], softmax[fp16], bnorm[fp16], rnn[fp16], gemm[fp16], ctc, dropout[fp16], "
           "tensorop, reduce[fp16|fp64], layernorm[bfp16|fp16], "
           "groupnorm[bfp16|fp16], cat[bfp16|fp16], addlayernorm[bfp16|fp16], "
           "t5layernorm[bfp16|fp16], adam[fp16], ampadam, reduceextreme[bfp16|fp16], "
           "adamw[fp16], ampadamw, transformersadamw[fp16], transformersampadamw, "
           "getitem[bfp16|fp16], reducecalculation[bfp16|fp16], rope[bfp16|fp16], "
           "prelu[bfp16|fp16], kthvalue[bfp16|fp16], glu[bfp16|fp16], softmarginloss[bfp16|fp16], "
           "multimarginloss[bfp16|fp16]\n");
    exit(e); // NOLINT (concurrency-mt-unsafe)
}

inline std::string ParseBaseArg(int argc, char* argv[])
{
    if(argc < 2)
    {
        printf("FAILED: Invalid Number of Input Arguments\n");
        Usage(EXIT_FAILURE);
    }

    std::string arg = argv[1];

    // List of valid base arguments
    static const std::vector<std::string> valid_args = {"conv",
                                                        "convfp16",
                                                        "convint8",
                                                        "convbfp16",
                                                        "CBAInfer",
                                                        "CBAInferfp16",
                                                        "CBAInferbfp16",
                                                        "CAInfer",
                                                        "CAInferfp16",
                                                        "CAInferbfp16",
                                                        "pool",
                                                        "poolfp16",
                                                        "lrn",
                                                        "lrnfp16",
                                                        "activ",
                                                        "activfp16",
                                                        "softmax",
                                                        "softmaxfp16",
                                                        "bnorm",
                                                        "bnormfp16",
                                                        "bnormbfp16",
                                                        "bnormfp16fp32",
                                                        "bnormbfp16fp32",
                                                        "rnn",
                                                        "rnnfp16",
                                                        "rnn_seq",
                                                        "rnn_seqfp16",
                                                        "gemm",
                                                        "gemmfp16",
                                                        "ctc",
                                                        "dropout",
                                                        "dropoutfp16",
                                                        "tensorop",
                                                        "reduce",
                                                        "reducefp16",
                                                        "reducefp64",
                                                        "layernorm",
                                                        "layernormfp16",
                                                        "layernormbfp16",
                                                        "groupnorm",
                                                        "groupnormfp16",
                                                        "groupnormbfp16",
                                                        "cat",
                                                        "catfp16",
                                                        "catbfp16",
                                                        "addlayernorm",
                                                        "addlayernormfp16",
                                                        "addlayernormbfp16",
                                                        "t5layernorm",
                                                        "t5layernormfp16",
                                                        "t5layernormbfp16",
                                                        "adam",
                                                        "adamfp16",
                                                        "ampadam",
                                                        "reduceextreme",
                                                        "reduceextremefp16",
                                                        "reduceextremebfp16",
                                                        "adamw",
                                                        "adamwfp16",
                                                        "ampadamw",
                                                        "transformersadamw",
                                                        "transformersadamwfp16",
                                                        "transformersampadamw",
                                                        "getitem",
                                                        "getitemfp16",
                                                        "getitembfp16",
                                                        "reducecalculation",
                                                        "reducecalculationfp16",
                                                        "reducecalculationbfp16",
                                                        "rope",
                                                        "ropefp16",
                                                        "ropebfp16",
                                                        "prelu",
                                                        "prelufp16",
                                                        "prelubfp16",
                                                        "kthvalue",
                                                        "kthvaluefp16",
                                                        "kthvaluebfp16",
                                                        "glu",
                                                        "glufp16",
                                                        "glubfp16",
                                                        "softmarginloss",
                                                        "softmarginlossfp16",
                                                        "softmarginlossbfp16",
                                                        "multimarginloss",
                                                        "multimarginlossfp16",
                                                        "multimarginlossbfp16",
                                                        "--version"};

    if(std::find(valid_args.begin(), valid_args.end(), arg) == valid_args.end())
    {
        printf("FAILED: Invalid Base Input Argument\n");
        Usage(EXIT_FAILURE);
    }
    else if(arg == "-h" || arg == "--help" || arg == "-?")
        Usage(EXIT_SUCCESS);
    else
        return arg;
}

class Driver
{
public:
    Driver()
    {
        data_type = miopenFloat;
#if MIOPEN_BACKEND_OPENCL
        miopenCreate(&handle);
#elif MIOPEN_BACKEND_HIP
        hipStream_t s;
        hipStreamCreate(&s);
        miopenCreateWithStream(&handle, s);
#endif

        miopenGetStream(handle, &q);
    }

    miopenHandle_t GetHandle() { return handle; }
    miopenDataType_t GetDataType() { return data_type; }

#if MIOPEN_BACKEND_OPENCL
    cl_command_queue& GetStream() { return q; }
#elif MIOPEN_BACKEND_HIP
    hipStream_t& GetStream() { return q; }
#endif
    virtual ~Driver() { miopenDestroy(handle); }

    // TODO: add timing APIs
    virtual int AddCmdLineArgs()                         = 0;
    virtual int ParseCmdLineArgs(int argc, char* argv[]) = 0;
    virtual InputFlags& GetInputFlags()                  = 0;
    virtual int GetandSetData()                          = 0;
    virtual int AllocateBuffersAndCopy()                 = 0;
    virtual int RunForwardGPU()                          = 0;
    virtual int VerifyForward()                          = 0;
    virtual int RunBackwardGPU()                         = 0;
    virtual int VerifyBackward()                         = 0;

protected:
    template <typename Tgpu>
    void InitDataType();
    void AddGpuBufferCheckFlag(InputFlags& inflags);
    GPUMem::Check GetGpuBufferCheck(const InputFlags& inflags) const;
    miopenHandle_t handle;
    miopenDataType_t data_type;

#if MIOPEN_BACKEND_OPENCL
    cl_command_queue q;
#elif MIOPEN_BACKEND_HIP
    hipStream_t q;
#endif
};

template <>
inline void Driver::InitDataType<int8_t>()
{
    data_type = miopenInt8;
}
template <>
inline void Driver::InitDataType<float>()
{
    data_type = miopenFloat;
}
template <>
inline void Driver::InitDataType<float16>()
{
    data_type = miopenHalf;
}
template <>
inline void Driver::InitDataType<bfloat16>()
{
    data_type = miopenBFloat16;
}
template <>
inline void Driver::InitDataType<float8_fnuz>()
{
    data_type = miopenFloat8_fnuz;
}
template <>
inline void Driver::InitDataType<bfloat8_fnuz>()
{
    data_type = miopenBFloat8_fnuz;
}
// "std::is_same<Tgpu, float>{}" used to avoid "static_assert" compilation error,
// which occurs when the condition does not depend in any way on the template parameters.
template <typename Tgpu>
inline void Driver::InitDataType()
{
    static_assert(std::is_same<Tgpu, float>{}, "unsupported Tgpu");
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vs)
{
    os << "{ size: " << vs.size() << ", entries: ";
    for(auto& v : vs)
        os << v << " ";
    os << "}";
    return os;
}

#endif // GUARD_MIOPEN_DRIVER_HPP
