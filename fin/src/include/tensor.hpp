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
#ifndef GUARD_FIN_TENSOR_HPP
#define GUARD_FIN_TENSOR_HPP

#include <miopen/config.h>
#if HIP_PACKAGE_VERSION_FLAT >= 5006000000ULL
#include <half/half.hpp>
#else
#include <half.hpp>
#endif
#include <miopen/bfloat16.hpp>

#include <gpu_mem.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_layout.hpp>

namespace fin {

using half_float::half;
typedef half float16;

template <typename T>
miopenDataType_t GetDataType();

template <>
miopenDataType_t GetDataType<float>();
template <>
miopenDataType_t GetDataType<float16>();
template <>
miopenDataType_t GetDataType<bfloat16>();
template <>
miopenDataType_t GetDataType<int8_t>();
#if FIN_BACKEND_OPENCL
#define STATUS_SUCCESS CL_SUCCESS
using status_t = cl_int;
#else // FIN_BACKEND_HIP
#define STATUS_SUCCESS 0
using status_t = int;
#endif

inline miopenTensorLayout_t GetMemLayout(const std::string& s)
{

    if(s == "NCHW")
        return miopenTensorLayout_t::miopenTensorNCHW;
    if(s == "NHWC")
        return miopenTensorLayout_t::miopenTensorNHWC;
    if(s == "CHWN")
        return miopenTensorLayout_t::miopenTensorCHWN;
    if(s == "NCHWc4")
        return miopenTensorLayout_t::miopenTensorNCHWc4;
    if(s == "NCHWc8")
        return miopenTensorLayout_t::miopenTensorNCHWc8;
    if(s == "CHWNc4")
        return miopenTensorLayout_t::miopenTensorCHWNc4;
    if(s == "CHWNc8")
        return miopenTensorLayout_t::miopenTensorCHWNc8;
    if(s == "NCDHW")
        return miopenTensorLayout_t::miopenTensorNCDHW;
    if(s == "NDHWC")
        return miopenTensorLayout_t::miopenTensorNDHWC;

    throw std::runtime_error("Unknown memory layout : " + s);
}

inline void LengthReorder(std::vector<int>& lens, const std::initializer_list<int>& indices)
{
    std::vector<int> out_lens;
    out_lens.reserve(indices.size());
    for(int index : indices)
    {
        assert(0 <= index && index < lens.size());
        out_lens.push_back(std::move(lens[index]));
    }
    lens = std::move(out_lens);
}

inline int SetTensorNdVector(miopenTensorDescriptor_t t,
                             std::vector<int>& len,
                             miopenTensorLayout_t layout,
                             miopenDataType_t data_type = miopenFloat)
{
    if(layout == miopenTensorNCHWc4 || layout == miopenTensorNCHWc8)
    {
        // Do nothing, MIOpen implicit logic that lens are in NCHW order.
    }
    else if(layout == miopenTensorCHWNc4 || layout == miopenTensorCHWNc8)
    {
        LengthReorder(len, {1, 2, 3, 0});
    }
    else
    {
        MIOPEN_THROW("We only support NCHWc4, NCHWc8, CHWNc4, CHWNc8 vectorized tensor layout.");
        return -1;
    }
    return miopenSetNdTensorDescriptorWithLayout(t, data_type, layout, len.data(), len.size());
}

inline int SetTensorNd(miopenTensorDescriptor_t t,
                       std::vector<int>& len,
                       miopenDataType_t data_type = miopenFloat)
{
    return miopenSetTensorDescriptor(t, data_type, len.size(), len.data(), nullptr);
}

inline int SetTensorNd(miopenTensorDescriptor_t t,
                       std::vector<int>& len,
                       std::vector<int>& strides,
                       miopenDataType_t data_type = miopenFloat)
{
    return miopenSetTensorDescriptor(t, data_type, len.size(), len.data(), strides.data());
}

inline int SetTensorNd(miopenTensorDescriptor_t t,
                       std::vector<int>& len,
                       const std::string& layout,
                       miopenDataType_t data_type = miopenFloat)
{
    if(layout.empty())
    {
        return SetTensorNd(t, len, data_type);
    }

    if(layout.size() != len.size() && layout.find("c") == std::string::npos)
    {
        MIOPEN_THROW("unmatched layout and dimension size");
    }

    if(layout.find("c") != std::string::npos)
    {
        return SetTensorNdVector(t, len, GetMemLayout(layout), data_type);
    }

    // Dimension lengths vector 'len' comes with a default layout.
    std::string len_layout = miopen::tensor_layout_get_default(layout.size());
    if(len_layout.empty())
    {
        return SetTensorNd(t, len, data_type);
    }

    std::vector<int> strides;
    miopen::tensor_layout_to_strides(len, len_layout, layout, strides);

    return SetTensorNd(t, len, strides, data_type);
}

template <typename Tgpu, typename Tcpu>
struct tensor
{
#if FIN_BACKEND_OPENCL
    using context_type       = cl_context;
    using accelerator_stream = cl_command_queue;
    context_type ctx;
#elif FIN_BACKEND_HIP
    using accelerator_stream = hipStream_t;
    using context_type       = uint32_t;
    context_type ctx         = 0;
#endif
    miopen::TensorDescriptor desc;
    std::vector<Tgpu> cpuData;    // version of the data for the CPU compute
    std::vector<Tgpu> deviceData; // home for the GPU data on the CPU side
    GPUMem gpuData;               // object representing the GPU data ON the GPU
    accelerator_stream q;
    bool is_input;
    bool is_output;

    tensor() {}
    // cppcheck-suppress uninitMemberVar
    template <typename U>
    tensor(accelerator_stream _q, const std::vector<U>& _plens, bool _is_input, bool _is_output)
        : desc(GetDataType<Tgpu>(), _plens),
          gpuData(),
          q(_q),
          is_input(_is_input),
          is_output(_is_output)
    {
#if FIN_BACKEND_OPENCL
        clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#endif
    }
    template <typename U>
    tensor(accelerator_stream _q,
           const std::string& layout,
           const std::vector<U>& _plens,
           bool _is_input,
           bool _is_output)
        : desc(GetDataType<Tgpu>(), GetMemLayout(layout), _plens),
          gpuData(),
          q(_q),
          is_input(_is_input),
          is_output(_is_output)
    {
#if FIN_BACKEND_OPENCL
        clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#endif
    }

    status_t FromDevice()
    {
        status_t status = 0;
        if(is_output)
            status = gpuData.FromGPU(q, deviceData.data());
        return status;
    }

    status_t ToDevice()
    {
        status_t status = 0;
        if(is_input)
            status = gpuData.ToGPU(q, cpuData.data());
        else if(is_output)
            status = gpuData.ToGPU(q, deviceData.data()); // to set the data to zero on the GPU
        return status;
    }

    void AllocateBuffers()
    {
        if(is_input)
            // TODO: check if the datatype is correct;
            cpuData = std::vector<Tgpu>(desc.GetNumBytes(), static_cast<Tgpu>(0));

        if(is_output)
            deviceData = std::vector<Tgpu>(desc.GetNumBytes(), static_cast<Tgpu>(0));

        gpuData = GPUMem{ctx, desc.GetNumBytes() / sizeof(Tgpu), sizeof(Tgpu)};
    }
    template <typename F>
    void FillBuffer(F f)
    {
        for(int i = 0; i < GetTensorSize(); i++)
        {
            if(is_input) // is input
                cpuData[i] = f(i);
            // Data_scale * RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0),
            // static_cast<Tgpu>(1.0));
            else /// \ref move_rand
                std::ignore = rand();
        }
    }

    size_t GetTensorSize() { return desc.GetElementSize(); }
};
} // namespace fin
#endif // GUARD_FIN_TENSOR_HPP
