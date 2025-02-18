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

#ifndef GUARD_MIOPEN_PAD_CONSTANT_DRIVER_HPP
#define GUARD_MIOPEN_PAD_CONSTANT_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "miopen/tensor_view_5d.hpp"
#include "tensor_driver.hpp"
#include "random.hpp"
#include "timer.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <../test/verify.hpp>
#include "../src/kernels/tensor_view_5d.hpp"

template <typename Tgpu, typename Tcheck>
void mloConstantPadForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                  miopenTensorDescriptor_t outputDesc,
                                  Tgpu* input,
                                  Tcheck* output_host,
                                  std::vector<int64_t>& padding_vec,
                                  Tgpu value)
{
    if(padding_vec.size() % 2 != 0)
        throw std::runtime_error("padding_vec size must be even");

    padding_5d_t padding;
    memset(padding.val, 0, sizeof(padding.val));
    for(auto i = 0; i < padding_vec.size(); i++)
        padding.val[10 - padding_vec.size() + i] = padding_vec[i];

    int64_t o[5];
    size_t output_size = miopen::deref(outputDesc).GetElementSize();

    tensor_view_5d_t input_tv  = get_inner_expanded_tv(miopen::deref(inputDesc));
    tensor_view_5d_t output_tv = get_inner_expanded_tv(miopen::deref(outputDesc));

    for(size_t gid = 0; gid < output_size; ++gid)
    {
        bool flag = true;
        getNCDHW(o, gid, output_tv.size);

        for(int i = 0; i < 5; i++)
        {
            o[i] = o[i] - padding.val[2 * i];
            flag *= (o[i] >= 0) && (o[i] < input_tv.size[i]);
        }

        auto val =
            flag ? get5DValueAt<Tgpu>(input, input_tv.stride, o[0], o[1], o[2], o[3], o[4]) : value;
        set5DValueAt(output_host, output_tv, gid, val);
    }
}

template <typename Tgpu, typename Tcheck>
void mloConstantPadBackwardRunHost(miopenTensorDescriptor_t backwardOutputDesc,
                                   miopenTensorDescriptor_t inputGradDesc,
                                   Tcheck* backward_output_host,
                                   Tgpu* input_grad,
                                   std::vector<int64_t>& padding_vec)
{
    if(padding_vec.size() % 2 != 0)
        throw std::runtime_error("padding size should be even");

    padding_5d_t padding;
    memset(padding.val, 0, sizeof(padding.val));
    for(auto i = 0; i < padding_vec.size(); i++)
        padding.val[10 - padding_vec.size() + i] = padding_vec[i];

    int64_t o[5];

    size_t backward_output_size = miopen::deref(backwardOutputDesc).GetElementSize();
    tensor_view_5d_t input_tv   = get_inner_expanded_tv(miopen::deref(inputGradDesc));
    tensor_view_5d_t output_tv  = get_inner_expanded_tv(miopen::deref(backwardOutputDesc));

    for(size_t gid = 0; gid < backward_output_size; ++gid)
    {
        bool flag = true;
        getNCDHW(o, gid, output_tv.size);

        for(int i = 0; i < 5; i++)
        {
            o[i] = o[i] + padding.val[2 * i];
            flag *= (o[i] >= 0) && (o[i] < input_tv.size[i]);
        }

        if(flag)
        {
            auto val =
                get5DValueAt<Tcheck>(input_grad, input_tv.stride, o[0], o[1], o[2], o[3], o[4]);
            set5DValueAt(backward_output_host, output_tv, gid, val);
        }
    }
}

template <typename T>
inline std::vector<T> GetStrides(std::vector<T> input, bool contiguous)
{
    if(!contiguous)
        std::swap(input.front(), input.back());
    std::vector<T> strides(input.size());
    strides.back() = 1;
    for(int i = input.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * input[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
class ConstantPadDriver : public Driver
{

public:
    ConstantPadDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&backwardOutputDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int64_t> GetPaddingsFromCmdLine(size_t input_dims_size);

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~ConstantPadDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(backwardOutputDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t backwardOutputDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> backward_output_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;
    std::vector<Tgpu> backward_output;
    std::vector<Tref> output_host;
    std::vector<Tref> backward_output_host;

    std::vector<int64_t> padding;
    Tgpu value;
};

template <typename Tgpu, typename Tref>
int32_t ConstantPadDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int32_t ConstantPadDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> input_dims = GetInputTensorLengthsFromCmdLine();
    padding                     = GetPaddingsFromCmdLine(input_dims.size());
    bool makeNonContiguous      = inflags.GetValueInt("contiguous") == 0;

    auto strides = GetStrides(input_dims, !makeNonContiguous);

    SetTensorNd(inputDesc, input_dims, strides, data_type);
    SetTensorNd(backwardOutputDesc, input_dims, strides, data_type);

    std::vector<int> output_dims;
    output_dims.reserve(input_dims.size());

    for(int i = 0; i < input_dims.size(); i++)
    {
        output_dims.push_back(input_dims[i] + padding[2 * i] + padding[2 * i + 1]);
    }

    SetTensorNd(outputDesc, output_dims, data_type);

    value = inflags.GetValueDouble("value");

    return 0;
}

inline std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim))
    {
        elems.push_back(item);
    }
    return elems;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "0", "Only run the Forward Pass (Default=1)", "int");
    inflags.AddInputFlag("contiguous", 'Z', "0", "Use contiguous tensor", "int");
    inflags.AddInputFlag("in", 'I', "3,3,3,3,3", "Input batch size (n,c,d,h,w)", "int");
    inflags.AddInputFlag("pad", 'P', "1,1,1,1,1,1", "Padding batch size (n,c,d,h,w)", "int");
    inflags.AddInputFlag("value", 'v', "0", "Padding value", "string");

    inflags.AddInputFlag("iter", 'i', "10", "Number of iterations", "int");
    inflags.AddInputFlag("verify", 'V', "0", "Verify results", "int");
    inflags.AddInputFlag("time", 't', "0", "Enable/Disable time measurement", "int");
    inflags.AddInputFlag("wall", 'l', "0", "Enable/Disable walltime measurement", "int");

    return 0;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConstantPadDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    auto in_str = inflags.GetValueStr("in");

    // Parse the comma_separated string
    std::vector<std::string> in = split(in_str, ',');

    switch(in.size())
    {
    case 5:
        return std::vector<int>({std::stoi(in[0]),
                                 std::stoi(in[1]),
                                 std::stoi(in[2]),
                                 std::stoi(in[3]),
                                 std::stoi(in[4])});

    case 4:
        return std::vector<int>(
            {std::stoi(in[0]), std::stoi(in[1]), std::stoi(in[2]), std::stoi(in[3])});

    case 3: return std::vector<int>({std::stoi(in[0]), std::stoi(in[1]), std::stoi(in[2])});

    case 2: return std::vector<int>({std::stoi(in[0]), std::stoi(in[1])});

    case 1: return std::vector<int>({std::stoi(in[0])});

    default:
        std::cerr << "Error: Invalid input dimensions" << std::endl;
        return std::vector({0});
        break;
    }
}

template <typename Tgpu, typename Tref>
std::vector<int64_t> ConstantPadDriver<Tgpu, Tref>::GetPaddingsFromCmdLine(size_t input_dims_size)
{
    // Input would be in PyTorch format (pad_left, pad_right, pad_up, pad_down, pad_front, pad_back,
    // etc.) We would need to flip it into ours (basically backwards of PyTorch format) Eg:
    // (1,2,1,2,1,2,0,0,0,0) -> (0,0,0,0,2,1,2,1,2,1)
    std::vector<int64_t> paddings =
        std::vector<int64_t>(static_cast<size_t>(input_dims_size) * 2, 0);
    auto pad                   = inflags.GetValueStr("pad");
    std::vector<std::string> p = split(pad, ',');
    assert(p.size() % 2 == 0);

    auto offset = paddings.size() - p.size();
    for(size_t i = 0; i < p.size(); i++)
    {
        paddings[i + offset] = std::stoi(p[i]);
    }
    // Reverse each pair
    for(size_t i = 0; i < paddings.size(); i += 2)
    {
        std::swap(paddings[i], paddings[i + 1]);
    }

    return paddings;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_size  = GetTensorSize(inputDesc);
    size_t output_size = GetTensorSize(outputDesc);

    in_dev              = std::unique_ptr<GPUMem>(new GPUMem(0, input_size, sizeof(Tgpu)));
    out_dev             = std::unique_ptr<GPUMem>(new GPUMem(0, output_size, sizeof(Tgpu)));
    backward_output_dev = std::unique_ptr<GPUMem>(new GPUMem(0, input_size, sizeof(Tgpu)));

    input       = std::vector<Tgpu>(input_size, static_cast<Tgpu>(0));
    output      = std::vector<Tgpu>(output_size, static_cast<Tgpu>(0));
    output_host = std::vector<Tref>(output_size, static_cast<Tref>(0));

    backward_output      = std::vector<Tgpu>(input_size, static_cast<Tgpu>(0));
    backward_output_host = std::vector<Tref>(input_size, static_cast<Tref>(0));

    for(int i = 0; i < input_size; i++)
    {
        input[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    if(in_dev->ToGPU(GetStream(), input.data()) != 0)
        std::cerr << "ConstantPadDriver: Error copying data to GPU, size: " << in_dev->GetSize()
                  << std::endl;

    return 0;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenPadConstantFwd(GetHandle(),
                             inputDesc,
                             outputDesc,
                             in_dev->GetMem(),
                             out_dev->GetMem(),
                             padding.data(),
                             padding.size(),
                             value);

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");

        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Elapsed: " << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_avg_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "Kernel Time Elapsed: " << kernel_avg_time << " ms" << std::endl;
    }

    if(out_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying data from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloConstantPadForwardRunHost(
        inputDesc, outputDesc, input.data(), output_host.data(), padding, value);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    for(int i = 0; i < output.size(); i++)
    {
        if(output[i] != output_host[i])
        {
            std::cout << "output[" << i << "] = " << output[i] << " != "
                      << "output_host[" << i << "] = " << output_host[i] << std::endl;
            std::cout << "ConstantPadDriver: Forward verification failed." << std::endl;
            return -1;
        }
    }
    std::cout << "ConstantPadDriver: Forward verification passed." << std::endl;
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloConstantPadBackwardRunHost(
        backwardOutputDesc, outputDesc, backward_output_host.data(), output.data(), padding);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenPadConstantBwd(GetHandle(),
                             backwardOutputDesc,
                             outputDesc,
                             backward_output_dev->GetMem(),
                             out_dev->GetMem(),
                             padding.data(),
                             padding.size());

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");

        if(WALL_CLOCK)
            std::cout << "Wall-clock Backward Time Elapsed: " << t.gettime_ms() / iter << " ms"
                      << std::endl;

        float kernel_avg_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "Kernel Backward Time Elapsed: " << kernel_avg_time << " ms" << std::endl;
    }

    if(backward_output_dev->FromGPU(GetStream(), backward_output.data()) != 0)
        std::cerr << "Error copying data from GPU, size: " << in_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();

    for(int i = 0; i < backward_output_host.size(); i++)
    {
        if(backward_output[i] != backward_output_host[i])
        {
            std::cout << "backward_output[" << i << "] = " << backward_output[i] << " != "
                      << "backward_output_host[" << i << "] = " << backward_output_host[i]
                      << std::endl;
            std::cout << "ConstantPadDriver: Backward verification failed." << std::endl;
            return -1;
        }
    }
    std::cout << "ConstantPadDriver: Backward verification passed." << std::endl;
    return miopenStatusSuccess;
}

#endif
