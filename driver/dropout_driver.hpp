/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_DROPOUT_DRIVER_HPP
#define GUARD_MIOPEN_DROPOUT_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "dropout_gpu_emulator.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "util_driver.hpp"
#include "util_file.hpp"

#include <../test/verify.hpp>

#include <miopen/dropout.hpp>
#include <miopen/env.hpp>
#include <miopen/miopen.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

template <typename Tgpu, typename Tref = Tgpu>
class DropoutDriver : public Driver
{
public:
    DropoutDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        miopenCreateDropoutDescriptor(&DropoutDesc);
        reservespace_dev = nullptr;
        data_type        = std::is_same<Tgpu, float16>{} ? miopenHalf : miopenFloat;
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine(std::string input_str);

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();
    int RunBackwardGPU() override;
    int RunBackwardCPU();
    int VerifyForward() override;
    int VerifyBackward() override;

    ~DropoutDriver() override
    {
        miopenDestroyTensorDescriptor(inputTensor);
        miopenDestroyTensorDescriptor(outputTensor);

        miopenDestroyDropoutDescriptor(DropoutDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t outputTensor;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> dout_dev;
    std::unique_ptr<GPUMem> din_dev;
    std::unique_ptr<GPUMem> reservespace_dev;
    std::unique_ptr<GPUMem> states_dev;

    tensor<Tgpu> in;
    tensor<Tgpu> out;
    tensor<Tgpu> dout;
    tensor<Tgpu> din;
    tensor<Tref> outhost;
    tensor<Tref> din_host;

    std::vector<rocrand_state_xorwow> states_host;
    std::vector<unsigned char> reservespace;
    std::vector<unsigned char> reservespace_host;

    miopenDropoutDescriptor_t DropoutDesc;

    float dropout;
    unsigned long long seed;
    bool use_mask;
};

template <typename Tgpu, typename Tref>
int DropoutDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DropoutDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine(inflags.GetValueStr("input_dim"));
    SetTensorNd(inputTensor, in_len, data_type);
    SetTensorNd(outputTensor, in_len, data_type);

    dropout  = static_cast<float>(inflags.GetValueDouble("dropout"));
    use_mask = static_cast<bool>(inflags.GetValueInt("use_mask"));

    auto seed_low  = static_cast<unsigned long long>(std::max(inflags.GetValueInt("seed_low"), 0));
    auto seed_high = static_cast<unsigned long long>(std::max(inflags.GetValueInt("seed_high"), 0));
    seed           = seed_high << 32 | seed_low;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DropoutDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag(
        "forw", 'F', "0", "Direction, Forward = 1, Backward = 2 , Both = 0 (Default=0)", "int");
    inflags.AddInputFlag(
        "input_dim", 'd', "4", "Input dimension (Default=4, support up to 5D)", "vector");
    inflags.AddInputFlag("dropout", 'p', "0.5", "Dropout rate (Default=0.5)", "float");
    inflags.AddInputFlag(
        "seed_low", 'l', "0", "Least significant 32 bits of seed (Default=0)", "int");
    inflags.AddInputFlag(
        "seed_high", 'm', "0", "Most significant 32 bits of seed (Default=0)", "int");
    inflags.AddInputFlag("use_mask",
                         'e',
                         "0",
                         "Use existing mask in reservespace: Use 1, Not use 0 (Default=0)",
                         "int");
    inflags.AddInputFlag(
        "gen_file",
        'f',
        "0",
        "Generate and write/overwrite PRNG skipahead files (1), No operation (0) (Default=0)",
        "int");
    inflags.AddInputFlag("iter", 'i', "1", "Number of Iterations (Default=1)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Dropout (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    inflags.AddInputFlag("dump_output", 'o', "0", "Dumps the output buffers (Default=0)", "int");

    return 0;
}

template <typename Tgpu, typename Tref>
std::vector<int> DropoutDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine(std::string input_str)
{
    std::vector<int> in_lens;
    std::stringstream ss(input_str);

    int cont = 0;
    int element;

    while(ss >> element)
    {
        if(cont++ >= 5)
        {
            std::cout << "Only support up to 5D-tensor dropout" << std::endl;
            break;
        }

        if(ss.peek() == ',' || ss.peek() == ' ')
        {
            ss.ignore();
        }

        in_lens.push_back(element);
    }

    return in_lens;
}

template <typename Tgpu, typename Tref>
int DropoutDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(inputTensor);
    size_t out_sz = GetTensorSize(outputTensor);

    size_t reserveSpaceSizeInBytes = 0;
    miopenDropoutGetReserveSpaceSize(inputTensor, &reserveSpaceSizeInBytes);
    size_t reserveSpaceSize = reserveSpaceSizeInBytes / sizeof(unsigned char);

    size_t statesSizeInBytes = 0;
    miopenDropoutGetStatesSize(GetHandle(), &statesSizeInBytes);
    size_t states_size = statesSizeInBytes / sizeof(rocrand_state_xorwow);

    DEFINE_CONTEXT(ctx);
#if MIOPEN_BACKEND_OPENCL
    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#endif

    states_dev =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, states_size, sizeof(rocrand_state_xorwow)));

    // if(inflags.GetValueInt("gen_file"))
    //     generate_skipahead_file();

    miopenSetDropoutDescriptor(DropoutDesc,
                               GetHandle(),
                               dropout,
                               states_dev->GetMem(),
                               states_dev->GetSize(),
                               seed,
                               use_mask,
                               false,
                               MIOPEN_RNG_PSEUDO_XORWOW);

    in_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    din_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    out_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    reservespace_dev =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, reserveSpaceSize, sizeof(unsigned char)));

    in   = tensor<Tgpu>(miopen::deref(inputTensor).GetLengths(),
                      miopen::deref(inputTensor).GetStrides());
    din  = tensor<Tgpu>(miopen::deref(inputTensor).GetLengths(),
                       miopen::deref(inputTensor).GetStrides());
    out  = tensor<Tgpu>(miopen::deref(outputTensor).GetLengths(),
                       miopen::deref(outputTensor).GetStrides());
    dout = tensor<Tgpu>(miopen::deref(outputTensor).GetLengths(),
                        miopen::deref(outputTensor).GetStrides());

    outhost  = tensor<Tref>(miopen::deref(outputTensor).GetLengths(),
                           miopen::deref(outputTensor).GetStrides());
    din_host = tensor<Tref>(miopen::deref(inputTensor).GetLengths(),
                            miopen::deref(inputTensor).GetStrides());

    reservespace      = std::vector<unsigned char>(reserveSpaceSize, static_cast<unsigned char>(1));
    reservespace_host = std::vector<unsigned char>(reserveSpaceSize, static_cast<unsigned char>(1));

    states_host = std::vector<rocrand_state_xorwow>(states_size);

    Tgpu Data_scale = static_cast<Tgpu>(0.01);

    for(int i = 0; i < in_sz; i++)
    {
        in.data[i] = prng::gen_0_to_B(Data_scale);
    }

    for(int i = 0; i < out_sz; i++)
    {
        dout.data[i] = prng::gen_0_to_B(Data_scale);
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_in.bin", in.data.data(), in_sz);
        dumpBufferToFile("dump_dout.bin", dout.data.data(), out_sz);
    }

    status_t status;
    status = in_dev->ToGPU(q, in.data.data());
    status |= din_dev->ToGPU(q, din.data.data());
    status |= out_dev->ToGPU(q, out.data.data());
    status |= dout_dev->ToGPU(q, dout.data.data());

    if(inflags.GetValueInt("use_mask") == 1)
    {
        for(int i = 0; i < reserveSpaceSize; i++)
        {
            reservespace[i]      = static_cast<uint8_t>(prng::gen_canonical<float>() > dropout);
            reservespace_host[i] = reservespace[i];
        }
        status |= reservespace_dev->ToGPU(q, reservespace.data());
    }

    if(status != STATUS_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DropoutDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME
    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenDropoutForward(GetHandle(),
                             DropoutDesc,
                             inputTensor,
                             inputTensor,
                             in_dev->GetMem(),
                             outputTensor,
                             out_dev->GetMem(),
                             reservespace_dev->GetMem(),
                             reservespace_dev->GetSize());

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        if(WALL_CLOCK)
            printf("Wall-clock Time Dropout Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));

        int iter = inflags.GetValueInt("iter");
        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward Dropout. Elapsed: %f ms (average)\n", kernel_average_time);
    }

    out_dev->FromGPU(GetStream(), out.data.data());
    reservespace_dev->FromGPU(GetStream(), reservespace.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DropoutDriver<Tgpu, Tref>::RunForwardCPU()
{
    InitKernelStateEmulator(states_host, DropoutDesc);
    RunDropoutForwardEmulator<Tgpu, Tref>(GetHandle(),
                                          DropoutDesc,
                                          inputTensor,
                                          inputTensor,
                                          in.data,
                                          outputTensor,
                                          outhost.data,
                                          reservespace_host,
                                          states_host);

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile<Tref>("dump_fwd_out_cpu.bin", outhost.data.data(), outhost.data.size());
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DropoutDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME
    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenDropoutBackward(GetHandle(),
                              DropoutDesc,
                              inputTensor,
                              outputTensor,
                              dout_dev->GetMem(),
                              inputTensor,
                              din_dev->GetMem(),
                              reservespace_dev->GetMem(),
                              reservespace_dev->GetSize());

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        if(WALL_CLOCK)
            printf("Wall-clock Time Backward Dropout Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));

        int iter = inflags.GetValueInt("iter");
        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Backward Dropout. Elapsed: %f ms (average)\n", kernel_average_time);
    }

    din_dev->FromGPU(GetStream(), din.data.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DropoutDriver<Tgpu, Tref>::RunBackwardCPU()
{
    RunDropoutBackwardEmulator<Tgpu, Tref>(
        DropoutDesc, outputTensor, dout.data, inputTensor, din_host.data, reservespace_host);

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile<Tref>("dump_bwd_out_cpu.bin", din_host.data.data(), din_host.data.size());
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DropoutDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    auto error = miopen::rms_range(outhost.data, out.data);

    const double tolerance = std::is_same<Tgpu, float16>{} ? 5e-4 : 1e-6;
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Dropout FAILED: " << error << std::endl;
    }
    else
    {
        std::cout << "Forward Dropout Verifies on CPU and GPU (" << error << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DropoutDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();

    auto error = miopen::rms_range(din_host.data, din.data);

    const double tolerance = std::is_same<Tgpu, float16>{} ? 5e-4 : 1e-6;
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward Dropout FAILED: " << error << std::endl;
    }
    else
    {
        std::cout << "Backward Dropout Verifies on CPU and GPU (" << error << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_DROPOUT_DRIVER_HPP
