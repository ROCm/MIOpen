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

#include "InputFlags.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <memory>
#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <numeric>
#include <rocrand_xorwow.h>

class GenerateRandomBitMaskDriver : public Driver
{
public:
    GenerateRandomBitMaskDriver() : Driver() { miopenCreateTensorDescriptor(&maskDesc); }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    int VerifyForward() override;
    int VerifyBackward() override;

    ~GenerateRandomBitMaskDriver() override { miopenDestroyTensorDescriptor(maskDesc); }

private:
    InputFlags inflags;

    int forw;

    std::vector<int> input_shape;
    miopenTensorDescriptor_t maskDesc;

    std::unique_ptr<GPUMem> pstate_dev;
    std::unique_ptr<GPUMem> mask_dev;

    std::vector<unsigned char> mask;

    std::vector<unsigned char> mask_host;

    size_t statesSizeInBytes;

    float p;
};

int GenerateRandomBitMaskDriver::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Only run forward pass (Default=1)", "int");
    inflags.AddInputFlag(
        "state-size-in-bytes",
        'S',
        "0",
        "Size of the prng_state in bytes. If size=0, auto run "
        "`miopenInitGenerateRandomBitMaskStates` to init inital states (Default=0)",
        "int");
    inflags.AddInputFlag(
        "mask-dims", 'M', "4x1", "Mask tensor dimensions (Default=4x1)", "tensor descriptor");
    inflags.AddInputFlag(
        "probability", 'p', "0.5", "Probability of an element to be zeroed (Default=0.5)", "float");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

int GenerateRandomBitMaskDriver::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    p                 = inflags.GetValueDouble("probability");
    statesSizeInBytes = inflags.GetValueInt("state-size-in-bytes");

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

int GenerateRandomBitMaskDriver::GetandSetData()
{
    auto mask_dims = inflags.GetValueTensor("mask-dims").lengths;

    if(statesSizeInBytes == 0)
    {
        auto status = miopenGetGenerateRandomBitMaskStatesSize(GetHandle(), &statesSizeInBytes);

        MIOPEN_THROW_IF(status != miopenStatusSuccess || statesSizeInBytes <= 0,
                        "Error in miopenGetGenerateRandomBitMaskStatesSize");
    }

    if(SetTensorNd(maskDesc, mask_dims, miopenInt8) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing mask tensor.");

    return miopenStatusSuccess;
}

int GenerateRandomBitMaskDriver::AllocateBuffersAndCopy()
{
    auto mask_size = GetTensorSize(maskDesc);

    size_t num_states = statesSizeInBytes / sizeof(rocrand_state_xorwow);

    uint32_t ctx = 0;

    // GPU Allocation
    pstate_dev = std::make_unique<GPUMem>(ctx, num_states, sizeof(rocrand_state_xorwow));

    // Initialize the random states
    auto status = miopenInitGenerateRandomBitMaskStates(
        GetHandle(), pstate_dev->GetMem(), statesSizeInBytes, 0);

    MIOPEN_THROW_IF(status != miopenStatusSuccess,
                    "Error in miopenInitGenerateRandomBitMaskStates");

    mask_dev = std::make_unique<GPUMem>(ctx, mask_size, sizeof(unsigned char));

    // GPU host allocation
    mask = std::vector<unsigned char>(mask_size);
    std::fill(mask.begin(), mask.end(), 0);

    // CPU allocation
    mask_host = std::vector<unsigned char>(mask_size);
    std::fill(mask_host.begin(), mask_host.end(), 0);

    if(mask_dev->ToGPU(GetStream(), mask.data()) != 0)
    {
        std::cerr << "Error copying (mask) to GPU, size: " << mask_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

int GenerateRandomBitMaskDriver::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenGenerateRandomBitMask(
            GetHandle(), pstate_dev->GetMem(), statesSizeInBytes, maskDesc, mask_dev->GetMem(), p);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenGenerateRandomBitMask");

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
            std::cout << "Wall-clock Time Backward Pdist Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward Pdist Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(mask_dev->FromGPU(GetStream(), mask.data()) != 0)
    {
        std::cerr << "Error copying (output) from GPU, size: " << mask_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

int GenerateRandomBitMaskDriver::RunForwardCPU() { return miopenStatusSuccess; }

int GenerateRandomBitMaskDriver::VerifyForward()
{
    // counting number bit 1 in mask
    int64_t count_1 = 0;
    for(size_t i = 0; i < mask.size(); i++)
    {
        unsigned char val = mask[i];
        for(int j = 0; j < 8; j++)
        {
            count_1 += val & 1; // Add the least significant bit
            val >>= 1;          // Right shift to process the next bit
        }
    }

    auto numel = GetTensorSize(maskDesc) * 8;

    double min_expected = (1 - p) * 0.95;
    double max_expected = (1 - p) * 1.05;

    double actual = static_cast<double>(count_1) / numel;

    if(actual < min_expected || actual > max_expected)
    {
        std::cout << "GenerateRandomBitMask FAILED: " << actual << " not in range [" << min_expected
                  << ", " << max_expected << "]" << std::endl;
        return EC_VerifyFwd;
    }

    std::cout << "GenerateRandomBitMask Verifies on CPU and GPU (ratio: " << actual << ")"
              << std::endl;

    return miopenStatusSuccess;
}

int GenerateRandomBitMaskDriver::RunBackwardGPU() { return miopenStatusNotImplemented; }

int GenerateRandomBitMaskDriver::RunBackwardCPU() { return miopenStatusNotImplemented; }

int GenerateRandomBitMaskDriver::VerifyBackward() { return miopenStatusNotImplemented; }
