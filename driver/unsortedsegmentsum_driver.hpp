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

#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <../test/ford.hpp>

#include "InputFlags.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <vector>

template <typename Tgpu, typename Tcheck, typename Tseg>
int32_t mloUnsortedSegmentSumForwardRunHost(const miopenTensorDescriptor_t InputDesc,
                                            const Tgpu* input,
                                            Tcheck* output,
                                            const Tseg* segment_ids,
                                            const uint64_t num_segments)
{
    uint64_t N = miopen::deref(InputDesc).GetElementSize();
    uint64_t inner_dim_size =
        miopen::deref(InputDesc).GetElementSize() / miopen::deref(InputDesc).GetLengths()[0];

    ford(N)([&](uint64_t gid) {
        uint64_t input_segment_index  = gid / inner_dim_size;
        uint64_t segment_offset       = gid % inner_dim_size;
        uint64_t output_segment_index = static_cast<uint64_t>(segment_ids[input_segment_index]);

        if(output_segment_index < num_segments)
        {
            uint64_t output_index = output_segment_index * inner_dim_size + segment_offset;
            double val            = static_cast<double>(output[output_index]);
            val += static_cast<double>(input[gid]);
            output[output_index] = static_cast<Tcheck>(val);
        }
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck, typename Tseg>
int32_t mloUnsortedSegmentSumBackwardRunHost(const miopenTensorDescriptor_t InputGradDesc,
                                             const Tgpu* output_grad,
                                             Tcheck* input_grad,
                                             const Tseg* segment_ids,
                                             const uint64_t num_segments)
{
    uint64_t N              = miopen::deref(InputGradDesc).GetElementSize();
    uint64_t inner_dim_size = miopen::deref(InputGradDesc).GetElementSize() /
                              miopen::deref(InputGradDesc).GetLengths()[0];

    par_ford(N)([&](uint64_t gid) {
        uint64_t input_segment_index  = gid / inner_dim_size;
        uint64_t segment_offset       = gid % inner_dim_size;
        uint64_t output_segment_index = static_cast<uint64_t>(segment_ids[input_segment_index]);

        if(output_segment_index < num_segments)
        {
            uint64_t output_index = output_segment_index * inner_dim_size + segment_offset;
            input_grad[gid]       = output_grad[output_index];
        }
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tseg>
class UnsortedSegmentSumDriver : public Driver
{
public:
    UnsortedSegmentSumDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&InputDesc);
        miopenCreateTensorDescriptor(&OutputDesc);
        miopenCreateTensorDescriptor(&InputGradDesc);
        miopenCreateTensorDescriptor(&OutputGradDesc);
        miopenCreateTensorDescriptor(&SegmentIdsDesc);

        data_type         = miopen_type<Tgpu>{};
        segment_data_type = miopen_type<Tseg>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~UnsortedSegmentSumDriver() override
    {
        miopenDestroyTensorDescriptor(InputDesc);
        miopenDestroyTensorDescriptor(OutputDesc);
        miopenDestroyTensorDescriptor(InputGradDesc);
        miopenDestroyTensorDescriptor(OutputGradDesc);
        miopenDestroyTensorDescriptor(SegmentIdsDesc);
    }

private:
    InputFlags inflags;
    miopenDataType_t segment_data_type;

    miopenTensorDescriptor_t InputDesc;
    miopenTensorDescriptor_t OutputDesc;
    miopenTensorDescriptor_t InputGradDesc;
    miopenTensorDescriptor_t OutputGradDesc;
    miopenTensorDescriptor_t SegmentIdsDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> input_grad_dev;
    std::unique_ptr<GPUMem> output_grad_dev;
    std::unique_ptr<GPUMem> segment_ids_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;
    std::vector<Tgpu> output_init;
    std::vector<Tgpu> input_grad;
    std::vector<Tgpu> output_grad;
    std::vector<Tseg> segment_ids;

    std::vector<Tref> output_host;
    std::vector<Tref> input_grad_host;

    int num_segments;
    std::vector<int> input_dims;
};

template <typename Tgpu, typename Tref, typename Tseg>
int UnsortedSegmentSumDriver<Tgpu, Tref, Tseg>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tseg>
int UnsortedSegmentSumDriver<Tgpu, Tref, Tseg>::GetandSetData()
{
    input_dims   = inflags.GetValueTensor("input_dims").lengths;
    num_segments = inflags.GetValueInt("num_segments");
    if(num_segments <= 0)
        MIOPEN_THROW(miopenStatusBadParm, "num_segments must be greater than 0.");

    std::vector<int> output_dims      = input_dims;
    output_dims[0]                    = num_segments;
    std::vector<int> segment_ids_dims = {input_dims[0]};

    if(SetTensorNd(InputDesc, input_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input_dims") + ".");
    if(SetTensorNd(OutputDesc, output_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor.");
    if(SetTensorNd(InputGradDesc, input_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input gradient tensor: " + inflags.GetValueStr("input_dims") +
                     ".");
    if(SetTensorNd(OutputGradDesc, output_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output gradient tensor.");
    if(SetTensorNd(SegmentIdsDesc, {segment_ids_dims}, segment_data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing segment ids tensor.");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tseg>
int UnsortedSegmentSumDriver<Tgpu, Tref, Tseg>::AddCmdLineArgs()
{
    inflags.AddInputFlag(
        "forw", 'F', "1", "Run only Forward UnsortedSegmentSum (Default=1)", "int");
    inflags.AddTensorFlag("input_dims",
                          'd',
                          "2x3x7",
                          "The dimensional lengths of the input tensor: N,C,D,H Example: 2x3x7.");
    inflags.AddInputFlag("num_segments",
                         's',
                         "1",
                         "Number of segmentations, must be a non-negative number (Default=1)",
                         "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time Each Layer (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tseg>
int UnsortedSegmentSumDriver<Tgpu, Tref, Tseg>::AllocateBuffersAndCopy()
{
    size_t input_element_size  = miopen::deref(InputDesc).GetElementSize();
    size_t output_element_size = miopen::deref(OutputDesc).GetElementSize();
    size_t segment_ids_size    = miopen::deref(SegmentIdsDesc).GetElementSize();

    uint32_t ctx = 0;

    input_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_element_size, sizeof(Tgpu)));
    output_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_element_size, sizeof(Tgpu)));
    input_grad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_element_size, sizeof(Tgpu)));
    output_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_element_size, sizeof(Tgpu)));
    segment_ids_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, segment_ids_size, sizeof(Tseg)));

    input       = std::vector<Tgpu>(input_element_size, static_cast<Tgpu>(0));
    output      = std::vector<Tgpu>(output_element_size, static_cast<Tgpu>(0));
    output_init = std::vector<Tgpu>(output_element_size, static_cast<Tgpu>(0));
    input_grad  = std::vector<Tgpu>(input_element_size, static_cast<Tgpu>(0));
    output_grad = std::vector<Tgpu>(output_element_size, static_cast<Tgpu>(0));
    segment_ids = std::vector<Tseg>(segment_ids_size, static_cast<Tgpu>(0));

    output_host     = std::vector<Tref>(output_element_size, static_cast<Tref>(0));
    input_grad_host = std::vector<Tref>(input_element_size, static_cast<Tref>(0));

    for(int i = 0; i < input_element_size; i++)
    {
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    for(int i = 0; i < segment_ids_size; i++)
    {
        segment_ids[i] = prng::gen_A_to_B<Tseg>(0, num_segments);
    }

    for(int i = 0; i < output_element_size; i++)
    {
        output_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
    {
        std::cerr << "Error copying input to GPU, size: " << input_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }
    if(segment_ids_dev->ToGPU(GetStream(), segment_ids.data()) != 0)
    {
        std::cerr << "Error copying segment_ids to GPU, size: " << segment_ids_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(output_dev->ToGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying output to GPU, size: " << output_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }
    if(output_grad_dev->ToGPU(GetStream(), output_grad.data()) != 0)
    {
        std::cerr << "Error copying output_grad to GPU, size: " << output_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(input_grad_dev->ToGPU(GetStream(), input_grad.data()) != 0)
    {
        std::cerr << "Error copying input_grad to GPU, size: " << input_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tseg>
int UnsortedSegmentSumDriver<Tgpu, Tref, Tseg>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        if(output_dev->ToGPU(GetStream(), output_init.data()) != 0)
        {
            std::cerr << "Error copying output to GPU, size: " << output_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
        auto status = miopenUnsortedSegmentSumForward(GetHandle(),
                                                      InputDesc,
                                                      input_dev->GetMem(),
                                                      OutputDesc,
                                                      output_dev->GetMem(),
                                                      SegmentIdsDesc,
                                                      segment_ids_dev->GetMem());
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenUnsortedSegmentSumForward");

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
            std::cout << "Wall-clock Time Forward UnsortedSegmentSum Elapsed: "
                      << t.gettime_ms() / iter << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward UnsortedSegmentSum Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying output_dev from GPU, size: " << output_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tseg>
int UnsortedSegmentSumDriver<Tgpu, Tref, Tseg>::RunForwardCPU()
{
    int status = miopenStatusSuccess;

    status = mloUnsortedSegmentSumForwardRunHost<Tgpu, Tref, Tseg>(
        InputDesc, input.data(), output_host.data(), segment_ids.data(), num_segments);
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloUnsortedSegmentSumForwardRunHost");

    return status;
}

template <typename Tgpu, typename Tref, typename Tseg>
int UnsortedSegmentSumDriver<Tgpu, Tref, Tseg>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenUnsortedSegmentSumBackward(GetHandle(),
                                                       OutputGradDesc,
                                                       output_grad_dev->GetMem(),
                                                       InputGradDesc,
                                                       input_grad_dev->GetMem(),
                                                       SegmentIdsDesc,
                                                       segment_ids_dev->GetMem());
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenUnsortedSegmentSumBackward");

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
            std::cout << "Wall-clock Time Backward UnsortedSegmentSum Elapsed: "
                      << t.gettime_ms() / iter << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward UnsortedSegmentSum Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(input_grad_dev->FromGPU(GetStream(), input_grad.data()) != 0)
    {
        std::cerr << "Error copying input_grad_dev from GPU, size: " << input_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tseg>
int UnsortedSegmentSumDriver<Tgpu, Tref, Tseg>::RunBackwardCPU()
{
    int status = miopenStatusSuccess;

    status = mloUnsortedSegmentSumBackwardRunHost<Tgpu, Tref, Tseg>(InputGradDesc,
                                                                    output_grad.data(),
                                                                    input_grad_host.data(),
                                                                    segment_ids.data(),
                                                                    num_segments);
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloUnsortedSegmentSumBackwardRunHost");

    return status;
}

template <typename Tgpu, typename Tref, typename Tseg>
Tref UnsortedSegmentSumDriver<Tgpu, Tref, Tseg>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref, typename Tseg>
int UnsortedSegmentSumDriver<Tgpu, Tref, Tseg>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(output_host, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward UnsortedSegmentSum Verifies FAILED: " << error << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward UnsortedSegmentSum Verifies OK on CPU reference " << "error:" << error
                  << " < " << tolerance << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tseg>
int UnsortedSegmentSumDriver<Tgpu, Tref, Tseg>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(input_grad_host, input_grad);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward UnsortedSegmentSum Verifies FAILED: " << error << " > " << tolerance
                  << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward UnsortedSegmentSum Verifies OK on CPU reference "
                  << "error:" << error << " < " << tolerance << std::endl;
    }

    return miopenStatusSuccess;
}
