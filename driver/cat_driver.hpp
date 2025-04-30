/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_CAT_DRIVER_HPP
#define GUARD_MIOPEN_CAT_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#ifndef MLO_CATHOST_H_
#define MLO_CATHOST_H_

template <typename Tgpu, typename Tcheck>
int32_t mloCatForwardRunHost(std::vector<miopenTensorDescriptor_t> inputDescs,
                             std::vector<Tgpu*> inputs,
                             miopenTensorDescriptor_t outputDesc,
                             Tcheck* outputhost,
                             uint32_t dim)
{
    auto shape             = miopen::deref(outputDesc).GetLengths();
    size_t outer_size      = 1;
    size_t inner_size      = 1;
    size_t output_dim_size = shape[dim];
    for(size_t i = 0; i < dim; i++)
    {
        outer_size *= shape[i];
    }

    for(size_t i = dim + 1; i < shape.size(); i++)
    {
        inner_size *= shape[i];
    }

    int32_t ret                = 0;
    size_t output_start_offset = 0;

    for(size_t i = 0; i < inputs.size(); i++)
    {
        auto input       = inputs[i];
        size_t dim_size  = miopen::deref(inputDescs[i]).GetLengths()[dim];
        size_t copy_size = inner_size * dim_size;
        for(size_t o = 0; o < outer_size; o++)
        {
            size_t output_offset = output_start_offset + (o * inner_size * output_dim_size);
            for(size_t j = 0; j < copy_size; j++)
            {
                outputhost[output_offset + j] = input[copy_size * o + j];
            }
        }
        output_start_offset += copy_size;
    }

    return ret;
}

#endif

template <typename Tgpu, typename Tref = Tgpu>
class CatDriver : public Driver
{
public:
    CatDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&outputDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<std::vector<int>> GetInputTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;

    int VerifyBackward() override;
    int VerifyForward() override;
    ~CatDriver() override
    {
        for(auto inputDesc : inputDescs)
        {
            miopenDestroyTensorDescriptor(inputDesc);
        }
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;
    uint32_t dim;

    std::vector<miopenTensorDescriptor_t> inputDescs;
    miopenTensorDescriptor_t outputDesc;

    std::vector<std::unique_ptr<GPUMem>> in_devs;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<std::vector<Tgpu>> ins;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;

    std::vector<void*> in_devs_ptr;
    std::vector<Tgpu*> ins_ptr;
};

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::GetandSetData()
{
    miopenTensorDescriptor_t inputDesc;
    size_t output_dim_size = 0;
    auto in_lens           = GetInputTensorLengthsFromCmdLine();
    dim                    = inflags.GetValueInt("dim");

    for(auto in_len : in_lens)
    {
        miopenCreateTensorDescriptor(&inputDesc);
        SetTensorNd(inputDesc, in_len, data_type);
        inputDescs.push_back(inputDesc);
        output_dim_size += in_len[dim];
    }
    auto out_len = in_lens[0];
    out_len[dim] = output_dim_size;

    SetTensorNd(outputDesc, out_len, data_type);

    return 0;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Cat (Default=1)", "int");
    inflags.AddTensorFlag("input1", '1', "2x32x128x128x128", "input1 tensor descriptor");
    inflags.AddTensorFlag("input2", '2', "2x32x128x128x128", "input2 tensor descriptor");
    inflags.AddTensorFlag("input3", '3', "", "input3 tensor descriptor");
    inflags.AddTensorFlag("input4", '4', "", "input4 tensor descriptor");
    inflags.AddTensorFlag("input5", '5', "", "input5 tensor descriptor");
    inflags.AddTensorFlag("input6", '6', "", "input6 tensor descriptor");
    inflags.AddTensorFlag("input7", '7', "", "input7 tensor descriptor");
    inflags.AddTensorFlag("input8", '8', "", "input8 tensor descriptor");
    inflags.AddInputFlag("dim", 'd', "0", "Concatenation dimension (Default=0)", "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<std::vector<int>> CatDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    const int max_input_count = 8;
    std::vector<std::vector<int>> ret;
    std::string name = "input";
    for(int i = 1; i < max_input_count; i++)
    {
        auto tensor = inflags.GetValueTensor(name + std::to_string(i));
        if(!tensor.lengths.empty())
            ret.push_back(tensor.lengths);
    }
    return ret;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;
    for(auto& inputDesc : inputDescs)
    {
        auto in_sz = GetTensorSize(inputDesc);
        in_devs.push_back(std::make_unique<GPUMem>(ctx, in_sz, sizeof(Tgpu)));
        ins.push_back(std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0)));
        auto& in    = ins.back();
        auto in_dev = in_devs.back().get();

        for(int i = 0; i < in_sz; i++)
        {
            in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        if(in_dev->ToGPU(GetStream(), in.data()) != 0)
            std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;
        in_devs_ptr.push_back(in_dev->GetMem());
        ins_ptr.push_back(in.data());
    }

    size_t out_sz = GetTensorSize(outputDesc);

    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    out     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenCatForward(GetHandle(),
                         inputDescs.size(),
                         inputDescs.data(),
                         in_devs_ptr.data(),
                         outputDesc,
                         out_dev->GetMem(),
                         dim);

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
            printf("Wall-clock Time Forward Cat Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward Cat Elapsed: %f ms\n", kernel_average_time);
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloCatForwardRunHost<Tgpu, Tref>(inputDescs, ins_ptr, outputDesc, outhost.data(), dim);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    auto error = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error != 0)
    {
        std::cout << "Forward Cat FAILED: " << error << " > 0" << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Cat Verifies OK on CPU reference" << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_CAT_DRIVER_HPP
