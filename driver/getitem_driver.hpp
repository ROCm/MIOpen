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
#ifndef GUARD_MIOPEN_GETITEM_DRIVER_HPP
#define GUARD_MIOPEN__DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
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

template <typename Tgpu, typename Tcheck>
int32_t mloGetitemBackwardRunHost(miopenTensorDescriptor_t dyDesc,
                                  miopenTensorDescriptor_t xDesc,
                                  miopenTensorDescriptor_t yDesc,
                                  miopenTensorDescriptor_t indexDesc,
                                  miopenTensorDescriptor_t dxDesc,
                                  Tgpu* x,
                                  Tgpu* y,
                                  int32_t* index,
                                  Tgpu* dy,
                                  Tref* dxhost,
                                  int32_t dim)
{
    // auto x_dims  = miopen::deref(xDesc).GetLengths();
    // auto y_dims = miopen::deref(yDesc).GetLengths();

    // int32_t reduce_size = static_cast<int32_t>(x_dims[dim]);
    // auto output_numel =
    //     std::accumulate(y_dims.begin(), y_dims.end(), 1L, std::multiplies<int64_t>());

    // auto inner_size = std::accumulate(
    //     x_dims.begin() + dim + 1, x_dims.end(), 1ULL, std::multiplies<uint64_t>());

    // int32_t ret = 0;

    // for(size_t o = 0; o < output_numel; o++)
    // {
    //     size_t x_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

    //     int32_t max_idx = 0;
    //     Tcheck max      = static_cast<Tcheck>(x[x_idx]);

    //     for(int32_t i = 1; i < reduce_size; i++)
    //     {
    //         x_idx += inner_size;
    //         Tcheck val = static_cast<Tcheck>(x[x_idx]);
    //         if(max < val)
    //         {
    //             max     = val;
    //             max_idx = i;
    //         }
    //     }
    //     yhost[o] = max_idx;
    // }
    return ret;
}

template <typename Tgpu, typename Tref>
class GetitemDriver : public Driver
{
public:
    GetitemDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&dyDesc);
        miopenCreateTensorDescriptor(&xDesc);
        miopenCreateTensorDescriptor(&yDesc);
        miopenCreateTensorDescriptor(&indexDesc);
        miopenCreateTensorDescriptor(&dxDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;

    int VerifyBackward() override;
    int VerifyForward() override;
    ~GetitemDriver() override
    {
        miopenDestroyTensorDescriptor(dyDesc);
        miopenDestroyTensorDescriptor(xDesc);
        miopenDestroyTensorDescriptor(yDesc);
        miopenDestroyTensorDescriptor(indexDesc);
        miopenDestroyTensorDescriptor(dxDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t dyDesc;
    miopenTensorDescriptor_t xDesc;
    miopenTensorDescriptor_t yDesc;
    miopenTensorDescriptor_t indexDesc;
    miopenTensorDescriptor_t dxDesc;

    std::unique_ptr<GPUMem> dy_dev;
    std::unique_ptr<GPUMem> x_dev;
    std::unique_ptr<GPUMem> y_dev;
    std::unique_ptr<GPUMem> index_dev;
    std::unique_ptr<GPUMem> dx_dev;

    std::vector<Tgpu> dy;
    std::vector<Tgpu> x;
    std::vector<Tgpu> y;
    std::vector<int32_t> index;
    std::vector<Tgpu> dx;
    std::vector<Tref> dxhost;

    int32_t dim;
};

template <typename Tgpu, typename Tref>
int GetitemDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GetitemDriver<Tgpu, Tref>::GetandSetData()
{
    auto dyTensorParam    = inflags.GetValueTensor("doutput");
    auto xTensorParam     = inflags.GetValueTensor("input");
    auto yTensorParam     = inflags.GetValueTensor("output");
    auto indexTensorParam = inflags.GetValueTensor("index");
    auto dxTensorParam    = inflags.GetValueTensor("dinput");
    dim                   = inflags.GetValueInt("Dim");

    dim_size = inflags.GetValueInt("Dim");

    if(SetTensorNd(dyDesc, dyTensorParam.lengths, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing doutput tensor: " + inflags.GetValueStr("doutput") + ".");

    if(SetTensorNd(xDesc, xTensorParam.lengths, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input") + ".");

    if(SetTensorNd(yDesc, yTensorParam.lengths, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor: " + inflags.GetValueStr("output") + ".");

    if(SetTensorNd(indexDesc, indexTensorParam.lengths, miopenInt32) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing index tensor: " + inflags.GetValueStr("index") + ".");

    if(SetTensorNd(dxDesc, dxTensorParam.lengths, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing dinput tensor: " + inflags.GetValueStr("dinput") + ".");

    return 0;
}

template <typename Tgpu, typename Tref>
int GetitemDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Getitem (Default=1)", "int");
    inflags.AddTensorFlag("doutput", 'O', "100x3x32x32", "doutput tensor descriptor");
    inflags.AddTensorFlag("input", 'X', "100x3x32x32", "input tensor descriptor");
    inflags.AddTensorFlag("output", 'Y', "100x3x32x32", "output tensor descriptor");
    inflags.AddTensorFlag("indexs", 'D', "100x3x32x32", "index tensors descriptor");
    inflags.AddTensorFlag("dinput", 'N', "100x3x32x32", "dinput tensor descriptor");

    inflags.AddInputFlag("Dim", '2', "0", "The dimension(Default=1)", "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GetitemDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t dy_sz    = GetTensorSize(dyDesc);
    size_t x_sz     = GetTensorSize(xDesc);
    size_t y_sz     = GetTensorSize(yDesc);
    size_t index_sz = GetTensorSize(indexDesc);
    size_t dx_sz    = GetTensorSize(dxDesc);

    uint32_t ctx = 0;

    dy_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, dy_sz, sizeof(Tgpu)));
    x_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, x_sz, sizeof(Tgpu)));
    y_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, y_sz, sizeof(Tgpu)));
    index_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, index_sz, sizeof(int32_t)));
    dx_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, dx_sz, sizeof(Tgpu)));

    dy     = std::vector<Tgpu>(index_sz, static_cast<Tgpu>(0));
    x      = std::vector<Tgpu>(x_sz, static_cast<Tgpu>(0));
    y      = std::vector<Tgpu>(y_sz, static_cast<Tgpu>(0));
    index  = std::vector<int32_t>(x_sz, static_cast<int32_t>(0));
    dx     = std::vector<Tgpu>(dy_sz, static_cast<Tgpu>(0));
    dxhost = std::vector<Tref>(dx_sz, static_cast<Tref>(0));

    for(int32_t i = 0; i < dy_sz; i++)
    {
        dy[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1.0), static_cast<Tgpu>(1.0));
    }

    for(int32_t i = 0; i < x_sz; i++)
    {
        x[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1.0), static_cast<Tgpu>(1.0));
    }

    for(int32_t i = 0; i < y_sz; i++)
    {
        y[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1.0), static_cast<Tgpu>(1.0));
    }

    for(int32_t i = 0; i < index_sz; i++)
    {
        index[i] = i;
    }

    if(dy_dev->ToGPU(GetStream(), dy.data()) != 0)
        std::cerr << "Error copying (dy) to GPU, size: " << dy_dev->GetSize() << std::endl;

    if(x_dev->ToGPU(GetStream(), x.data()) != 0)
        std::cerr << "Error copying (x) to GPU, size: " << x_dev->GetSize() << std::endl;

    if(y_dev->ToGPU(GetStream(), y.data()) != 0)
        std::cerr << "Error copying (y) to GPU, size: " << y_dev->GetSize() << std::endl;

    if(index_dev->ToGPU(GetStream(), index.data()) != 0)
        std::cerr << "Error copying (index) to GPU, size: " << index_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GetitemDriver<Tgpu, Tref>::RunForwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GetitemDriver<Tgpu, Tref>::RunForwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GetitemDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenGetitemForward(GetHandle(),
                             dyDesc,
                             dy_dev->GetMem(),
                             xDesc,
                             x_dev->GetMem(),
                             yDesc,
                             x_dev->GetMem(),
                             indexDesc,
                             index_dev->GetMem(),
                             dim,
                             dxDesc,
                             dx_dev->GetMem());

        float time = 0;
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
            std::cout << "Wall-clock Time Forward Getitem Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Getitem Elapsed: " << kernel_average_time << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GetitemDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloGetitemBackwardRunHost<Tgpu, Tref>(dyDesc,
                                          xDesc,
                                          yDesc,
                                          indexDesc,
                                          dxDesc,
                                          dy.data(),
                                          x.data(),
                                          y.data(),
                                          index.data(),
                                          dxhost.data(),
                                          dim);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref GetitemDriver<Tgpu, Tref>::GetTolerance()
{
    // Computation error of fp16 is ~2^13 (=8192) bigger than
    // the one of fp32 because mantissa is shorter by 13 bits.
    auto tolerance = std::is_same<Tgpu, float>::value ? 1.5e-6 : 8.2e-3;

    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 8.0;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int GetitemDriver<Tgpu, Tref>::VerifyForward()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GetitemDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();

    auto error = miopen::rms_range(dxhost, dx);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward Getitem FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward Getitem Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_GETITEM_DRIVER_HPP
