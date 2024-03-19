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
#ifndef GUARD_MIOPEN_REDUCEEXTREME_DRIVER_HPP
#define GUARD_MIOPEN_REDUCEEXTREME_DRIVER_HPP

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
int32_t mloArgmaxForwardRunHost(miopenTensorDescriptor_t xDesc,
                                miopenTensorDescriptor_t indiceDesc,
                                Tgpu* x,
                                int32_t* indicehost,
                                int32_t dim)
{
    auto x_dims      = miopen::deref(xDesc).GetLengths();
    auto indice_dims = miopen::deref(indiceDesc).GetLengths();

    int32_t reduce_size = static_cast<int32_t>(x_dims[dim]);
    auto indice_numel =
        std::accumulate(indice_dims.begin(), indice_dims.end(), 1L, std::multiplies<int64_t>());

    auto inner_size =
        std::accumulate(x_dims.begin() + dim + 1, x_dims.end(), 1ULL, std::multiplies<uint64_t>());

    int32_t ret = 0;

    for(size_t o = 0; o < indice_numel; o++)
    {
        size_t x_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

        int32_t max_idx = 0;
        Tcheck max      = static_cast<Tcheck>(x[x_idx]);

        for(int32_t i = 1; i < reduce_size; i++)
        {
            x_idx += inner_size;
            Tcheck val = static_cast<Tcheck>(x[x_idx]);
            if(max < val)
            {
                max     = val;
                max_idx = i;
            }
        }
        indicehost[o] = max_idx;
    }
    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloArgminForwardRunHost(miopenTensorDescriptor_t xDesc,
                                miopenTensorDescriptor_t indiceDesc,
                                Tgpu* x,
                                int32_t* indicehost,
                                int32_t dim)
{
    auto x_dims      = miopen::deref(xDesc).GetLengths();
    auto indice_dims = miopen::deref(indiceDesc).GetLengths();

    int32_t reduce_size = static_cast<int32_t>(x_dims[dim]);
    auto indice_numel =
        std::accumulate(indice_dims.begin(), indice_dims.end(), 1L, std::multiplies<int64_t>());

    auto inner_size =
        std::accumulate(x_dims.begin() + dim + 1, x_dims.end(), 1ULL, std::multiplies<uint64_t>());

    int32_t ret = 0;

    for(size_t o = 0; o < indice_numel; o++)
    {
        size_t x_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

        int32_t min_idx = 0;
        Tcheck min      = static_cast<Tcheck>(x[x_idx]);

        for(int32_t i = 1; i < reduce_size; i++)
        {
            x_idx += inner_size;
            Tcheck val = static_cast<Tcheck>(x[x_idx]);
            if(min > val)
            {
                min     = val;
                min_idx = i;
            }
        }
        indicehost[o] = min_idx;
    }
    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloMaxForwardRunHost(miopenTensorDescriptor_t xDesc,
                             miopenTensorDescriptor_t yDesc,
                             Tgpu* x,
                             Tgpu* yhost,
                             int32_t* indicehost,
                             int32_t dim)
{
    auto x_dims = miopen::deref(xDesc).GetLengths();
    auto y_dims = miopen::deref(yDesc).GetLengths();

    int32_t reduce_size = static_cast<int32_t>(x_dims[dim]);
    auto output_numel =
        std::accumulate(y_dims.begin(), y_dims.end(), 1L, std::multiplies<int64_t>());

    auto inner_size =
        std::accumulate(x_dims.begin() + dim + 1, x_dims.end(), 1ULL, std::multiplies<uint64_t>());

    int32_t ret = 0;

    for(size_t o = 0; o < output_numel; o++)
    {
        size_t x_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

        int32_t max_idx = 0;
        Tcheck max      = static_cast<Tcheck>(x[x_idx]);

        for(int32_t i = 1; i < reduce_size; i++)
        {
            x_idx += inner_size;
            Tcheck val = static_cast<Tcheck>(x[x_idx]);
            if(max < val)
            {
                max     = val;
                max_idx = i;
            }
        }
        indicehost[o] = max_idx;
        yhost[o]      = max;
    }
    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloMinForwardRunHost(miopenTensorDescriptor_t xDesc,
                             miopenTensorDescriptor_t yDesc,
                             Tgpu* x,
                             Tgpu* yhost,
                             int32_t* indicehost,
                             int32_t dim)
{
    auto x_dims = miopen::deref(xDesc).GetLengths();
    auto y_dims = miopen::deref(yDesc).GetLengths();

    int32_t reduce_size = static_cast<int32_t>(x_dims[dim]);
    auto output_numel =
        std::accumulate(y_dims.begin(), y_dims.end(), 1L, std::multiplies<int64_t>());

    auto inner_size =
        std::accumulate(x_dims.begin() + dim + 1, x_dims.end(), 1ULL, std::multiplies<uint64_t>());

    int32_t ret = 0;

    for(size_t o = 0; o < output_numel; o++)
    {
        size_t x_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

        int32_t min_idx = 0;
        Tcheck min      = static_cast<Tcheck>(x[x_idx]);

        for(int32_t i = 1; i < reduce_size; i++)
        {
            x_idx += inner_size;
            Tcheck val = static_cast<Tcheck>(x[x_idx]);
            if(min > val)
            {
                min     = val;
                min_idx = i;
            }
        }
        indicehost[o] = min_idx;
        yhost[o]      = min;
    }
    return ret;
}

template <typename Tgpu, typename Tref>
class ReduceExtremeDriver : public Driver
{
public:
    ReduceExtremeDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&xDesc);
        miopenCreateTensorDescriptor(&yDesc);
        miopenCreateTensorDescriptor(&indiceDesc);

        data_type        = miopen_type<Tgpu>{};
        indice_data_type = miopen_type<int32_t>{};
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
    ~ReduceExtremeDriver() override
    {
        miopenDestroyTensorDescriptor(xDesc);
        miopenDestroyTensorDescriptor(yDesc);
        miopenDestroyTensorDescriptor(indiceDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t xDesc;
    miopenTensorDescriptor_t yDesc;
    miopenTensorDescriptor_t indiceDesc;

    std::unique_ptr<GPUMem> x_dev;
    std::unique_ptr<GPUMem> indice_dev;
    std::unique_ptr<GPUMem> y_dev;

    std::vector<Tgpu> x;
    std::vector<Tgpu> y;
    std::vector<Tgpu> yhost;
    std::vector<int32_t> indice;
    std::vector<int32_t> indicehost;

    int dim;
    miopenReduceExtremeOp_t reduceExtremeOp;

    miopenDataType_t indice_data_type;
};

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();
    dim                     = inflags.GetValueInt("DimToReduce");
    reduceExtremeOp = static_cast<miopenReduceExtremeOp_t>(inflags.GetValueInt("ReduceExtremeOp"));

    SetTensorNd(xDesc, in_len, data_type);

    std::vector<int> out_len;

    for(int i = 0; i < in_len.size(); i++)
    {
        if(i != dim)
        {
            out_len.push_back(in_len[i]);
        }
    }

    if(out_len.empty())
        out_len.push_back(1);

    SetTensorNd(yDesc, out_len, data_type);
    SetTensorNd(indiceDesc, out_len, indice_data_type);

    return 0;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward ReduceExtreme (Default=1)", "int");
    inflags.AddInputFlag("batchsize", 'n', "21", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "500", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_d", 'D', "0", "Input Depth (Default=0)", "int");
    inflags.AddInputFlag("in_h", 'H', "0", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "375", "Input Width (Default=32)", "int");
    inflags.AddInputFlag(
        "DimToReduce", 'R', "0", "The indice of the dimensions to be reduced(Default=1)", "int");
    inflags.AddInputFlag("ReduceExtremeOp",
                         'O',
                         "0",
                         "Reduce Extreme Operation Type (check the enum miopenReduceExtremeOp_t in "
                         "miopen.h) (Default=0 to Find the the minimum index)",
                         "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> ReduceExtremeDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_w = inflags.GetValueInt("in_w");
    int in_h = inflags.GetValueInt("in_h");
    int in_d = inflags.GetValueInt("in_d");

    if((in_n != 0) && (in_c != 0) && (in_d != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_d, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_w});
    }
    else if((in_n != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_w});
    }
    else if(in_n != 0)
    {
        return std::vector<int>({in_n});
    }
    else
    {
        std::cerr << "Error Input Tensor Lengths\n" << std::endl;
        return std::vector<int>({0});
    }
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(xDesc);
    size_t out_sz = GetTensorSize(yDesc);

    uint32_t ctx = 0;

    x_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    indice_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(int32_t)));

    x          = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    indice     = std::vector<int32_t>(out_sz, static_cast<int32_t>(0));
    indicehost = std::vector<int32_t>(out_sz, static_cast<int32_t>(0));

    for(int32_t i = 0; i < in_sz; i++)
    {
        x[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    if(x_dev->ToGPU(GetStream(), x.data()) != 0)
        std::cerr << "Error copying (x) to GPU, size: " << x_dev->GetSize() << std::endl;

    if(indice_dev->ToGPU(GetStream(), indice.data()) != 0)
        std::cerr << "Error copying (indice) to GPU, size: " << indice_dev->GetSize() << std::endl;

    if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
       (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX))
    {
        y_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
        y     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
        yhost = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

        if(y_dev->ToGPU(GetStream(), y.data()) != 0)
            std::cerr << "Error copying (y) to GPU, size: " << y_dev->GetSize() << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int32_t i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
           (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX))
        {
            miopenReduceExtremeForward(GetHandle(),
                                       xDesc,
                                       x_dev->GetMem(),
                                       dim,
                                       reduceExtremeOp,
                                       yDesc,
                                       y_dev->GetMem(),
                                       indiceDesc,
                                       indice_dev->GetMem());
        }
        else
        {
            miopenReduceExtremeForward(GetHandle(),
                                       xDesc,
                                       x_dev->GetMem(),
                                       dim,
                                       reduceExtremeOp,
                                       nullptr,
                                       nullptr,
                                       indiceDesc,
                                       indice_dev->GetMem());
        }

        float time = 0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int32_t iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Forward ReduceExtreme Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward ReduceExtreme Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(indice_dev->FromGPU(GetStream(), indice.data()) != 0)
        std::cerr << "Error copying (indice_dev) from GPU, size: " << indice_dev->GetSize()
                  << std::endl;

    if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
       (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX))
    {
        if(y_dev->FromGPU(GetStream(), y.data()) != 0)
            std::cerr << "Error copying (y_dev) from GPU, size: " << y_dev->GetSize() << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_ARGMIN)
    {
        mloArgminForwardRunHost<Tgpu, Tref>(xDesc, indiceDesc, x.data(), indicehost.data(), dim);
    }
    else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_ARGMAX)
    {
        mloArgmaxForwardRunHost<Tgpu, Tref>(xDesc, indiceDesc, x.data(), indicehost.data(), dim);
    }
    else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN)
    {
        mloMinForwardRunHost<Tgpu, Tref>(
            xDesc, yDesc, x.data(), yhost.data(), indicehost.data(), dim);
    }
    else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX)
    {
        mloMaxForwardRunHost<Tgpu, Tref>(
            xDesc, yDesc, x.data(), yhost.data(), indicehost.data(), dim);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
       (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX))
    {
        auto error = miopen::rms_range(yhost, y);

        if(!std::isfinite(error) || std::abs(static_cast<float>(error)) != 0.0f)
        {
            std::cout << "Forward ReduceExtreme FAILED: Result does not equal" << std::endl;
            return EC_VerifyFwd;
        }
        else
        {
            std::cout << "Forward ReduceExtreme Verifies on CPU and GPU (err=" << error << ")\n";
        }
    }
    auto indice_error = miopen::rms_range(indicehost, indice);

    if(!std::isfinite(indice_error) || std::abs(static_cast<float>(indice_error)) != 0.0f)
    {
        std::cout << "Forward ReduceExtreme FAILED: Indice does not equal" << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward ReduceExtreme Incide Verifies on CPU and GPU (err=" << indice_error
                  << ")\n";
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_REDUCEEXTREME_DRIVER_HPP
