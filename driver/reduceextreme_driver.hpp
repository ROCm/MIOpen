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
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <numeric>
#include <vector>
#include <../test/ford.hpp>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>
#include "../src/kernels/MIOpenReduceExtreme.hpp"

template <typename T>
bool compare_equal(T r1, T r2)
{
    return r1 == r2;
}

template <typename Tgpu, typename Tcheck, ReduceExtremeOp_t op>
int32_t mloReduceExtremeForwardRunHost(miopenTensorDescriptor_t xDesc,
                                       miopenTensorDescriptor_t yDesc,
                                       miopenTensorDescriptor_t indiceDesc,
                                       Tgpu* x,
                                       Tcheck* yhost,
                                       int32_t* indicehost,
                                       int32_t dim)
{
    auto x_dims = miopen::deref(xDesc).GetLengths();
    std::vector<std::size_t> indice_dims;
    if(yhost)
        indice_dims = miopen::deref(yDesc).GetLengths();
    else
        indice_dims = miopen::deref(indiceDesc).GetLengths();

    int32_t reduce_size = static_cast<int32_t>(x_dims[dim]);
    auto indice_numel =
        std::accumulate(indice_dims.begin(), indice_dims.end(), 1LL, std::multiplies<int64_t>());

    auto inner_size =
        std::accumulate(x_dims.begin() + dim + 1, x_dims.end(), 1ULL, std::multiplies<uint64_t>());

    int32_t ret = miopenStatusSuccess;

    for(size_t o = 0; o < indice_numel; ++o)
    {
        size_t x_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

        int32_t extreme_idx = 0;
        Tcheck extreme      = static_cast<Tcheck>(x[x_idx]);

        for(int32_t i = 1; i < reduce_size; ++i)
        {
            x_idx += inner_size;
            Tcheck val = static_cast<Tcheck>(x[x_idx]);
            reduce_func<Tcheck, int32_t, op>{}.calculate(extreme, val, extreme_idx, i);
        }
        indicehost[o] = extreme_idx;
        if(yhost)
            yhost[o] = extreme;
    }
    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloReduceExtremeAminmaxBackwardRunHost(const miopenTensorDescriptor_t xDesc,
                                               const miopenTensorDescriptor_t xGradDesc,
                                               const miopenTensorDescriptor_t yDesc,
                                               const miopenTensorDescriptor_t yGradDesc,
                                               const miopenTensorDescriptor_t countDesc,
                                               const Tgpu* x,
                                               Tcheck* x_grad,
                                               const Tgpu* y,
                                               const Tgpu* y_grad,
                                               const int32_t* count,
                                               const int32_t* dims)
{
    auto x_tv      = miopen::get_inner_expanded_tv<5>(miopen::deref(xDesc));
    auto x_grad_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(xGradDesc));
    auto y_tv      = miopen::get_inner_expanded_tv<5>(miopen::deref(yDesc));
    auto y_grad_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(yGradDesc));
    auto count_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(countDesc));

    auto N = miopen::deref(xDesc).GetElementSize();
    par_ford(N)([&](size_t gid) {
        uint64_t oN, oC, oD, oH, oW;
        tensor_layout_t<5> tensor_layout(x_tv, gid);

        oN = dims[0] ? 0 : tensor_layout.layout[0];
        oC = dims[1] ? 0 : tensor_layout.layout[1];
        oD = dims[2] ? 0 : tensor_layout.layout[2];
        oH = dims[3] ? 0 : tensor_layout.layout[3];
        oW = dims[4] ? 0 : tensor_layout.layout[4];

        int32_t minmax_count = count[count_tv.get_tensor_view_idx({oN, oC, oD, oH, oW})];

        double temp =
            (static_cast<double>(x[x_tv.get_tensor_view_idx(tensor_layout)]) ==
             static_cast<double>(y[y_tv.get_tensor_view_idx({oN, oC, oD, oH, oW})]))
                ? static_cast<double>(y_grad[y_grad_tv.get_tensor_view_idx({oN, oC, oD, oH, oW})]) /
                      minmax_count
                : 0;
        x_grad[x_grad_tv.get_tensor_view_idx(tensor_layout)] = static_cast<Tcheck>(temp);
    });

    return miopenStatusSuccess;
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
        miopenCreateTensorDescriptor(&xGradDesc);
        miopenCreateTensorDescriptor(&yGradDesc);
        miopenCreateTensorDescriptor(&countDesc);
        miopenCreateTensorDescriptor(&dimsDesc);

        data_type       = miopen_type<Tgpu>{};
        int32_data_type = miopen_type<int32_t>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> input);
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
    ~ReduceExtremeDriver() override
    {
        miopenDestroyTensorDescriptor(xDesc);
        miopenDestroyTensorDescriptor(yDesc);
        miopenDestroyTensorDescriptor(indiceDesc);
        miopenDestroyTensorDescriptor(xGradDesc);
        miopenDestroyTensorDescriptor(yGradDesc);
        miopenDestroyTensorDescriptor(countDesc);
        miopenDestroyTensorDescriptor(dimsDesc);
    }

private:
    InputFlags inflags;
    int forw;

    miopenTensorDescriptor_t xDesc;
    miopenTensorDescriptor_t yDesc;
    miopenTensorDescriptor_t indiceDesc;
    miopenTensorDescriptor_t xGradDesc;
    miopenTensorDescriptor_t yGradDesc;
    miopenTensorDescriptor_t countDesc;
    miopenTensorDescriptor_t dimsDesc;

    std::unique_ptr<GPUMem> x_dev;
    std::unique_ptr<GPUMem> indice_dev;
    std::unique_ptr<GPUMem> y_dev;
    std::unique_ptr<GPUMem> x_grad_dev;
    std::unique_ptr<GPUMem> y_grad_dev;
    std::unique_ptr<GPUMem> count_dev;
    std::unique_ptr<GPUMem> dims_dev;

    std::vector<Tgpu> x;
    std::vector<Tgpu> y;
    std::vector<Tref> yhost;
    std::vector<int32_t> indice;
    std::vector<int32_t> indicehost;
    std::vector<Tgpu> x_grad;
    std::vector<Tgpu> y_grad;
    std::vector<Tref> x_gradhost;
    std::vector<int32_t> count;
    std::vector<int32_t> dims;

    int dim;
    miopenReduceExtremeOp_t reduceExtremeOp;
    miopenDataType_t int32_data_type;
    bool isContiguous;
};

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    isContiguous     = inflags.GetValueInt("is-contiguous") == 1 ? true : false;
    forw             = inflags.GetValueInt("forw");
    reduceExtremeOp  = static_cast<miopenReduceExtremeOp_t>(inflags.GetValueInt("ReduceExtremeOp"));
    dim              = inflags.GetValueInt("DimToReduce");
    auto dims_parsed = inflags.GetValueTensor("DimsToReduce").lengths;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    if((static_cast<ReduceExtremeOp_t>(inflags.GetValueInt("ReduceExtremeOp")) <
        ReduceExtremeOp_t::First_) ||
       (static_cast<ReduceExtremeOp_t>(inflags.GetValueInt("ReduceExtremeOp")) >
        ReduceExtremeOp_t::Last_))
    {
        std::cerr << "Error ReduceExtremeOp(1-6)" << std::endl;
        return miopenStatusBadParm;
    }

    auto in_len = inflags.GetValueTensor("input").lengths;

    if(reduceExtremeOp != MIOPEN_REDUCE_EXTREME_AMAX &&
       reduceExtremeOp != MIOPEN_REDUCE_EXTREME_AMIN)
    {
        if((inflags.GetValueInt("DimToReduce") < 0) ||
           (inflags.GetValueInt("DimToReduce") > in_len.size() - 1))
        {
            std::cerr << "Error DimToReduce(0-" << in_len.size() - 1 << ")" << std::endl;
            return miopenStatusBadParm;
        }
    }
    else
    {
        for(int each_dim : dims_parsed)
        {
            if((each_dim < 0) || (each_dim > in_len.size() - 1))
            {
                std::cerr << "Error DimsToReduce(0-" << in_len.size() - 1 << ")" << std::endl;
                return miopenStatusBadParm;
            }
        }
        // Sort and check if the dimensions to reduce are unique
        std::sort(dims_parsed.begin(), dims_parsed.end());
        if(std::adjacent_find(dims_parsed.begin(), dims_parsed.end()) != dims_parsed.end())
        {
            std::cerr << "Error DimsToReduce must be unique" << std::endl;
            return miopenStatusBadParm;
        }

        // one-hot dims
        dims = std::vector<int32_t>(in_len.size(), 0);
        for(auto&& d : dims_parsed)
        {
            dims[d] = 1;
        }
    }

    if(((forw == 0 || forw == 1) && (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_AMIN ||
                                     reduceExtremeOp == MIOPEN_REDUCE_EXTREME_AMAX)) ||
       ((forw == 0 || forw == 2) && (reduceExtremeOp != MIOPEN_REDUCE_EXTREME_AMIN &&
                                     reduceExtremeOp != MIOPEN_REDUCE_EXTREME_AMAX)))
    {
        std::cerr << "Error: MIN and MAX, ARGMIN and ARGMAX are only supported in forward mode, "
                     "AMIN and AMAX are only supported in backward mode"
                  << std::endl;
        return miopenStatusNotImplemented;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len    = inflags.GetValueTensor("input").lengths;
    std::vector<int> in_stride = ComputeStrides(in_len);
    std::vector<int> out_len;

    if(reduceExtremeOp != MIOPEN_REDUCE_EXTREME_AMIN &&
       reduceExtremeOp != MIOPEN_REDUCE_EXTREME_AMAX)
    {
        for(int i = 0; i < in_len.size(); ++i)
        {
            if(i != dim)
            {
                out_len.push_back(in_len[i]);
            }
        }
    }
    else
    {
        for(int i = 0; i < in_len.size(); ++i)
        {
            if(dims[i] == 0)
            {
                out_len.push_back(in_len[i]);
            }
        }
    }
    if(out_len.empty())
        out_len.push_back(1);

    if(SetTensorNd(xDesc, in_len, in_stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing x tensor: " + inflags.GetValueStr("input") + ".");
    if(SetTensorNd(yDesc, out_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting y tensor.");
    if(SetTensorNd(indiceDesc, out_len, int32_data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting indice tensor.");
    if(SetTensorNd(xGradDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting xGrad tensor.");
    if(SetTensorNd(yGradDesc, out_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting yGrad tensor.");
    if(SetTensorNd(countDesc, out_len, int32_data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting count tensor.");
    if(SetTensorNd(dimsDesc, out_len, int32_data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting dims tensor.");

    return 0;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> ReduceExtremeDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
{
    if(!isContiguous)
        std::swap(inputDim.front(), inputDim.back());
    std::vector<int> strides(inputDim.size());
    strides.back() = 1;
    for(int i = inputDim.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * inputDim[i + 1];
    if(!isContiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward ReduceExtreme (Default=1)", "int");
    inflags.AddTensorFlag("input", 'X', "21x500x375", "input tensor descriptor");
    inflags.AddInputFlag(
        "DimToReduce", 'R', "0", "The indice of the dimensions to be reduced(Default=0)", "int");
    inflags.AddTensorFlag("DimsToReduce",
                          'D',
                          "0",
                          "The indices of the dimensions to be reduced. This option must be used "
                          "in backward mode and run both backward and forward (Default=0).");
    inflags.AddInputFlag("ReduceExtremeOp",
                         'O',
                         "1",
                         "Reduce Extreme Operation Type (check the enum miopenReduceExtremeOp_t in "
                         "miopen.h) (Default=1 to Find the the minimum index)",
                         "int");
    inflags.AddInputFlag("is-contiguous", 'C', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
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

    for(int32_t i = 0; i < in_sz; ++i)
    {
        x[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1.0), static_cast<Tgpu>(1.0));
    }

    if(x_dev->ToGPU(GetStream(), x.data()) != 0)
    {
        std::cerr << "Error copying (x) to GPU, size: " << x_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }
    if(indice_dev->ToGPU(GetStream(), indice.data()) != 0)
    {
        std::cerr << "Error copying (indice) to GPU, size: " << indice_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }
    if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
       (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX))
    {
        y_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
        y     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
        yhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

        if(y_dev->ToGPU(GetStream(), y.data()) != 0)
        {
            std::cerr << "Error copying (y) to GPU, size: " << y_dev->GetSize() << std::endl;
            return miopenStatusAllocFailed;
        }
    }
    else if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_AMIN) ||
            (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_AMAX))
    {
        y_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
        x_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        y_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
        count_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(int32_t)));
        dims_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(int32_t)));

        y          = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
        x_grad     = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
        y_grad     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
        x_gradhost = std::vector<Tref>(in_sz, static_cast<Tref>(0));
        count      = std::vector<int32_t>(out_sz, static_cast<int32_t>(0));

        for(int32_t i = 0; i < out_sz; ++i)
        {
            y[i]      = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1.0), static_cast<Tgpu>(1.0));
            y_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1.0), static_cast<Tgpu>(1.0));
            count[i]  = prng::gen_A_to_B<int32_t>(1, 10);
        }

        if(y_dev->ToGPU(GetStream(), y.data()) != 0)
        {
            std::cerr << "Error copying (y) to GPU, size: " << y_dev->GetSize() << std::endl;
            return miopenStatusAllocFailed;
        }
        if(x_grad_dev->ToGPU(GetStream(), x_grad.data()) != 0)
        {
            std::cerr << "Error copying (x_grad) to GPU, size: " << x_grad_dev->GetSize()
                      << std::endl;
            return miopenStatusAllocFailed;
        }
        if(y_grad_dev->ToGPU(GetStream(), y_grad.data()) != 0)
        {
            std::cerr << "Error copying (y_grad) to GPU, size: " << y_grad_dev->GetSize()
                      << std::endl;
            return miopenStatusAllocFailed;
        }
        if(count_dev->ToGPU(GetStream(), count.data()) != 0)
        {
            std::cerr << "Error copying (count) to GPU, size: " << count_dev->GetSize()
                      << std::endl;
            return miopenStatusAllocFailed;
        }
        if(dims_dev->ToGPU(GetStream(), dims.data()) != 0)
        {
            std::cerr << "Error copying (dims) to GPU, size: " << dims_dev->GetSize() << std::endl;
            return miopenStatusAllocFailed;
        }
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

    for(int32_t i = 0; i < inflags.GetValueInt("iter"); ++i)
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
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward ReduceExtreme Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(indice_dev->FromGPU(GetStream(), indice.data()) != 0)
    {
        std::cerr << "Error copying (indice_dev) from GPU, size: " << indice_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
       (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX))
    {
        if(y_dev->FromGPU(GetStream(), y.data()) != 0)
        {
            std::cerr << "Error copying (y_dev) from GPU, size: " << y_dev->GetSize() << std::endl;
            return miopenStatusInternalError;
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_ARGMIN)
    {
        return mloReduceExtremeForwardRunHost<Tgpu, Tref, ReduceExtremeOp_t::Argmin>(
            xDesc, nullptr, indiceDesc, x.data(), nullptr, indicehost.data(), dim);
    }
    else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_ARGMAX)
    {
        return mloReduceExtremeForwardRunHost<Tgpu, Tref, ReduceExtremeOp_t::Argmax>(
            xDesc, nullptr, indiceDesc, x.data(), nullptr, indicehost.data(), dim);
    }
    else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN)
    {
        return mloReduceExtremeForwardRunHost<Tgpu, Tref, ReduceExtremeOp_t::Min>(
            xDesc, yDesc, indiceDesc, x.data(), yhost.data(), indicehost.data(), dim);
    }
    else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX)
    {
        return mloReduceExtremeForwardRunHost<Tgpu, Tref, ReduceExtremeOp_t::Max>(
            xDesc, yDesc, indiceDesc, x.data(), yhost.data(), indicehost.data(), dim);
    }

    return miopenStatusInternalError;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int32_t i = 0; i < inflags.GetValueInt("iter"); ++i)
    {
        if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_AMIN) ||
           (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_AMAX))
        {
            miopenReduceExtremeBackward(GetHandle(),
                                        xDesc,
                                        x_dev->GetMem(),
                                        xGradDesc,
                                        x_grad_dev->GetMem(),
                                        yDesc,
                                        y_dev->GetMem(),
                                        yGradDesc,
                                        y_grad_dev->GetMem(),
                                        dimsDesc,
                                        dims_dev->GetMem(),
                                        reduceExtremeOp,
                                        countDesc,
                                        count_dev->GetMem());
        }
        // leave the else for future backward ops

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
            std::cout << "Wall-clock Time Backward ReduceExtreme Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward ReduceExtreme Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(x_grad_dev->FromGPU(GetStream(), x_grad.data()) != 0)
    {
        std::cerr << "Error copying (x_grad_dev) from GPU, size: " << x_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::RunBackwardCPU()
{
    if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_AMIN ||
       reduceExtremeOp == MIOPEN_REDUCE_EXTREME_AMAX)
    {
        return mloReduceExtremeAminmaxBackwardRunHost<Tgpu, Tref>(xDesc,
                                                                  xGradDesc,
                                                                  yDesc,
                                                                  yGradDesc,
                                                                  countDesc,
                                                                  x.data(),
                                                                  x_gradhost.data(),
                                                                  y.data(),
                                                                  y_grad.data(),
                                                                  count.data(),
                                                                  dims.data());
    }

    return miopenStatusInternalError;
}

template <typename Tgpu, typename Tref>
Tref ReduceExtremeDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
       (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX))
    {
        const Tref tolerance = GetTolerance();
        auto error           = miopen::rms_range(yhost, y);

        if(!std::isfinite(error) || error > tolerance)
        {
            std::cout << "Forward ReduceExtreme FAILED: " << error << " > " << tolerance
                      << std::endl;
            return EC_VerifyFwd;
        }
        else
        {
            std::cout << "Forward ReduceExtreme Verifies on CPU (" << error << " < " << tolerance
                      << ')' << std::endl;
        }
    }
    auto error_idx = miopen::mismatch_idx(indicehost, indice, compare_equal<int32_t>);

    if(error_idx < miopen::range_distance(indicehost))
    {
        std::cout << "Forward ReduceExtreme FAILED: Indice does not equal at " << error_idx
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward ReduceExtreme Indice Verifies on CPU and GPU" << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();

    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(x_gradhost, x_grad);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward ReduceExtreme FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward ReduceExtreme Verifies on CPU (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
