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
#define GUARD_MIOPEN_GETITEM_DRIVER_HPP

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
#include "tensor_view.h"
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

tensor_view_5d_t get_inner_expanded_tv(const miopen::TensorDescriptor Desc)
{
    auto dims    = Desc.GetLengths();
    auto strides = Desc.GetStrides();

    tensor_view_5d_t tv_5d;
    for(size_t i = 0; i < strides.size(); ++i)
    {
        tv_5d.stride[i] = strides[i];
        tv_5d.size[i]   = dims[i];
    }
    auto rest = strides.size();
    for(size_t j = rest; j < 5; ++j)
    {
        tv_5d.stride[j] = (rest == 0 ? 1 : strides[rest - 1]);
        tv_5d.size[j]   = 1;
    }
    return tv_5d;
}

void slice_tv(tensor_view_5d_t& tv_5d, int32_t sliceCount, const int32_t* slices)
{
    for(int32_t i = 0; i < sliceCount; i++)
    {
        int32_t dim   = slices[4 * i + 0];
        int32_t start = slices[4 * i + 1];
        int32_t end   = slices[4 * i + 2];
        int32_t step  = slices[4 * i + 3];

        if(end > static_cast<int32_t>(tv_5d.size[dim]))
            end = tv_5d.size[dim];

        auto len = end - start;

        tv_5d.size[dim] = (len + step - 1) / step;
        tv_5d.stride[dim] *= step;
    }
}

template <typename Tgpu, typename Tcheck>
int32_t mloGetitemBackwardRunHost(miopenTensorDescriptor_t dyDesc,
                                  miopenTensorDescriptor_t xDesc,
                                  int32_t indexCount,
                                  miopenTensorDescriptor_t* indexDescs,
                                  miopenTensorDescriptor_t yDesc,
                                  miopenTensorDescriptor_t dxDesc,
                                  miopenTensorDescriptor_t errorDesc,
                                  Tgpu* dy,
                                  Tgpu* x,
                                  Tgpu* y,
                                  int32_t** indexs,
                                  Tcheck* dxhost,
                                  int32_t* errorhost,
                                  int32_t dimCount,
                                  int32_t* dims,
                                  int32_t sliceCount,
                                  int32_t* slices,
                                  int32_t offset)
{
    auto dy_dims    = miopen::deref(dyDesc).GetLengths();
    auto dy_strides = miopen::deref(dyDesc).GetStrides();
    auto dy_numel =
        std::accumulate(dy_dims.begin(), dy_dims.end(), 1ULL, std::multiplies<int64_t>());
    auto dx_dims    = miopen::deref(dxDesc).GetLengths();
    auto index_dims = miopen::deref(indexDescs[0]).GetLengths();
    auto index_numel =
        std::accumulate(index_dims.begin(), index_dims.end(), 1ULL, std::multiplies<int64_t>());
    auto element_index = std::vector<int32_t>(indexCount * index_numel + indexCount);

    std::vector<int32_t> output_dims;
    for(int32_t i = 0; i < dimCount; i++)
    {
        output_dims.push_back(dx_dims[dims[i]]);
    }

    auto dim_info_offset = indexCount > 0 ? indexCount * index_dims[0] : 0;
    auto start_dim       = dims[0];

    auto dy_tv     = get_inner_expanded_tv(miopen::deref(dyDesc));
    auto dxhost_tv = get_inner_expanded_tv(miopen::deref(dxDesc));
    slice_tv(dxhost_tv, sliceCount, slices);

    int32_t ret = 0;

    // Get element index form indexs
    for(size_t j = 0; j < indexCount; j++)
    {
        auto index_dim = dims[j];
        auto dim_size  = output_dims[j];

        for(size_t o = 0; o < index_numel; o++)
        {
            int32_t getitem_index = indexs[j][o];

            if(getitem_index >= 0 && getitem_index < dim_size)
            {
                element_index[(o * indexCount) + j] = getitem_index;
            }
            else if(getitem_index >= -dim_size && getitem_index < 0)
            {
                element_index[(o * indexCount) + j] = getitem_index + dim_size;
            }
            else
            {
                errorhost[j] = -1;
            }

            if(o == 0)
            {
                element_index[dim_info_offset + j] = index_dim;
            }
        }
    }

    // GetItem
    for(size_t o = 0; o < dy_numel; o++)
    {
        size_t NCDHW[5], idx[5];
        GET_NCDHW(NCDHW[0], NCDHW[1], NCDHW[2], NCDHW[3], NCDHW[4], o, dy_tv);

        for(int i = 0; i < 5; i++)
        {
            idx[i] = NCDHW[i];
        }

        if(indexCount > 0)
        {
            size_t dim_cursor = NCDHW[start_dim];
            size_t i          = start_dim;
            size_t j          = 0;

            for(; i < start_dim + indexCount; ++i, ++j)
            {
                size_t dim_idx = element_index[dim_info_offset + j];
                idx[dim_idx]   = element_index[(dim_cursor * indexCount) + j];
            }

            i          = element_index[dim_info_offset + indexCount - 1] + 1;
            dim_cursor = start_dim + 1;
            for(; i < 5; ++i, ++dim_cursor)
            {
                idx[i] = NCDHW[dim_cursor];
            }
        }

        dxhost[TV5D_IDX(dxhost_tv, idx[0] + offset, idx[1], idx[2], idx[3], idx[4])] +=
            dy[TV5D_IDX(dy_tv, NCDHW[0] + offset, NCDHW[1], NCDHW[2], NCDHW[3], NCDHW[4])];
    }

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
        miopenCreateTensorDescriptor(&dxDesc);
        miopenCreateTensorDescriptor(&errorDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();

    int VerifyBackward() override;
    int VerifyForward() override;
    ~GetitemDriver() override
    {
        miopenDestroyTensorDescriptor(dyDesc);
        miopenDestroyTensorDescriptor(xDesc);
        miopenDestroyTensorDescriptor(yDesc);
        for(auto indexDesc : indexDescs)
        {
            miopenDestroyTensorDescriptor(indexDesc);
        }
        miopenDestroyTensorDescriptor(dxDesc);
        miopenDestroyTensorDescriptor(errorDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t dyDesc;
    miopenTensorDescriptor_t xDesc;
    miopenTensorDescriptor_t yDesc;
    std::vector<miopenTensorDescriptor_t> indexDescs;
    miopenTensorDescriptor_t dxDesc;
    miopenTensorDescriptor_t errorDesc;

    std::unique_ptr<GPUMem> dy_dev;
    std::unique_ptr<GPUMem> x_dev;
    std::unique_ptr<GPUMem> y_dev;
    std::vector<std::unique_ptr<GPUMem>> index_devs;
    std::unique_ptr<GPUMem> dx_dev;
    std::unique_ptr<GPUMem> error_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> dy;
    std::vector<Tgpu> x;
    std::vector<Tgpu> y;
    std::vector<std::vector<int32_t>> indexs;
    std::vector<Tgpu> dx;
    std::vector<int32_t> error;
    std::vector<int32_t> workspace;
    std::vector<Tref> dxhost;
    std::vector<int32_t> errorhost;

    size_t ws_sizeInBytes;

    std::vector<int32_t> dims;
    std::vector<std::vector<int32_t>> slices;
    std::vector<int32_t> slices_flat;
    int32_t offset;

    std::vector<int32_t> output_dims;
    std::vector<void*> index_devs_ptr;
    std::vector<int32_t*> indexs_ptr;
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
    auto dyTensorParam   = inflags.GetValueTensor("doutput");
    auto xTensorParam    = inflags.GetValueTensor("input");
    auto yTensorParam    = inflags.GetValueTensor("output");
    auto dxTensorParam   = inflags.GetValueTensor("dinput");
    auto indexCountParam = inflags.GetValueInt("indexcount");
    auto dimCountParam   = inflags.GetValueInt("dimcount");
    auto sliceCountParam = inflags.GetValueInt("slicecount");
    offset               = inflags.GetValueInt("offset");

    auto indexTensorLengths = inflags.GetValue2dVectorInt("indexs");
    if(indexTensorLengths.size() != indexCountParam)
        MIOPEN_THROW("Error parsing indexs tensor: " + inflags.GetValueStr("indexs") + ".");

    dims = inflags.GetValueVectorInt("dims");
    if(dims.size() != dimCountParam)
        MIOPEN_THROW("Error parsing dims tensor: " + inflags.GetValueStr("dims") + ".");

    for(auto dim : dims)
    {
        output_dims.push_back(dxTensorParam.lengths[dim]);
    }

    slices = inflags.GetValue2dVectorInt("slices");
    if(slices.size() != sliceCountParam)
        MIOPEN_THROW("Error parsing slices: " + inflags.GetValueStr("slices") + ".");

    for(auto slice : slices)
    {
        for(int32_t i = 0; i < 4; i++)
        {
            slices_flat.push_back(slice[i]);
        }
    }

    if(SetTensorNd(dyDesc, dyTensorParam.lengths, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing doutput tensor: " + inflags.GetValueStr("doutput") + ".");

    if(SetTensorNd(xDesc, xTensorParam.lengths, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input") + ".");

    if(SetTensorNd(yDesc, yTensorParam.lengths, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor: " + inflags.GetValueStr("output") + ".");

    for(auto indexTensorLength : indexTensorLengths)
    {
        miopenTensorDescriptor_t indexDesc;
        miopenCreateTensorDescriptor(&indexDesc);
        if(SetTensorNd(indexDesc, indexTensorLength, miopenInt32) != miopenStatusSuccess)
            MIOPEN_THROW("Error parsing indexs tensor: " + inflags.GetValueStr("indexs") + ".");
        indexDescs.push_back(indexDesc);
    }

    if(SetTensorNd(dxDesc, dxTensorParam.lengths, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing dinput tensor: " + inflags.GetValueStr("dinput") + ".");

    std::vector<int32_t> error_length;
    error_length.push_back(indexCountParam);
    if(SetTensorNd(errorDesc, error_length, miopen_type<int32_t>{}) != miopenStatusSuccess)
        MIOPEN_THROW("Error making error tensor: " + inflags.GetValueStr("indexcount") + ".");

    return 0;
}

template <typename Tgpu, typename Tref>
int GetitemDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "0", "Run only Forward Getitem (Default=0)", "int");
    inflags.AddTensorFlag("doutput", 'O', "128x128", "doutput tensor descriptor");
    inflags.AddTensorFlag("input", 'X', "128x128", "input tensor descriptor");
    inflags.AddTensorFlag("output", 'Y', "128x128", "output tensor descriptor");
    inflags.AddTensorFlag("indexs", 'D', "128", "indexs tensor descriptor");
    inflags.AddTensorFlag("dinput", 'N', "128x128", "dinput tensor descriptor");

    inflags.AddInputFlag("indexcount", '1', "1", "the number of indexs tensor(Default=1)", "int");
    inflags.AddInputFlag("dimcount", '2', "1", "The dimensions(Default=1)", "int");
    inflags.AddInputFlag("dims", '3', "0", "The dimensions(Default=0)", "vector<int>");
    inflags.AddInputFlag("slicecount", '4', "0", "The number of slices(Default=0)", "int");
    inflags.AddInputFlag("slices",
                         '5',
                         "",
                         "The slices(Default=\'\'"
                         ")",
                         "vector<vector<int>>");
    inflags.AddInputFlag("offset", '6', "0", "The offset of output(Default=0)", "int");

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
    size_t dx_sz    = GetTensorSize(dxDesc);
    size_t error_sz = GetTensorSize(errorDesc);

    miopenGetGetitemWorkspaceSize(
        GetHandle(), indexDescs.size(), indexDescs.data(), &ws_sizeInBytes);
    if(ws_sizeInBytes == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    uint32_t ctx = 0;

    dy_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, dy_sz, sizeof(Tgpu)));
    x_dev         = std::unique_ptr<GPUMem>(new GPUMem(ctx, x_sz, sizeof(Tgpu)));
    y_dev         = std::unique_ptr<GPUMem>(new GPUMem(ctx, y_sz, sizeof(Tgpu)));
    dx_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, dx_sz, sizeof(Tgpu)));
    error_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, error_sz, sizeof(int32_t)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));

    dy        = std::vector<Tgpu>(dy_sz, static_cast<Tgpu>(0));
    x         = std::vector<Tgpu>(x_sz, static_cast<Tgpu>(0));
    y         = std::vector<Tgpu>(y_sz, static_cast<Tgpu>(0));
    dx        = std::vector<Tgpu>(dx_sz, static_cast<Tgpu>(0));
    error     = std::vector<int32_t>(error_sz, static_cast<int32_t>(0));
    workspace = std::vector<int32_t>(ws_sizeInBytes / sizeof(int32_t), static_cast<int32_t>(0));
    dxhost    = std::vector<Tref>(dx_sz, static_cast<Tref>(0));
    errorhost = std::vector<int32_t>(error_sz, static_cast<int32_t>(0));

    for(int32_t i = 0; i < dy_sz; i++)
    {
        dy[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-0.01), static_cast<Tgpu>(0.01));
    }

    for(int32_t i = 0; i < x_sz; i++)
    {
        x[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-0.01), static_cast<Tgpu>(0.01));
    }

    for(int32_t i = 0; i < y_sz; i++)
    {
        y[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-0.01), static_cast<Tgpu>(0.01));
    }

    for(int32_t i = 0; i < error_sz; i++)
    {
        errorhost[i] = 1;
    }

    for(int32_t i = 0; i < ws_sizeInBytes / sizeof(int32_t); i++)
    {
        workspace[i] = 0;
    }

    for(int32_t i = 0; i < dx_sz; i++)
    {
        dx[i]     = 0;
        dxhost[i] = 0;
    }

    for(int32_t i = 0; i < indexDescs.size(); i++)
    {
        size_t index_sz = GetTensorSize(indexDescs[i]);
        index_devs.push_back(std::unique_ptr<GPUMem>(new GPUMem(ctx, index_sz, sizeof(int32_t))));
        indexs.push_back(std::vector<int32_t>(index_sz, static_cast<int32_t>(0)));
        auto& index    = indexs.back();
        auto index_dev = index_devs.back().get();

        index[i] = prng::gen_A_to_B<int32_t>(static_cast<int32_t>(0),
                                             static_cast<int32_t>(output_dims[i]));

        if(index_dev->ToGPU(GetStream(), index.data()) != 0)
            std::cerr << "Error copying (index) to GPU, size: " << index_dev->GetSize()
                      << std::endl;
        index_devs_ptr.push_back(index_dev->GetMem());
        indexs_ptr.push_back(index.data());
    }

    if(dy_dev->ToGPU(GetStream(), dy.data()) != 0)
        std::cerr << "Error copying (dy) to GPU, size: " << dy_dev->GetSize() << std::endl;

    if(x_dev->ToGPU(GetStream(), x.data()) != 0)
        std::cerr << "Error copying (x) to GPU, size: " << x_dev->GetSize() << std::endl;

    if(y_dev->ToGPU(GetStream(), y.data()) != 0)
        std::cerr << "Error copying (y) to GPU, size: " << y_dev->GetSize() << std::endl;

    if(workspace_dev->ToGPU(GetStream(), workspace.data()) != 0)
        std::cerr << "Error copying (workspace) to GPU, size: " << workspace_dev->GetSize()
                  << std::endl;

    if(error_dev->ToGPU(GetStream(), errorhost.data()) != 0)
        std::cerr << "Error copying (error) to GPU, size: " << error_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GetitemDriver<Tgpu, Tref>::RunForwardGPU()
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

    for(int32_t i = 0; i < inflags.GetValueInt("iter"); i++)
    {

        if(dx_dev->ToGPU(GetStream(), dx.data()) != 0)
            std::cerr << "Error copying (dx) to GPU, size: " << dx_dev->GetSize() << std::endl;

        miopenGetitemBackward(GetHandle(),
                              workspace_dev->GetMem(),
                              ws_sizeInBytes,
                              dyDesc,
                              dy_dev->GetMem(),
                              xDesc,
                              x_dev->GetMem(),
                              indexDescs.size(),
                              indexDescs.data(),
                              index_devs_ptr.data(),
                              yDesc,
                              y_dev->GetMem(),
                              dxDesc,
                              dx_dev->GetMem(),
                              errorDesc,
                              error_dev->GetMem(),
                              dims.size(),
                              dims.data(),
                              slices.size(),
                              slices_flat.data(),
                              offset);

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
            std::cout << "Wall-clock Time Forward Getitem Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Getitem Elapsed: " << kernel_average_time << " ms\n";
    }

    if(dx_dev->FromGPU(GetStream(), dx.data()) != 0)
        std::cerr << "Error copying (dx_dev) from GPU, size: " << dx_dev->GetSize() << std::endl;

    if(error_dev->FromGPU(GetStream(), error.data()) != 0)
        std::cerr << "Error copying (error_dev) from GPU, size: " << error_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GetitemDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloGetitemBackwardRunHost<Tgpu, Tref>(dyDesc,
                                          xDesc,
                                          indexDescs.size(),
                                          indexDescs.data(),
                                          yDesc,
                                          dxDesc,
                                          errorDesc,
                                          dy.data(),
                                          x.data(),
                                          y.data(),
                                          indexs_ptr.data(),
                                          dxhost.data(),
                                          errorhost.data(),
                                          dims.size(),
                                          dims.data(),
                                          slices.size(),
                                          slices_flat.data(),
                                          offset);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref GetitemDriver<Tgpu, Tref>::GetTolerance()
{
    // Computation error of fp16 is ~2^13 (=8192) bigger than
    // the one of fp32 because mantissa is shorter by 13 bits.
    auto tolerance =
        std::is_same<Tgpu, float>::value ? 1.5e-6 : 8.2e-3;

    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    // If there is an atomic operation on the GPU kernel, a large error occurs depending on the
    // calculation order, so it is multiplied by 10 times.
    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 80.0;
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

    auto error_dx = miopen::rms_range(dxhost, dx);

    if(!std::isfinite(error_dx) || error_dx > tolerance)
    {
        std::cout << "Backward Getitem FAILED: " << error_dx << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward Getitem Verifies OK on CPU reference (" << error_dx << " < "
                  << tolerance << ')' << std::endl;
    }

    auto error_error = miopen::rms_range(errorhost, error);

    if(!std::isfinite(error_error) || std::abs(static_cast<float>(error_error)) != 0.0f)
    {
        std::cout << "Backward Getitem FAILED: Result does not equal" << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward Getitem Verifies OK on CPU and GPU (err=" << error << ")\n";
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_GETITEM_DRIVER_HPP
