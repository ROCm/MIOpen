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
#include <miopen/tensor_view_utils.hpp>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloGetitemBackwardRunHost(miopenTensorDescriptor_t dyDesc,
                                  uint32_t indexCount,
                                  miopenTensorDescriptor_t* indexDescs,
                                  miopenTensorDescriptor_t dxDesc,
                                  miopenTensorDescriptor_t errorDesc,
                                  Tgpu* dy,
                                  int32_t** indexs,
                                  Tcheck* dxhost,
                                  int32_t* errorhost,
                                  uint32_t dimCount,
                                  int32_t* dims,
                                  uint32_t sliceCount,
                                  int32_t* slices,
                                  uint32_t offset)
{
    auto dy_dims  = miopen::deref(dyDesc).GetLengths();
    auto dy_numel = std::accumulate(dy_dims.begin(), dy_dims.end(), 1L, std::multiplies<int64_t>());
    auto dx_dims  = miopen::deref(dxDesc).GetLengths();
    auto index_dims = miopen::deref(indexDescs[0]).GetLengths();
    auto index_numel =
        std::accumulate(index_dims.begin(), index_dims.end(), 1L, std::multiplies<int64_t>());
    auto element_index = std::vector<int32_t>(indexCount * index_numel + indexCount);

    std::vector<size_t> output_dims;
    for(int32_t i = 0; i < dimCount; i++)
    {
        output_dims.push_back(dx_dims[dims[i]]);
    }

    auto dim_info_offset = indexCount > 0 ? indexCount * index_dims[0] : 0;
    auto start_dim       = dims[0];

    auto dy_tv     = miopen::get_inner_expanded_tv<5>(miopen::deref(dyDesc));
    auto dxhost_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(dxDesc));
    miopen::slice_tv<5>(dxhost_tv, sliceCount, slices);

    int32_t ret = 0;

    // Get element index form indexs
    for(size_t j = 0; j < indexCount; j++)
    {
        const auto& index_dim = dims[j];
        const auto& dim_size  = output_dims[j];

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
        tensor_layout_t<5> ncdhw(dy_tv, o);
        tensor_layout_t<5> idx(ncdhw);

        if(indexCount > 0)
        {
            size_t dim_cursor = ncdhw.layout[start_dim];
            size_t i          = start_dim;
            size_t j          = 0;

            for(; i < start_dim + indexCount; ++i, ++j)
            {
                size_t dim_idx      = element_index[dim_info_offset + j];
                idx.layout[dim_idx] = element_index[(dim_cursor * indexCount) + j];
            }

            i          = element_index[dim_info_offset + indexCount - 1] + 1;
            dim_cursor = start_dim + 1;
            for(; i < 5; ++i, ++dim_cursor)
            {
                idx.layout[i] = ncdhw.layout[dim_cursor];
            }
        }

        dxhost[dxhost_tv.get_tensor_view_idx(idx)] += dy[dy_tv.get_tensor_view_idx(ncdhw)];
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
        miopenCreateTensorDescriptor(&dxDesc);
        miopenCreateTensorDescriptor(&errorDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

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
        for(auto indexDesc : indexDescs)
        {
            miopenDestroyTensorDescriptor(indexDesc);
        }
        miopenDestroyTensorDescriptor(dxDesc);
        miopenDestroyTensorDescriptor(errorDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t dyDesc;
    std::vector<miopenTensorDescriptor_t> indexDescs;
    miopenTensorDescriptor_t dxDesc;
    miopenTensorDescriptor_t errorDesc;

    std::unique_ptr<GPUMem> dy_dev;
    std::vector<std::unique_ptr<GPUMem>> index_devs;
    std::unique_ptr<GPUMem> dx_dev;
    std::unique_ptr<GPUMem> error_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> dy;
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
    uint32_t offset;

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

    if(inflags.GetValueInt("indexcount") < 0)
        MIOPEN_THROW("Index count is negative: " + inflags.GetValueStr("indexcount") + ".");

    if(inflags.GetValueInt("dimcount") < 0)
        MIOPEN_THROW("Dim count is negative: " + inflags.GetValueStr("dimcount") + ".");

    if(inflags.GetValueInt("slicecount") < 0)
        MIOPEN_THROW("Slice count is negative: " + inflags.GetValueStr("slicecount") + ".");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GetitemDriver<Tgpu, Tref>::GetandSetData()
{
    auto dyTensorParam   = inflags.GetValueTensorUint64("doutput");
    auto dxTensorParam   = inflags.GetValueTensorUint64("dinput");
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
    size_t dx_sz    = GetTensorSize(dxDesc);
    size_t error_sz = GetTensorSize(errorDesc);

    miopenGetGetitemWorkspaceSize(
        GetHandle(), indexDescs.size(), indexDescs.data(), &ws_sizeInBytes);

    uint32_t ctx = 0;

    dy_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, dy_sz, sizeof(Tgpu)));
    dx_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, dx_sz, sizeof(Tgpu)));
    error_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, error_sz, sizeof(int32_t)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));

    dy        = std::vector<Tgpu>(dy_sz, static_cast<Tgpu>(0));
    dx        = std::vector<Tgpu>(dx_sz, static_cast<Tgpu>(0));
    error     = std::vector<int32_t>(error_sz, static_cast<int32_t>(0));
    workspace = std::vector<int32_t>(ws_sizeInBytes / sizeof(int32_t), static_cast<int32_t>(0));
    dxhost    = std::vector<Tref>(dx_sz, static_cast<Tref>(0));
    errorhost = std::vector<int32_t>(error_sz, static_cast<int32_t>(0));

    for(int32_t i = 0; i < dy_sz; i++)
    {
        dy[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1), static_cast<Tgpu>(1));
    }

    for(int32_t i = 0; i < indexDescs.size(); i++)
    {
        size_t index_sz = GetTensorSize(indexDescs[i]);
        index_devs.push_back(std::unique_ptr<GPUMem>(new GPUMem(ctx, index_sz, sizeof(int32_t))));
        indexs.push_back(std::vector<int32_t>(index_sz, static_cast<int32_t>(0)));
        auto& index    = indexs.back();
        auto index_dev = index_devs.back().get();

        for(int j = 0; j < index_sz; j++)
        {
            index[j] = prng::gen_A_to_B<int32_t>(static_cast<int32_t>(0),
                                                 static_cast<int32_t>(output_dims[i]));
        }
        if(index_dev->ToGPU(GetStream(), index.data()) != 0)
            std::cerr << "Error copying (index) to GPU, size: " << index_dev->GetSize()
                      << std::endl;
        index_devs_ptr.push_back(index_dev->GetMem());
        indexs_ptr.push_back(index.data());
    }

    if(dy_dev->ToGPU(GetStream(), dy.data()) != 0)
        std::cerr << "Error copying (dy) to GPU, size: " << dy_dev->GetSize() << std::endl;

    if(workspace_dev->ToGPU(GetStream(), workspace.data()) != 0)
        std::cerr << "Error copying (workspace) to GPU, size: " << workspace_dev->GetSize()
                  << std::endl;

    if(dx_dev->ToGPU(GetStream(), dx.data()) != 0)
        std::cerr << "Error copying (dx) to GPU, size: " << dx_dev->GetSize() << std::endl;

    if(error_dev->ToGPU(GetStream(), error.data()) != 0)
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
                              indexDescs.size(),
                              indexDescs.data(),
                              index_devs_ptr.data(),
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
            std::cout << "Wall-clock Time Backward Getitem Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward Getitem Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
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
                                          indexDescs.size(),
                                          indexDescs.data(),
                                          dxDesc,
                                          errorDesc,
                                          dy.data(),
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
    // In the case of layernorm, there is a cumulative sum operation, and in the case of
    // floating point operation, the result value can change if the order of the summed values
    // is changed. So apply a threshold that is 10 times larger than other operations.
    auto tolerance = std::is_same<Tgpu, float>::value ? 1.5e-4 : 8.2e-1;

    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    // If there is an atomic operation on the GPU kernel, a large error occurs depending on the
    // calculation order, so it is multiplied by 10 times.
    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 8000.0;
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
        std::cout << "Backward Getitem Verifies OK on CPU and GPU" << std::endl;
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_GETITEM_DRIVER_HPP
