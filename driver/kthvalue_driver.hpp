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

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/tensor_view_utils.hpp>
#include <miopen/miopen.h>
#include <miopen/errors.hpp>

#include <vector>

template <typename TIO>
void mloKthvalueFwdRunHost(TIO* input,
                           miopenTensorDescriptor_t pInputDesc,
                           TIO* outputHost,
                           miopenTensorDescriptor_t outputDesc,
                           size_t* indices,
                           miopenTensorDescriptor_t indicesDesc,
                           size_t k,
                           int dim)
{
    auto inputDesc         = miopen::deref(pInputDesc);
    size_t inputSize       = inputDesc.GetElementSize();
    size_t dimSize         = inputDesc.GetLengths()[dim];
    size_t dimStride       = inputDesc.GetStrides()[dim];
    auto inputTv           = miopen::get_inner_expanded_tv<5>(miopen::deref(pInputDesc));
    auto inputTvWithoutDim = miopen::get_tv_without_dim<5>(inputTv, dim);
    auto outputTv          = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));
    auto indicesTv         = miopen::get_inner_expanded_tv<5>(miopen::deref(indicesDesc));

    size_t numSlice = inputSize / dimSize;

    std::vector<float> elements;
    std::vector<size_t> ids(dimSize);
    for(int i = 0; i < dimSize; ++i)
    {
        ids[i] = i;
    }

    for(int slideID = 0; slideID < numSlice; ++slideID)
    {
        elements.clear();
        tensor_layout_t<4> layout(inputTvWithoutDim, slideID);
        auto idx = inputTvWithoutDim.get_tensor_view_idx(layout);

        for(int j = 0; j < dimSize; ++j)
        {
            elements.push_back(static_cast<float>(input[idx + j * dimStride]));
        }

        std::sort(ids.begin(), ids.end(), [=](size_t x, size_t y) -> bool {
            return elements[x] < elements[y];
        });
        auto output_layout  = tensor_layout_t<5>(outputTv, slideID);
        auto indices_layout = tensor_layout_t<5>(indicesTv, slideID);
        outputHost[outputTv.get_tensor_view_idx(output_layout)] =
            static_cast<TIO>(elements[ids[k - 1]]);
        indices[indicesTv.get_tensor_view_idx(indices_layout)] = ids[k - 1];
    }
}

template <typename TIO>
class KthvalueDriver : public Driver
{
public:
    KthvalueDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&indicesDesc);
        miopenCreateTensorDescriptor(&outputDesc);

        data_type = miopen_type<TIO>{};
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

    int VerifyBackward() override;
    int VerifyForward() override;
    ~KthvalueDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(indicesDesc);
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t indicesDesc;
    miopenTensorDescriptor_t outputDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> indices_dev;
    std::unique_ptr<GPUMem> output_dev;

    std::vector<TIO> input;
    std::vector<size_t> indices;
    std::vector<size_t> indicesHost;
    std::vector<TIO> output;
    std::vector<TIO> outputHost;

    bool isContiguous;
    int dim;
    size_t k;
    bool keepDim;
};

template <typename TIO>
int KthvalueDriver<TIO>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    isContiguous = inflags.GetValueInt("is-contiguous") == 1 ? true : false;
    k            = inflags.GetValueInt("k");
    dim          = inflags.GetValueInt("dim");
    keepDim      = inflags.GetValueInt("keep-dim") == 1 ? true : false;
    auto inDims  = inflags.GetValueTensor("dim-lengths").lengths;
    int num_dim  = inDims.size();
    if(dim < -num_dim || dim >= num_dim)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Kthvalue: dim doesn't not exist");
    }

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename TIO>
int KthvalueDriver<TIO>::GetandSetData()
{
    auto inDims               = inflags.GetValueTensor("dim-lengths").lengths;
    std::vector<int> inStride = ComputeStrides(inDims);
    auto outDims              = inflags.GetValueTensor("dim-lengths").lengths;

    if(dim < 0)
    {
        dim += inDims.size();
    }
    if(!keepDim)
    {
        outDims.erase(outDims.begin() + dim);
        if(outDims.empty())
            outDims.push_back(1);
    }
    else
    {
        outDims[dim] = 1;
    }

    SetTensorNd(inputDesc, inDims, inStride, data_type);
    SetTensorNd(outputDesc, outDims, data_type);
    SetTensorNd(indicesDesc, outDims, miopenInt64);

    return 0;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename TIO>
std::vector<int> KthvalueDriver<TIO>::ComputeStrides(std::vector<int> inputDim)
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

template <typename TIO>
int KthvalueDriver<TIO>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (Default=1)", "int");
    inflags.AddTensorFlag(
        "dim-lengths", 'D', "256x4x2", "The dimensional lengths of the input tensor");
    inflags.AddInputFlag("k", 'k', "1", "k (Default=1)", "int");
    inflags.AddInputFlag("dim", 'd', "-1", "dim (Default=-1)", "int");
    inflags.AddInputFlag("keep-dim",
                         'K',
                         "0",
                         "Whether the output tensor has dim retained or not (Default=0)",
                         "int");
    inflags.AddInputFlag("is-contiguous", 'c', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename TIO>
int KthvalueDriver<TIO>::AllocateBuffersAndCopy()
{
    size_t in_sz  = miopen::deref(inputDesc).GetElementSize();
    size_t idx_sz = miopen::deref(indicesDesc).GetElementSize();
    size_t out_sz = miopen::deref(outputDesc).GetElementSize();

    uint32_t ctx = 0;

    input_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(TIO)));
    indices_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, idx_sz, sizeof(size_t)));
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(TIO)));

    input       = std::vector<TIO>(in_sz, static_cast<TIO>(0));
    indices     = std::vector<size_t>(idx_sz, 0);
    indicesHost = std::vector<size_t>(idx_sz, 0);
    output      = std::vector<TIO>(out_sz, static_cast<TIO>(0));
    outputHost  = std::vector<TIO>(out_sz, static_cast<TIO>(0));

    for(int i = 0; i < in_sz; i++)
    {
        input[i] = prng::gen_A_to_B<TIO>(static_cast<TIO>(-10), static_cast<TIO>(10));
    }

    fill(output.begin(), output.end(), static_cast<TIO>(0));
    fill(indices.begin(), indices.end(), static_cast<size_t>(0));

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << input_dev->GetSize() << std::endl;

    if(indices_dev->ToGPU(GetStream(), indices.data()) != 0)
        std::cerr << "Error copying (idx) to GPU, size: " << indices_dev->GetSize() << std::endl;

    if(output_dev->ToGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << output_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO>
int KthvalueDriver<TIO>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenKthvalueForward(GetHandle(),
                              inputDesc,
                              input_dev->GetMem(),
                              outputDesc,
                              output_dev->GetMem(),
                              indicesDesc,
                              (size_t*)indices_dev->GetMem(),
                              k,
                              dim,
                              keepDim);
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
            std::cout << "Wall-clock Time Kthvalue Fwd Elapsed: " << t.gettime_ms() / iter << " ms"
                      << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Kthvalue Fwd Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;

    if(indices_dev->FromGPU(GetStream(), indices.data()) != 0)
        std::cerr << "Error copying (indices_dev) from GPU, size: " << indices_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO>
int KthvalueDriver<TIO>::RunForwardCPU()
{
    mloKthvalueFwdRunHost<TIO>(input.data(),
                               inputDesc,
                               outputHost.data(),
                               outputDesc,
                               indicesHost.data(),
                               indicesDesc,
                               k,
                               dim);

    return miopenStatusSuccess;
}

template <typename TIO>
int KthvalueDriver<TIO>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename TIO>
int KthvalueDriver<TIO>::RunBackwardCPU()
{
    return miopenStatusSuccess;
}

template <typename TIO>
int KthvalueDriver<TIO>::VerifyForward()
{
    RunForwardCPU();

    double tolerance = std::numeric_limits<TIO>::epsilon() * 10;
    auto errorOutput = miopen::rms_range(outputHost, output);

    if(!std::isfinite(errorOutput) || errorOutput > tolerance)
    {
        std::cout << "Forward Kthvalue output FAILED: " << errorOutput << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Kthvalue Verifies OK on CPU reference (" << errorOutput << "< "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename TIO>
int KthvalueDriver<TIO>::VerifyBackward()
{
    return miopenStatusSuccess;
}
