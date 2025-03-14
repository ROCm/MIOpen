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

#include "../test/tensor_holder.hpp"
#include "../test/verify.hpp"
#include "InputFlags.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <cassert>
#include <cmath>
#include <memory>

#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>

template <typename T = float>
int mloConvertRGBToHSV(const T r, const T g, const T b, T* h, T* s, T* v)
{
    T minc = std::min(r, std::min(g, b));
    T maxc = std::max(r, std::max(g, b));

    *v = maxc;

    T cr     = maxc - minc;
    bool eqc = (cr == 0);

    *s = cr / (eqc ? 1.0f : maxc);

    T cr_divisor = eqc ? static_cast<T>(1.0f) : cr;
    T rc         = (maxc - r) / cr_divisor;
    T gc         = (maxc - g) / cr_divisor;
    T bc         = (maxc - b) / cr_divisor;

    T hr = (maxc == r) * (bc - gc);
    T hg = ((maxc == g) && (maxc != r)) * (static_cast<T>(2.0f) + rc - bc);
    T hb = ((maxc != g) && (maxc != r)) * (static_cast<T>(4.0f) + gc - rc);

    *h = fmod((hr + hg + hb) / 6.0 + 1.0, 1.0);

    return 0;
}

template <typename T = float>
int mloConvertHSVToRGB(const T h, const T s, const T v, T* r, T* g, T* b)
{
    T i        = floor(h * 6.0);
    T f        = h * 6.0 - i;
    int i_case = (static_cast<int>(i) + 6) % 6;

    T p = std::clamp(v * (1.0 - s), 0.0, 1.0);
    T q = std::clamp(v * (1.0 - s * f), 0.0, 1.0);
    T t = std::clamp(v * (1.0 - s * (1.0 - f)), 0.0, 1.0);

    switch(i_case)
    {
    case 0:
        *r = v;
        *g = t;
        *b = p;
        break;
    case 1:
        *r = q;
        *g = v;
        *b = p;
        break;
    case 2:
        *r = p;
        *g = v;
        *b = t;
        break;
    case 3:
        *r = p;
        *g = q;
        *b = v;
        break;
    case 4:
        *r = t;
        *g = p;
        *b = v;
        break;
    case 5:
        *r = v;
        *g = p;
        *b = q;
        break;
    default:
        // This case should never happen (i_case is guaranteed to be in range [0,5])
        // Just in case this ever does, panic immediately
        MIOPEN_THROW("i_case out of range");
    }

    return 0;
}

template <typename Tgpu, typename Tref>
int mloRunImageAdjustHueHost(const Tgpu* input,
                             Tref* output,
                             const miopenTensorDescriptor_t inputTensorDesc,
                             const miopenTensorDescriptor_t outputTensorDesc,
                             float hue_factor)
{
    size_t N       = miopen::deref(inputTensorDesc).GetElementSize() / 3;
    auto input_tv  = miopen::get_inner_expanded_tv<4>(miopen::deref(inputTensorDesc));
    auto output_tv = miopen::get_inner_expanded_tv<4>(miopen::deref(outputTensorDesc));

    for(auto gid = 0; gid < N; gid++)
    {
        tensor_layout_t<4> input_layout(input_tv, gid);
        auto n = input_layout.layout[0];
        auto c = input_layout.layout[1];
        auto h = input_layout.layout[2];
        auto w = input_layout.layout[3];

        n = n * 3 + c;

        Tref r = static_cast<Tref>(input[input_tv.get_tensor_view_idx({n, 0, h, w})]);
        Tref g = static_cast<Tref>(input[input_tv.get_tensor_view_idx({n, 1, h, w})]);
        Tref b = static_cast<Tref>(input[input_tv.get_tensor_view_idx({n, 2, h, w})]);

        Tref hue, sat, val;
        mloConvertRGBToHSV(r, g, b, &hue, &sat, &val);
        hue = fmod(hue + hue_factor, 1.0);
        mloConvertHSVToRGB(hue, sat, val, &r, &g, &b);

        output[output_tv.get_tensor_view_idx({n, 0, h, w})] = r;
        output[output_tv.get_tensor_view_idx({n, 1, h, w})] = g;
        output[output_tv.get_tensor_view_idx({n, 2, h, w})] = b;
    }

    return 0;
}

template <typename Tgpu, typename Tref>
class ImageAdjustHueDriver : public Driver
{
public:
    ImageAdjustHueDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensorDesc);
        miopenCreateTensorDescriptor(&outputTensorDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    std::vector<int> ComputeStrides(std::vector<int> inputDim);
    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    int VerifyForward() override;
    int VerifyBackward() override;

    ~ImageAdjustHueDriver() override
    {
        miopenDestroyTensorDescriptor(inputTensorDesc);
        miopenDestroyTensorDescriptor(outputTensorDesc);
    }

private:
    InputFlags inflags;

    int forw;
    bool isContiguous;

    miopenTensorDescriptor_t inputTensorDesc;
    miopenTensorDescriptor_t outputTensorDesc;

    std::unique_ptr<GPUMem> input_gpu;
    std::unique_ptr<GPUMem> output_gpu;

    std::vector<Tgpu> in_host;
    std::vector<Tgpu> out_host;

    std::vector<Tref> out_ref;

    float hue;
};

template <typename Tgpu, typename Tref>
int ImageAdjustHueDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only the forward pass (Default=1)", "int");

    inflags.AddTensorFlag("input", 'I', "1x3x96x96", "Input Tensor Size (Default=1x3x96x96)");
    inflags.AddInputFlag("contiguous", 'Z', "1", "Use Contiguous Tensors (Default=1)", "int");
    inflags.AddInputFlag("hue", 'H', "0.2", "Hue (Default=0.2)", "double");

    inflags.AddInputFlag("iter", 'i', "10", "Number of iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustHueDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    hue          = inflags.GetValueDouble("hue");
    isContiguous = inflags.GetValueInt("contiguous") == 0 ? false : true;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    forw = inflags.GetValueInt("forw");
    if(forw != 1)
    {
        MIOPEN_THROW("Only forward pass is supported");
    }

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> ImageAdjustHueDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int ImageAdjustHueDriver<Tgpu, Tref>::GetandSetData()
{
    auto input_dims = inflags.GetValueTensor("input").lengths;
    auto strides    = ComputeStrides(input_dims);

    SetTensorNd(inputTensorDesc, input_dims, strides, data_type);
    SetTensorNd(outputTensorDesc, input_dims, strides, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustHueDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_size  = GetTensorSize(inputTensorDesc);
    size_t out_size = GetTensorSize(outputTensorDesc);

    uint32_t ctx = 0;

    input_gpu  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_size, sizeof(Tgpu)));
    output_gpu = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_size, sizeof(Tgpu)));

    in_host  = std::vector<Tgpu>(in_size, static_cast<Tgpu>(0));
    out_host = std::vector<Tgpu>(out_size, static_cast<Tgpu>(0));
    out_ref  = std::vector<Tref>(out_size, static_cast<Tref>(0));

    for(auto i = 0; i < in_size; i++)
    {
        in_host[i] = static_cast<Tgpu>(prng::gen_A_to_B(0.0f, 1.0f));
    }

    if(input_gpu->ToGPU(GetStream(), in_host.data()) != miopenStatusSuccess)
        std::cerr << "Error copying buffer to GPU with size " << input_gpu->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustHueDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenImageAdjustHue(GetHandle(),
                             inputTensorDesc,
                             outputTensorDesc,
                             input_gpu->GetMem(),
                             output_gpu->GetMem(),
                             hue);
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
            std::cout << "Image Adjust Hue Forward Time Elapsed: " << kernel_total_time << " ms"
                      << std::endl;

        float kernel_avg_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_total_time;
        std::cout << "Image Adjust Hue Kernel Time Elapsed: " << kernel_avg_time << " ms"
                  << std::endl;
    }

    if(output_gpu->FromGPU(GetStream(), out_host.data()) != miopenStatusSuccess)
    {
        std::cerr << "Error copying buffer from GPU with size " << output_gpu->GetSize()
                  << std::endl;
        return -1;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustHueDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int ImageAdjustHueDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloRunImageAdjustHueHost(
        in_host.data(), out_ref.data(), inputTensorDesc, outputTensorDesc, hue);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustHueDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int ImageAdjustHueDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    auto threashold = std::numeric_limits<Tgpu>::epsilon();
    auto error      = miopen::rms_range(out_ref, out_host);

    if(!std::isfinite(error) || error > threashold)
    {
        std::cout << "Forward Image Adjust Hue FAILED: " << error << std::endl;
    }
    else
    {
        std::cout << "Forward Image Adjust Hue Verifies on CPU and GPU (" << error << ')'
                  << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustHueDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusNotImplemented;
}
