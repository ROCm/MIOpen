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
#include "tensor_holder.hpp"
#include "tensor_view.hpp"

template <typename T>
T clamp(T val, T min, T max)
{
    val = val < min ? min : val;
    val = val > max ? max : val;
    return val;
}

template <typename T = float>
void mloConvertRGBToHSV(const T r, const T g, const T b, T* h, T* s, T* v)
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

    T hr = static_cast<T>((maxc == r) * (bc - gc));
    T hg = static_cast<T>(((maxc == g) && (maxc != r)) * (static_cast<T>(2.0f) + rc - bc));
    T hb = static_cast<T>(((maxc != g) && (maxc != r)) * (static_cast<T>(4.0f) + gc - rc));

    *h = fmod((hr + hg + hb) / 6.0 + 1.0, 1.0);
}

template <typename T = float>
void mloConvertHSVToRGB(const T h, const T s, const T v, T* r, T* g, T* b)
{
    T i        = static_cast<T>(floor(h * 6.0));
    T f        = static_cast<T>(h * 6.0 - i);
    int i_case = (static_cast<int>(i) + 6) % 6;

    T p = static_cast<T>(clamp(v * (1.0f - s), 0.0f, 1.0f));
    T q = static_cast<T>(clamp(v * (1.0f - s * f), 0.0f, 1.0f));
    T t = static_cast<T>(clamp(v * (1.0f - s * (1.0f - f)), 0.0f, 1.0f));

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
}

template <typename T>
void mloRunImageAdjustHueHost(const T* input,
                              T* output,
                              miopen::TensorDescriptor inputTensorDesc,
                              miopen::TensorDescriptor outputTensorDesc,
                              float hue_factor)
{
    size_t N       = inputTensorDesc.GetElementSize() / 3;
    auto input_tv  = miopen::get_inner_expanded_tv<4>(inputTensorDesc);
    auto output_tv = miopen::get_inner_expanded_tv<4>(outputTensorDesc);

    for(auto gid = 0; gid < N; gid++)
    {
        tensor_layout_t<4> input_layout(input_tv, gid);
        auto n = input_layout.layout[0];
        auto c = input_layout.layout[1];
        auto h = input_layout.layout[2];
        auto w = input_layout.layout[3];

        n = n * 3 + c;

        T r = static_cast<T>(input[input_tv.get_tensor_view_idx({n, 0, h, w})]);
        T g = static_cast<T>(input[input_tv.get_tensor_view_idx({n, 1, h, w})]);
        T b = static_cast<T>(input[input_tv.get_tensor_view_idx({n, 2, h, w})]);

        T hue, sat, val;

        mloConvertRGBToHSV(r, g, b, &hue, &sat, &val);
        hue = fmod(hue + hue_factor, 1.0);
        mloConvertHSVToRGB(hue, sat, val, &r, &g, &b);

        output[output_tv.get_tensor_view_idx({n, 0, h, w})] = r;
        output[output_tv.get_tensor_view_idx({n, 1, h, w})] = g;
        output[output_tv.get_tensor_view_idx({n, 2, h, w})] = b;
    }
}

template <typename T>
void cpu_image_adjust_hue(const tensor<T>& input, tensor<T>& output, float hue)
{
    mloRunImageAdjustHueHost(input.data.data(), output.data.data(), input.desc, output.desc, hue);
}

template <typename T>
void mloImageAdjustBrightnessRunHost(const T* input,
                                     T* output,
                                     miopen::TensorDescriptor inputDesc,
                                     miopen::TensorDescriptor outputDesc,
                                     const float brightness_factor)
{
    auto input_tv  = miopen::get_inner_expanded_tv<4>(inputDesc);
    auto output_tv = miopen::get_inner_expanded_tv<4>(outputDesc);

    size_t N = inputDesc.GetElementSize();

    for(size_t gid = 0; gid < N; gid++)
    {
        tensor_layout_t<4> input_layout(input_tv, gid);
        T pixel  = input[input_tv.get_tensor_view_idx(input_layout)];
        T result = static_cast<T>(clamp(static_cast<float>(pixel) * brightness_factor, 0.0f, 1.0f));
        output[output_tv.get_tensor_view_idx(input_layout)] = result;
    }
}

template <typename T>
void cpu_image_adjust_brightness(const tensor<T>& input, tensor<T>& output, float brightness)
{
    mloImageAdjustBrightnessRunHost(
        input.data.data(), output.data.data(), input.desc, output.desc, brightness);
}

template <typename T>
void RGBToGrayscale(
    const T* src, T* dst, tensor_view_t<4> src_tv, tensor_view_t<4> dst_tv, size_t N)
{
    for(size_t gid = 0; gid < N; gid++)
    {
        tensor_layout_t<4> dst_layout(dst_tv, gid);
        auto n = dst_layout.layout[0];
        auto h = dst_layout.layout[2];
        auto w = dst_layout.layout[3];

        T r = src[src_tv.get_tensor_view_idx({n, 0, h, w})];
        T g = src[src_tv.get_tensor_view_idx({n, 1, h, w})];
        T b = src[src_tv.get_tensor_view_idx({n, 2, h, w})];

        T value = static_cast<T>(0.2989 * r + 0.587 * g + 0.114 * b);

        // We expect the workspace here to always stay contiguous
        dst[gid] = value;
    }
}

template <typename T>
void Blend(const T* img1,
           const T* img2,
           T* output,
           tensor_view_t<4> img1_tv,
           tensor_view_t<4> output_tv,
           const size_t n_stride,
           const size_t c_stride,
           const size_t N,
           float ratio,
           float bound)

{
    for(size_t gid = 0; gid < N; gid++)
    {
        const size_t n        = gid / n_stride;
        const size_t img2_idx = n * c_stride + gid % c_stride;

        tensor_layout_t<4> img1_layout(img1_tv, gid);
        T img1_v = img1[img1_tv.get_tensor_view_idx(img1_layout)];
        T img2_v = img2[img2_idx];

        T result = static_cast<T>(clamp((ratio * img1_v + (1.0f - ratio) * img2_v), 0.0f, bound));

        tensor_layout_t<4> output_layout(output_tv, gid);
        output[output_tv.get_tensor_view_idx(output_layout)] = result;
    }
}

template <typename T>
void mloImageAdjustSaturationRunHost(miopen::TensorDescriptor inputDesc,
                                     miopen::TensorDescriptor outputDesc,
                                     const T* input,
                                     T* output,
                                     float saturation_factor)

{
    auto input_tv  = miopen::get_inner_expanded_tv<4>(inputDesc);
    auto output_tv = miopen::get_inner_expanded_tv<4>(outputDesc);

    // temporary view for workspace (basically a contiguous vector with same size as input_tv)
    std::vector<T> workspace = std::vector<T>(inputDesc.GetElementSize(), static_cast<T>(0.0f));
    miopen::TensorDescriptor wsDesc =
        miopen::TensorDescriptor{inputDesc.GetType(), inputDesc.GetLengths()};

    auto ws_tv = miopen::get_inner_expanded_tv<4>(wsDesc);

    auto N        = inputDesc.GetElementSize();
    auto c_stride = input_tv.size[2] * input_tv.size[3];
    auto n_stride = c_stride * input_tv.size[1];

    float bound = 1.0f;

    RGBToGrayscale(input, workspace.data(), input_tv, ws_tv, N);
    Blend(input,
          workspace.data(),
          output,
          input_tv,
          output_tv,
          n_stride,
          c_stride,
          N,
          saturation_factor,
          bound);
}

template <typename T>
void cpu_image_adjust_saturation(const tensor<T>& input, tensor<T>& output, float saturation_factor)
{
    mloImageAdjustSaturationRunHost(
        input.desc, output.desc, input.data.data(), output.data.data(), saturation_factor);
}

template <typename T>
void mloImageNormalizeRunHost(miopen::TensorDescriptor inputDesc,
                              miopen::TensorDescriptor outputDesc,
                              const T* input,
                              T* output,
                              const T* mean,
                              const T* stdvar)
{
    auto input_tv  = miopen::get_inner_expanded_tv<4>(inputDesc);
    auto output_tv = miopen::get_inner_expanded_tv<4>(outputDesc);

    auto N         = inputDesc.GetElementSize();
    auto C         = input_tv.size[1];
    auto c_strides = input_tv.stride[1];

    for(size_t gid = 0; gid < N; gid++)
    {
        auto c = gid / c_strides % C;

        tensor_layout_t<4> input_layout(input_tv, gid);
        T pixel  = input[input_tv.get_tensor_view_idx(input_layout)];
        T result = (pixel - static_cast<T>(mean[c])) / static_cast<T>(stdvar[c]);
        tensor_layout_t<4> output_layout(output_tv, gid);
        output[output_tv.get_tensor_view_idx(output_layout)] = result;
    }
}

template <typename T>
void cpu_image_normalize(const tensor<T>& input,
                         tensor<T>& output,
                         const tensor<T>& mean,
                         const tensor<T>& stdvar)
{
    mloImageNormalizeRunHost(input.desc,
                             output.desc,
                             input.data.data(),
                             output.data.data(),
                             mean.data.data(),
                             stdvar.data.data());
}
