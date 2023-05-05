/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/common.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct ConvTensors
{
    TensorDescriptor xDesc;
    ConstData_t x = nullptr;
    TensorDescriptor wDesc;
    ConstData_t w = nullptr;
    TensorDescriptor yDesc;
    ConstData_t y = nullptr;
};

struct ConvFwdTensors
{
    TensorDescriptor xDesc;
    ConstData_t x = nullptr;
    TensorDescriptor wDesc;
    ConstData_t w = nullptr;
    TensorDescriptor yDesc;
    Data_t y = nullptr;

    ConstData_t in() const { return x; }
    Data_t out() const { return y; }

    operator ConvTensors() const { return {xDesc, x, wDesc, w, yDesc, y}; }
};

struct ConvBwdTensors
{
    TensorDescriptor dyDesc;
    ConstData_t dy = nullptr;
    TensorDescriptor wDesc;
    ConstData_t w = nullptr;
    TensorDescriptor dxDesc;
    Data_t dx = nullptr;

    ConstData_t in() const { return dy; }
    Data_t out() const { return dx; }

    operator ConvTensors() const { return {dxDesc, dx, wDesc, w, dyDesc, dy}; }
};

struct ConvDataTensors
{
    TensorDescriptor inDesc;
    ConstData_t in = nullptr;
    TensorDescriptor wDesc;
    ConstData_t w = nullptr;
    TensorDescriptor outDesc;
    Data_t out = nullptr;

    ConvDataTensors(TensorDescriptor inDesc_,
                    ConstData_t in_,
                    TensorDescriptor wDesc_,
                    ConstData_t w_,
                    TensorDescriptor outDesc_,
                    Data_t out_)
        : inDesc(std::move(inDesc_)),
          in(in_),
          wDesc(std::move(wDesc_)),
          w(w_),
          outDesc(std::move(outDesc_)),
          out(out_)
    {
    }

    ConvDataTensors(ConvFwdTensors tensors)
        : inDesc(std::move(tensors.xDesc)),
          in(tensors.x),
          wDesc(std::move(tensors.wDesc)),
          w(tensors.w),
          outDesc(std::move(tensors.yDesc)),
          out(tensors.y)
    {
    }

    ConvDataTensors(ConvBwdTensors tensors)
        : inDesc(std::move(tensors.dyDesc)),
          in(tensors.dy),
          wDesc(std::move(tensors.wDesc)),
          w(tensors.w),
          outDesc(std::move(tensors.dxDesc)),
          out(tensors.dx)
    {
    }
};

struct ConvWrwTensors
{
    TensorDescriptor dyDesc;
    ConstData_t dy = nullptr;
    TensorDescriptor xDesc;
    ConstData_t x = nullptr;
    TensorDescriptor dwDesc;
    Data_t dw = nullptr;

    operator ConvTensors() const { return {xDesc, x, dwDesc, dw, dyDesc, dy}; }
};

struct FusedConvDataTensors
{
    TensorDescriptor inDesc;
    ConstData_t in = nullptr;
    TensorDescriptor wDesc;
    ConstData_t w = nullptr;
    TensorDescriptor outDesc;
    Data_t out       = nullptr;
    ConstData_t bias = nullptr;
};

} // namespace miopen
