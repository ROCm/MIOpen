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

namespace miopen {

struct TensorDescriptor;

struct ConvTensors
{
    const TensorDescriptor& xDesc;
    ConstData_t x;
    const TensorDescriptor& wDesc;
    ConstData_t w;
    const TensorDescriptor& yDesc;
    ConstData_t y;
};

struct ConvFwdTensors
{
    const TensorDescriptor& xDesc;
    ConstData_t x;
    const TensorDescriptor& wDesc;
    ConstData_t w;
    const TensorDescriptor& yDesc;
    Data_t y;

    ConstData_t in() const { return x; }
    Data_t out() const { return y; }

    operator ConvTensors() const { return {xDesc, x, wDesc, w, yDesc, y}; }
};

struct ConvBwdTensors
{
    const TensorDescriptor& dyDesc;
    ConstData_t dy;
    const TensorDescriptor& wDesc;
    ConstData_t w;
    const TensorDescriptor& dxDesc;
    Data_t dx;

    ConstData_t in() const { return dy; }
    Data_t out() const { return dx; }

    operator ConvTensors() const { return {dxDesc, dx, wDesc, w, dyDesc, dy}; }
};

struct ConvWrwTensors
{
    const TensorDescriptor& dyDesc;
    ConstData_t dy;
    const TensorDescriptor& xDesc;
    ConstData_t x;
    const TensorDescriptor& dwDesc;
    Data_t dw;

    operator ConvTensors() const { return {xDesc, x, dwDesc, dw, dyDesc, dy}; }
};

} // namespace miopen
