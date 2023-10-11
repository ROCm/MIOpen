/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include "tensor_holder.hpp"

#include <miopen/seq_tensor.hpp>

template <class T>
struct seqTensor
{
    miopen::SeqTensorDescriptor desc;
    
//private:
    std::vector<T> data;
//public:
    
    size_t GetDataByteSize() const
    {
        return data.empty() ? desc.GetTensorRealByteSpace() : data.size() * sizeof(T);
    }
    
    size_t GetSize() const { 
        return desc.GetTensorRealByteSpace() / sizeof(T);
    }
    
    std::vector<T>& GetDataPtr() const { return data.data(); }
    
    //size_t GetNotPaddedDataCnt() { return desc.GetElementCount();}

    seqTensor(const miopen::SeqTensorDescriptor& tensor_desc)
        : desc(tensor_desc), data(desc.GetTensorRealByteSpace() / sizeof(T))
    {
    }

    template <class X>
    seqTensor(const std::vector<X>& dims)
        : desc(miopen_type<T>{}, dims), data(desc.GetTensorRealByteSpace() / sizeof(T))
    {
    }

    seqTensor(miopenDataType_t t,
              miopenTensorLayout_t layout,
              std::size_t batch,
              std::size_t seq,
              std::size_t vector_in)
        : seqTensor(t, layout, {batch, seq, vector_in})
    {
    }

    template <class X>
    seqTensor(miopenDataType_t t, miopenTensorLayout_t layout, const std::vector<X>& dims)
        : desc(t, layout, dims), data(desc.GetTensorRealByteSpace() / sizeof(T))
    {
    }

    
};

