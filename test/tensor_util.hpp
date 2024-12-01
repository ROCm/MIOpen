/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#ifndef GUARD_TENSOR_UTIL_HPP
#define GUARD_TENSOR_UTIL_HPP

#include <type_traits>

#include <miopen/miopen.h>
#include <miopen/filesystem.hpp>
#include <miopen/tensor.hpp>
#include "tensor_holder.hpp"

namespace fs = miopen::fs;

// unary operation
template <class DataOp, typename Container>
void operate_over_subtensor(DataOp&& dataOp,
                            Container& srcSuperTensor,
                            const miopen::TensorDescriptor& srcSubDesc,
                            const int64_t srcOffset)
{
    const auto& srcStrides = srcSubDesc.GetStrides();
    const auto& srcLens    = srcSubDesc.GetLengths();

    auto operate_over_subtensor_impl =
        [&, dataOp, max_dim = srcLens.size() - 1](
            auto&& self, const size_t current_dim, const int64_t srcOff) -> void {
        const auto current_stride = srcStrides[current_dim];

        int64_t index = srcOff;

        for(size_t i = 0; i < srcLens[current_dim]; ++i)
        {
            if(current_dim < max_dim)
            {
                self(self, current_dim + 1, index);
            }
            else
            {
                dataOp(srcSuperTensor[index]);
            }
            index += current_stride;
        }
    };
    operate_over_subtensor_impl(operate_over_subtensor_impl, 0, srcOffset);
}

// binary operation, it implies cast operation
template <class DataOp, typename DstContainer, typename SrcContainer>
void operate_over_subtensor(DataOp&& dataOp,
                            DstContainer& dstSuperTensor,
                            SrcContainer& srcSuperTensor,
                            const miopen::TensorDescriptor& dstSubDesc,
                            const miopen::TensorDescriptor& srcSubDesc,
                            const int64_t dstOffset,
                            const int64_t srcOffset)
{
    const auto& dstStrides = dstSubDesc.GetStrides();
    const auto& srcStrides = srcSubDesc.GetStrides();

    const auto& srcLens = srcSubDesc.GetLengths();

    auto operate_over_subtensor_impl =
        [&, dataOp, max_dim = srcLens.size() - 1](auto&& self,
                                                  const size_t current_dim,
                                                  const int64_t dstOff,
                                                  const int64_t srcOff) -> void {
        const auto dstStride = dstStrides[current_dim];
        const auto srcStride = srcStrides[current_dim];

        int64_t dstIdx = dstOff;
        int64_t srcIdx = srcOff;

        for(size_t i = 0; i < srcLens[current_dim]; ++i)
        {
            if(current_dim < max_dim)
            {
                self(self, current_dim + 1, dstIdx, srcIdx);
            }
            else
            {
                dataOp(dstSuperTensor[dstIdx], srcSuperTensor[srcIdx]);
            }
            dstIdx += dstStride;
            srcIdx += srcStride;
        }
    };
    operate_over_subtensor_impl(operate_over_subtensor_impl, 0, dstOffset, srcOffset);
}

// ternary operation, it implies broadcasting for src2
template <typename DataOp, typename Container>
void operate_over_subtensor(DataOp&& dataOp,
                            Container& dstSuperTensor,
                            const Container& src1SuperTensor,
                            const Container& src2SuperTensor,
                            const miopen::TensorDescriptor& dstSubDesc,
                            const miopen::TensorDescriptor& src1SubDesc,
                            const miopen::TensorDescriptor& src2SubDesc,
                            const int64_t dstOffset,
                            const int64_t src1Offset,
                            const int64_t src2Offset)
{
    const auto& dstStrides  = dstSubDesc.GetStrides();
    const auto& src1Strides = src1SubDesc.GetStrides();
    const auto& src2Strides = src2SubDesc.GetStrides();

    const auto& src1Lens = src1SubDesc.GetLengths();
    const auto& src2Lens = src2SubDesc.GetLengths();

    auto operate_over_subtensor_impl =
        [&, dataOp, max_dim = src1Lens.size() - 1](auto&& self,
                                                   const size_t current_dim,
                                                   const int64_t dstOff,
                                                   const int64_t src1Off,
                                                   const int64_t src2Off) -> void {
        const auto dstStride  = dstStrides[current_dim];
        const auto src1Stride = src1Strides[current_dim];
        const auto src2Stride = src2Strides[current_dim];
        const bool squashed   = src1Lens[current_dim] != src2Lens[current_dim];

        int64_t dstIdx  = dstOff;
        int64_t src1Idx = src1Off;
        int64_t src2Idx = src2Off;

        for(size_t i = 0; i < src1Lens[current_dim]; ++i)
        {
            if(current_dim < max_dim)
            {
                self(self, current_dim + 1, dstIdx, src1Idx, src2Idx);
            }
            else
            {
                dataOp(dstSuperTensor[dstIdx], src1SuperTensor[src1Idx], src2SuperTensor[src2Idx]);
            }
            dstIdx += dstStride;
            src1Idx += src1Stride;
            src2Idx += squashed ? 0 : src2Stride;
        }
    };
    operate_over_subtensor_impl(operate_over_subtensor_impl, 0, dstOffset, src1Offset, src2Offset);
}

template <typename T>
void output_tensor_to_csv(const tensor<T>& x, const fs::path& filename)
{
    int dim = x.desc.GetSize();
    std::vector<int> index(dim);

    std::ofstream file{filename};

    for(int j = 0; j < dim; ++j)
        file << "d" << j << ", ";
    file << "x" << std::endl;

    for(int i = 0; i < x.data.size(); ++i)
    {
        int is = i;
        for(int j = 0; j < dim; ++j)
        {
            index[j] = is / x.desc.GetStrides()[j];
            is -= index[j] * x.desc.GetStrides()[j];
        }

        for(int j = 0; j < dim; ++j)
        {
            file << index[j] << ", ";
        }
        file << x[i] << std::endl;
    }
}

template <typename T>
void output_tensor_to_bin(const fs::path& fileName, T* data, size_t dataNumItems)
{
    std::ofstream outFile(fileName, std::ios::binary);
    if(outFile.is_open())
    {
        outFile.write(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
        outFile.close();
    }
    else
    {
        std::cerr << "Could not open file " << fileName << " for writing" << std::endl;
    }
}

template <typename T>
void print_tensor(const tensor<T>& tensor_val,
                  std::string header_msg = "start",
                  size_t set_precision   = 2)
{
    std::cout << "\n================= " << header_msg << " =====================\n";

    const auto lens = tensor_val.desc.GetLengths();
    size_t dim      = lens.size();
    if(dim == 2)
    {
        ford(lens[0], lens[1])([&](int ii, int jj) {
            std::cout << std::fixed << std::setprecision(set_precision) << tensor_val(ii, jj)
                      << ", ";
            if(jj == lens[1] - 1)
            {
                std::cout << "\n";
            }
        });
    }
    else if(dim == 3)
    {
        ford(lens[0], lens[1], lens[2])([&](int ii, int jj, int kk) {
            std::cout << std::fixed << std::setprecision(set_precision) << tensor_val(ii, jj, kk)
                      << ", ";
            if(kk == lens[2] - 1)
            {
                std::cout << "\n";
            }
            if(kk == lens[2] - 1 && jj == lens[1] - 1)
            {
                std::cout << "\n";
            }
        });
    }
    else if(dim == 4)
    {
        ford(lens[0], lens[1], lens[2], lens[3])([&](int ii, int jj, int kk, int ll) {
            std::cout << std::fixed << std::setprecision(set_precision)
                      << tensor_val(ii, jj, kk, ll) << ", ";
            if(ll == lens[3] - 1)
            {
                std::cout << "\n";
            }
            if(ll == lens[3] - 1 && kk == lens[2] - 1)
            {
                std::cout << "\n";
            }
            if(ll == lens[3] - 1 && kk == lens[2] - 1 && jj == lens[1] - 1)
            {
                std::cout << "\n";
            }
        });
    }
    else
    {
        std::cerr << "Need to handle print for dim : " << dim << std::endl;
    }

    std::cout << "\n=================end=====================\n";
}

#endif
