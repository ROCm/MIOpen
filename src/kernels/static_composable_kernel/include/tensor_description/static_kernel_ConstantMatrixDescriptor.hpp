/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#ifndef CK_CONSTANT_MATRIX_DESCRIPTOR_HPP
#define CK_CONSTANT_MATRIX_DESCRIPTOR_HPP

#include "static_kernel_common_header.hpp"
#include "static_kernel_ConstantTensorDescriptor_deprecated.hpp"
#include "static_kernel_tensor_descriptor.hpp"

namespace ck {

template <index_t NRow_, index_t NCol_, index_t RowStride_>
struct ConstantMatrixDescriptor
{
    __host__ __device__ constexpr ConstantMatrixDescriptor()
    {
        static_assert(NCol_ <= RowStride_, "wrong! NCol > RowStride!");
    }

    __host__ __device__ static constexpr index_t NRow() { return NRow_; }

    __host__ __device__ static constexpr index_t NCol() { return NCol_; }

    __host__ __device__ static constexpr index_t RowStride() { return RowStride_; }

    __host__ __device__ static constexpr auto GetLengths() { return Sequence<NRow_, NCol_>{}; }

    __host__ __device__ static constexpr index_t GetElementSize() { return NRow_ * NCol_; }

    __host__ __device__ static constexpr index_t GetElementSpace() { return NRow_ * RowStride_; }

    __host__ __device__ static index_t GetOffsetFromMultiIndex(index_t irow, index_t icol)
    {
        return irow * RowStride_ + icol;
    }

    __host__ __device__ static index_t CalculateOffset(index_t irow, index_t icol)
    {
        return GetOffsetFromMultiIndex(irow, icol);
    }

    template <index_t SubNRow, index_t SubNCol>
    __host__ __device__ static constexpr auto MakeSubMatrixDescriptor(Number<SubNRow>,
                                                                      Number<SubNCol>)
    {
        return ConstantMatrixDescriptor<SubNRow, SubNCol, RowStride_>{};
    }
};

template <index_t NRow, index_t NCol>
__host__ __device__ constexpr auto make_ConstantMatrixDescriptor_packed(Number<NRow>, Number<NCol>)
{
    return ConstantMatrixDescriptor<NRow, NCol, NCol>{};
}

template <index_t NRow, index_t NCol, index_t RowStride>
__host__ __device__ constexpr auto
    make_ConstantMatrixDescriptor(Number<NRow>, Number<NCol>, Number<RowStride>)
{
    return ConstantMatrixDescriptor<NRow, NCol, RowStride>{};
}

template <typename... Ts>
__host__ __device__ constexpr auto
    make_ConstantMatrixDescriptor(ConstantTensorDescriptor_deprecated<Ts...>)
{
    using TDesc = ConstantTensorDescriptor_deprecated<Ts...>;
    static_assert(TDesc::GetNumOfDimension() == 2, "wrong");
    static_assert(TDesc::GetStrides()[1] == 1, "wrong");
    return ConstantMatrixDescriptor<TDesc::GetLengths()[0],
                                    TDesc::GetLengths()[1],
                                    TDesc::GetStrides()[0]>{};
}

template <typename... Ts>
__host__ __device__ constexpr auto make_ConstantMatrixDescriptor(NativeTensorDescriptor<Ts...>)
{
    using TDesc = NativeTensorDescriptor<Ts...>;
    static_assert(TDesc::GetNumOfDimension() == 2, "wrong");
    static_assert(TDesc::GetStrides()[1] == 1, "wrong");
    return ConstantMatrixDescriptor<TDesc::GetLengths()[0],
                                    TDesc::GetLengths()[1],
                                    TDesc::GetStrides()[0]>{};
}

template <typename TDesc>
__host__ __device__ void print_ConstantMatrixDescriptor(TDesc, const char* s)
{
    printf(
        "%s NRow %u NCol %u RowStride %u\n", s, TDesc::NRow(), TDesc::NCol(), TDesc::RowStride());
}

} // namespace ck

#endif
