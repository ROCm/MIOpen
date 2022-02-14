/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "order.hpp"

#ifndef TENSOR_REORDER_OCCUPANCY
#define TENSOR_REORDER_OCCUPANCY 4
#endif

inline __device__ uint32_t magic_div_u32(const uint32_t& numer,
                                         const uint32_t& magic,
                                         const uint32_t& shift)
{
    uint32_t tmp = __umulhi(numer, magic);
    return (tmp + numer) >> shift;
}

template <typename T, typename dst_order>
inline __device__ void general_4d_reorder_1x256(T* dst,
                                                T* src,
                                                uint32_t dim_0,
                                                uint32_t dim_1,
                                                uint32_t dim_2,
                                                uint32_t dim_3,
                                                uint32_t dim_stride,
                                                uint32_t dim_total,
                                                uint32_t magic_stride0,
                                                uint32_t shift_stride0,
                                                uint32_t magic_stride1,
                                                uint32_t shift_stride1,
                                                uint32_t magic_stride2,
                                                uint32_t shift_stride2)
{
    constexpr auto dorder = dst_order{};
    uint32_t pixel_total  = dim_0 * dim_1 * dim_2 * dim_3;
    uint32_t src_index, dst_index;
    const uint64_t src_dim[4] = {dim_0, dim_1, dim_2, dim_3};
    const uint64_t dst_dim[4] = {
        src_dim[dorder.at(0)], src_dim[dorder.at(1)], src_dim[dorder.at(2)], src_dim[dorder.at(3)]};
    const uint64_t src_stride[4] = {
        src_dim[1] * src_dim[2] * src_dim[3], src_dim[2] * src_dim[3], src_dim[3], 1};
    const uint64_t dst_stride[4] = {
        dst_dim[1] * dst_dim[2] * dst_dim[3], dst_dim[2] * dst_dim[3], dst_dim[3], 1};

    uint32_t i_src[4] = {0, 0, 0, 0};
    uint32_t i_dst[4] = {0, 0, 0, 0};

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        for(uint32_t k = 0; k < 1; k++)
        {
            // unroll k         block          thread
            src_index = k * dim_total * 256 + dim_id * 256 + threadIdx.x;
            if(src_index < pixel_total)
            {
                i_src[0] = magic_div_u32(src_index, magic_stride0, shift_stride0);
                i_src[1] = magic_div_u32(
                    src_index - i_src[0] * src_stride[0], magic_stride1, shift_stride1);
                i_src[2] =
                    magic_div_u32(src_index - i_src[0] * src_stride[0] - i_src[1] * src_stride[1],
                                  magic_stride2,
                                  shift_stride2);
                i_src[3] = src_index - i_src[0] * src_stride[0] - i_src[1] * src_stride[1] -
                           i_src[2] * src_stride[2];

                i_dst[0] = i_src[dorder.at(0)];
                i_dst[1] = i_src[dorder.at(1)];
                i_dst[2] = i_src[dorder.at(2)];
                i_dst[3] = i_src[dorder.at(3)];

                dst_index = i_dst[0] * dst_stride[0] + i_dst[1] * dst_stride[1] +
                            i_dst[2] * dst_stride[2] + i_dst[3] * dst_stride[3];
                dst[dst_index] = src[src_index];
            }
        }
    }
}

template <typename T, typename dst_order>
inline __device__ void general_4d_reorder_2x256(T* dst,
                                                T* src,
                                                uint32_t dim_0,
                                                uint32_t dim_1,
                                                uint32_t dim_2,
                                                uint32_t dim_3,
                                                uint32_t dim_stride,
                                                uint32_t dim_total,
                                                uint32_t magic_stride0,
                                                uint32_t shift_stride0,
                                                uint32_t magic_stride1,
                                                uint32_t shift_stride1,
                                                uint32_t magic_stride2,
                                                uint32_t shift_stride2)
{
    constexpr auto dorder = dst_order{};
    uint32_t pixel_total  = dim_0 * dim_1 * dim_2 * dim_3;
    uint32_t src_index, dst_index;
    const uint64_t src_dim[4] = {dim_0, dim_1, dim_2, dim_3};
    const uint64_t dst_dim[4] = {
        src_dim[dorder.at(0)], src_dim[dorder.at(1)], src_dim[dorder.at(2)], src_dim[dorder.at(3)]};
    const uint64_t src_stride[4] = {
        src_dim[1] * src_dim[2] * src_dim[3], src_dim[2] * src_dim[3], src_dim[3], 1};
    const uint64_t dst_stride[4] = {
        dst_dim[1] * dst_dim[2] * dst_dim[3], dst_dim[2] * dst_dim[3], dst_dim[3], 1};

    uint32_t i_src[4] = {0, 0, 0, 0};
    uint32_t i_dst[4] = {0, 0, 0, 0};

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        for(uint32_t k = 0; k < 2; k++)
        {
            // unroll k         block          thread
            src_index = k * dim_total * 256 + dim_id * 256 + threadIdx.x;
            if(src_index < pixel_total)
            {
                i_src[0] = magic_div_u32(src_index, magic_stride0, shift_stride0);
                i_src[1] = magic_div_u32(
                    src_index - i_src[0] * src_stride[0], magic_stride1, shift_stride1);
                i_src[2] =
                    magic_div_u32(src_index - i_src[0] * src_stride[0] - i_src[1] * src_stride[1],
                                  magic_stride2,
                                  shift_stride2);
                i_src[3] = src_index - i_src[0] * src_stride[0] - i_src[1] * src_stride[1] -
                           i_src[2] * src_stride[2];

                i_dst[0] = i_src[dorder.at(0)];
                i_dst[1] = i_src[dorder.at(1)];
                i_dst[2] = i_src[dorder.at(2)];
                i_dst[3] = i_src[dorder.at(3)];

                dst_index = i_dst[0] * dst_stride[0] + i_dst[1] * dst_stride[1] +
                            i_dst[2] * dst_stride[2] + i_dst[3] * dst_stride[3];
                dst[dst_index] = src[src_index];
            }
        }
    }
}

template <typename T, typename dst_order>
inline __device__ void general_4d_reorder_4x256(T* dst,
                                                T* src,
                                                uint32_t dim_0,
                                                uint32_t dim_1,
                                                uint32_t dim_2,
                                                uint32_t dim_3,
                                                uint32_t dim_stride,
                                                uint32_t dim_total,
                                                uint32_t magic_stride0,
                                                uint32_t shift_stride0,
                                                uint32_t magic_stride1,
                                                uint32_t shift_stride1,
                                                uint32_t magic_stride2,
                                                uint32_t shift_stride2)
{
    constexpr auto dorder = dst_order{};
    uint32_t pixel_total  = dim_0 * dim_1 * dim_2 * dim_3;
    uint32_t src_index, dst_index;
    const uint64_t src_dim[4] = {dim_0, dim_1, dim_2, dim_3};
    const uint64_t dst_dim[4] = {
        src_dim[dorder.at(0)], src_dim[dorder.at(1)], src_dim[dorder.at(2)], src_dim[dorder.at(3)]};
    const uint64_t src_stride[4] = {
        src_dim[1] * src_dim[2] * src_dim[3], src_dim[2] * src_dim[3], src_dim[3], 1};
    const uint64_t dst_stride[4] = {
        dst_dim[1] * dst_dim[2] * dst_dim[3], dst_dim[2] * dst_dim[3], dst_dim[3], 1};

    uint32_t i_src[4] = {0, 0, 0, 0};
    uint32_t i_dst[4] = {0, 0, 0, 0};

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        for(uint32_t k = 0; k < 4; k++)
        {
            // unroll k         block          thread
            src_index = k * dim_total * 256 + dim_id * 256 + threadIdx.x;
            if(src_index < pixel_total)
            {
                i_src[0] = magic_div_u32(src_index, magic_stride0, shift_stride0);
                i_src[1] = magic_div_u32(
                    src_index - i_src[0] * src_stride[0], magic_stride1, shift_stride1);
                i_src[2] =
                    magic_div_u32(src_index - i_src[0] * src_stride[0] - i_src[1] * src_stride[1],
                                  magic_stride2,
                                  shift_stride2);
                i_src[3] = src_index - i_src[0] * src_stride[0] - i_src[1] * src_stride[1] -
                           i_src[2] * src_stride[2];

                i_dst[0] = i_src[dorder.at(0)];
                i_dst[1] = i_src[dorder.at(1)];
                i_dst[2] = i_src[dorder.at(2)];
                i_dst[3] = i_src[dorder.at(3)];

                dst_index = i_dst[0] * dst_stride[0] + i_dst[1] * dst_stride[1] +
                            i_dst[2] * dst_stride[2] + i_dst[3] * dst_stride[3];
                dst[dst_index] = src[src_index];
            }
        }
    }
}

template <typename T, typename dst_order>
inline __device__ void general_4d_reorder_8x256(T* dst,
                                                T* src,
                                                uint32_t dim_0,
                                                uint32_t dim_1,
                                                uint32_t dim_2,
                                                uint32_t dim_3,
                                                uint32_t dim_stride,
                                                uint32_t dim_total,
                                                uint32_t magic_stride0,
                                                uint32_t shift_stride0,
                                                uint32_t magic_stride1,
                                                uint32_t shift_stride1,
                                                uint32_t magic_stride2,
                                                uint32_t shift_stride2)
{
    constexpr auto dorder = dst_order{};
    uint32_t pixel_total  = dim_0 * dim_1 * dim_2 * dim_3;
    uint32_t src_index, dst_index;
    const uint64_t src_dim[4] = {dim_0, dim_1, dim_2, dim_3};
    const uint64_t dst_dim[4] = {
        src_dim[dorder.at(0)], src_dim[dorder.at(1)], src_dim[dorder.at(2)], src_dim[dorder.at(3)]};
    const uint64_t src_stride[4] = {
        src_dim[1] * src_dim[2] * src_dim[3], src_dim[2] * src_dim[3], src_dim[3], 1};
    const uint64_t dst_stride[4] = {
        dst_dim[1] * dst_dim[2] * dst_dim[3], dst_dim[2] * dst_dim[3], dst_dim[3], 1};

    uint32_t i_src[4] = {0, 0, 0, 0};
    uint32_t i_dst[4] = {0, 0, 0, 0};

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        for(uint32_t k = 0; k < 8; k++)
        {
            // unroll k         block          thread
            src_index = k * dim_total * 256 + dim_id * 256 + threadIdx.x;
            if(src_index < pixel_total)
            {
                i_src[0] = magic_div_u32(src_index, magic_stride0, shift_stride0);
                i_src[1] = magic_div_u32(
                    src_index - i_src[0] * src_stride[0], magic_stride1, shift_stride1);
                i_src[2] =
                    magic_div_u32(src_index - i_src[0] * src_stride[0] - i_src[1] * src_stride[1],
                                  magic_stride2,
                                  shift_stride2);
                i_src[3] = src_index - i_src[0] * src_stride[0] - i_src[1] * src_stride[1] -
                           i_src[2] * src_stride[2];

                i_dst[0] = i_src[dorder.at(0)];
                i_dst[1] = i_src[dorder.at(1)];
                i_dst[2] = i_src[dorder.at(2)];
                i_dst[3] = i_src[dorder.at(3)];

                dst_index = i_dst[0] * dst_stride[0] + i_dst[1] * dst_stride[1] +
                            i_dst[2] * dst_stride[2] + i_dst[3] * dst_stride[3];
                dst[dst_index] = src[src_index];
            }
        }
    }
}

template <typename T, typename dst_order>
inline __device__ void general_4d_reorder_16x256(T* dst,
                                                 T* src,
                                                 uint32_t dim_0,
                                                 uint32_t dim_1,
                                                 uint32_t dim_2,
                                                 uint32_t dim_3,
                                                 uint32_t dim_stride,
                                                 uint32_t dim_total,
                                                 uint32_t magic_stride0,
                                                 uint32_t shift_stride0,
                                                 uint32_t magic_stride1,
                                                 uint32_t shift_stride1,
                                                 uint32_t magic_stride2,
                                                 uint32_t shift_stride2)
{
    constexpr auto dorder = dst_order{};
    uint32_t pixel_total  = dim_0 * dim_1 * dim_2 * dim_3;
    uint32_t src_index, dst_index;
    const uint64_t src_dim[4] = {dim_0, dim_1, dim_2, dim_3};
    const uint64_t dst_dim[4] = {
        src_dim[dorder.at(0)], src_dim[dorder.at(1)], src_dim[dorder.at(2)], src_dim[dorder.at(3)]};
    const uint64_t src_stride[4] = {
        src_dim[1] * src_dim[2] * src_dim[3], src_dim[2] * src_dim[3], src_dim[3], 1};
    const uint64_t dst_stride[4] = {
        dst_dim[1] * dst_dim[2] * dst_dim[3], dst_dim[2] * dst_dim[3], dst_dim[3], 1};

    uint32_t i_src[4] = {0, 0, 0, 0};
    uint32_t i_dst[4] = {0, 0, 0, 0};

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        for(uint32_t k = 0; k < 16; k++)
        {
            // unroll k         block          thread
            src_index = k * dim_total * 256 + dim_id * 256 + threadIdx.x;
            if(src_index < pixel_total)
            {
                i_src[0] = magic_div_u32(src_index, magic_stride0, shift_stride0);
                i_src[1] = magic_div_u32(
                    src_index - i_src[0] * src_stride[0], magic_stride1, shift_stride1);
                i_src[2] =
                    magic_div_u32(src_index - i_src[0] * src_stride[0] - i_src[1] * src_stride[1],
                                  magic_stride2,
                                  shift_stride2);
                i_src[3] = src_index - i_src[0] * src_stride[0] - i_src[1] * src_stride[1] -
                           i_src[2] * src_stride[2];

                i_dst[0] = i_src[dorder.at(0)];
                i_dst[1] = i_src[dorder.at(1)];
                i_dst[2] = i_src[dorder.at(2)];
                i_dst[3] = i_src[dorder.at(3)];

                dst_index = i_dst[0] * dst_stride[0] + i_dst[1] * dst_stride[1] +
                            i_dst[2] * dst_stride[2] + i_dst[3] * dst_stride[3];
                dst[dst_index] = src[src_index];
            }
        }
    }
}

#define DEFINE_GENERAL_4D_REORDER_KERNEL(tile_trait,                                               \
                                         dst_order,                                                \
                                         accept_data_type,                                         \
                                         cast_data_type,                                           \
                                         lb_threads_per_block,                                     \
                                         lb_blocks_per_cu)                                         \
    extern "C" __global__ void __launch_bounds__(lb_threads_per_block, lb_blocks_per_cu)           \
        general_4d_reorder_##tile_trait##_##accept_data_type##_##dst_order(void* dst,              \
                                                                           void* src,              \
                                                                           uint32_t dim_0,         \
                                                                           uint32_t dim_1,         \
                                                                           uint32_t dim_2,         \
                                                                           uint32_t dim_3,         \
                                                                           uint32_t dim_stride,    \
                                                                           uint32_t dim_total,     \
                                                                           uint32_t magic_stride0, \
                                                                           uint32_t shift_stride0, \
                                                                           uint32_t magic_stride1, \
                                                                           uint32_t shift_stride1, \
                                                                           uint32_t magic_stride2, \
                                                                           uint32_t shift_stride2) \
    {                                                                                              \
        general_4d_reorder_##tile_trait<cast_data_type, dst_order>(                                \
            reinterpret_cast<cast_data_type*>(dst),                                                \
            reinterpret_cast<cast_data_type*>(src),                                                \
            dim_0,                                                                                 \
            dim_1,                                                                                 \
            dim_2,                                                                                 \
            dim_3,                                                                                 \
            dim_stride,                                                                            \
            dim_total,                                                                             \
            magic_stride0,                                                                         \
            shift_stride0,                                                                         \
            magic_stride1,                                                                         \
            shift_stride1,                                                                         \
            magic_stride2,                                                                         \
            shift_stride2);                                                                        \
    }
// default order is 0 1 2 3
using r0132 = order<0, 1, 3, 2>;
using r0213 = order<0, 2, 1, 3>; // nhwc2nchwc
using r0231 = order<0, 2, 3, 1>; // nchw2nchwc
using r0312 = order<0, 3, 1, 2>; // nhwc2nchw
using r0321 = order<0, 3, 2, 1>;
using r1023 = order<1, 0, 2, 3>;
using r1032 = order<1, 0, 3, 2>;
using r1203 = order<1, 2, 0, 3>;
using r1230 = order<1, 2, 3, 0>;
using r1302 = order<1, 3, 0, 2>; // nchw2chwnc
using r1320 = order<1, 3, 2, 0>;
using r2013 = order<2, 0, 1, 3>;
using r2031 = order<2, 0, 3, 1>;
using r2103 = order<2, 1, 0, 3>; // nhwc2chwnc
using r2130 = order<2, 1, 3, 0>;
using r2301 = order<2, 3, 0, 1>;
using r2310 = order<2, 3, 1, 0>;
using r3012 = order<3, 0, 1, 2>;
using r3021 = order<3, 0, 2, 1>;
using r3102 = order<3, 1, 0, 2>;
using r3120 = order<3, 1, 2, 0>;
using r3201 = order<3, 2, 0, 1>;
using r3210 = order<3, 2, 1, 0>;

DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0132, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0213, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0231, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0312, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0321, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1023, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1032, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1203, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1230, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1302, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1320, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2013, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2031, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2103, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2130, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2301, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2310, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3012, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3021, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3102, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3120, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3201, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3210, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0132, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0213, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0231, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0312, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0321, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1023, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1032, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1203, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1230, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1302, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1320, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2013, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2031, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2103, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2130, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2301, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2310, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3012, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3021, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3102, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3120, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3201, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3210, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0132, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0213, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0231, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0312, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0321, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1023, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1032, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1203, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1230, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1302, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1320, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2013, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2031, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2103, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2130, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2301, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2310, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3012, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3021, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3102, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3120, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3201, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3210, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0132, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0213, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0231, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0312, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0321, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1023, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1032, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1203, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1230, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1302, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1320, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2013, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2031, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2103, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2130, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2301, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2310, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3012, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3021, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3102, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3120, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3201, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3210, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0132, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0213, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0231, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0312, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0321, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1023, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1032, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1203, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1230, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1302, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1320, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2013, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2031, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2103, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2130, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2301, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2310, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3012, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3021, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3102, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3120, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3201, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3210, dwordx2, double, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0132, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0213, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0231, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0312, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0321, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1023, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1032, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1203, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1230, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1302, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1320, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2013, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2031, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2103, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2130, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2301, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2310, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3012, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3021, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3102, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3120, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3201, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3210, dword, float, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0132, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0213, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0231, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0312, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0321, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1023, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1032, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1203, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1230, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1302, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1320, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2013, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2031, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2103, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2130, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2301, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2310, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3012, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3021, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3102, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3120, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3201, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3210, dword, float, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0132, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0213, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0231, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0312, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0321, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1023, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1032, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1203, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1230, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1302, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1320, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2013, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2031, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2103, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2130, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2301, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2310, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3012, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3021, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3102, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3120, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3201, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3210, dword, float, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0132, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0213, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0231, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0312, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0321, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1023, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1032, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1203, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1230, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1302, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1320, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2013, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2031, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2103, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2130, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2301, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2310, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3012, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3021, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3102, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3120, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3201, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3210, dword, float, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0132, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0213, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0231, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0312, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0321, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1023, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1032, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1203, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1230, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1302, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1320, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2013, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2031, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2103, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2130, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2301, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2310, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3012, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3021, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3102, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3120, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3201, dword, float, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3210, dword, float, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0132, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0213, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0231, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0312, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0321, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1023, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1032, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1203, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1230, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1302, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1320, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2013, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2031, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2103, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2130, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2301, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2310, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3012, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3021, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3102, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3120, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3201, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3210, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0132, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0213, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0231, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0312, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0321, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1023, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1032, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1203, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1230, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1302, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1320, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2013, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2031, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2103, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2130, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2301, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2310, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3012, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3021, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3102, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3120, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3201, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3210, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0132, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0213, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0231, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0312, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0321, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1023, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1032, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1203, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1230, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1302, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1320, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2013, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2031, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2103, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2130, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2301, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2310, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3012, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3021, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3102, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3120, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3201, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3210, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0132, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0213, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0231, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0312, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0321, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1023, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1032, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1203, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1230, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1302, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1320, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2013, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2031, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2103, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2130, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2301, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2310, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3012, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3021, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3102, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3120, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3201, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3210, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0132, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0213, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0231, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0312, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0321, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1023, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1032, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1203, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1230, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1302, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1320, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2013, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2031, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2103, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2130, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2301, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2310, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3012, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3021, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3102, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3120, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3201, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3210, half, ushort, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0132, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0213, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0231, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0312, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r0321, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1023, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1032, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1203, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1230, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1302, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r1320, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2013, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2031, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2103, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2130, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2301, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r2310, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3012, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3021, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3102, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3120, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3201, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1x256, r3210, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0132, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0213, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0231, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0312, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r0321, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1023, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1032, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1203, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1230, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1302, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r1320, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2013, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2031, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2103, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2130, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2301, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r2310, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3012, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3021, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3102, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3120, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3201, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2x256, r3210, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0132, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0213, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0231, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0312, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r0321, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1023, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1032, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1203, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1230, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1302, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r1320, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2013, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2031, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2103, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2130, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2301, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r2310, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3012, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3021, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3102, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3120, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3201, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4x256, r3210, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0132, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0213, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0231, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0312, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r0321, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1023, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1032, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1203, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1230, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1302, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r1320, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2013, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2031, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2103, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2130, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2301, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r2310, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3012, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3021, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3102, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3120, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3201, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8x256, r3210, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0132, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0213, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0231, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0312, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r0321, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1023, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1032, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1203, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1230, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1302, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r1320, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2013, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2031, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2103, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2130, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2301, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r2310, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3012, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3021, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3102, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3120, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3201, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16x256, r3210, byte, uchar, 256, TENSOR_REORDER_OCCUPANCY)
