/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2022 Advanced Micro Devices, Inc.
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
#include "general_tensor_reorder_kernel_util.hpp"
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
