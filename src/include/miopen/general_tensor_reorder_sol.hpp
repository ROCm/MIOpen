/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_TENSOR_REORDER_SOL_HPP
#define GUARD_MIOPEN_TENSOR_REORDER_SOL_HPP

#include <miopen/miopen.h>
#include <miopen/kernel_info.hpp>
#include <miopen/op_kernel_args.hpp>
#include <miopen/execution_context.hpp>
#include <vector>
#include <../kernels/gpu_general_tensor_reorder_kernel/order.hpp>

#include <miopen/tensor.hpp>
#include <miopen/magic_div.hpp>
#include <miopen/float_equal.hpp>
#include <string>
#include <limits>
#include <iostream>
#include <sstream>

#define TENSOR_REORDER_BLOCK_SIZE 256
#define TENSOR_REORDER_PERSISTENT 0

#if TENSOR_REORDER_PERSISTENT
#define TENSOR_REORDER_OCCUPANCY 4
#endif

namespace miopen {

struct GeneralReorderParam
{
    int tile_x{0};
    int tile_y{0};
    int pack_x{0};
    int pack_y{0};
    int ediv_x{0};
    int ediv_y{0};
};

template<typename dst_order>
struct GeneralReorderSolution
{
    GeneralReorderSolution(const ExecutionContext& ctx_,
                                miopenDataType_t data_type_,
                                uint32_t dim_0_,
                                uint32_t dim_1_,
                                uint32_t dim_2_,
                                uint32_t dim_3_);
    solver::KernelInfo GetKernel() const;
    std::vector<OpKernelArg> GetKernelArg() const;
    std::string GetKernelName() const;
    bool IsSkippable() const;
    size_t GetSize() const;

    miopenDataType_t data_type;
    uint32_t dim_0;
    uint32_t dim_1;
    uint32_t dim_2;
    uint32_t dim_3;
    int num_cu;

    GeneralReorderParam kernel_param_heuristic;
};
namespace tensor_reorder {

static inline std::string GetNameTrait(std::size_t type_size)
{
    if(type_size == 1)
        return "byte";
    if(type_size == 2)
        return "half";
    if(type_size == 4)
        return "dword";
    MIOPEN_THROW("data type not supported");
}

static inline const std::vector<GeneralReorderParam>& GetKernelList(std::size_t data_size)
{
    if(data_size == 1)
    {
        static const std::vector<GeneralReorderParam> byte_kernel_list{
            // clang-format off
            {1, 256, 1, 1, 1, 1},
            {2, 256, 1, 1, 1, 1},
            {4, 256, 1, 1, 1, 1},
            {8, 256, 1, 1, 1, 1},
            {16, 256, 1, 1, 1, 1},
            // clang-format on
        };
        return byte_kernel_list;
    }
    if(data_size == 2)
    {
        static const std::vector<GeneralReorderParam> half_kernel_list{
            // clang-format off
            {1, 256, 1, 1, 1, 1},
            {2, 256, 1, 1, 1, 1},
            {4, 256, 1, 1, 1, 1},
            {8, 256, 1, 1, 1, 1},
            {16, 256, 1, 1, 1, 1},
            // clang-format on
        };
        return half_kernel_list;
    }
    if(data_size == 4)
    {
        static const std::vector<GeneralReorderParam> dword_kernel_list{
            // clang-format off
            {1, 256, 1, 1, 1, 1},
            {2, 256, 1, 1, 1, 1},
            {4, 256, 1, 1, 1, 1},
            {8, 256, 1, 1, 1, 1},
            {16, 256, 1, 1, 1, 1},
            // clang-format on
        };
        return dword_kernel_list;
    }
    MIOPEN_THROW("data type not supported");
}

static inline bool IsApplicable(uint32_t /* batch */,
                                uint32_t height,
                                uint32_t width,
                                const GeneralReorderParam* kparam)
{
    return width % kparam->ediv_x == 0 && height % kparam->ediv_y == 0;
}

static inline bool IsSameSide(uint32_t height, uint32_t width, const GeneralReorderParam* kparam)
{
    float radio = 0;
    if(width > height)
        radio = static_cast<float>(kparam->tile_x) / kparam->tile_y;
    else
        radio = static_cast<float>(kparam->tile_y) / kparam->tile_x;

    // E.g. for cases like width=1000, height=10
    // Allow at least 32x64, 64x64... 16x64 not allowed
    return radio >= 0.4;
}

template <typename T>
static inline float GetNormalizedRadio(T x, T y)
{
    if(y > x)
        return static_cast<float>(y) / x;
    return static_cast<float>(x) / y;
}
template<typename dst_order>
static inline std::string GetKernelName(std::size_t data_size, const GeneralReorderParam* kparam)
{
    std::ostringstream kernel_name;
    std::string type_trait = GetNameTrait(data_size);
    kernel_name << "general_4d_reorder_" << kparam->tile_x << "x" << kparam->tile_y << "_";
    if(!(kparam->pack_x == 1 && kparam->pack_y == 1 && kparam->ediv_x == 1 && kparam->ediv_y == 1))
    {
        kernel_name << "pack_" << kparam->pack_x << "x" << kparam->pack_y << "_ediv_"
                    << kparam->ediv_x << "x" << kparam->ediv_y << "_";
    }
    kernel_name << type_trait<<"_r"<<dst_order::at(0)<<dst_order::at(1)<<dst_order::at(2)<<dst_order::at(3);
    return kernel_name.str();
}

static inline std::size_t GetExtraPaddingSize(uint32_t /* batch */,
                                              uint32_t height,
                                              uint32_t width,
                                              const GeneralReorderParam* kparam)
{
    // For simplicity and speed, we ignore batch, only compute h*w
    uint32_t padded_h = ((height + kparam->tile_y - 1) / kparam->tile_y) * kparam->tile_y;
    uint32_t padded_w = ((width + kparam->tile_x - 1) / kparam->tile_x) * kparam->tile_x;
    return static_cast<std::size_t>(padded_h) * padded_w - static_cast<std::size_t>(height) * width;
}

static inline GeneralReorderParam
HeuristicGet(std::size_t data_size, uint32_t dim_0, uint32_t dim_1, uint32_t dim_2, uint32_t dim_3)
{
    /*
     * TODO:
     * Design a algorithm to determine general tensor reorder tile size.
     */

    if(dim_0 >= 1 && dim_1 >= 1 && dim_2 >= 1 && dim_3 >= 1 && data_size<=4)
    {
        if(dim_3 >= 16)
        {
            return GeneralReorderParam{16, 256, 1, 1, 1, 1};
        }
        else if(dim_3 >= 8)
        {
            return GeneralReorderParam{8, 256, 1, 1, 1, 1};
        }
        else if(dim_3 >= 4)
        {
            return GeneralReorderParam{4, 256, 1, 1, 1, 1};
        }
        else if(dim_3 >= 2)
        {
            return GeneralReorderParam{2, 256, 1, 1, 1, 1};
        }
        else
        {
            return GeneralReorderParam{1, 256, 1, 1, 1, 1};
        }
    }
    MIOPEN_THROW("data type not supported");
}

} // namespace tensor_reorder
template<typename dst_order>
GeneralReorderSolution<dst_order>::GeneralReorderSolution(const ExecutionContext& ctx,
                                                          miopenDataType_t data_type_,
                                                          uint32_t dim_0_,
                                                          uint32_t dim_1_,
                                                          uint32_t dim_2_,
                                                          uint32_t dim_3_)
    : data_type(data_type_), dim_0(dim_0_), dim_1(dim_1_), dim_2(dim_2_), dim_3(dim_3_)
{
    if(data_type == miopenInt8x4 || data_type == miopenDouble)
        MIOPEN_THROW("These data type are not supported");
    num_cu                 = ctx.GetStream().GetMaxComputeUnits();
    std::size_t data_size  = miopen::GetTypeSize(data_type);
    kernel_param_heuristic = tensor_reorder::HeuristicGet(data_size, dim_0, dim_1, dim_2, dim_3);
}

template<typename dst_order>
solver::KernelInfo GeneralReorderSolution<dst_order>::GetKernel() const
{
    std::size_t block_size = TENSOR_REORDER_BLOCK_SIZE;
#if TENSOR_REORDER_PERSISTENT
    std::size_t grid_size = num_cu * TENSOR_REORDER_OCCUPANCY;
#else
    uint32_t pixel_total = dim_0 * dim_1 * dim_2 * dim_3;
    uint32_t dim_total = (pixel_total + block_size * kernel_param_heuristic.tile_x - 1) / (block_size * kernel_param_heuristic.tile_x);
    std::size_t grid_size = dim_total;
#endif
    std::string kernel_name = GetKernelName();
    solver::KernelInfo kernel;
    kernel.kernel_file = "general_tensor_reorder.cpp";
    kernel.kernel_name = kernel_name;
    kernel.g_wk.clear();
    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    MIOPEN_LOG_I2("GeneralReorderSolution use kernel: " + kernel_name);

    return kernel;
}

template<typename dst_order>
std::vector<OpKernelArg> GeneralReorderSolution<dst_order>::GetKernelArg() const
{
    std::size_t block_size = TENSOR_REORDER_BLOCK_SIZE;
    uint32_t pixel_total = dim_0 * dim_1 * dim_2 * dim_3;
    uint32_t dim_total = (pixel_total + block_size * kernel_param_heuristic.tile_x - 1) / (block_size * kernel_param_heuristic.tile_x);
#if TENSOR_REORDER_PERSISTENT
    std::size_t grid_size = num_cu * TENSOR_REORDER_OCCUPANCY;
#else
    std::size_t grid_size = dim_total;
#endif

    magic_div_u32_t magic_stride0 = magic_div_u32_gen(dim_1 * dim_2 * dim_3);
    magic_div_u32_t magic_stride1 = magic_div_u32_gen(dim_2 * dim_3);
    magic_div_u32_t magic_stride2 = magic_div_u32_gen(dim_3);

    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(dim_0);
    opArgs.emplace_back(dim_1);
    opArgs.emplace_back(dim_2);
    opArgs.emplace_back(dim_3);
    opArgs.emplace_back(static_cast<uint32_t>(grid_size));
    opArgs.emplace_back(dim_total);
    opArgs.emplace_back(magic_stride0.magic);
    opArgs.emplace_back(static_cast<uint32_t>(magic_stride0.shift));
    opArgs.emplace_back(magic_stride1.magic);
    opArgs.emplace_back(static_cast<uint32_t>(magic_stride1.shift));
    opArgs.emplace_back(magic_stride2.magic);
    opArgs.emplace_back(static_cast<uint32_t>(magic_stride2.shift));

    return opArgs;
}

template<typename dst_order>
std::string GeneralReorderSolution<dst_order>::GetKernelName() const
{
    std::size_t data_size = miopen::GetTypeSize(data_type);
    std::ostringstream kernel_name;
    std::string type_trait = tensor_reorder::GetNameTrait(data_size);
    kernel_name << "general_4d_reorder_" << kernel_param_heuristic.tile_x << "x" << kernel_param_heuristic.tile_y << "_";
    if(!(kernel_param_heuristic.pack_x == 1 && kernel_param_heuristic.pack_y == 1 && kernel_param_heuristic.ediv_x == 1 && kernel_param_heuristic.ediv_y == 1))
    {
        kernel_name << "pack_" << kernel_param_heuristic.pack_x << "x" << kernel_param_heuristic.pack_y << "_ediv_"
                    << kernel_param_heuristic.ediv_x << "x" << kernel_param_heuristic.ediv_y << "_";
    }
    kernel_name << type_trait<<"_r"<<dst_order::at(0)<<dst_order::at(1)<<dst_order::at(2)<<dst_order::at(3);
    return kernel_name.str();
    //return tensor_reorder::GetKernelName(data_size, &kernel_param_heuristic);
}

template<typename dst_order>
bool GeneralReorderSolution<dst_order>::IsSkippable() const
{
    // Disable the IsSkippable funciton
    return dim_0 == 0 || dim_1 == 0 || dim_2 == 0 || dim_3 == 0 ;
}

template<typename dst_order>
size_t GeneralReorderSolution<dst_order>::GetSize() const
{
    return miopen::GetTypeSize(data_type) * dim_0 * dim_1 * dim_2 * dim_3;
}
} // namespace miopen
#endif
