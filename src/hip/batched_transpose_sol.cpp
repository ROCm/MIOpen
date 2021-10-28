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

#include <miopen/batched_transpose_sol.hpp>
#include <miopen/datatype.hpp>
#include <miopen/magic_div.hpp>
#include <string>
#include <vector>
#include <limits>
#include <iostream>
#include <sstream>

#define BATCHED_TRANSPOSE_BLOCK_SIZE 256
#define BATCHED_TRANSPOSE_OCCUPANCY 4

namespace miopen{

std::string transpose_kernel_get_name_trait(std::size_t type_size){
    if(type_size == 1)
        return "byte";
    if(type_size == 2)
        return "half";
    if(type_size == 4)
        return "dword";
    MIOPEN_THROW("data type not supported");
}

static inline const std::vector<BatchedTransposeParam>& get_transpose_kernel_list(std::size_t data_size)
{
    if(data_size == 1){
        static const std::vector<BatchedTransposeParam> byte_kernel_list {
            {16, 16, 1, 1, 1, 1},
            {16, 32, 1, 1, 1, 1},
            {32, 16, 1, 1, 1, 1},
            {32, 32, 1, 1, 1, 1},
        };
        return byte_kernel_list;
    }
    if(data_size == 2){
        static const std::vector<BatchedTransposeParam> half_kernel_list {
            {16, 16, 1, 1, 1, 1},
            {32, 16, 1, 1, 1, 1},
            {16, 32, 1, 1, 1, 1},
            {32, 32, 1, 1, 1, 1},

            {32, 32, 2, 2, 1, 1},
            {32, 32, 2, 2, 1, 2},
            {32, 32, 2, 2, 2, 1},
            {32, 32, 2, 2, 2, 2},

            {16, 64, 1, 4, 1, 2},
            {64, 16, 4, 1, 2, 1},

            {32, 64, 2, 4, 1, 2},
            {32, 64, 2, 4, 2, 2},
            {32, 64, 2, 4, 2, 4},

            {64, 32, 4, 2, 2, 1},
            {64, 32, 4, 2, 2, 2},
            {64, 32, 4, 2, 4, 2},

            {64, 64, 4, 4, 2, 2},
            {64, 64, 4, 4, 4, 4},
        };
        return half_kernel_list;
    }
    if(data_size == 4){
        static const std::vector<BatchedTransposeParam> dword_kernel_list {
            {16, 16, 1, 1, 1, 1},
            {16, 32, 1, 1, 1, 1},
            {32, 16, 1, 1, 1, 1},
            {32, 32, 1, 1, 1, 1},
        };
        return dword_kernel_list;
    }
    MIOPEN_THROW("data type not supported");
}

static inline bool transpose_kernel_is_valid(uint32_t /* batch */, uint32_t height, uint32_t width, const BatchedTransposeParam * kparam)
{
    return width % kparam->ediv_x == 0 && height % kparam->ediv_y == 0;
}

static inline std::string get_transpose_kernel_name(std::size_t data_size, const BatchedTransposeParam * kparam){
    std::ostringstream kernel_name;
    std::string type_trait = transpose_kernel_get_name_trait(data_size);
    kernel_name << "batched_transpose_" << kparam->tile_x << "x" <<  kparam->tile_y << "_";
    if(!(kparam->pack_x == 1 && kparam->pack_y == 1 && kparam->ediv_x == 1 && kparam->ediv_y == 1)){
        kernel_name << "pack_" << kparam->pack_x << "x" << kparam->pack_y << "_ediv_" << kparam->ediv_x << "x" << kparam->ediv_y << "_";
    }
    kernel_name << type_trait;
    return kernel_name.str();
}

static inline std::size_t get_extra_padding_size(uint32_t /* batch */, uint32_t height, uint32_t width, const BatchedTransposeParam * kparam)
{
    // for simplicity and speed, we ignore batch, only compute h*w
    uint32_t padded_h = ((height + kparam->tile_y - 1) / kparam->tile_y) * kparam->tile_y;
    uint32_t padded_w = ((width + kparam->tile_x - 1) / kparam->tile_x) * kparam->tile_x;
    return static_cast<std::size_t>(padded_h) * padded_w - static_cast<std::size_t>(height) * width;
}

static inline BatchedTransposeParam heuristic_get_transpose_kernel(std::size_t data_size, uint32_t batch, uint32_t height, uint32_t width)
{
    /*
    * iterate from big tile size to small tile size, and try match ediv first
    * if every kernel is applicable, then will pick up the bigest one
    * if need extra padding in h/w (due to tile size), then will pick up kernel that waste the samllest.
    */

    const auto & kernel_list = get_transpose_kernel_list(data_size);
    BatchedTransposeParam best_kernel;
    std::size_t extra_padding_size =  std::numeric_limits<std::size_t>::max();
    
    for(auto it = kernel_list.rbegin(); it != kernel_list.rend(); it++){
        if(!transpose_kernel_is_valid(batch, height, width, &(*it)))
            continue;
        std::size_t current_padding_size = get_extra_padding_size(batch, height, width, &(*it));
        if(current_padding_size < extra_padding_size){
            extra_padding_size = current_padding_size;
            best_kernel = *it;
        }
    }

    assert(extra_padding_size != std::numeric_limits<std::size_t>::max());   // impossible
    return best_kernel;
}

BatchedTransposeSolution::BatchedTransposeSolution(const ExecutionContext & ctx, miopenDataType_t data_type_, uint32_t batch_, uint32_t height_, uint32_t width_)
: data_type(data_type_), batch(batch_), height(height_), width(width_)
{
    if(data_type == miopenInt8x4 || data_type == miopenDouble)
        MIOPEN_THROW("These data type are not supported");
    num_cu = ctx.GetStream().GetMaxComputeUnits();
    std::size_t data_size = get_data_size(data_type);
    kernel_param_heuristic = heuristic_get_transpose_kernel(data_size, batch, height, width);
}

solver::KernelInfo BatchedTransposeSolution::GetKernel() const {
    std::size_t block_size = BATCHED_TRANSPOSE_BLOCK_SIZE;
    std::size_t grid_size = num_cu * BATCHED_TRANSPOSE_OCCUPANCY;
    std::size_t data_size = get_data_size(data_type);

    std::string kernel_name = get_transpose_kernel_name(data_size, &kernel_param_heuristic);
    solver::KernelInfo kernel;
    kernel.kernel_file = "batched_transpose.cpp";
    kernel.kernel_name = kernel_name;
    kernel.g_wk.clear();
    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    MIOPEN_LOG_T("BatchedTransposeSolution use kernel: " + kernel_name);

    return kernel;
}

std::vector<OpKernelArg> BatchedTransposeSolution::GetKernelArg() const
{
    std::size_t grid_size = num_cu * BATCHED_TRANSPOSE_OCCUPANCY;
    uint32_t dim_h = (height + kernel_param_heuristic.tile_y - 1) / kernel_param_heuristic.tile_y;
    uint32_t dim_w = (width + kernel_param_heuristic.tile_x - 1) / kernel_param_heuristic.tile_x;
    uint32_t dim_total = batch * dim_h * dim_w;

    magic_div_u32_t magic_h = magic_div_u32_gen(dim_h);
    magic_div_u32_t magic_w = magic_div_u32_gen(dim_w);

    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(height);
    opArgs.emplace_back(width);
    opArgs.emplace_back(static_cast<uint32_t>(grid_size));
    opArgs.emplace_back(dim_total);
    opArgs.emplace_back(magic_h.magic);
    opArgs.emplace_back(magic_h.shift);
    opArgs.emplace_back(magic_w.magic);
    opArgs.emplace_back(magic_w.shift);

    return opArgs;
}

}
