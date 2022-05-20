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
#ifndef GUARD_GENERAL_MIOPEN_TENSOR_REORDER_SOL_HPP
#define GUARD_GENERAL_MIOPEN_TENSOR_REORDER_SOL_HPP

#include <miopen/kernel_info.hpp>
#include <miopen/op_kernel_args.hpp>
#include <miopen/execution_context.hpp>
#include <cstdint>
#include <vector>

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

struct GenericReorderSolutionImpl
{
    GenericReorderSolutionImpl(miopenDataType_t data_type_,
                               uint32_t dim_0_,
                               uint32_t dim_1_,
                               uint32_t dim_2_,
                               uint32_t dim_3_,
                               uint32_t order_0_,
                               uint32_t order_1_,
                               uint32_t order_2_,
                               uint32_t order_3_);
    // TODO batched transpose API
    solver::KernelInfo GetKernelInfo() const;
    std::vector<OpKernelArg> GetKernelArg() const;
    std::string GetKernelFileName() const;
    std::string GetKernelName() const;
    bool IsSkippable() const;
    size_t GetOutputTensorSize() const;

    miopenDataType_t data_type;
    uint32_t dim_0;
    uint32_t dim_1;
    uint32_t dim_2;
    uint32_t dim_3;
    uint32_t order_0;
    uint32_t order_1;
    uint32_t order_2;
    uint32_t order_3;

    GeneralReorderParam kernel_param_heuristic;
};
} // namespace miopen
#endif
