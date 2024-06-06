/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <miopen/prelu/utils.hpp>

namespace miopen {

namespace solver {

namespace prelu {

KernelInfo make_hip_kernel(std::vector<size_t> localsize,
                           std::vector<size_t> gridsize,
                           std::string kernel_file,
                           std::string kernel_name,
                           KernelBuildParameters build_params)
{
    while(localsize.size() < 3)
        localsize.push_back(1);
    while(gridsize.size() < 3)
        gridsize.push_back(1);
    for(int i = 0; i < localsize.size(); ++i)
        gridsize[i] = AlignUp(gridsize[i], localsize[i]);
    return KernelInfo{
        build_params.GenerateFor(kbp::HIP{}), localsize, gridsize, kernel_file, kernel_name};
}

size_t get_reqd_work_item_cnt(const ExecutionContext& context, size_t local_size)
{
    // At least 4 WGs per one CU
    return static_cast<size_t>(local_size * context.GetStream().GetMaxComputeUnits() * 4);
}

size_t get_reqd_work_item_cnt(const Handle& handle, size_t local_size)
{
    // At least 4 WGs per one CU
    return static_cast<size_t>(local_size * handle.GetMaxComputeUnits() * 4);
}

size_t get_parallelism_size(size_t reqd_work_item_cnt, size_t output_numel, size_t reduce_size)
{
    size_t parallelism_size = 1ULL;
    while(parallelism_size * output_numel < reqd_work_item_cnt &&
          parallelism_size < std::sqrt(reduce_size))
    {
        parallelism_size *= 2ULL;
    }
    return parallelism_size;
}

bool is_parallelism(size_t reqd_work_item_cnt, size_t output_numel, size_t reduce_size)
{
    return !(output_numel > reqd_work_item_cnt) &&
           (output_numel * reduce_size > reqd_work_item_cnt);
}

} // namespace prelu
} // namespace solver
} // namespace miopen
