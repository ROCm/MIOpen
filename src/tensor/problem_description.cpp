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

#include <miopen/tensor/problem_description.hpp>
#include <miopen/names.hpp>

namespace miopen {

namespace tensor {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;
    const auto tensor_dim = aTensorDesc.GetLengths().size();

    auto alens = aTensorDesc.GetLengths();
    auto blens = bTensorDesc.GetLengths();
    auto clens = cTensorDesc.GetLengths();

    size_t local_threads = 256;
    int max_num_wg       = 4096;

    ss << std::to_string(bTensorDesc.GetType()) << "-" << std::to_string(aTensorDesc.GetType())
       << "-" << std::to_string(tensorOp);

    if(tensor_dim == 1)
    {
        int num_wg            = std::clamp(clens[0] / local_threads, size_t(1), size_t(max_num_wg));
        num_wg                = num_wg > max_num_wg ? max_num_wg : num_wg;
        size_t global_threads = num_wg * local_threads;
        ss << "-" << std::to_string(global_threads) << "-" << std::to_string(local_threads);

        if(aTensorDesc.AllDimsFitIntoInt())
        {
            ss << "-32bit";
        }
        else
        {
            ss << "-64bit";
        }
    }
    else if(tensor_dim == 2)
    {
        local_threads = 32;
        int num_wg =
            std::clamp((clens[0] * clens[1]) / local_threads, size_t(1), size_t(max_num_wg));
        num_wg                = num_wg > max_num_wg ? max_num_wg : num_wg;
        size_t global_threads = num_wg * local_threads;
        ss << "-" << std::to_string(global_threads) << "-" << std::to_string(local_threads);
    }
    else if(tensor_dim == 3)
    {

        size_t RD_BLCK        = (clens[2] % 4 == 0) ? 4 : (clens[2] % 2 == 0) ? 2 : 1;
        size_t total_work     = std::max(clens[2] / RD_BLCK, size_t(1));
        size_t grp_sz         = (total_work + local_threads - 1) / local_threads;
        size_t local_threads2 = 64;
        size_t total_work2    = clens[1];
        size_t grp_sz2        = (total_work2 + local_threads2 - 1) / local_threads2;
        grp_sz2               = std::min(size_t(max_num_wg / grp_sz), grp_sz2);

        bool lite_applicable = grp_sz <= size_t(max_num_wg);

        bool is_lite = clens[0] == 1 && blens[0] == 1 && alens[0] == 1 &&
                       (blens[1] == clens[1] || blens[1] == 1) && blens[2] == clens[2];

        if(lite_applicable && is_lite)
        {
            ss << "-" << std::to_string(RD_BLCK) << "x" << std::to_string(local_threads) << "x"
               << std::to_string(grp_sz) << std::to_string(local_threads2)
               << std::to_string(grp_sz2);
        }
    }

    return NetworkConfig{ss.str()};
}

} // namespace tensor

} // namespace miopen
