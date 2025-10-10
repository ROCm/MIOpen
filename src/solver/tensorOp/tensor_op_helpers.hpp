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
#pragma once

#include <miopen/tensorOp/problem_description.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/datatype.hpp>

#include <tuple>

namespace miopen {

namespace solver {

namespace tensorOp {

inline void GetCommonParams(KernelBuildParameters& build_params,
                            const miopen::tensorOp::ProblemDescription& problem,
                            bool is64bSupported)
{
    miopenDataType_t data_type = problem.GetBTensorDesc().GetType();

    build_params.Define("MIOPEN_TYPE", miopen::GetDataType(data_type));

    switch(problem.GetTensorOp())
    {
    case 0: build_params.Define("MIOPEN_TENSOR_OP", "miopenAdd"); break;
    case 1: build_params.Define("MIOPEN_TENSOR_OP", "miopenMul"); break;
    case 2: build_params.Define("MIOPEN_TENSOR_OP", "miopenMin"); break;
    case 3: build_params.Define("MIOPEN_TENSOR_OP", "miopenMax"); break;
    }

    if(is64bSupported && problem.GetATensorDesc().AllDimsFitIntoInt())
    {
        build_params.Define("DIM_TYPE", "uint32_t");
    }
    else
    {
        build_params.Define("DIM_TYPE", "uint64_t");
    }
}

inline std::tuple<size_t, std::string> GetRDBLCKandREADTYPE(size_t len, miopenDataType_t type)
{
    const std::string data_type = GetDataType(type);
    size_t RD_BLCK              = (len % 4 == 0) ? 4 : (len % 2 == 0) ? 2 : 1;

    if(data_type == "half" && RD_BLCK == 4)
    {
        RD_BLCK = 2;
    }

    return std::make_tuple(RD_BLCK,
                           (RD_BLCK == 1) ? data_type : data_type + std::to_string(RD_BLCK));
}

inline std::tuple<size_t, std::string> GetRDBLCKandREADTYPEHIP(size_t len, miopenDataType_t type)
{
    if(type == miopenHalf)
    {
        return (len % 2 == 0) ? std::make_tuple(2U, "half2") : std::make_tuple(1U, "half");
    }
    const std::string data_type = GetDataType(type);
    size_t RD_BLCK              = (len % 4 == 0) ? 4 : (len % 2 == 0) ? 2 : 1;
    return std::make_tuple(RD_BLCK,
                           (RD_BLCK == 1) ? data_type : data_type + std::to_string(RD_BLCK));
}

inline std::tuple<int, int, unsigned int> GetBitmapAndWgInfo(const std::vector<size_t>& blens,
                                                             const std::vector<size_t>& clens)
{
    // first_not_one is incorrect if btensor size equal to 1
    auto first_not_one = std::find_if(blens.rbegin(), blens.rend(), [](int i) { return i != 1; });
    auto d             = std::distance(blens.begin(), first_not_one.base());

    // quick fix
    int num_wg = first_not_one != blens.rend()
                     ? static_cast<int>(*first_not_one == 0 ? 1 : *first_not_one)
                     : 1;

    int work_per_wg = std::accumulate(clens.begin() + d, clens.end(), 1, std::multiplies<int>());

    unsigned int bitmap = 0;
    // update bitmap for first_not_one
    bitmap |= (1 << (blens.size() - d));

    for(int i = (d - 2); i >= 0; i--)
    {
        if(blens[i] != 1)
        {
            bitmap |= (1 << (blens.size() - (i + 1)));
            num_wg *= blens[i];
        }
        else
        {
            work_per_wg *= clens[i];
        }
    }

    return std::make_tuple(num_wg, work_per_wg, bitmap);
}

inline bool IsBitmapLeadingOnes(unsigned int bitmap, int n_size, int first_not_one)
{
    bool leading_ones = true;
    for(int i = first_not_one; i >= 0; i--)
    {
        bool is_one = (bitmap & (1 << (n_size - 1 - i))) != 0u;
        leading_ones &= is_one;
    }
    return leading_ones;
}

inline std::tuple<int, int, int, unsigned int, size_t, size_t>
Get4dParams(const miopen::tensorOp::ProblemDescription& problem, bool is4dLite)
{
    const auto& bTensorDesc = problem.GetBTensorDesc();
    const auto& cTensorDesc = problem.GetCTensorDesc();

    const auto& blens = bTensorDesc.GetLengths();
    const auto& clens = cTensorDesc.GetLengths();

    auto dims = clens.size();

    // first_not_one is incorrect if btensor size equal to 1
    auto first_not_one = std::find_if(blens.rbegin(), blens.rend(), [](int i) { return i != 1; });
    auto d             = std::distance(blens.begin(), first_not_one.base());

    // quick fix
    int num_wg = first_not_one != blens.rend()
                     ? static_cast<int>(*first_not_one == 0 ? 1 : *first_not_one)
                     : 1;

    int work_per_wg = std::accumulate(clens.begin() + d, clens.end(), 1, std::multiplies<int>());

    unsigned int bitmap = 0;
    // update bitmap for first_not_one
    bitmap |= (1 << (blens.size() - d));

    for(int i = (d - 2); i >= 0; i--)
    {
        if(blens[i] != 1)
        {
            bitmap |= (1 << (blens.size() - (i + 1)));
            num_wg *= blens[i];
        }
        else
        {
            work_per_wg *= clens[i];
        }
    }

    // quick fix for btensor = <1, 1, 1, 1>
    if(bTensorDesc.GetElementSize() == 1)
        bitmap = 4;

    int incr_wg = 0;
    // Forward Convolution Bias specialization
    // for fwd-bias, bitmap looks like <0, 1, 0, 0>
    // Is the no. of work-groups and the work for each wg balanced?
    auto fwd_conv_bias = bitmap == (1 << 2) ? 1 : 0;
    // This block gives off indexing for 5d tensors, skipping
    if(fwd_conv_bias == 1 && dims < 5 && num_wg < 640 && work_per_wg > 256 && clens[0] > 0)
    { // 640 workgroups of size 256 needed to completely fill the GPU

        work_per_wg /= clens[0]; // c_n;
        num_wg *= clens[0];      // c_n;
        incr_wg = 1;
    }

    int num_wg_orig = num_wg;
    int max_num_wg  = 4096;
    num_wg          = num_wg > max_num_wg ? max_num_wg : num_wg;

    size_t local_threads = 256;

    bool leading_ones = IsBitmapLeadingOnes(bitmap, clens.size(), static_cast<int>(d - 2));

    if(leading_ones && work_per_wg < 64)
    {
        local_threads = 64;
    }

    // Special case for adding tensors in place
    size_t global_threads =
        (static_cast<int>(leading_ones) == 1 && (d - 1) == 3) ? num_wg : num_wg * local_threads;
    global_threads = (global_threads < local_threads) ? local_threads : global_threads;

    if(is4dLite)
    {
        size_t TENS_LEN = cTensorDesc.GetElementSize();
        size_t RD_BLCK  = (TENS_LEN % 4 == 0) ? 4 : (TENS_LEN % 2 == 0) ? 2 : 1;

        size_t total_work = std::max(TENS_LEN / RD_BLCK, size_t(1));
        size_t grp_sz     = (total_work + local_threads - 1) / local_threads;
        grp_sz            = std::min(size_t(max_num_wg), grp_sz);
        size_t glb_sz     = local_threads * grp_sz;

        global_threads = glb_sz;
    }

    return std::make_tuple(
        num_wg_orig, work_per_wg, incr_wg, bitmap, local_threads, global_threads);
}

} // namespace tensorOp

} // namespace solver

} // namespace miopen
