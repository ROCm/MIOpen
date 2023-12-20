/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef CK_ASM_IMPLICITGEMM_HPP_
#define CK_ASM_IMPLICITGEMM_HPP_

#include <miopen/config.h>

#include <string>
#include <cmath>
#include <ostream>
#include <tuple>
#include <vector>
#include <limits>

/// W/A for issue 1979: igemm solver does not support group conv. See:
/// https://github.com/ROCm/MIOpen/issues/1979
#define WORKAROUND_ISSUE_1979 1

/// W/A for issue 2624: asm igemm fwd error when c <=4 and dilation_y > 1
/// https://github.com/ROCm/MIOpen/issues/2624
#define WORKAROUND_ISSUE_2624 1

namespace miopen {

namespace solver {

struct TunableImplicitGemmGTCDynamic_t
{
    std::string direction      = " ";
    miopenDataType_t precision = miopenFloat;
    int nxb                    = 0;
    int nxe                    = 0;

    int gemm_m_per_block = 0;
    int gemm_n_per_block = 0;
    int gemm_k_per_block = 0;

    int wave_tile_m   = 0;
    int wave_tile_n   = 0;
    int wave_tile_k   = 0;
    int wave_step_m   = 0;
    int wave_step_n   = 0;
    int wave_repeat_m = 0;
    int wave_repeat_n = 0;

    int tensor_a_thread_lengths[4]  = {0, 0, 0, 0};
    int tensor_a_cluster_lengths[4] = {0, 0, 0, 0};
    int tensor_b_thread_lengths[4]  = {0, 0, 0, 0};
    int tensor_b_cluster_lengths[4] = {0, 0, 0, 0};
    int gemm_k_global_split         = 0;

    int GetBlockSize() const
    {
        const auto WaveSize  = 64;
        const auto divisor_m = wave_tile_m * wave_step_m * wave_repeat_m;
        const auto divisor_n = wave_tile_n * wave_step_n * wave_repeat_n;
        assert(divisor_m != 0 && divisor_n != 0);
        return (gemm_m_per_block / divisor_m) * (gemm_n_per_block / divisor_n) * WaveSize;
    }

    std::string GetKernelName() const
    {
        std::ostringstream kernel_name;
        std::string kernel_precision = precision == miopenFloat ? "fp32" : "fp16";
        kernel_name << "igemm_" << direction << "_gtcx_nchw_" << kernel_precision << "_bx" << nxb
                    << "_ex" << nxe << "_bt" << gemm_m_per_block << "x" << gemm_n_per_block << "x"
                    << gemm_k_per_block << "_wt" << wave_tile_m << "x" << wave_tile_n << "x"
                    << wave_tile_k << "_ws" << wave_step_m << "x" << wave_step_n << "_wr"
                    << wave_repeat_m << "x" << wave_repeat_n << "_ta" << tensor_a_thread_lengths[0]
                    << "x" << tensor_a_thread_lengths[1] << "x" << tensor_a_thread_lengths[2] << "x"
                    << tensor_a_thread_lengths[3] << "_" << tensor_a_cluster_lengths[0] << "x"
                    << tensor_a_cluster_lengths[1] << "x" << tensor_a_cluster_lengths[2] << "x"
                    << tensor_a_cluster_lengths[3] << "_tb" << tensor_b_thread_lengths[0] << "x"
                    << tensor_b_thread_lengths[1] << "x" << tensor_b_thread_lengths[2] << "x"
                    << tensor_b_thread_lengths[3] << "_" << tensor_b_cluster_lengths[0] << "x"
                    << tensor_b_cluster_lengths[1] << "x" << tensor_b_cluster_lengths[2] << "x"
                    << tensor_b_cluster_lengths[3];
        if(this->gemm_k_global_split != 0)
            kernel_name << "_gkgs";

        return kernel_name.str();
    }
};

// calculate log2_gemm_k_global_splits
// with assumption that dimension_0, _1 will merge into a single dimension, and do split only along
// dimension_0
static inline size_t ComputeLog2GemmKGlobalSplitsWith2DMerge(size_t current_grid_size,
                                                             size_t max_grid_size,
                                                             size_t merge_dimension_0,
                                                             size_t merge_dimensoin_1,
                                                             size_t gemm_k_per_block,
                                                             size_t max_log2_splits)
{
    size_t log2_gemm_k_global_splits = 0;
    for(size_t gs = 0; gs < max_log2_splits; gs++)
    {
        if((current_grid_size << gs) > max_grid_size)
            break;

        if((merge_dimension_0 % (1 << gs)) != 0)
        {
            break;
        }

        if((merge_dimension_0 >> gs) * merge_dimensoin_1 % gemm_k_per_block != 0)
        {
            break;
        }
        log2_gemm_k_global_splits = gs;
    }
    return log2_gemm_k_global_splits;
}

// calculate gemm_k_global_splits
// with assumption that some dimensions will merge into a single dimension
static inline size_t ComputeGemmKGlobalSplitsWith2DMerge(size_t current_grid_size,
                                                         // size_t merge_dimension,
                                                         // size_t gemm_k_per_block,
                                                         size_t occupancy,
                                                         size_t num_cus)
{
    size_t gemm_k_global_splits = num_cus * occupancy / current_grid_size;
    // int gemm_k_per_wg = math::integer_divide_ceil(merge_dimension / gemm_k_global_splits);
    // gemm_k_per_wg = (gemm_k_per_wg + gemm_k_per_block - 1) / gemm_k_per_block * gemm_k_per_block;
    // gemm_k_global_splits = math::integer_divide_ceil(merge_dimension / gemm_k_per_wg);

    return gemm_k_global_splits;
}

static inline size_t
ComputeMatrixPadSize(size_t col, size_t col_per_block, size_t row, size_t row_per_block)
{
    size_t col_padded = ((col + col_per_block - 1) / col_per_block) * col_per_block;
    size_t row_padded = ((row + row_per_block - 1) / row_per_block) * row_per_block;
    size_t col_extra  = col_padded - col;
    size_t row_extra  = row_padded - row;

    return col_extra * row + row_extra * col + col_extra * row_extra;
}

static inline std::tuple<int, int, int> // m_per_block, n_per_block, k_per_block
HeuristicInitMacroTileNoPadGemmK(size_t gemm_m,
                                 size_t gemm_n,
                                 size_t gemm_k,
                                 const std::vector<std::tuple<int, int, int>>& tile_list)
{
    int m_per_block, n_per_block, k_per_block;
    bool found = false;

    // find exact divide
    for(const auto& tile : tile_list)
    {
        int mpb, npb, kpb;
        std::tie(mpb, npb, kpb) = tile;
        if(gemm_m % mpb == 0 && gemm_n % npb == 0 && gemm_k % kpb == 0)
        {
            m_per_block = mpb;
            n_per_block = npb;
            k_per_block = kpb;
            found       = true;
            break;
        }
    }

    if(!found)
    {
        size_t min_pad_pixel = std::numeric_limits<std::size_t>::max();
        int mpb_pad          = 0;
        int npb_pad          = 0;
        // first try gemm_m, gemm_n padding
        for(const auto& tile : tile_list)
        {
            int mpb, npb, kpb;
            std::tie(mpb, npb, kpb) = tile;
            if(gemm_k % kpb != 0)
                continue;
            size_t cur_pad_pixel = ComputeMatrixPadSize(gemm_m, mpb, gemm_k, kpb) +
                                   ComputeMatrixPadSize(gemm_n, npb, gemm_k, kpb) +
                                   ComputeMatrixPadSize(gemm_m, mpb, gemm_n, npb);
            if(cur_pad_pixel < min_pad_pixel)
            {
                min_pad_pixel = cur_pad_pixel;
                mpb_pad       = mpb;
                npb_pad       = npb;
            }
        }

        // second, we need find the max k_per_block among the same mpb/npb per block
        for(const auto& tile : tile_list)
        {
            int mpb, npb, kpb;
            std::tie(mpb, npb, kpb) = tile;
            if(mpb == mpb_pad && npb == npb_pad)
            {
                if(gemm_k % kpb == 0)
                {
                    m_per_block = mpb;
                    n_per_block = npb;
                    k_per_block = kpb;
                    found       = true;
                    break;
                }
            }
        }
    }

    if(found)
        return std::make_tuple(m_per_block, n_per_block, k_per_block);
    else
        return std::make_tuple(0, 0, 0);
}

// This is to support big tensor > 4G. Need to decide how many splits needed.
// Return the number of splits.
static inline int igemm_split_batch_size(const int hi,
                                         const int wi,
                                         const int ho,
                                         const int wo,
                                         const int n,
                                         const int k,
                                         const int c,
                                         const int data_byte)
{
    size_t image_size_input          = static_cast<size_t>(c) * hi * wi * data_byte;
    size_t image_size_output         = static_cast<size_t>(k) * ho * wo * data_byte;
    constexpr size_t max_tensor_size = 0xffffffffUL;

    size_t image_size = std::max(image_size_input, image_size_output);

    // When image size is larger than max tensor size, max batch applicable is 0, so 0 is returned.
    if(image_size >= max_tensor_size)
        return 0;

    // Round up splits: we must find the largest multiple of n, max_n, s.t.
    // max_n * image_size <= max_tensor_size
    size_t max_n = max_tensor_size / image_size;
    if(max_n > n)
    {
        max_n = n % max_n;
    }
    else if(max_n < n)
    {
        // find the smallest multiple m of n such that (n / m) * image_size <= max_tensor_size.
        // once m is known, max_n := (n / m)
        size_t m       = std::ceil(n / max_n); // m >= n * (image_size / max_tensor_size)
        size_t _sqrt_n = std::sqrt(n);
        while(n % max_n != 0)
        {
            if(n % m == 0)
            {
                max_n = n / m;
            }
            else
            {
                m += 1;
                if(m > _sqrt_n)
                {
                    // if m > sqrt_n, then there must exist u < sqrt_n s.t. u * m = sqrt_n, but
                    // such a u cannot exist since m is the smallest multiple of n. Thus, the
                    // search is over, and we know m = max_n
                    max_n = 1;
                }
            }
        }
    }

    return n / max_n;
}

template <int L, int H>
inline static bool IsLinear(const int v)
{
    static_assert(L <= H, "L <= H");
    return L <= v && v <= H;
}

template <int L, int H>
inline static bool NextLinear(int& v)
{
    assert((IsLinear<L, H>(v)));
    if(H == v)
    {
        v = L;
        return true;
    }
    ++v;
    return false;
}

} // namespace solver
} // namespace miopen
#endif
