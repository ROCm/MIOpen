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
#include <string>
#include <ostream>
namespace miopen {

namespace solver {

struct TunableImplicitGemmGTCDynamic_t
{
    std::string direction;
    std::string precision;
    int nxb;
    int nxe;

    int gemm_m_per_block;
    int gemm_n_per_block;
    int gemm_k_per_block;

    int wave_tile_m;
    int wave_tile_n;
    int wave_tile_k;
    int wave_step_m;
    int wave_step_n;
    int wave_repeat_m;
    int wave_repeat_n;

    int tensor_a_thread_lengths[4];
    int tensor_a_cluster_lengths[4];
    int tensor_b_thread_lengths[4];
    int tensor_b_cluster_lengths[4];
    int gemm_k_global_split;

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
        kernel_name << "igemm_" << direction << "_gtcx_nchw_" << precision << "_bx" << nxb << "_ex"
                    << nxe << "_bt" << gemm_m_per_block << "x" << gemm_n_per_block << "x"
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
        if(gemm_k_global_split != 0)
            kernel_name << "_gkgs";

        return kernel_name.str();
    }
};
} // namespace solver
} // namespace miopen
#endif
