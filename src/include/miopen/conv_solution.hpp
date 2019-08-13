/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_GUARD_MLOPEN_CONV_SOLUTION_HPP
#define MIOPEN_GUARD_MLOPEN_CONV_SOLUTION_HPP

#include <miopen/miopen.h>

#include <string>
#include <vector>

namespace miopen {
namespace solver {

/// Describes a kernel source and whatever information required in order
/// to build and run it (the former is unused for binary kernels).
struct KernelInfo
{
    std::string comp_options;
    std::vector<size_t> l_wk;
    std::vector<size_t> g_wk;
    std::string kernel_file;
    std::string kernel_name;
    friend std::ostream& operator<<(std::ostream& os, const KernelInfo& k);
};

/// Information required to build and run a kernel (or a set of kernels),
/// which is expected to perform computatons as per the problem config.
///
/// TODO: Currently best suits a subset of existing solvers,
/// namely some OpenCL-written forward direct convolutions.
/// Shall be refactored (possibly, to a class hierarchy).
struct ConvSolution
{
    /// \todo Use better name than construction_params.
    std::vector<KernelInfo> construction_params; // impl may consist of multiple kernels.
    miopenStatus_t status;
    std::string solver_id;

    size_t workspce_sz;
    int grp_tile1;       // total number ALUs per group
    int grp_tile0;       // total number ALUs per group
    int in_tile1;        // size of in-tile in local memory
    int in_tile0;        // size of in-tile in local memory
    int out_pix_tile1;   // # of generated pixels per output per wk-item  (ALU)
    int out_pix_tile0;   // # of generated pixels per output per wk-item  (ALU)
    int n_out_pix_tiles; // # output pixel tiles per wk-item (ALU)
    int n_in_data_tiles; // # of blocks of different inputs in LDS
    int n_stacks;        // # of diff stacks (part of batch).

    ConvSolution(miopenStatus_t status_ = miopenStatusSuccess)
        : status(status_),
          solver_id("<unknown>"),
          workspce_sz(0),
          grp_tile1(-1),
          grp_tile0(-1),
          in_tile1(-1),
          in_tile0(-1),
          out_pix_tile1(-1),
          out_pix_tile0(-1),
          n_out_pix_tiles(-1),
          n_in_data_tiles(-1),
          n_stacks(-1)
    {
    }

    inline bool Succeeded() const { return status == miopenStatusSuccess; }
};

std::ostream& operator<<(std::ostream& os, const ConvSolution& s);

} // namespace solver
} // namespace miopen

#endif
