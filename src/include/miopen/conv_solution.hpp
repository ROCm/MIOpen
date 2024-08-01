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
#include <miopen/kernel_info.hpp>
#include <miopen/invoker.hpp>

#include <boost/optional.hpp>

#include <string>
#include <vector>
#include <ostream>

namespace miopen {

namespace solver {

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
    boost::optional<InvokerFactory> invoker_factory;

    size_t workspace_sz;
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
          invoker_factory(boost::none),
          workspace_sz(0),
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

void PrecompileSolutions(const Handle& h,
                         const std::vector<const ConvSolution*>& sols,
                         bool force_attach_binary = false);

} // namespace solver
} // namespace miopen

#endif
